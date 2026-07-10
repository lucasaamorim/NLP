import keras
import numpy as np
from keras import layers, ops

from ..utils.data_loader import load_parsing_data
from ..utils.evaluation import print_evalb_results, run_evalb
from ..utils.model_utils import (
    autoregressive_decode,
    compile_char_model,
    default_early_stopping,
    make_token_embedding,
    make_transformer_decoder,
)
from ..utils.preprocessing import (
    build_char_vocabulary,
    decode_char_sequence,
    encode_decoder_only_sequences,
    tokenize_sentences,
)


def build_decoder_only_parser(max_len, char_vocab_size, embed_dim=128):
    inputs = keras.Input(shape=(max_len,), dtype="int32", name="sequence_input")
    padding_mask = ops.not_equal(inputs, 0)

    x = make_token_embedding(
        inputs, char_vocab_size, max_len, embed_dim,
    )

    x = make_transformer_decoder(x, padding_mask)

    outputs = layers.Dense(char_vocab_size, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="Transformer_DecoderOnly_Parser")


if __name__ == "__main__":
    print("1. Carregando Dados...")
    data = load_parsing_data()
    X_train_raw = data["train"]["sentences"]
    X_val_raw = data["val"]["sentences"]
    X_test_raw = data["test"]["sentences"]
    train_tree_strs = data["train"]["tree_strings"]
    val_tree_strs = data["val"]["tree_strings"]
    test_tree_strs = data["test"]["tree_strings"]

    sent_strs_train = [" ".join(s) for s in X_train_raw]
    sent_strs_val = [" ".join(s) for s in X_val_raw]
    sent_strs_test = [" ".join(s) for s in X_test_raw]

    print(f"   Amostra: {sent_strs_train[0]}")
    print(f"   Árvore: {train_tree_strs[0][:120]}...")

    max_word_len = min(50, max(len(s) for s in X_train_raw))
    full_lens = [
        len(sent) + 2 + len(tree) + 1
        for sent, tree in zip(sent_strs_train, train_tree_strs)
    ]
    p95 = int(np.percentile(full_lens, 95))
    p99 = int(np.percentile(full_lens, 99))
    max_seq_len = 768

    n_full = len(X_train_raw)
    ok_indices = [
        i for i in range(n_full)
        if len(X_train_raw[i]) <= 50 and full_lens[i] <= max_seq_len
    ]
    X_train_raw = [X_train_raw[i] for i in ok_indices]
    train_tree_strs = [train_tree_strs[i] for i in ok_indices]
    sent_strs_train = [sent_strs_train[i] for i in ok_indices]

    print(f"   Max seq: {max_seq_len} chars (P95: {p95}, P99: {p99})")
    print(f"   Max palavras: {max_word_len}")
    print(f"   Treino: {len(X_train_raw)}/{n_full} cabem")

    print("2. Vocabulário de Caracteres...")
    char_lookup = build_char_vocabulary(
        train_tree_strs, sentence_strings=sent_strs_train,
    )
    char_vocab = char_lookup.get_vocabulary()
    print(f"   Vocab de caracteres: {len(char_vocab)} tokens")

    print("3. Codificando Sequências...")
    X_train, Y_train, mask_train = encode_decoder_only_sequences(
        X_train_raw, train_tree_strs, char_lookup, max_seq_len,
    )
    X_val, Y_val, mask_val = encode_decoder_only_sequences(
        X_val_raw, val_tree_strs, char_lookup, max_seq_len,
    )
    X_test, Y_test, mask_test = encode_decoder_only_sequences(
        X_test_raw, test_tree_strs, char_lookup, max_seq_len,
    )
    print(f"   X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"   % tokens árvore (train): {mask_train.mean()*100:.1f}%")

    sep_id = int(char_lookup("[SEP]").numpy())
    end_id = int(char_lookup("[END]").numpy())

    print("4. Construindo Modelo Decoder-Only...")
    model = build_decoder_only_parser(max_seq_len, len(char_vocab), embed_dim=128)
    model.summary()

    print("5. Compilando e Treinando...")
    compile_char_model(model)

    model.fit(
        x=X_train, y=Y_train, sample_weight=mask_train,
        validation_data=(X_val, Y_val, mask_val),
        epochs=30, batch_size=32, verbose=1,
        callbacks=[default_early_stopping()],
    )

    BATCH_SIZE = 64

    print("6. Validação (EVALB)...")
    predictions_val = []
    for start in range(0, len(X_val), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_val))
        batch = X_val[start:end]
        sep_pos = np.argmax(batch == sep_id, axis=1)
        prefix_len = int(sep_pos.max()) + 1
        prefix = np.asarray(batch[:, :prefix_len], dtype=np.int32)

        seq = np.pad(
            prefix, ((0, 0), (0, max_seq_len - prefix.shape[1])),
        )
        dec_out = autoregressive_decode(
            model, seq, start_pos=prefix_len, end_id=end_id,
        )
        for i in range(dec_out.shape[0]):
            sep_i = int(np.argmax(dec_out[i] == sep_id)) + 1
            predictions_val.append(
                decode_char_sequence(dec_out[i, sep_i:], char_vocab),
            )

    print_evalb_results(run_evalb(predictions_val, val_tree_strs))

    print("7. Teste (EVALB)...")
    predictions_test = []
    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))
        batch = X_test[start:end]
        sep_pos = np.argmax(batch == sep_id, axis=1)
        prefix_len = int(sep_pos.max()) + 1
        prefix = np.asarray(batch[:, :prefix_len], dtype=np.int32)

        seq = np.pad(
            prefix, ((0, 0), (0, max_seq_len - prefix.shape[1])),
        )
        dec_out = autoregressive_decode(
            model, seq, start_pos=prefix_len, end_id=end_id,
        )
        for i in range(dec_out.shape[0]):
            sep_i = int(np.argmax(dec_out[i] == sep_id)) + 1
            predictions_test.append(
                decode_char_sequence(dec_out[i, sep_i:], char_vocab),
            )

    print_evalb_results(run_evalb(predictions_test, test_tree_strs))
