import keras
import numpy as np
from keras import layers, ops

from ..utils.data_loader import load_parsing_data
from ..utils.evaluation import print_evalb_results, run_evalb
from ..utils.model_utils import (
    autoregressive_decode,
    compile_char_model,
    default_early_stopping,
    load_glove,
    make_token_embedding,
    make_transformer_decoder,
    make_transformer_encoder,
)
from ..utils.preprocessing import (
    build_char_vocabulary,
    decode_char_sequence,
    encode_char_sequences,
    tokenize_sentences,
)


def build_encoder_decoder_parser(
    max_word_len, max_tree_len,
    word_vocab_size, char_vocab_size,
    embed_dim_words=300, embed_dim_chars=128,
    embedding_matrix=None,
):
    encoder_inputs = keras.Input(
        shape=(max_word_len,), dtype="int32", name="encoder_input",
    )
    enc_mask = ops.not_equal(encoder_inputs, 0)

    enc_out = make_token_embedding(
        encoder_inputs, word_vocab_size, max_word_len, embed_dim_words,
        embedding_matrix=embedding_matrix, name="word_embedding",
    )
    enc_out = make_transformer_encoder(enc_out, enc_mask)

    decoder_inputs = keras.Input(
        shape=(max_tree_len,), dtype="int32", name="decoder_input",
    )
    dec_mask = ops.not_equal(decoder_inputs, 0)

    dec_out = make_token_embedding(
        decoder_inputs, char_vocab_size, max_tree_len, embed_dim_chars,
        name="char_embedding",
    )

    dec_out = make_transformer_decoder(
        dec_out, dec_mask,
        encoder_output=enc_out, encoder_mask=enc_mask,
    )

    outputs = layers.Dense(char_vocab_size, activation="softmax", name="output")(dec_out)
    return keras.Model(
        [encoder_inputs, decoder_inputs], outputs,
        name="Transformer_EncoderDecoder_Parser",
    )


if __name__ == "__main__":
    print("1. Carregando Dados...")
    data = load_parsing_data()
    X_train_raw = data["train"]["sentences"]
    X_val_raw = data["val"]["sentences"]
    X_test_raw = data["test"]["sentences"]
    train_tree_strs = data["train"]["tree_strings"]
    val_tree_strs = data["val"]["tree_strings"]
    test_tree_strs = data["test"]["tree_strings"]

    print(f"   Amostra (treino): {train_tree_strs[0][:120]}...")

    print("2. Tokenização de Palavras...")
    vectorizer, max_word_len = tokenize_sentences(X_train_raw, 15000)

    all_tree_lens = [len(s) + 1 for s in train_tree_strs]
    p95 = int(np.percentile(all_tree_lens, 95))
    p99 = int(np.percentile(all_tree_lens, 99))
    max_tree_len = 512

    n_full = len(X_train_raw)
    ok_indices = [
        i for i, (s, t) in enumerate(zip(X_train_raw, train_tree_strs))
        if len(s) <= 50 and len(t) + 1 <= max_tree_len
    ]
    X_train_raw = [X_train_raw[i] for i in ok_indices]
    train_tree_strs = [train_tree_strs[i] for i in ok_indices]
    max_word_len = min(50, max(len(s) for s in X_train_raw))
    print(f"   Max palavras: {max_word_len}, Max árvore: {max_tree_len} chars")
    print(f"   P95: {p95} chars, P99: {p99} chars")
    print(f"   Treino: {len(train_tree_strs)}/{n_full} cabem")

    print("3. Vocabulário de Caracteres...")
    char_lookup = build_char_vocabulary(train_tree_strs)
    char_vocab = char_lookup.get_vocabulary()
    print(f"   Vocab de caracteres: {len(char_vocab)} tokens")

    Y_train, Y_train_shifted = encode_char_sequences(
        train_tree_strs, char_lookup, max_tree_len, return_shifted=True,
    )
    Y_val, Y_val_shifted = encode_char_sequences(
        val_tree_strs, char_lookup, max_tree_len, return_shifted=True,
    )

    X_train = vectorizer([" ".join(s) for s in X_train_raw])
    X_val = vectorizer([" ".join(s) for s in X_val_raw])
    X_test = vectorizer([" ".join(s) for s in X_test_raw])

    start_id = int(char_lookup("[START]").numpy())
    end_id = int(char_lookup("[END]").numpy())

    embedding_dim = 300
    embedding_matrix = load_glove(vectorizer, embedding_dim)

    print("4. Construindo Modelo Encoder-Decoder...")
    model = build_encoder_decoder_parser(
        max_word_len, max_tree_len,
        vectorizer.vocabulary_size(), len(char_vocab),
        embed_dim_words=embedding_dim, embed_dim_chars=128,
        embedding_matrix=embedding_matrix,
    )
    model.summary()

    print("5. Compilando e Treinando...")
    compile_char_model(model)

    model.fit(
        x=[X_train, Y_train_shifted], y=Y_train,
        validation_data=([X_val, Y_val_shifted], Y_val),
        epochs=30, batch_size=32, verbose=1,
        callbacks=[default_early_stopping()],
    )

    BATCH_SIZE = 64

    print("6. Validação (EVALB)...")
    dec_seed = np.full((len(X_val), max_tree_len), 0, dtype=np.int32)
    dec_seed[:, 0] = start_id
    dec_out = autoregressive_decode(
        model, dec_seed, start_pos=1, end_id=end_id,
        encoder_inputs=X_val,
    )
    predictions_val = [
        decode_char_sequence(dec_out[i], char_vocab)
        for i in range(dec_out.shape[0])
    ]
    print_evalb_results(run_evalb(predictions_val, val_tree_strs))

    print("7. Teste (EVALB)...")
    dec_seed = np.full((len(X_test), max_tree_len), 0, dtype=np.int32)
    dec_seed[:, 0] = start_id
    dec_out = autoregressive_decode(
        model, dec_seed, start_pos=1, end_id=end_id,
        encoder_inputs=X_test,
    )
    predictions_test = [
        decode_char_sequence(dec_out[i], char_vocab)
        for i in range(dec_out.shape[0])
    ]
    print_evalb_results(run_evalb(predictions_test, test_tree_strs))
