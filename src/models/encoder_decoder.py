import keras
import numpy as np
from keras import layers, ops

from ..utils.data_loader import load_tagging_data
from ..utils.evaluation import generate_confusion_matrix, MaskedAccuracy
from ..utils.model_utils import (
    autoregressive_decode,
    compile_tagging_model,
    default_early_stopping,
    load_glove,
    make_token_embedding,
    make_transformer_decoder,
    make_transformer_encoder,
    print_metrics,
)
from ..utils.preprocessing import tokenize_sentences, vectorize_tags


def build_encoder_decoder_model(
    max_length, vocab_size, num_tags,
    embed_dim_words, embed_dim_tags,
    embedding_matrix=None,
):
    encoder_inputs = keras.Input(
        shape=(max_length,), dtype="int32", name="encoder_input",
    )
    enc_mask = ops.not_equal(encoder_inputs, 0)

    enc_out = make_token_embedding(
        encoder_inputs, vocab_size, max_length, embed_dim_words,
        embedding_matrix=embedding_matrix, name="word_embedding",
    )
    enc_out = make_transformer_encoder(enc_out, enc_mask)

    decoder_inputs = keras.Input(
        shape=(max_length,), dtype="int32", name="decoder_input",
    )
    dec_mask = ops.not_equal(decoder_inputs, 0)

    dec_out = make_token_embedding(
        decoder_inputs, num_tags, max_length, embed_dim_tags,
        name="tag_embedding",
    )

    dec_out = make_transformer_decoder(
        dec_out, dec_mask,
        encoder_output=enc_out, encoder_mask=enc_mask,
    )

    outputs = layers.Dense(num_tags, activation="softmax", name="final_classifier")(dec_out)
    return keras.Model(
        [encoder_inputs, decoder_inputs], outputs, name="Transformer_EncoderDecoder",
    )


if __name__ == "__main__":
    print("1. Carregando Dados...")
    data = load_tagging_data()
    X_train_raw = data["train"]["sentences"]
    Y_train_raw = data["train"]["tags"]
    X_val_raw = data["val"]["sentences"]
    Y_val_raw = data["val"]["tags"]
    X_test_raw = data["test"]["sentences"]
    Y_test_raw = data["test"]["tags"]

    print("2. Pré-processando e Tokenizando (com Teacher Forcing)...")
    vectorizer, max_len = tokenize_sentences(X_train_raw, 15000)

    tag_lookup, Y_train, Y_train_shifted = vectorize_tags(
        Y_train_raw, max_len, return_shifted=True,
    )
    _, Y_val, Y_val_shifted = vectorize_tags(
        Y_val_raw, max_len, existing_lookup=tag_lookup, return_shifted=True,
    )
    _, Y_test = vectorize_tags(
        Y_test_raw, max_len, existing_lookup=tag_lookup,
    )

    X_train = vectorizer([" ".join(s) for s in X_train_raw])
    X_val = vectorizer([" ".join(s) for s in X_val_raw])
    X_test = vectorizer([" ".join(s) for s in X_test_raw])

    start_id = int(tag_lookup("[START]").numpy())

    embedding_dim = 300
    embedding_matrix = load_glove(vectorizer, embedding_dim)

    print("3. Construindo Arquitetura...")
    model = build_encoder_decoder_model(
        max_len, vectorizer.vocabulary_size(), tag_lookup.vocabulary_size(),
        embed_dim_words=embedding_dim, embed_dim_tags=64,
        embedding_matrix=embedding_matrix,
    )
    model.summary()

    print("4. Compilando e Treinando...")
    ignore_indices = compile_tagging_model(model, tag_lookup)

    model.fit(
        x=[X_train, Y_train_shifted], y=Y_train,
        validation_data=([X_val, Y_val_shifted], Y_val),
        epochs=30, batch_size=32, verbose=1,
        callbacks=[default_early_stopping()],
    )

    print("5. Avaliação (Validação - Autoregressiva)...")
    dec_seed = np.zeros((len(X_val), max_len), dtype=np.int32)
    dec_seed[:, 0] = start_id
    Y_val_pred = autoregressive_decode(
        model, dec_seed, start_pos=1, end_id=None, encoder_inputs=X_val,
    )
    Y_val_pred = Y_val_pred[:, 1:]
    Y_val_pred = np.pad(Y_val_pred, ((0, 0), (0, 1)), constant_values=0)
    generate_confusion_matrix(Y_val, Y_val_pred, tag_lookup,
                              ignore_indices, "enc_dec_val.png")

    val_ar_metric = MaskedAccuracy(ignore_classes=ignore_indices)
    val_ar_metric.update_state(Y_val, Y_val_pred)
    print(f"  MASKED_ACC AUTOREGRESSIVO: {float(val_ar_metric.result()) * 100:.2f}%")

    resultados = model.evaluate(
        [X_val, Y_val_shifted], Y_val, batch_size=32, verbose=0,
    )
    print_metrics(model.metrics_names, resultados, "VALIDAÇÃO (Teacher Forcing)")

    print("6. Avaliação (Teste - Autoregressiva)...")
    dec_seed = np.zeros((len(X_test), max_len), dtype=np.int32)
    dec_seed[:, 0] = start_id
    Y_test_pred = autoregressive_decode(
        model, dec_seed, start_pos=1, end_id=None, encoder_inputs=X_test,
    )
    Y_test_pred = Y_test_pred[:, 1:]
    Y_test_pred = np.pad(Y_test_pred, ((0, 0), (0, 1)), constant_values=0)
    generate_confusion_matrix(Y_test, Y_test_pred, tag_lookup,
                              ignore_indices, "enc_dec_test.png")

    test_acc_metric = MaskedAccuracy(
        ignore_classes=ignore_indices, name="masked_acc",
    )
    test_acc_metric.update_state(Y_test, Y_test_pred)
    test_acc = float(test_acc_metric.result())
    print("\n" + "=" * 30)
    print("      RESULTADOS TESTE (Autoregressivo)")
    print("=" * 30)
    print(f"MASKED_ACC: {test_acc * 100:.2f}%")
    print("=" * 30 + "\n")
