import keras
import numpy as np
from keras import layers, ops

from ..utils.data_loader import load_tagging_data
from ..utils.evaluation import generate_confusion_matrix
from ..utils.model_utils import (
    compile_tagging_model,
    default_early_stopping,
    load_glove,
    make_token_embedding,
    make_transformer_encoder,
    print_metrics,
)
from ..utils.preprocessing import tokenize_sentences, vectorize_tags


def build_encoder_only_model(
    max_length, vocab_size, num_tags, embed_dim,
    num_encoder_blocks=2, embedding_matrix=None,
):
    inputs = keras.Input(shape=(max_length,), dtype="int32", name="sentence_input")
    padding_mask = ops.not_equal(inputs, 0)

    x = make_token_embedding(
        inputs, vocab_size, max_length, embed_dim,
        embedding_matrix=embedding_matrix,
    )

    x = make_transformer_encoder(
        x, padding_mask, num_blocks=num_encoder_blocks,
    )

    outputs = layers.Dense(num_tags, activation="softmax", name="tag_classifier")(x)
    return keras.Model(inputs, outputs, name="Transformer_EncoderOnly")


if __name__ == "__main__":
    print("1. Carregando os Dados do Penn Treebank...")
    data = load_tagging_data()
    X_train_raw = data["train"]["sentences"]
    Y_train_raw = data["train"]["tags"]
    X_val_raw = data["val"]["sentences"]
    Y_val_raw = data["val"]["tags"]
    X_test_raw = data["test"]["sentences"]
    Y_test_raw = data["test"]["tags"]

    print("2. Pré-processando e Tokenizando...")
    vectorizer, max_len = tokenize_sentences(X_train_raw, 15000)

    tag_lookup, Y_train = vectorize_tags(Y_train_raw, max_len)
    _, Y_val = vectorize_tags(Y_val_raw, max_len, existing_lookup=tag_lookup)
    _, Y_test = vectorize_tags(Y_test_raw, max_len, existing_lookup=tag_lookup)

    X_train = vectorizer([" ".join(s) for s in X_train_raw])
    X_val = vectorizer([" ".join(s) for s in X_val_raw])
    X_test = vectorizer([" ".join(s) for s in X_test_raw])

    embedding_dim = 300
    embedding_matrix = load_glove(vectorizer, embedding_dim)

    print("3. Construindo a Arquitetura Transformer...")
    model = build_encoder_only_model(
        max_len, vectorizer.vocabulary_size(), tag_lookup.vocabulary_size(),
        embed_dim=embedding_dim, embedding_matrix=embedding_matrix,
    )
    model.summary()

    print("4. Compilando o Modelo...")
    ignore_indices = compile_tagging_model(model, tag_lookup)

    print("5. Iniciando o Treinamento...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=30, batch_size=32, verbose=1,
        callbacks=[default_early_stopping()],
    )

    print("6. Avaliação (Validação)...")
    Y_val_pred = np.argmax(model.predict(X_val), axis=-1)
    generate_confusion_matrix(Y_val, Y_val_pred, tag_lookup,
                              ignore_indices, "encoder_val.png")
    resultados = model.evaluate(X_val, Y_val, batch_size=32, verbose=0)
    print_metrics(model.metrics_names, resultados, "VALIDAÇÃO")

    print("7. Avaliação (Teste)...")
    Y_test_pred = np.argmax(model.predict(X_test), axis=-1)
    generate_confusion_matrix(Y_test, Y_test_pred, tag_lookup,
                              ignore_indices, "encoder_test.png")
    resultados = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
    print_metrics(model.metrics_names, resultados, "TESTE")
