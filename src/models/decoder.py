import keras
import keras_nlp
import numpy as np
from keras import layers

from ..utils.data_loader import load_pretrained_embeddings, load_tagging_data
from ..utils.evaluation import (
    MaskedAccuracy,
    generate_confusion_matrix,
    get_ignore_indices,
)
from ..utils.preprocessing import (
    build_embedding_matrix,
    tokenize_sentences,
    vectorize_tags,
)


def build_decoder_only_model(
    max_length, vocab_size, num_tags, embed_dim, embedding_matrix=None
):
    inputs = keras.Input(shape=(max_length,), dtype="int32", name="sentence_input")
    padding_mask = keras.ops.not_equal(inputs, 0)

    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=max_length,
        embedding_dim=embed_dim,
        mask_zero=False,
        name="token_and_position_embedding",
    )
    x = embedding_layer(inputs)

    if embedding_matrix is not None:
        embedding_layer.token_embedding.set_weights([embedding_matrix])

    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=256, num_heads=4, dropout=0.2, name="decoder_block"
    )(decoder_sequence=x, decoder_padding_mask=padding_mask)

    outputs = layers.Dense(num_tags, activation="softmax", name="tag_classifier")(x)
    return keras.Model(inputs, outputs, name="Transformer_DecoderOnly")


if __name__ == "__main__":
    print("1. Carregando Dados...")
    data = load_tagging_data()
    X_train_raw, Y_train_raw = data["train"]["sentences"], data["train"]["tags"]
    X_val_raw, Y_val_raw = data["val"]["sentences"], data["val"]["tags"]

    print("2. Pré-processando e Tokenizando...")
    vectorizer, max_len = tokenize_sentences(X_train_raw, 15000)

    tag_lookup, Y_train = vectorize_tags(Y_train_raw, max_len)
    X_train = vectorizer([" ".join(s) for s in X_train_raw])

    X_val = vectorizer([" ".join(s) for s in X_val_raw])
    _, Y_val = vectorize_tags(Y_val_raw, max_len, existing_lookup=tag_lookup)

    vocab_size = vectorizer.vocabulary_size()
    num_tags = tag_lookup.vocabulary_size()
    embedding_dim = 300

    glove_path = (
        f"/content/drive/MyDrive/NLP/data/glove.6B/glove.6B.{embedding_dim}d.txt"
    )

    embedding_matrix = None
    try:
        glove_index = load_pretrained_embeddings(glove_path)
        embedding_matrix = build_embedding_matrix(
            vectorizer, glove_index, embedding_dim
        )
        print(f"Dimensões da Matriz de Embedding: {embedding_matrix.shape}")
    except FileNotFoundError:
        print(f"Arquivo GloVe não encontrado em {glove_path}.")

    print("3. Construindo Arquitetura...")
    model = build_decoder_only_model(
        max_len, vocab_size, num_tags, embedding_dim, embedding_matrix
    )
    model.summary()

    print("4. Compilando e Treinando...")
    ignore_indices = get_ignore_indices(tag_lookup)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[MaskedAccuracy(ignore_classes=ignore_indices, name="masked_acc")],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_masked_acc", patience=3, restore_best_weights=True, mode="max"
    )

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=30,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping],
    )

    print("5. Avaliação Final...")
    Y_pred = np.argmax(model.predict(X_val), axis=-1)
    generate_confusion_matrix(
        y_true=Y_val,
        y_pred=Y_pred,
        tag_lookup=tag_lookup,
        ignore_classes=ignore_indices,
        filepath="decoder_confusion_matrix.png",
    )

    resultados = model.evaluate(X_val, Y_val, batch_size=32, verbose=0)
    print("\n" + "=" * 30)
    print("      RESULTADOS FINAIS")
    print("=" * 30)
    for nome, valor in zip(model.metrics_names, resultados):
        if "acc" in nome.lower() or "metric" in nome.lower():
            print(f"{nome.upper()}: {valor * 100:.2f}%")
        else:
            print(f"{nome.upper()}: {valor:.4f}")
    print("=" * 30 + "\n")
