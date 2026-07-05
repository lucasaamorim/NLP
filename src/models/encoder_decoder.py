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


def build_encoder_decoder_model(
    max_length,
    vocab_size,
    num_tags,
    embed_dim_words,
    embed_dim_tags,
    embedding_matrix=None,
):
    # Encoder
    encoder_inputs = keras.Input(
        shape=(max_length,), dtype="int32", name="encoder_input"
    )
    enc_padding_mask = keras.ops.not_equal(encoder_inputs, 0)

    word_embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=max_length,
        embedding_dim=embed_dim_words,
        mask_zero=False,
        name="word_embedding",
    )
    word_embedding = word_embedding_layer(encoder_inputs)

    if embedding_matrix is not None:
        word_embedding_layer.token_embedding.set_weights([embedding_matrix])

    encoder_output = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=256, num_heads=4, dropout=0.2, name="encoder_block"
    )(word_embedding, padding_mask=enc_padding_mask)

    # Decoder
    decoder_inputs = keras.Input(
        shape=(max_length,), dtype="int32", name="decoder_input"
    )
    dec_padding_mask = keras.ops.not_equal(decoder_inputs, 0)

    tag_embedding = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=num_tags,
        sequence_length=max_length,
        embedding_dim=embed_dim_tags,
        mask_zero=False,
        name="tag_embedding",
    )(decoder_inputs)

    decoder_output = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=256, num_heads=4, dropout=0.2, name="decoder_block"
    )(
        decoder_sequence=tag_embedding,
        encoder_sequence=encoder_output,
        decoder_padding_mask=dec_padding_mask,
        encoder_padding_mask=enc_padding_mask,
    )

    outputs = layers.Dense(num_tags, activation="softmax", name="final_classifier")(
        decoder_output
    )
    return keras.Model(
        [encoder_inputs, decoder_inputs], outputs, name="Transformer_EncoderDecoder"
    )


if __name__ == "__main__":
    print("1. Carregando Dados...")
    data = load_tagging_data()
    X_train_raw, Y_train_raw = data["train"]["sentences"], data["train"]["tags"]
    X_val_raw, Y_val_raw = data["val"]["sentences"], data["val"]["tags"]

    print("2. Pré-processando e Tokenizando (com Teacher Forcing)...")
    vectorizer, max_len = tokenize_sentences(X_train_raw, 15000)

    tag_lookup, Y_train, Y_train_shifted = vectorize_tags(
        Y_train_raw, max_len, return_shifted=True
    )
    X_train = vectorizer([" ".join(s) for s in X_train_raw])

    X_val = vectorizer([" ".join(s) for s in X_val_raw])
    _, Y_val, Y_val_shifted = vectorize_tags(
        Y_val_raw, max_len, existing_lookup=tag_lookup, return_shifted=True
    )

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
    except FileNotFoundError:
        print("Arquivo GloVe não encontrado.")

    print("3. Construindo Arquitetura...")
    model = build_encoder_decoder_model(
        max_len,
        vocab_size,
        num_tags,
        embed_dim_words=embedding_dim,
        embed_dim_tags=64,
        embedding_matrix=embedding_matrix,
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
        x=[X_train, Y_train_shifted],
        y=Y_train,
        validation_data=([X_val, Y_val_shifted], Y_val),
        epochs=1,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping],
    )

    print("5. Avaliação Final...")
    Y_pred = np.argmax(model.predict([X_val, Y_val_shifted]), axis=-1)
    generate_confusion_matrix(
        y_true=Y_val,
        y_pred=Y_pred,
        tag_lookup=tag_lookup,
        ignore_classes=ignore_indices,
        filepath="enc_dec_confusion_matrix.png",
    )

    resultados = model.evaluate([X_val, Y_val_shifted], Y_val, batch_size=32, verbose=0)
    print("\n" + "=" * 30)
    print("      RESULTADOS FINAIS")
    print("=" * 30)
    for nome, valor in zip(model.metrics_names, resultados):
        if "acc" in nome.lower() or "metric" in nome.lower():
            print(f"{nome.upper()}: {valor * 100:.2f}%")
        else:
            print(f"{nome.upper()}: {valor:.4f}")
    print("=" * 30 + "\n")
