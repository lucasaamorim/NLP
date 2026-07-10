import os

import keras_nlp
import numpy as np

from .data_loader import load_pretrained_embeddings
from .evaluation import MaskedAccuracy, get_ignore_indices
from .preprocessing import build_embedding_matrix


def make_token_embedding(inputs, vocab_size, seq_len, embed_dim,
                         embedding_matrix=None, name="token_embedding"):
    layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=seq_len,
        embedding_dim=embed_dim,
        mask_zero=False,
        name=name,
    )
    x = layer(inputs)
    if embedding_matrix is not None:
        layer.token_embedding.set_weights([embedding_matrix])
    return x


def make_transformer_encoder(x, padding_mask, intermediate_dim=256, num_heads=4,
                              dropout=0.2, num_blocks=1, name_prefix="encoder"):
    for i in range(num_blocks):
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=dropout,
            name=f"{name_prefix}_{i}",
        )(x, padding_mask=padding_mask)
    return x


def make_transformer_decoder(decoder_sequence, decoder_mask,
                              encoder_output=None, encoder_mask=None,
                              intermediate_dim=256, num_heads=4, dropout=0.2,
                              name="decoder"):
    return keras_nlp.layers.TransformerDecoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=dropout,
        name=name,
    )(
        decoder_sequence=decoder_sequence,
        encoder_sequence=encoder_output,
        decoder_padding_mask=decoder_mask,
        encoder_padding_mask=encoder_mask,
    )


def autoregressive_decode(model, seed, start_pos=1, end_id=None, encoder_inputs=None):
    seed = np.asarray(seed, dtype=np.int32)
    batch = seed.shape[0]
    max_len = seed.shape[1]
    finished = np.zeros(batch, dtype=bool)

    for t in range(start_pos, max_len):
        if encoder_inputs is not None:
            probs = model.predict([encoder_inputs, seed], verbose=0)
        else:
            probs = model.predict(seed, verbose=0)
        next_ids = np.argmax(probs[:, t - 1, :], axis=-1)
        next_ids = np.where(finished, 0, next_ids)
        seed[:, t] = next_ids
        if end_id is not None:
            finished |= next_ids == end_id
            if finished.all():
                break

    return seed


def load_glove(vectorizer, embedding_dim=300, glove_path=None):
    if glove_path is None:
        base = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "glove.6B"
        )
        glove_path = os.path.join(base, f"glove.6B.{embedding_dim}d.txt")

    try:
        glove_index = load_pretrained_embeddings(glove_path)
        matrix = build_embedding_matrix(vectorizer, glove_index, embedding_dim)
        print(f"Dimensões da Matriz de Embedding: {matrix.shape}")
        return matrix
    except FileNotFoundError:
        print(f"Arquivo GloVe não encontrado em {glove_path}.")
        return None


def compile_tagging_model(model, tag_lookup, lr=0.001):
    import keras
    ignore_indices = get_ignore_indices(tag_lookup)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[MaskedAccuracy(ignore_classes=ignore_indices, name="masked_acc")],
    )
    return ignore_indices


def compile_char_model(model, lr=0.001):
    import keras
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        metrics=[MaskedAccuracy(ignore_classes=[0], name="masked_acc")],
    )


def default_early_stopping():
    import keras
    return keras.callbacks.EarlyStopping(
        monitor="val_masked_acc",
        patience=3,
        restore_best_weights=True,
        mode="max",
    )


def print_metrics(metrics_names, metrics_values, title="RESULTADOS"):
    print("\n" + "=" * 30)
    print(f"      {title}")
    print("=" * 30)
    for nome, valor in zip(metrics_names, metrics_values):
        if "acc" in nome.lower() or "metric" in nome.lower():
            print(f"{nome.upper()}: {valor * 100:.2f}%")
        else:
            print(f"{nome.upper()}: {valor:.4f}")
    print("=" * 30 + "\n")
