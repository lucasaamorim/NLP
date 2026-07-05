import keras
import numpy as np
from decoder import build_decoder_only_model
from encoder import build_encoder_only_model
from encoder_decoder import build_encoder_decoder_model

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


def run_model(
    model_name,
    model,
    X_train,
    Y_train,
    Y_train_shifted,
    X_val,
    Y_val,
    Y_val_shifted,
    tag_lookup,
):
    print(f"\n{'=' * 50}")
    print(f"INICIANDO PIPELINE: {model_name}")
    print(f"{'=' * 50}\n")

    is_enc_dec = "EncoderDecoder" in model.name

    if is_enc_dec:
        train_x = [X_train, Y_train_shifted]
        val_x = [X_val, Y_val_shifted]
    else:
        train_x = X_train
        val_x = X_val

    print(f"[{model_name}] 1. Iniciando o Treinamento...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_masked_acc",
        patience=3,
        restore_best_weights=True,
        mode="max",
    )

    history = model.fit(
        x=train_x,
        y=Y_train,
        validation_data=(val_x, Y_val),
        epochs=30,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping],
    )

    print(f"\n[{model_name}] 2. Gerando Avaliação e Matriz de Confusão...")
    Y_pred_probs = model.predict(val_x)
    Y_pred = np.argmax(Y_pred_probs, axis=-1)

    ignore_indices = get_ignore_indices(tag_lookup)

    generate_confusion_matrix(
        y_true=Y_val,
        y_pred=Y_pred,
        tag_lookup=tag_lookup,
        ignore_classes=ignore_indices,
        filepath=f"{model_name.lower().replace('-', '_')}_confusion_matrix.png",
    )

    print(f"[{model_name}] 3. Avaliação Final do Modelo...")
    resultados = model.evaluate(val_x, Y_val, batch_size=32, verbose=0)
    nomes_metricas = model.metrics_names

    print("\n" + "=" * 35)
    print(f" RESULTADOS FINAIS: {model_name}")
    print("=" * 35)

    for nome, valor in zip(nomes_metricas, resultados):
        if "acc" in nome.lower() or "metric" in nome.lower():
            print(f"{nome.upper()}: {valor * 100:.2f}%")
        else:
            print(f"{nome.upper()}: {valor:.4f}")

    print("=" * 35 + "\n")
    return history


if __name__ == "__main__":
    print("1. Carregando os Dados do Penn Treebank...")
    data = load_tagging_data()
    X_train_raw, Y_train_raw = data["train"]["sentences"], data["train"]["tags"]
    X_val_raw, Y_val_raw = data["val"]["sentences"], data["val"]["tags"]

    print("2. Pré-processando e Tokenizando...")
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
        print(f"Dimensões da Matriz de Embedding: {embedding_matrix.shape}")
    except FileNotFoundError:
        print(f"Arquivo GloVe não encontrado em {glove_path}. Pulei o teste da matriz.")

    print("\n3. Construindo as Arquiteturas Transformer...")
    encoder_model = build_encoder_only_model(
        max_len,
        vocab_size,
        num_tags,
        embed_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
    )
    decoder_model = build_decoder_only_model(
        max_len, vocab_size, num_tags, embedding_dim, embedding_matrix
    )
    encoder_decoder_model = build_encoder_decoder_model(
        max_len,
        vocab_size,
        num_tags,
        embed_dim_words=embedding_dim,
        embed_dim_tags=64,
        embedding_matrix=embedding_matrix,
    )

    print("\n4. Compilando os Modelos...")
    compile_kwargs = {
        "optimizer": keras.optimizers.AdamW(learning_rate=0.001),
        "loss": keras.losses.SparseCategoricalCrossentropy(),
        "metrics": [
            MaskedAccuracy(
                ignore_classes=get_ignore_indices(tag_lookup), name="masked_acc"
            )
        ],
    }

    encoder_model.compile(**compile_kwargs)
    decoder_model.compile(**compile_kwargs)
    encoder_decoder_model.compile(**compile_kwargs)

    modelos_para_treinar = {
        "Encoder-Only": encoder_model,
        "Decoder-Only": decoder_model,
        "Encoder-Decoder": encoder_decoder_model,
    }

    historicos = {}

    for nome, modelo in modelos_para_treinar.items():
        historicos[nome] = run_model(
            model_name=nome,
            model=modelo,
            X_train=X_train,
            Y_train=Y_train,
            Y_train_shifted=Y_train_shifted,
            X_val=X_val,
            Y_val=Y_val,
            Y_val_shifted=Y_val_shifted,
            tag_lookup=tag_lookup,
        )
