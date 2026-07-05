import keras
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization


def custom_standardization(input_string: str):
    return tf.strings.lower(input_string)


def tokenize_sentences(sentences, max_vocab_size=2000):
    """
    Tokenização de sentenças (somente embeddings de palavras inteiras).
    """
    text_data = [" ".join(sentence) for sentence in sentences]

    max_length = max(len(sentence) for sentence in sentences)

    vectorizer = TextVectorization(
        max_tokens=max_vocab_size,
        output_sequence_length=max_length,
        standardize=custom_standardization,
        split="whitespace",
    )

    vectorizer.adapt(text_data)

    return vectorizer, max_length


def vectorize_tags(tags_lists, max_len, existing_lookup=None, return_shifted=False):
    cleaned_tags = [[str(tag).strip() for tag in sentence] for sentence in tags_lists]

    if existing_lookup is None:
        vocab = sorted(
            list(set([tag for sentence in cleaned_tags for tag in sentence]))
        )

        # Injeta o token especial [START] no vocabulário se precisarmos fazer seq2seq
        if return_shifted and "[START]" not in vocab:
            vocab.insert(0, "[START]")

        tag_lookup = keras.layers.StringLookup(
            vocabulary=vocab, mask_token="[PAD]", num_oov_indices=1, name="tag_lookup"
        )
    else:
        tag_lookup = existing_lookup

    padded_tags = []
    shifted_tags = []

    for sentence in cleaned_tags:
        if len(sentence) > max_len:
            padded_tags.append(sentence[:max_len])
        else:
            padded_tags.append(sentence + ["[PAD]"] * (max_len - len(sentence)))

        if return_shifted:
            if len(sentence) > max_len - 1:
                shifted = ["[START]"] + sentence[: max_len - 1]
            else:
                shifted = (
                    ["[START]"] + sentence + ["[PAD]"] * (max_len - 1 - len(sentence))
                )
            shifted_tags.append(shifted)

    Y = np.array(tag_lookup(padded_tags))

    if return_shifted:
        Y_shifted = np.array(tag_lookup(shifted_tags))
        return tag_lookup, Y, Y_shifted

    return tag_lookup, Y


def tokenize_sentences_subwords(sentences, tags, tokenizer, tag_lookup, max_length):
    """
    Tokenização de sentenças e tags para modelos Transformer (subwords).
    Se uma palavra for dividida em múltiplos tokens, a tag correta é
    atribuída à primeira subword, e as demais recebem -100 para serem ignoradas na Loss.
    """
    X_subwords = []
    Y_aligned = []

    for sentence, sentence_tags in zip(sentences, tags):
        tokenized_sentence = []
        aligned_tags = []

        for word, tag in zip(sentence, sentence_tags):
            subwords = tokenizer.tokenize(word)
            tokenized_sentence.extend(subwords)

            tag_id = tag_lookup(tag)

            aligned_tags.append(tag_id)
            aligned_tags.extend([-100] * (len(subwords) - 1))

        if len(tokenized_sentence) > max_length:
            tokenized_sentence = tokenized_sentence[:max_length]
            aligned_tags = aligned_tags[:max_length]
        else:
            pad_len = max_length - len(tokenized_sentence)
            tokenized_sentence.extend(["<PAD>"] * pad_len)
            aligned_tags.extend([-100] * pad_len)

        X_subwords.append(tokenized_sentence)
        Y_aligned.append(aligned_tags)

    return X_subwords, np.array(Y_aligned)


def build_embedding_matrix(vectorizer, embeddings_index, embedding_dim):
    """
    Constrói a matriz de pesos para a EmbeddingLayer do Keras.
    Faz o cruzamento entre o vocabulário gerado pelo Keras e o dicionário fornecido.
    OBS: É assumido que o dicionário fornecido segue o mesmo formato do GloVe.
    """
    vocabulary = vectorizer.get_vocabulary()
    num_tokens = len(vocabulary)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_tokens, embedding_dim))

    # <PAD> não tem significado, logo é tudo zero
    embedding_matrix[0] = np.zeros(embedding_dim)

    hits = 0
    misses = 0

    for i, word in enumerate(vocabulary):
        if i == 0:
            continue  # ignora <PAD>

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    print(
        f"Matriz de Embedding: {hits} tokens encontrados no GloVe | {misses} tokens inicializados aleatoriamente."
    )

    return embedding_matrix


if __name__ == "__main__":
    from data_loader import load_tagging_data

    print("Carregando dados para teste...")
    data = load_tagging_data()
    train_sentences = data["train"]["sentences"]
    train_tags = data["train"]["tags"]

    print(f"Total de sentenças de treino carregadas: {len(train_sentences)}")

    print("\n--- Testando Tokenização de Palavras Inteiras ---")
    vectorizer, max_len = tokenize_sentences(train_sentences, 100000)
    print(f"Tamanho máximo da sentença calculado: {max_len}")
    print(f"Tamanho do vocabulário de palavras: {vectorizer.vocabulary_size()}")

    print("\n--- Testando Vetorização das Tags ---")
    tag_lookup, y_train = vectorize_tags(train_tags, max_len)
    print(f"Tamanho do vocabulário de tags: {tag_lookup.vocabulary_size()}")
    print(f"Formato da matriz de tags (Y_train): {y_train.shape}")

    print("PSA: <PAD> é o índice 0 e <UNK> é o índice 1")

    if len(train_sentences) > 0:
        sample_idx = 0
        sample_text = " ".join(train_sentences[sample_idx]).lower()
        print(f"\nExemplo da sentença {sample_idx} normalizada: {sample_text}")
        print(f"Vetorização de X[{sample_idx}]:\n {vectorizer([sample_text])}")
        print(f"Vetorização de Y[{sample_idx}]:\n {y_train[sample_idx]}")

    from data_loader import load_pretrained_embeddings

    print("\n--- Testando Matriz de Embeddings ---")
    embedding_dim = 50
    glove_path = f"/home/lucasaamorim/Code/Github/NLP/data/glove.6B/glove.6B.{embedding_dim}d.txt"

    try:
        glove_index = load_pretrained_embeddings(glove_path)
        embedding_matrix = build_embedding_matrix(
            vectorizer, glove_index, embedding_dim
        )
        print(f"Dimensões da Matriz de Embedding final: {embedding_matrix.shape}")
    except FileNotFoundError:
        print(f"Arquivo GloVe não encontrado em {glove_path}. Pulei o teste da matriz.")
