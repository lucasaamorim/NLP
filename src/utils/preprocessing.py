import keras
import numpy as np
import tensorflow as tf
from keras.layers import StringLookup, TextVectorization


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


def vectorize_tags(tags, max_length):
    """
    Mapeia as strings das tags para IDs inteiros e aplica o padding correspondente.
    """
    all_tags = [tag for sentence in tags for tag in sentence]

    tag_lookup = StringLookup(mask_token="<PAD>")
    tag_lookup.adapt(all_tags)

    Y_vectorized = []
    for sentence_tags in tags:
        tag_ids = tag_lookup(sentence_tags)

        pad_length = max_length - len(tag_ids)
        if pad_length > 0:
            padded_tags = keras.ops.pad(tag_ids, [[0, pad_length]])
        else:
            padded_tags = tag_ids

        Y_vectorized.append(padded_tags)

    return tag_lookup, np.array(Y_vectorized)


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
    OBS: É assumido que o diconário fornecido segue o mesmo formato do GloVe.
    """
    vocabulary = vectorizer.get_vocabulary()
    num_tokens = len(vocabulary)

    # Extrai todos os vetores do GloVe para descobrir o desvio padrão e média
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    # Inicializando a matriz com valores aleatórios seguindo uma distribuição normal baseada
    # na distribuição do GloVe.
    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_tokens, embedding_dim))

    # <PAD> não tem significado, logo é tudo zero
    embedding_matrix[0] = np.zeros(embedding_dim)

    hits = 0
    misses = 0

    for i, word in enumerate(vocabulary):
        if i == 0:
            continue  # Ignora o <PAD>, já zerado

        # O vetor pode não existir no dicionário
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # Encontrou: sobrescreve o valor aleatório com o vetor do GloVe
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            # Não encontrou: mantém a inicialização aleatória
            misses += 1

    print(
        f"Matriz de Embedding: {hits} tokens encontrados no GloVe | {misses} tokens inicializados aleatoriamente."
    )

    return embedding_matrix


if __name__ == "__main__":
    from data_loader import load_tagging_data

    print("Carregando dados para teste...")
    # Carregando os dados relevantes para os taggers (train, val, test)
    data = load_tagging_data()
    train_sentences = data["train"]["sentences"]
    train_tags = data["train"]["tags"]

    print(f"Total de sentenças de treino carregadas: {len(train_sentences)}")

    # 1. Tokenização de Palavras Inteiras
    print("\n--- Testando Tokenização de Palavras Inteiras ---")
    # Lembrando que o segundo parâmtro determina um limite no tamanho do vocabulário
    # 2k palavras é muito menor que o vocabulário do PTB, então é razoavelmente comum encontrar tokens UNK
    # Caso n seja passado um limite explícito (ou o limite seja baixo)
    vectorizer, max_len = tokenize_sentences(train_sentences, 100000)
    print(f"Tamanho máximo da sentença calculado: {max_len}")
    print(f"Tamanho do vocabulário de palavras: {vectorizer.vocabulary_size()}")

    # 2. Vetorização das Tags
    print("\n--- Testando Vetorização das Tags ---")
    tag_lookup, y_train = vectorize_tags(train_tags, max_len)
    print(f"Tamanho do vocabulário de tags: {tag_lookup.vocabulary_size()}")
    print(f"Formato da matriz de tags (Y_train): {y_train.shape}")

    print("PSA: <PAD> é o índice 0 e <UNK> é o índice 1")

    # 3. Exibindo resultados do primeiro elemento do batch
    if len(train_sentences) > 0:
        sample_idx = 0
        sample_text = " ".join(train_sentences[sample_idx]).lower()
        print(f"\nExemplo da sentença {sample_idx} normalizada: {sample_text}")
        print(f"Vetorização de X[{sample_idx}]:\n {vectorizer([sample_text])}")
        print(f"Vetorização de Y[{sample_idx}]:\n {y_train[sample_idx]}")

    from data_loader import load_pretrained_embeddings

    # Testando a criação da matriz de embeddings
    print("\n--- Testando Matriz de Embeddings ---")
    embedding_dim = 50
    glove_path = f"/home/lucasaamorim/Code/Github/NLP/data/glove.6B/glove.6B.{embedding_dim}d.txt"  # Colocar o caminho para os embeddings do GloVe aqui (coloquei absoluto pq relativo tava bugado)

    try:
        glove_index = load_pretrained_embeddings(glove_path)

        # Constrói a matriz passando o vectorizer do treino
        embedding_matrix = build_embedding_matrix(
            vectorizer, glove_index, embedding_dim
        )

        print(f"Dimensões da Matriz de Embedding final: {embedding_matrix.shape}")

    except FileNotFoundError:
        print(f"Arquivo GloVe não encontrado em {glove_path}. Pulei o teste da matriz.")
