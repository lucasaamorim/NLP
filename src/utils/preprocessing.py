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


def vectorize_tags(tags_lists, max_len, existing_lookup=None):
    # 1. Limpeza Rigorosa: Remove espaços em branco invisíveis de TODAS as tags
    cleaned_tags = [[str(tag).strip() for tag in sentence] for sentence in tags_lists]

    # 2. Cria o Vocabulário APENAS se for o conjunto de treino
    if existing_lookup is None:
        vocab = sorted(
            list(set([tag for sentence in cleaned_tags for tag in sentence]))
        )

        tag_lookup = keras.layers.StringLookup(
            vocabulary=vocab,
            mask_token="[PAD]",
            num_oov_indices=1,  # Desconhecidos vão para o índice 1
            name="tag_lookup",
        )
    else:
        tag_lookup = existing_lookup

    # 3. Padding Manual
    padded_tags = []
    for sentence in cleaned_tags:
        if len(sentence) > max_len:
            padded_tags.append(sentence[:max_len])
        else:
            padded_tags.append(sentence + ["[PAD]"] * (max_len - len(sentence)))

    # 4. Converte para array numérico de forma segura para a GPU
    Y = np.array(tag_lookup(padded_tags))

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
