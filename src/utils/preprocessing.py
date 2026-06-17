from typing import Any

import keras
from keras.layers import TextVectorization


def tokenize_sentences(sentences):
    """Tokenização de sentencas (somente embeddings de palavras inteiras)"""
    max_length = max((len(sentence) for sentence in sentences))
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=max_length)

    vectorizer.adapt(sentences)
    vocab = vectorizer.get_vocabulary()
    word_idx = dict(zip(vocab, range(len(vocab))))

# TODO: Ver como fazer isso (e se dá pra juntar tokenização para embeddings de palavras inteiras e subwords numa mesma função)
def tokenize_sentences_subwords(sentences):
