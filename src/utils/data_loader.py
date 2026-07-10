import os

import nltk
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def tagged_sentences_from_file(filename: str):
    """Extrai listas de sentenças e tags de um arquivo."""

    path = os.path.join(DATA_DIR, filename)
    with open(path) as file:
        sentences = []
        for sentence in file:
            if sentence.strip():
                sentences.append(
                    [tuple(item.split("_")) for item in sentence.split(" ")]
                )

    X = []
    Y = []
    for sentence in sentences:
        words, tags = map(list, zip(*sentence))
        X.append(words)
        Y.append(tags)

    return X, Y


def _remove_none_nodes(tree):
    """Remove subárvores -NONE- (traços, elementos nulos) da árvore."""
    for i in range(len(tree) - 1, -1, -1):
        if isinstance(tree[i], nltk.Tree):
            if tree[i].label() == "-NONE-":
                tree.pop(i)
            else:
                _remove_none_nodes(tree[i])
    return tree


def _strip_functional_tags(tree):
    """Remove tags funcionais (NP-SBJ -> NP) da árvore."""
    label = tree.label()
    if not label.startswith("-"):
        for sep in ("-", "="):
            idx = label.find(sep)
            if idx != -1:
                tree.set_label(label[:idx])
                break
    for child in tree:
        if isinstance(child, nltk.Tree):
            _strip_functional_tags(child)


def _remove_empty_nodes(tree):
    for i in range(len(tree) - 1, -1, -1):
        if isinstance(tree[i], nltk.Tree):
            _remove_empty_nodes(tree[i])
            if len(tree[i]) == 0:
                tree.pop(i)


def tree_to_bracket_string(tree):
    """Converte nltk.Tree para string de brackets linearizada.

    Remove functional tags (NP-SBJ -> NP), empty nodes (leftovers from
    -NONE- removal), and produces a single-line S-expression.
    """
    tree = tree.copy(deep=True)
    tree.set_label("TOP")
    _strip_functional_tags(tree)
    _remove_empty_nodes(tree)
    return tree.pformat(margin=10 ** 9)


def tree_from_file(filename: str):
    """Extrai a lista de sentenças e as árvores sintáticas de um arquivo.

    Cada linha do arquivo contém uma árvore no formato do Penn Treebank.
    """

    path = os.path.join(DATA_DIR, filename)
    sentences = []
    trees = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Artefato da formatação do arquivo
            for tree_str in line.replace("))(TOP", "))\n(TOP").splitlines():
                tree_str = tree_str.strip()
                if not tree_str:
                    continue
                tree = nltk.Tree.fromstring(tree_str)
                tree = _remove_none_nodes(tree)
                sentences.append(tree.leaves())
                trees.append(tree)

    return sentences, trees


def load_tagging_data():
    """Carrega os dados relevantes para os taggers."""

    X_train, Y_train = tagged_sentences_from_file("train0-18.txt")
    X_val, Y_val = tagged_sentences_from_file("val19-21.txt")
    X_test, Y_test = tagged_sentences_from_file("test22-24.txt")

    return {
        "train": {"sentences": X_train, "tags": Y_train},
        "val": {"sentences": X_val, "tags": Y_val},
        "test": {"sentences": X_test, "tags": Y_test},
    }


def load_parsing_data():
    """Carrega os dados para os parsers."""

    X_train, trees_train = tree_from_file("train_constituency_0-18.txt")
    X_val, trees_val = tree_from_file("val_constituency_19-21.txt")
    X_test, trees_test = tree_from_file("test_constituency_22-24.txt")

    train_strs = [tree_to_bracket_string(t) for t in trees_train]
    val_strs = [tree_to_bracket_string(t) for t in trees_val]
    test_strs = [tree_to_bracket_string(t) for t in trees_test]

    return {
        "train": {
            "sentences": X_train,
            "trees": trees_train,
            "tree_strings": train_strs,
        },
        "val": {
            "sentences": X_val,
            "trees": trees_val,
            "tree_strings": val_strs,
        },
        "test": {
            "sentences": X_test,
            "trees": trees_test,
            "tree_strings": test_strs,
        },
    }


def load_pretrained_embeddings(filepath):
    """Carrega Embeddings pré treinados para um dicionário."""
    embeddings_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print(f"Embeddings em {filepath} carregados com sucesso\n")

    return embeddings_index


if __name__ == "__main__":
    data = load_parsing_data()
    for split in ("train", "val", "test"):
        print(
            f"{split}: {len(data[split]['sentences'])} sentences, {len(data[split]['trees'])} trees"
        )
