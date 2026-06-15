import os

import nltk

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def tagged_sentences_from_file(filename: str):
    """Extrai listas de sentenças e tags de um arquivo.

    @return Uma lista de palavras X e uma lista de tags Y
    """

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


def tree_from_file(filename: str):
    """Extrai a lista de sentenças e as árvores sintáticas de um arquivo.

    Cada linha do arquivo contém uma árvore no formato do Penn Treebank.

    @return Uma lista de palavras X e uma lista de árvores Y
    """

    path = os.path.join(DATA_DIR, filename)
    sentences = []
    trees = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
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
    """Carrega os Dados Relevantes para os taggers.

    @return 3 dicionários (train,val, test), onde cada um contém as entradas:
        - 'sentences': Lista de sentenças
        - 'tags': Lista de tags de cada sentença
    """

    X_train, Y_train = tagged_sentences_from_file("train0-18.txt")
    X_val, Y_val = tagged_sentences_from_file("val19-21.txt")
    X_test, Y_test = tagged_sentences_from_file("test22-24.txt")

    return {
        "train": {"sentences": X_train, "tags": Y_train},
        "val": {"sentences": X_val, "tags": Y_val},
        "test": {"sentences": X_test, "tags": Y_test},
    }


def load_parsing_data():
    """Carrega os dados para os parsers.

    @return 3 dicionários (train, val, test), onde cada um contém as entradas:
        - 'sentences': Lista de palavras
        - 'tree': Árvore sintática de cada sentença
    """

    X_train, trees_train = tree_from_file("train_constituency_0-18.txt")
    X_val, trees_val = tree_from_file("val_constituency_19-21.txt")
    X_test, trees_test = tree_from_file("test_constituency_22-24.txt")

    return {
        "train": {"sentences": X_train, "trees": trees_train},
        "val": {"sentences": X_val, "trees": trees_val},
        "test": {"sentences": X_test, "trees": trees_test},
    }


if __name__ == "__main__":
    data = load_parsing_data()
    for split in ("train", "val", "test"):
        print(
            f"{split}: {len(data[split]['sentences'])} sentences, {len(data[split]['trees'])} trees"
        )
