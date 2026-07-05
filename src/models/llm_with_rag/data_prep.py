import argparse
import json
import random
from pathlib import Path

import nltk

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ensure_corpus():
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank")


def clean_sentence(tagged_sent):
    """
        Remove tokens com tag -NONE-
    """
    return [(w, t) for w, t in tagged_sent if t != "-NONE-" and w.strip() != ""]


def load_clean_sentences():
    """
        Descarta sentenças que ficaram vazias ou triviais (<=1 token) após a limpeza
    """
    from nltk.corpus import treebank

    raw = treebank.tagged_sents()
    cleaned = [clean_sentence(s) for s in raw]
    
    cleaned = [s for s in cleaned if len(s) > 1]
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=250,
                         help="quantidade de sentenças no conjunto de teste")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_corpus()
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    sentences = load_clean_sentences()
    random.Random(args.seed).shuffle(sentences)

    test = sentences[: args.test_size]
    train = sentences[args.test_size:] 

    def to_records(sents):
        return [{"tokens": [w for w, _ in s], "tags": [t for _, t in s]} for s in sents]

    train_records = to_records(train)
    test_records = to_records(test)

    with open(DATA_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    all_tags = sorted({t for s in sentences for _, t in s})
    with open(DATA_DIR / "tagset.json", "w", encoding="utf-8") as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)

    print(f"Total de sentenças (limpas): {len(sentences)}")
    print(f"  -> treino/pool de exemplos: {len(train_records)}")
    print(f"  -> teste: {len(test_records)}")
    print(f"  -> tagset: {len(all_tags)} tags -> {all_tags}")


if __name__ == "__main__":
    main()
