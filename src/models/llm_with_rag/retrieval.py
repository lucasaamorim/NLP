import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KNNRetriever:
    def __init__(self, train_records):
        self.train_records = train_records
        self.corpus = [" ".join(r["tokens"]) for r in train_records]
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        return cls(records)

    def top_k(self, query_tokens, k=4):
        query = " ".join(query_tokens)
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [self.train_records[i] for i in top_idx]


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "data"
    retriever = KNNRetriever.from_json(base / "train.json")

    with open(base / "test.json", encoding="utf-8") as f:
        test_records = json.load(f)

    query = test_records[0]["tokens"]
    print("Query:", query)
    for ex in retriever.top_k(query, k=3):
        print("  ->", ex["tokens"])
