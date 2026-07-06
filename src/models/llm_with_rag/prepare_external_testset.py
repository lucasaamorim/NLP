import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_ptb_tagged_file(path):
    """
        Parseia o formato 'token_TAG token_TAG ...' por linha.
        Usa rsplit('_', 1) em vez de split('_', 1) porque alguns tokens do
        PTB (ex. '-LRB-', '-RRB-') contêm hífens mas não underscores, então
        isso é seguro; mas separar pelo ÚLTIMO underscore é mais robusto
        caso apareça algum token com underscore embutido.
    """
    records = []
    n_malformed = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tokens, tags = [], []
            for piece in line.split(" "):
                if "_" not in piece:
                    n_malformed += 1
                    continue
                tok, tag = piece.rsplit("_", 1)
                tokens.append(tok)
                tags.append(tag)
            if tokens:
                records.append({"tokens": tokens, "tags": tags})

    if n_malformed:
        print(f"Aviso: {n_malformed} tokens malformados (sem '_') foram ignorados.")
    return records


def sentence_key(record):
    return " ".join(record["tokens"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="caminho do arquivo .txt no formato token_TAG")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random

    external_records = parse_ptb_tagged_file(args.input)
    print(f"Sentenças no arquivo externo: {len(external_records)}")

    # tagset check
    external_tags = sorted({t for r in external_records for t in r["tags"]})
    existing_tagset = json.load(open(DATA_DIR / "tagset.json", encoding="utf-8"))
    diff_new = sorted(set(external_tags) - set(existing_tagset))
    diff_missing = sorted(set(existing_tagset) - set(external_tags))
    print(f"Tags no arquivo externo: {len(external_tags)}")
    if diff_new:
        print(f"  ATENÇÃO: tags novas não previstas no tagset.json atual: {diff_new}")
    if diff_missing:
        print(f"  (tags do tagset.json que não aparecem no arquivo externo: {diff_missing})")

    # dedup / remover vazamento do train.json
    external_keys = {sentence_key(r) for r in external_records}
    train_path = DATA_DIR / "train.json"
    train_records = json.load(open(train_path, encoding="utf-8"))
    before = len(train_records)
    train_records = [r for r in train_records if sentence_key(r) not in external_keys]
    removed = before - len(train_records)
    print(f"Sentenças removidas do train.json por vazamento (apareciam no arquivo externo): {removed}")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    # amostra para o novo test.json
    rng = random.Random(args.seed)
    shuffled = external_records[:]
    rng.shuffle(shuffled)
    sample = shuffled[: args.sample_size]

    test_path = DATA_DIR / "test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Novo data/test.json salvo com {len(sample)} sentenças (amostradas do arquivo externo).")

    # guarda o arquivo externo completo já parseado, para referência/uso futuro
    full_path = DATA_DIR / "test22-24_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(external_records, f, ensure_ascii=False, indent=2)
    print(f"Arquivo externo completo (parseado) salvo em: {full_path}")


if __name__ == "__main__":
    main()
