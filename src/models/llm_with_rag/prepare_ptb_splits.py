import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_ptb_tagged_file(path):
    records = []
    n_malformed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
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
        print(f"  aviso: {n_malformed} tokens malformados ignorados em {path}")
    return records


def sentence_key(record):
    return " ".join(record["tokens"])


def report_tagset(name, records, reference=None):
    tags = sorted({t for r in records for t in r["tags"]})
    print(f"  {name}: {len(records)} sentenças, {len(tags)} tags únicas")
    if reference is not None:
        extra = sorted(set(tags) - set(reference))
        missing = sorted(set(reference) - set(tags))
        if extra:
            print(f"    ATENÇÃO - tags extras não vistas antes: {extra}")
        if missing:
            print(f"    (tags do outro split ausentes aqui: {missing})")
    return tags


def check_overlap(name_a, records_a, name_b, records_b):
    keys_a = {sentence_key(r) for r in records_a}
    keys_b = {sentence_key(r) for r in records_b}
    overlap = keys_a & keys_b
    if overlap:
        print(f"  ATENÇÃO: {len(overlap)} sentenças em comum entre {name_a} e {name_b}!")
    else:
        print(f"  OK: nenhuma sentença em comum entre {name_a} e {name_b}.")
    return overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="data/raw/train0-18.txt")
    parser.add_argument("--val", required=True, help="data/raw/val-19-21.txt")
    parser.add_argument("--test", required=True, help="data/raw/test22-24.txt")
    parser.add_argument("--test-sample-size", type=int, default=250)
    parser.add_argument("--val-sample-size", type=int, default=100,
                         help="amostra do val para tuning rápido/barato (val completo também é salvo)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    rng = random.Random(args.seed)

    print("Carregando splits...")
    train_records = parse_ptb_tagged_file(args.train)
    val_records = parse_ptb_tagged_file(args.val)
    test_records = parse_ptb_tagged_file(args.test)

    print("\nVerificação de tagset:")
    train_tags = report_tagset("train", train_records)
    report_tagset("val", val_records, reference=train_tags)
    report_tagset("test", test_records, reference=train_tags)

    print("\nVerificação de vazamento (overlap) entre splits:")
    check_overlap("train", train_records, "val", val_records)
    check_overlap("train", train_records, "test", test_records)
    check_overlap("val", val_records, "test", test_records)

    # --- salva splits completos, para referência/uso futuro ---
    with open(DATA_DIR / "train_full.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "val_full.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "test_full.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    # --- data/train.json: pool completo (usado por retrieval.py e pelos exemplos estáticos) ---
    with open(DATA_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    # --- data/val.json: amostra para tuning (mais barato que rodar o val inteiro) ---
    shuffled_val = val_records[:]
    rng.shuffle(shuffled_val)
    val_sample = shuffled_val[: args.val_sample_size]
    with open(DATA_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_sample, f, ensure_ascii=False, indent=2)

    # --- data/test.json: amostra para avaliação final ---
    shuffled_test = test_records[:]
    rng.shuffle(shuffled_test)
    test_sample = shuffled_test[: args.test_sample_size]
    with open(DATA_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_sample, f, ensure_ascii=False, indent=2)

    # --- tagset.json: união de todos os splits ---
    all_tags = sorted({t for recs in (train_records, val_records, test_records) for r in recs for t in r["tags"]})
    with open(DATA_DIR / "tagset.json", "w", encoding="utf-8") as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)

    print("\nResumo final:")
    print(f"  data/train.json -> {len(train_records)} sentenças (pool completo p/ few-shot/RAG)")
    print(f"  data/val.json   -> {len(val_sample)} sentenças (amostra p/ tuning; "
          f"{len(val_records)} disponíveis em val_full.json)")
    print(f"  data/test.json  -> {len(test_sample)} sentenças (amostra p/ avaliação final; "
          f"{len(test_records)} disponíveis em test_full.json)")
    print(f"  data/tagset.json -> {len(all_tags)} tags")


if __name__ == "__main__":
    main()
