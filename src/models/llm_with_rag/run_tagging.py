import argparse
import json
import random
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemini_client import tag_tokens
from prompts import build_zero_shot_prompt, build_few_shot_prompt 
from retrieval import KNNRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

STATIC_EXAMPLE_INDICES = [0, 15, 42, 100, 233]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_static_examples(train_records, n):
    idxs = STATIC_EXAMPLE_INDICES[:n]
    return [train_records[i] for i in idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["zero-shot", "few-shot", "rag"])
    parser.add_argument("--split", default="test", choices=["val", "test"],
                         help="'val' para testar/ajustar decisões de design (ex. n_examples), "
                              "'test' para a avaliação final (default)")
    parser.add_argument("--n-examples", type=int, default=5,
                         help="número de exemplos no prompt (few-shot/rag)")
    parser.add_argument("--limit", type=int, default=None,
                         help="rodar só nas N primeiras sentenças (debug/custo)")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="pausa entre chamadas, em segundos (evitar rate limit)")
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "mistral"),
                        choices=["mistral"],
                        help="provedor de IA usado para o tagging")
    parser.add_argument("--model", default=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
                        help="modelo do provedor de IA")
    args = parser.parse_args()

    split_file = DATA_DIR / f"{args.split}.json"
    test_records = load_json(split_file)
    print(f"Usando split: {args.split} ({split_file.name}, {len(test_records)} sentenças disponíveis)")
    if args.limit:
        test_records = test_records[: args.limit]

    train_records = None
    retriever = None
    static_examples = None

    if args.mode == "few-shot":
        train_records = load_json(DATA_DIR / "train.json")
        static_examples = get_static_examples(train_records, args.n_examples)
    elif args.mode == "rag":
        retriever = KNNRetriever.from_json(DATA_DIR / "train.json")

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    suffix = "" if args.split == "test" else f"_{args.split}"
    out_path = RESULTS_DIR / f"{args.mode}{suffix}.json"

    results = []
    for i, record in enumerate(test_records, start=1):
        tokens = record["tokens"]
        gold = record["tags"]

        if args.mode == "zero-shot":
            prompt = build_zero_shot_prompt(tokens)
        elif args.mode == "few-shot":
            prompt = build_few_shot_prompt(tokens, static_examples)
        else:  # rag
            examples = retriever.top_k(tokens, k=args.n_examples)
            prompt = build_few_shot_prompt(tokens, examples)

        try:
            pred = tag_tokens(
                prompt,
                n_tokens=len(tokens),
                provider=args.provider,
                model=args.model,
            )
            status = "ok"
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(test_records)}] ERRO: {e}")
            pred = ["<PARSE_ERROR>"] * len(tokens)
            status = "error"

        results.append({"tokens": tokens, "gold": gold, "pred": pred, "status": status})
        print(f"[{i}/{len(test_records)}] {args.mode} - {status}")

        time.sleep(args.sleep)

        # salva incrementalmente: se a API cair no meio, você não perde tudo
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    n_errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nConcluído. {len(results)} sentenças processadas, {n_errors} com erro.")
    print(f"Resultados salvos em: {out_path}")


if __name__ == "__main__":
    main()
