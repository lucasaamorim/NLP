import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_results(mode):
    path = RESULTS_DIR / f"{mode}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten(results):
    """
        Achata gold/pred em duas listas paralelas de tags, alinhadas por token.
        Sentenças com erro de parsing (status == error) ou com tamanho
        diferente do esperado são descartadas da métrica de acurácia fina,
        mas contadas separadamente como "falha de geração" — isso é
        importante reportar no trabalho, pois é uma limitação real do LLM,
        não um erro de tagging propriamente dito.
    """
    gold_flat, pred_flat = [], []
    n_malformed = 0

    for r in results:
        if r["status"] == "error" or len(r["pred"]) != len(r["gold"]):
            n_malformed += 1
            continue
        gold_flat.extend(r["gold"])
        pred_flat.extend(r["pred"])

    return gold_flat, pred_flat, n_malformed


def evaluate_mode(mode, plot_confusion=True):
    results = load_results(mode)
    gold_flat, pred_flat, n_malformed = flatten(results)

    n_total_sents = len(results)
    n_tokens = len(gold_flat)
    accuracy = sum(g == p for g, p in zip(gold_flat, pred_flat)) / n_tokens if n_tokens else 0.0

    print(f"\n{'=' * 60}")
    print(f"Modo: {mode}")
    print(f"{'=' * 60}")
    print(f"Sentenças avaliadas: {n_total_sents} "
          f"({n_malformed} descartadas por erro de geração/alinhamento)")
    print(f"Tokens avaliados: {n_tokens}")
    print(f"Acurácia (token-level): {accuracy:.4f}")
    print("\nClassification report (por tag):")
    print(classification_report(gold_flat, pred_flat, zero_division=0))

    if plot_confusion and n_tokens:
        _plot_confusion_matrix(gold_flat, pred_flat, mode)

    return {
        "mode": mode,
        "accuracy": accuracy,
        "n_tokens": n_tokens,
        "n_malformed": n_malformed,
        "n_total_sents": n_total_sents,
    }


def _plot_confusion_matrix(gold_flat, pred_flat, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = sorted(set(gold_flat) | set(pred_flat))
    cm = confusion_matrix(gold_flat, pred_flat, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.35), max(9, len(labels) * 0.35)))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real (gold)")
    ax.set_title(f"Matriz de confusão (normalizada) - {mode}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = RESULTS_DIR / f"confusion_matrix_{mode}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {out_path}")


def compare_modes(modes):
    rows = []
    for mode in modes:
        try:
            summary = evaluate_mode(mode, plot_confusion=True)
            rows.append(summary)
        except FileNotFoundError:
            print(f"Aviso: results/{mode}.json não encontrado, pulando.")

    print(f"\n{'=' * 60}")
    print("COMPARAÇÃO ENTRE ABORDAGENS")
    print(f"{'=' * 60}")
    print(f"{'Modo':<15}{'Acurácia':>12}{'Tokens':>12}{'Sent. c/ erro':>16}")
    for r in rows:
        print(f"{r['mode']:<15}{r['accuracy']:>12.4f}{r['n_tokens']:>12}{r['n_malformed']:>16}")

    with open(RESULTS_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="avalia um único modo")
    parser.add_argument("--compare", nargs="+", help="compara vários modos")
    args = parser.parse_args()

    if args.compare:
        compare_modes(args.compare)
    elif args.mode:
        evaluate_mode(args.mode)
    else:
        parser.error("Use --mode <nome> ou --compare <nome1> <nome2> ...")


if __name__ == "__main__":
    main()
