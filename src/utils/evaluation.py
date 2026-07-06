import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import ops
from sklearn.metrics import confusion_matrix


def get_ignore_indices(tag_lookup):
    tags_to_ignore = [
        "[PAD]",
        "[START]",
        "[UNK]",
        ".",
        ",",
        ":",
        "''",
        "``",
        "-LRB-",
        "-RRB-",
        "#",
        "$",
    ]

    ignore_indices = []
    for tag in tags_to_ignore:
        idx = int(tag_lookup(tag).numpy())
        if idx not in ignore_indices:
            ignore_indices.append(idx)
    return ignore_indices


class MaskedAccuracy(keras.metrics.Metric):
    """
    Métrica de acurácia customizada para POS Tagging.
    Calcula a taxa de acerto ignorando classes como Padding, subwords e pontuações.
    """

    def __init__(self, name="masked_accuracy", ignore_classes=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if ignore_classes is None:
            ignore_classes = [0, -100]

        self.ignore_classes = ignore_classes
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = ops.argmax(y_pred, axis=-1)

        y_true = ops.cast(y_true, dtype=y_pred_classes.dtype)

        matches = ops.equal(y_true, y_pred_classes)
        matches = ops.cast(matches, dtype="float32")

        mask = ops.ones_like(y_true, dtype="bool")

        for ignore_id in self.ignore_classes:
            ignore_tensor = ops.cast(ignore_id, dtype=y_true.dtype)
            mask = ops.logical_and(mask, ops.not_equal(y_true, ignore_tensor))

        mask = ops.cast(mask, dtype="float32")

        self.correct.assign_add(ops.sum(matches * mask))
        self.total.assign_add(ops.sum(mask))

    def result(self):
        # OBS: + 1e-7 evita divisão por zero
        return ops.divide(self.correct, ops.add(self.total, 1e-7))

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def generate_confusion_matrix(
    y_true, y_pred, tag_lookup, ignore_classes=None, filepath="confusion_matrix.png"
):
    """
    Gera e salva o Heatmap da Matriz de Confusão para visualizar os erros do modelo.
    """
    if ignore_classes is None:
        ignore_classes = [0, -100]

    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    mask = ~np.isin(y_true_flat, ignore_classes)
    y_true_filtered = y_true_flat[mask]
    y_pred_filtered = y_pred_flat[mask]

    vocab = tag_lookup.get_vocabulary()

    unique_classes = np.unique(np.concatenate([y_true_filtered, y_pred_filtered]))
    labels_text = [vocab[int(idx)] for idx in unique_classes]

    cm = confusion_matrix(
        y_true_filtered, y_pred_filtered, labels=unique_classes, normalize="true"
    )

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels_text,
        yticklabels=labels_text,
    )

    plt.title("Matriz de Confusão - POS Tagging")
    plt.xlabel("Tag Prevista")
    plt.ylabel("Tag Real")
    plt.tight_layout()

    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Heatmap da Matriz de Confusão salvo em: {filepath}")
"""
evaluate.py
-----------
Calcula métricas para uma ou mais execuções salvas em results/*.json e
gera:
  - acurácia global por token
  - classification report (precision/recall/f1 por tag)
  - matriz de confusão (salva como PNG)
  - tabela comparativa entre modos (se mais de um results/*.json existir)

Uso:
  python src/evaluate.py --mode zero-shot
  python src/evaluate.py --compare zero-shot few-shot rag
"""
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
    Achata gold/pred em duas listas paralelas de tags, ALINHADAS por token.
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
