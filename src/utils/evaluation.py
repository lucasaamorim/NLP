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
