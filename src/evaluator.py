import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(model, x_test, y_test, output_dir: Path, confusion_matrix_path: Path):
    predictions = model.predict(x_test)
    labels = sorted(set(y_test) | set(predictions))

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=labels,
        cmap="Blues",
    )
    display.ax_.set_title("Universal Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=160)
    plt.close()

    return metrics, predictions


def save_metrics(metrics, metrics_path: Path):
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
