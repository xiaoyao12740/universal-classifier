import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from models import build_model
from preprocessing import build_preprocessor


DEFAULT_MODELS = ["svm", "random_forest", "knn"]


def compare_models(features, labels, model_types, test_size, random_state):
    stratify = labels if labels.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    records = []
    trained_models = {}
    predictions_by_model = {}

    for model_type in model_types:
        pipeline = make_pipeline(
            build_preprocessor(features),
            build_model(model_type),
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
        }
        records.append({"model_type": model_type, **metrics})
        trained_models[model_type] = pipeline
        predictions_by_model[model_type] = predictions

    records = sorted(records, key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)
    best_record = records[0]
    best_model = trained_models[best_record["model_type"]]
    best_predictions = predictions_by_model[best_record["model_type"]]
    return records, best_record, best_model, x_test, y_test, best_predictions


def save_comparison(records, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def save_best_model(model, model_path: Path):
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)


def save_prediction_table(x_test, y_test, predictions, output_path: Path):
    table = x_test.copy()
    table["true_label"] = y_test.values
    table["predicted_label"] = predictions
    output_path.parent.mkdir(exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8")
    return table


def comparison_dataframe(records):
    return pd.DataFrame(records)
