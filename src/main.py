import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from data_loader import load_tabular_csv
from data_analysis import analyze_dataframe, save_data_report
from evaluator import evaluate_classifier, save_metrics
from model_registry import latest_model_path, register_model, timestamp_id
from reporter import build_report
from trainer import train_classifier


def load_config(project_root: Path, config_path: str):
    path = project_root / config_path
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(project_root: Path, relative_path: str):
    return project_root / relative_path


def train(config_path: str):
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root, config_path)

    csv_path = resolve(project_root, config["data"]["csv_path"])
    data, features, labels = load_tabular_csv(csv_path, config["data"]["target_column"])
    data_quality = analyze_dataframe(data, target_column=config["data"]["target_column"])

    output_dir = project_root / "outputs"
    model_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    model = config["model"]["type"]
    pipeline, x_test, y_test = train_classifier(
        features,
        labels,
        model_type=model,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    metrics, predictions = evaluate_classifier(
        pipeline,
        x_test,
        y_test,
        output_dir,
        resolve(project_root, config["outputs"]["confusion_matrix_path"]),
    )
    save_metrics(metrics, resolve(project_root, config["outputs"]["metrics_path"]))
    save_data_report(data_quality, resolve(project_root, config["outputs"]["data_report_path"]))

    prediction_table = x_test.copy()
    prediction_table["true_label"] = y_test.values
    prediction_table["predicted_label"] = predictions
    prediction_table.to_csv(resolve(project_root, config["outputs"]["prediction_path"]), index=False, encoding="utf-8")

    model_path = model_dir / f"model_{timestamp_id()}.joblib"
    joblib.dump(pipeline, model_path)
    register_model(
        project_root,
        model_path=model_path,
        model_type=model,
        dataset_name=config["data"]["csv_path"],
        metrics=metrics,
    )
    build_report(
        dataset_name=config["data"]["csv_path"],
        model_type=model,
        row_count=len(data),
        feature_count=features.shape[1],
        target_column=config["data"]["target_column"],
        metrics=metrics,
        report_path=resolve(project_root, config["outputs"]["report_path"]),
        data_quality=data_quality,
    )

    print(f"Model type: {model}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro precision: {metrics['macro_precision']:.3f}")
    print(f"Macro recall: {metrics['macro_recall']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved report to: {resolve(project_root, config['outputs']['report_path'])}")


def predict(config_path: str, input_csv: str):
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root, config_path)
    model_path = latest_model_path(project_root) or resolve(project_root, config["outputs"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run train first.")

    pipeline = joblib.load(model_path)
    input_path = resolve(project_root, input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")
    data = pd.read_csv(input_path)
    expected_columns = list(getattr(pipeline, "feature_names_in_", []))
    missing_columns = [column for column in expected_columns if column not in data.columns]
    if missing_columns:
        raise ValueError("Prediction CSV is missing required columns: " + ", ".join(missing_columns))

    predictions = pipeline.predict(data)

    output = data.copy()
    output["predicted_label"] = predictions
    output_path = resolve(project_root, config["outputs"]["new_prediction_path"])
    output_path.parent.mkdir(exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Universal tabular classification platform")
    parser.add_argument("--config", default="config.json")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input-csv", required=True)

    args = parser.parse_args()
    try:
        if args.command == "predict":
            predict(args.config, args.input_csv)
        else:
            train(args.config)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
