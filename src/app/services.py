import json
from pathlib import Path

import joblib
import pandas as pd

from data_analysis import (
    analyze_dataframe,
    plot_numeric_distributions,
    plot_target_distribution,
    save_data_report,
)
from column_descriptions import load_column_descriptions
from data_loader import load_tabular_csv
from evaluator import evaluate_classifier, save_metrics
from model_comparison import (
    DEFAULT_MODELS,
    compare_models,
    comparison_dataframe,
    save_best_model,
    save_comparison,
    save_prediction_table,
)
from model_registry import latest_model_path, load_registry, register_model, timestamp_id
from reporter import build_report
from trainer import train_classifier


def load_config(project_root: Path):
    return json.loads((project_root / "config.json").read_text(encoding="utf-8"))


def save_uploaded_csv(uploaded_file, destination: Path):
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(uploaded_file.getvalue())
    return pd.read_csv(destination)


def update_training_config(config, csv_path, target_column, model_type):
    config["data"]["csv_path"] = csv_path
    config["data"]["target_column"] = target_column
    config["model"]["type"] = model_type
    return config


def resolve(project_root: Path, relative_path: str):
    return project_root / relative_path


def run_training(project_root: Path, config):
    csv_path = resolve(project_root, config["data"]["csv_path"])
    data, features, labels = load_tabular_csv(csv_path, config["data"]["target_column"])
    data_quality = analyze_dataframe(data, target_column=config["data"]["target_column"])
    save_data_report(data_quality, resolve(project_root, config["outputs"]["data_report_path"]))

    output_dir = project_root / "outputs"
    model_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    model_type = config["model"]["type"]
    pipeline, x_test, y_test = train_classifier(
        features,
        labels,
        model_type=model_type,
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

    prediction_table = x_test.copy()
    prediction_table["true_label"] = y_test.values
    prediction_table["predicted_label"] = predictions
    prediction_table.to_csv(resolve(project_root, config["outputs"]["prediction_path"]), index=False, encoding="utf-8")

    model_path = model_dir / f"model_{timestamp_id()}.joblib"
    joblib.dump(pipeline, model_path)
    registry_record = register_model(
        project_root,
        model_path=model_path,
        model_type=model_type,
        dataset_name=config["data"]["csv_path"],
        metrics=metrics,
    )

    report_path = resolve(project_root, config["outputs"]["report_path"])
    build_report(
        dataset_name=config["data"]["csv_path"],
        model_type=model_type,
        row_count=len(data),
        feature_count=features.shape[1],
        target_column=config["data"]["target_column"],
        metrics=metrics,
        report_path=report_path,
        data_quality=data_quality,
    )

    return {
        "metrics": metrics,
        "model_path": model_path,
        "registry_record": registry_record,
        "report_path": report_path,
        "confusion_matrix_path": resolve(project_root, config["outputs"]["confusion_matrix_path"]),
        "predictions_path": resolve(project_root, config["outputs"]["prediction_path"]),
    }


def run_prediction(project_root: Path, config, uploaded_file):
    input_path = project_root / "data" / "uploaded_predict.csv"
    data = save_uploaded_csv(uploaded_file, input_path)

    model_path = latest_model_path(project_root) or resolve(project_root, config["outputs"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError("No trained model was found. Train a model first.")

    pipeline = joblib.load(model_path)
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
    return output, output_path


def model_history(project_root: Path):
    return load_registry(project_root)


def column_descriptions(project_root: Path):
    return load_column_descriptions(project_root)


def run_data_analysis(project_root: Path, config, data, target_column=None):
    report = analyze_dataframe(data, target_column=target_column)
    save_data_report(report, resolve(project_root, config["outputs"]["data_report_path"]))

    target_plot = None
    if target_column:
        target_plot = plot_target_distribution(
            data,
            target_column,
            resolve(project_root, config["outputs"]["target_distribution_path"]),
        )

    feature_plots = plot_numeric_distributions(
        data,
        resolve(project_root, config["outputs"]["feature_distribution_dir"]),
    )

    return {
        "report": report,
        "report_path": resolve(project_root, config["outputs"]["data_report_path"]),
        "target_plot": target_plot,
        "feature_plots": feature_plots,
    }


def run_model_comparison(project_root: Path, config):
    csv_path = resolve(project_root, config["data"]["csv_path"])
    data, features, labels = load_tabular_csv(csv_path, config["data"]["target_column"])
    output_dir = project_root / "outputs"
    model_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    records, best_record, best_model, x_test, y_test, best_predictions = compare_models(
        features,
        labels,
        DEFAULT_MODELS,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    save_comparison(records, resolve(project_root, config["outputs"]["model_comparison_path"]))

    model_path = model_dir / f"model_{timestamp_id()}_{best_record['model_type']}_best.joblib"
    save_best_model(best_model, model_path)
    save_prediction_table(
        x_test,
        y_test,
        best_predictions,
        resolve(project_root, config["outputs"]["prediction_path"]),
    )

    registry_record = register_model(
        project_root,
        model_path=model_path,
        model_type=best_record["model_type"],
        dataset_name=config["data"]["csv_path"],
        metrics=best_record,
    )

    return {
        "records": records,
        "comparison_table": comparison_dataframe(records),
        "best_record": best_record,
        "model_path": model_path,
        "registry_record": registry_record,
        "comparison_path": resolve(project_root, config["outputs"]["model_comparison_path"]),
    }
