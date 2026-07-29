import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_dataframe(data, target_column=None):
    missing_by_column = data.isna().sum().to_dict()
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in data.columns if column not in numeric_columns]

    report = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "feature_count": int(len(data.columns) - 1) if target_column in data.columns else int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values_total": int(data.isna().sum().sum()),
        "missing_rate": float(data.isna().sum().sum() / max(data.size, 1)),
        "duplicate_rate": float(data.duplicated().sum() / max(len(data), 1)),
        "missing_values_by_column": {key: int(value) for key, value in missing_by_column.items()},
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "target_summary": None,
        "scale_evaluation": evaluate_scale(len(data), len(data.columns)),
    }

    if target_column and target_column in data.columns:
        counts = data[target_column].value_counts(dropna=False)
        total = len(data)
        report["target_summary"] = {
            "class_count": int(counts.size),
            "class_counts": {str(key): int(value) for key, value in counts.items()},
            "class_ratios": {str(key): float(value / total) for key, value in counts.items()},
        }

    return report


def evaluate_scale(row_count, column_count):
    if row_count < 50:
        size = "small"
        advice = "Dataset is very small. Results may be unstable."
    elif row_count < 1000:
        size = "medium"
        advice = "Dataset size is suitable for a lightweight baseline."
    else:
        size = "large"
        advice = "Dataset is large enough for more reliable model comparison."

    return {
        "size": size,
        "row_count": int(row_count),
        "column_count": int(column_count),
        "advice": advice,
    }


def save_data_report(report, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_target_distribution(data, target_column, output_path: Path):
    if not target_column or target_column not in data.columns:
        return None

    counts = data[target_column].value_counts()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color="#2563eb")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.title("Target Class Distribution")
    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_numeric_distributions(data, output_dir: Path, limit=4):
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()[:limit]
    paths = []
    output_dir.mkdir(exist_ok=True)

    for column in numeric_columns:
        output_path = output_dir / f"distribution_{safe_filename(column)}.png"
        plt.figure(figsize=(7, 4))
        data[column].dropna().hist(bins=20, color="#0f766e", edgecolor="white")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.title(f"Distribution: {column}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        paths.append(output_path)

    return paths


def safe_filename(value):
    allowed = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "feature"
