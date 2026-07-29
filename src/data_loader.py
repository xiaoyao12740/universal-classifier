from pathlib import Path

import pandas as pd


def load_tabular_csv(csv_path: Path, target_column: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}")

    features = data.drop(columns=[target_column])
    labels = data[target_column]
    if labels.isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")
    return data, features, labels
