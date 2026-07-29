import json
from datetime import datetime
from pathlib import Path


def timestamp_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def registry_path(project_root: Path):
    return project_root / "models" / "model_registry.json"


def load_registry(project_root: Path):
    path = registry_path(project_root)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(project_root: Path, records):
    path = registry_path(project_root)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def register_model(project_root: Path, model_path: Path, model_type, dataset_name, metrics):
    records = load_registry(project_root)
    record = {
        "model_name": model_path.name,
        "model_path": model_path.relative_to(project_root).as_posix(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": model_type,
        "dataset": dataset_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }
    records.append(record)
    save_registry(project_root, records)
    return record


def latest_model_path(project_root: Path):
    records = load_registry(project_root)
    if not records:
        return None
    latest = sorted(records, key=lambda item: item.get("created_at", ""))[-1]
    return project_root / latest["model_path"]
