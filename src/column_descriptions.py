import json
from pathlib import Path


def load_column_descriptions(project_root: Path):
    path = project_root / "config" / "column_descriptions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def describe_column(column, descriptions):
    info = descriptions.get(column, {})
    zh_name = info.get("zh_name", column)
    description = info.get("description", "")
    return {
        "column": column,
        "display_name": f"{zh_name} ({column})" if zh_name != column else column,
        "zh_name": zh_name,
        "description": description,
    }
