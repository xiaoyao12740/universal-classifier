from html import escape
from pathlib import Path


def build_report(
    dataset_name,
    model_type,
    row_count,
    feature_count,
    target_column,
    metrics,
    report_path: Path,
    data_quality=None,
    model_comparison=None,
):
    accuracy = metrics["accuracy"]
    macro_precision = metrics["macro_precision"]
    macro_recall = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    data_quality_html = build_data_quality_html(data_quality)
    comparison_html = build_comparison_html(model_comparison)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Universal Classifier Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    .panel {{ border: 1px solid #d6dbe5; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    .metric {{ font-size: 28px; font-weight: bold; color: #2563eb; }}
    img {{ max-width: 680px; width: 100%; border: 1px solid #d6dbe5; border-radius: 8px; }}
    code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Universal Classifier Report</h1>
  <section class="panel">
    <h2>Dataset</h2>
    <p><strong>File:</strong> {escape(dataset_name)}</p>
    <p><strong>Rows:</strong> {row_count}</p>
    <p><strong>Features:</strong> {feature_count}</p>
    <p><strong>Target column:</strong> <code>{escape(target_column)}</code></p>
  </section>
  <section class="panel">
    <h2>Model</h2>
    <p>{escape(model_type)}</p>
  </section>
  {data_quality_html}
  {comparison_html}
  <section class="panel">
    <h2>Metrics</h2>
    <p>Accuracy</p>
    <div class="metric">{accuracy:.3f}</div>
    <p>Macro Precision</p>
    <div class="metric">{macro_precision:.3f}</div>
    <p>Macro Recall</p>
    <div class="metric">{macro_recall:.3f}</div>
    <p>Macro F1</p>
    <div class="metric">{macro_f1:.3f}</div>
  </section>
  <section class="panel">
    <h2>Confusion Matrix</h2>
    <img src="confusion_matrix.png" alt="Confusion matrix">
  </section>
  <section class="panel">
    <h2>How To Read This</h2>
    <p>Accuracy measures the overall correct prediction ratio. Macro precision measures how reliable positive predictions are across classes. Macro recall measures how many real samples are found across classes. Macro F1 balances precision and recall. The confusion matrix shows which classes are confused with each other.</p>
  </section>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def build_data_quality_html(data_quality):
    if not data_quality:
        return ""
    scale = data_quality.get("scale_evaluation", {})
    return f"""
  <section class="panel">
    <h2>Data Quality</h2>
    <p><strong>Missing values:</strong> {data_quality.get("missing_values_total", 0)}</p>
    <p><strong>Missing rate:</strong> {data_quality.get("missing_rate", 0):.2%}</p>
    <p><strong>Duplicate rows:</strong> {data_quality.get("duplicate_rows", 0)}</p>
    <p><strong>Duplicate rate:</strong> {data_quality.get("duplicate_rate", 0):.2%}</p>
    <p><strong>Scale:</strong> {escape(str(scale.get("size", "unknown")))}</p>
    <p>{escape(str(scale.get("advice", "")))}</p>
  </section>
"""


def build_comparison_html(model_comparison):
    if not model_comparison:
        return ""
    rows = []
    for item in model_comparison:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item['model_type']))}</td>"
            f"<td>{item['accuracy']:.3f}</td>"
            f"<td>{item['macro_precision']:.3f}</td>"
            f"<td>{item['macro_recall']:.3f}</td>"
            f"<td>{item['macro_f1']:.3f}</td>"
            "</tr>"
        )
    return f"""
  <section class="panel">
    <h2>Model Comparison</h2>
    <table border="1" cellspacing="0" cellpadding="8">
      <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Macro F1</th></tr>
      {''.join(rows)}
    </table>
  </section>
"""
