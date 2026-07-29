# 01 - Universal Classifier

This is the upgraded version of `01-iris-classifier`. It turns the original Iris-only demo into a reusable tabular classification project.

## What It Does

Given a CSV file and a target column, the project can:

- Load tabular data from CSV
- Split train/test data
- Train a selected classification model
- Save the trained model
- Generate metrics, predictions, a confusion matrix, and an HTML report

## Project Structure

```text
01-universal-classifier/
  config/
    column_descriptions.json
  data/
    iris.csv
  models/
    classifier.joblib
  outputs/
    metrics.json
    predictions.csv
    confusion_matrix.png
    report.html
  src/
    data_loader.py
    preprocessing.py
    models.py
    trainer.py
    evaluator.py
    reporter.py
    main.py
  config.json
  requirements.txt
  README.md
  EXPERIMENT.md
  Dockerfile
  Dockerfile.local
  .dockerignore
```

## Run

Run from the `ML-DL-Projects` root folder:

```powershell
.\.venv\Scripts\python.exe .\01-universal-classifier\src\main.py train
```

Example output:

```text
Model type: svm
Accuracy: 0.947
Macro precision: 0.949
Macro recall: 0.949
Macro F1: 0.949
```

The run generates:

- `models/classifier.joblib`
- `models/model_YYYYMMDD_HHMMSS.joblib`
- `models/model_registry.json`
- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/confusion_matrix.png`
- `outputs/report.html`

## Change Model

Edit `config.json`:

```json
{
  "model": {
    "type": "svm"
  }
}
```

Supported model types:

- `svm`
- `random_forest`
- `knn`

## Change Dataset

Put a new CSV file in `data/`, then edit:

```json
{
  "data": {
    "csv_path": "data/your_file.csv",
    "target_column": "your_label_column"
  }
}
```

This project is designed for tabular classification, such as flower classification, risk labels, medical indicator categories, or customer segmentation labels.

It is not designed for image classification or text generation. Those are handled by later projects.

## Predict With Saved Model

After training, prepare a CSV file with the same feature columns but without the target column. Then run:

```powershell
.\.venv\Scripts\python.exe .\01-universal-classifier\src\main.py predict --input-csv data\new_samples.csv
```

Predictions are saved to:

```text
outputs/new_predictions.csv
```

Example prediction result:

```text
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),predicted_label
5.1,3.5,1.4,0.2,setosa
6.0,2.9,4.5,1.5,versicolor
6.5,3.0,5.8,2.2,virginica
```

## Why This Matters

The project now has two separate modes:

- `train`: learn from labeled CSV data, save a versioned model, and register model metadata.
- `predict`: load the latest registered model and predict labels for new CSV data.

This is closer to real machine learning work: models are trained periodically, then reused for prediction instead of retraining every time new samples arrive.

## Local Web App

This project also provides a Streamlit app. It keeps the machine learning logic in `src/` and uses `app.py` only as the local Web interface.

Start the app from the `01-universal-classifier` folder:

```powershell
cd D:\360MoveData\Users\11142\Desktop\ML-DL-Projects\01-universal-classifier
..\.venv\Scripts\python.exe -m streamlit run app.py
```

If your shell does not accept the path above, run from the `ML-DL-Projects` root folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\01-universal-classifier\app.py
```

The app has four tabs:

- `Data Analysis`: upload a CSV and inspect data quality before training.
- `Train`: upload a CSV, preview data, choose target column, choose model, train, and view metrics.
- `Predict`: upload a new CSV, load the latest registered model, predict labels, and download `new_predictions.csv`.
- `Model History`: review saved model versions, algorithms, datasets, and metrics.

The first version is intentionally local-only. It does not include login, database, Docker, cloud deployment, or deep learning models.

## Model Registry

Each training run creates a new model file:

```text
models/model_YYYYMMDD_HHMMSS.joblib
```

The registry file records model metadata:

```text
models/model_registry.json
```

It includes model name, creation time, algorithm, dataset path, accuracy, precision, recall, and macro F1.

Current scope: this is a single-user local project. `model_registry.json` is not designed for concurrent multi-user writes.

The latest model is selected by `created_at`, not by raw JSON order.

## Data Analysis Report

The data analysis tab generates:

- row count
- column count
- feature count
- missing value count
- duplicate row count
- missing rate
- duplicate rate
- data scale evaluation
- column data types
- optional column display names and descriptions from `config/column_descriptions.json`
- target class counts and ratios
- target class distribution chart
- numeric feature distribution charts

The JSON report is saved to:

```text
outputs/data_report.json
```

## Auto Model Comparison

The training tab also supports an automatic comparison mode. It trains:

- SVM
- RandomForest
- KNN

The app ranks models by Macro F1 and Accuracy, saves the best model, and registers it in `models/model_registry.json`.

Comparison results are saved to:

```text
outputs/model_comparison.json
```

Iris example:

```text
Best model: svm
Accuracy: 0.947
Macro F1: 0.949
```

## Column Descriptions

Optional field descriptions live in:

```text
config/column_descriptions.json
```

They let the UI show a friendly display name and description while keeping the original CSV column names unchanged.

## Prediction Field Check

Before prediction, the app checks whether the uploaded CSV contains all feature columns required by the trained model. If a column is missing, the user sees a clear message such as:

```text
Prediction CSV is missing required columns: petal width (cm)
```

## Missing Value Strategy

The loader does not silently drop rows with missing feature values. Missing values are handled in preprocessing:

- numeric features: median imputation
- categorical features: most frequent value imputation

Rows are rejected only when the target column itself contains missing labels.

## Docker

Standard build:

```powershell
docker build -t universal-classifier:latest .
docker run --rm -p 8502:8501 universal-classifier:latest
```

Open:

```text
http://localhost:8502
```
