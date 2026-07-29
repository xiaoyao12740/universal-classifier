# Experiment Record

## Goal

Upgrade the original Iris classifier into a reusable tabular classification platform.

## Main Upgrade

The original project used:

```python
load_iris()
```

This version uses:

```text
data/iris.csv
```

That means the model is no longer tied to one hard-coded dataset.

## Method

- Data format: CSV
- Target column: `label`
- Models: SVM, Random Forest, KNN
- Preprocessing:
  - numeric columns: StandardScaler
  - categorical columns: OneHotEncoder
- Model persistence: joblib
- Outputs:
  - JSON metrics
  - CSV predictions
  - confusion matrix image
  - HTML report

## Result

```text
Model type: svm
Accuracy: 0.947
Macro precision: 0.949
Macro recall: 0.949
Macro F1: 0.949
```

Prediction demo:

```text
5.1,3.5,1.4,0.2 -> setosa
6.0,2.9,4.5,1.5 -> versicolor
6.5,3.0,5.8,2.2 -> virginica
```

## What To Understand

- A real ML project usually separates data loading, preprocessing, model training, evaluation, and reporting.
- Training and prediction should be separated. You train once, save the model, then reuse the model for future prediction.
- Config files make the project reusable without editing source code.
- This project is for tabular classification, not image or NLP tasks.

## Stage 3 Web App

Added a local Streamlit interface:

- `app.py`: Web entry point
- `src/app/services.py`: thin service layer that calls existing ML modules

The Web app supports:

- CSV upload for training
- data preview
- target column selection
- model selection
- training metrics display
- confusion matrix display
- CSV upload for prediction
- prediction result table
- downloadable prediction CSV

The app does not duplicate the core training code. It calls the existing project modules.

## Stage 4 Model Registry

Added lightweight local MLOps features:

- each training run creates a versioned model file
- model metadata is stored in `models/model_registry.json`
- prediction uses the latest registered model by default
- the Streamlit app includes a model history tab

This avoids overwriting previous trained models and makes experiment tracking easier.

## Stage 4 Data Quality Analysis

Added a pre-training data analysis module:

- `src/data_analysis.py`
- Streamlit `Data Analysis` tab
- JSON output: `outputs/data_report.json`
- class distribution chart: `outputs/target_distribution.png`
- numeric feature distributions: `outputs/feature_distributions/`

Iris analysis result:

```text
rows: 150
columns: 5
missing values: 0
duplicate rows: 1
target classes: 3
```

This step matters because data quality often determines the upper bound of model performance.

## Stage 4 Auto Model Comparison

Added a lightweight AutoML comparison mode:

- trains SVM, RandomForest, and KNN on the same train/test split
- ranks models by Macro F1 and Accuracy
- saves the best model as a versioned model file
- registers the best model in `models/model_registry.json`
- writes the leaderboard to `outputs/model_comparison.json`

Iris comparison result:

```text
1. svm: accuracy=0.947, macro_f1=0.949
2. knn: accuracy=0.921, macro_f1=0.922
3. random_forest: accuracy=0.895, macro_f1=0.897
```

## Product Polish

Added product-facing improvements:

- `config/column_descriptions.json` for friendly column names and field descriptions
- data quality report now includes missing rate, duplicate rate, and scale evaluation
- prediction checks required feature columns before calling the model
- CLI and Streamlit show clearer prediction errors
- HTML report includes a data quality summary

Bad prediction CSV test:

```text
Error: Prediction CSV is missing required columns: petal width (cm)
```

## Final Audit Fixes

Code audit fixes:

- pinned project dependencies in `requirements.txt`
- changed latest model lookup to sort by `created_at`
- stored registry model paths in portable POSIX style
- removed silent `dropna()` from data loading
- added median imputation for numeric features
- added most-frequent imputation for categorical features
- kept target labels strict: missing target labels raise an error
- confirmed Auto Model Comparison saves the best model, not the last trained model
- confirmed Docker context excludes virtual environments, outputs, and model history artifacts
