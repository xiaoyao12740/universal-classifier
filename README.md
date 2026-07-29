# Universal Classifier / Tabular Classification Platform

Universal Classifier is a lightweight AutoML-style platform for tabular classification. It supports CSV ingestion, data quality analysis, automated preprocessing, model training, model comparison, model registry, prediction, bilingual Streamlit UI, and Docker-based local deployment.

Universal Classifier 是一个面向表格分类任务的轻量级 AutoML 平台，支持 CSV 数据导入、数据质量分析、自动预处理、模型训练、模型比较、模型注册、预测推理、双语 Streamlit 页面和 Docker 本地部署。

## Positioning / 项目定位

This project packages a standard tabular classification workflow into a reusable local machine learning application. It is designed for small to medium CSV classification scenarios where users need fast model training, evaluation, prediction, and traceable model records without setting up a heavy ML platform.

本项目将表格分类流程封装成可复用的本地机器学习应用，适用于中小型 CSV 分类场景：快速训练、评估、预测，并保留可追踪的模型记录，而不需要部署复杂的机器学习平台。

## Highlights / 项目亮点

- CSV upload and target-column selection
- Data quality analysis before training
- Numeric and categorical preprocessing
- SVM, Random Forest, and KNN classifiers
- Automated model comparison leaderboard
- Train/predict workflow separation
- Versioned model registry
- Prediction schema validation
- Bilingual field description system
- HTML report generation
- Streamlit local web interface
- Docker configuration for local deployment

## Core Workflow / 核心流程

```text
CSV data
  -> data analysis
  -> preprocessing
  -> model training
  -> evaluation
  -> model registry
  -> prediction
  -> report and web display
```

## Outputs / 输出内容

- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/confusion_matrix.png`
- `outputs/report.html`
- `models/classifier.joblib`
- `models/model_registry.json`

## Project Structure / 项目结构

```text
01-universal-classifier/
  app.py
  config.json
  Dockerfile
  requirements.txt
  config/
    column_descriptions.json
  data/
    iris.csv
    new_samples.csv
  models/
    classifier.joblib
    model_registry.json
  outputs/
    metrics.json
    predictions.csv
    confusion_matrix.png
    report.html
  src/
    app/
      services.py
    column_descriptions.py
    data_analysis.py
    data_loader.py
    evaluator.py
    main.py
    model_comparison.py
    model_registry.py
    models.py
    preprocessing.py
    reporter.py
    trainer.py
```

## Run Locally / 本地运行

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train from CLI:

```powershell
.\.venv\Scripts\python.exe .\src\main.py train
```

Predict from CLI:

```powershell
.\.venv\Scripts\python.exe .\src\main.py predict
```

Start the Streamlit app:

```powershell
.\.venv\Scripts\streamlit.exe run .\app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Docker / 容器运行

```powershell
docker build -t universal-classifier .
docker run --rm -p 8501:8501 universal-classifier
```

## Model Options / 模型选项

- `svm`
- `random_forest`
- `knn`

The model can be changed through configuration or the Streamlit interface.

## Known Limits / 已知限制

- Designed for tabular classification, not image, audio, or generative tasks.
- Current implementation is single-user and local-first.
- Model registry is file-based, not database-backed.
- Advanced AutoML features such as hyperparameter search and cross-validation can be added later.

## Resume Summary / 简历描述

Built Universal Classifier, a local AutoML-style tabular classification platform with CSV ingestion, automated preprocessing, model comparison, model registry, train/predict separation, bilingual Streamlit UI, HTML reports, and Docker deployment support.
