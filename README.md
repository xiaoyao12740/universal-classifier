# Universal Classifier

Universal Classifier is a lightweight AutoML-style platform for tabular classification. It supports CSV ingestion, data quality analysis, automated preprocessing, model training, model comparison, model registry, prediction, Streamlit interaction, HTML reporting, and Docker-based local deployment.

Universal Classifier 是一个面向表格分类任务的轻量级 AutoML 风格平台。它支持 CSV 数据导入、数据质量分析、自动预处理、模型训练、模型对比、模型注册、预测推理、Streamlit 交互页面、HTML 报告生成和 Docker 本地部署。

## Positioning

This project packages a standard tabular classification workflow into a reusable local machine learning application. It is designed for small to medium CSV classification scenarios where users need fast model training, evaluation, prediction, and traceable model records without setting up a heavy ML platform.

本项目将标准表格分类流程封装成可复用的本地机器学习应用，适合中小型 CSV 分类场景：用户可以快速完成训练、评估、预测，并保留可追踪的模型记录，而不需要搭建复杂的机器学习平台。

It starts from the Iris dataset as a clear demonstration case, but the structure is intentionally generalized so the workflow can be adapted to other tabular classification datasets.

项目使用 Iris 数据集作为清晰的演示案例，但整体结构刻意做成通用形式，便于迁移到其他表格分类数据集。

## Highlights

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

中文概览：

- 支持 CSV 上传和目标列选择
- 训练前进行数据质量分析
- 同时处理数值特征和类别特征
- 内置 SVM、Random Forest、KNN 分类器
- 自动生成模型对比排行榜
- 分离训练流程和预测流程
- 提供带版本记录的模型注册表
- 预测前校验输入字段结构
- 提供中英文字段说明系统
- 自动生成 HTML 报告
- 提供 Streamlit 本地网页界面
- 支持 Docker 本地部署

## Core Workflow

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

中文流程：

```text
CSV 数据
  -> 数据分析
  -> 自动预处理
  -> 模型训练
  -> 模型评估
  -> 模型注册
  -> 预测推理
  -> 报告与网页展示
```

## Outputs

- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/confusion_matrix.png`
- `outputs/report.html`
- `models/classifier.joblib`
- `models/model_registry.json`

这些输出覆盖了模型指标、预测结果、混淆矩阵、HTML 报告、已保存模型和模型版本记录，便于复现运行结果并检查项目完整性。

These outputs cover model metrics, prediction results, confusion matrix visualization, HTML reporting, saved model artifacts, and model version records, making each run easier to reproduce and inspect.

## Project Structure

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

## Run Locally

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

中文说明：

- 先创建虚拟环境并安装依赖
- 可通过命令行训练模型和执行预测
- 也可以启动 Streamlit 页面进行交互式训练、预测和结果查看
- 默认使用本地 `8501` 端口

## Docker

```powershell
docker build -t universal-classifier .
docker run --rm -p 8501:8501 universal-classifier
```

Docker configuration is included for local deployment and repeatable demos.

项目提供 Docker 配置，便于本地部署和可复现演示。

## Model Options

- `svm`
- `random_forest`
- `knn`

The model can be changed through configuration or the Streamlit interface.

模型可以通过配置文件或 Streamlit 页面切换。

## Known Limits

- Designed for tabular classification, not image, audio, or generative tasks.
- Current implementation is single-user and local-first.
- Model registry is file-based, not database-backed.
- Advanced AutoML features such as hyperparameter search and cross-validation can be added later.

中文补充：

- 当前项目面向表格分类，不适用于图像、音频或生成式任务
- 当前实现以本地单用户演示为主
- 模型注册表基于文件保存，不是数据库系统
- 超参数搜索、交叉验证等高级 AutoML 能力可在后续版本加入
