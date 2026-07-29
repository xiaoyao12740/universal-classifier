# Universal Classifier / 通用表格分类平台

一个轻量级 AutoML/MLOps 表格分类项目，支持 CSV 数据分析、模型训练、自动模型比较、模型版本管理、预测推理、Streamlit Web 展示和 Docker 运行。

A lightweight AutoML/MLOps platform for tabular classification with CSV data analysis, model training, auto model comparison, model registry, prediction, Streamlit UI, and Docker support.

## 项目定位 / Project Positioning

这个项目不是单纯的 Iris 分类脚本，而是把一个教学分类任务逐步工程化为可复用的表格分类平台。

This project is not just an Iris classifier script. It upgrades a teaching demo into a reusable tabular classification platform.

适用场景：

- 表格数据分类
- 机器学习工程入门
- sklearn Pipeline 实践
- 轻量 AutoML 原型
- 本地 MLOps 学习项目

Suitable for:

- tabular classification
- machine learning engineering practice
- sklearn Pipeline practice
- lightweight AutoML prototyping
- local MLOps learning

## 核心功能 / Features

- CSV 数据读取 / CSV data loading
- 数据质量分析 / Data quality analysis
- 数值与类别特征自动预处理 / automatic numeric and categorical preprocessing
- SVM、RandomForest、KNN 模型训练 / SVM, RandomForest, and KNN training
- 自动模型比较与排行榜 / automatic model comparison and leaderboard
- 训练/预测分离 / separated training and prediction
- 模型版本管理 / versioned model registry
- 预测字段检查 / prediction schema validation
- HTML 实验报告 / HTML experiment report
- 中英文 Streamlit 页面 / bilingual Streamlit UI
- Docker 支持 / Docker support

## 项目结构 / Project Structure

```text
01-universal-classifier/
  app.py
  config.json
  requirements.txt
  Dockerfile
  .dockerignore
  config/
    column_descriptions.json
  data/
    iris.csv
    new_samples.csv
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

## 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 命令行训练 / Train From CLI

```powershell
.\.venv\Scripts\python.exe src\main.py train
```

示例输出 / Example output:

```text
Model type: svm
Accuracy: 0.947
Macro precision: 0.949
Macro recall: 0.949
Macro F1: 0.949
```

### 3. 命令行预测 / Predict From CLI

```powershell
.\.venv\Scripts\python.exe src\main.py predict --input-csv data\new_samples.csv
```

示例预测 / Example prediction:

```text
5.1,3.5,1.4,0.2 -> setosa
6.0,2.9,4.5,1.5 -> versicolor
6.5,3.0,5.8,2.2 -> virginica
```

### 4. 启动 Web 页面 / Run Streamlit App

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开 / Open:

```text
http://localhost:8501
```

Web 页面包含四个 Tab / The Web app includes four tabs:

- 数据分析 / Data Analysis
- 模型训练 / Train
- 模型预测 / Predict
- 模型历史 / Model History

## 配置模型 / Configure Model

编辑 `config.json` 中的模型类型：

Edit the model type in `config.json`:

```json
{
  "model": {
    "type": "svm"
  }
}
```

支持模型 / Supported models:

- `svm`
- `random_forest`
- `knn`

## 更换数据集 / Change Dataset

把新的 CSV 放到 `data/` 目录，并修改：

Put a new CSV file into `data/`, then update:

```json
{
  "data": {
    "csv_path": "data/your_file.csv",
    "target_column": "your_label_column"
  }
}
```

本项目适用于表格分类任务，不适用于图像分类或文本生成任务。

This project is designed for tabular classification, not image classification or text generation.

## 数据质量分析 / Data Quality Analysis

数据分析页面会生成：

The data analysis tab generates:

- 样本数量 / row count
- 列数量 / column count
- 特征数量 / feature count
- 缺失值数量 / missing value count
- 重复行数量 / duplicate row count
- 缺失率 / missing rate
- 重复率 / duplicate rate
- 数据规模评价 / data scale evaluation
- 字段类型 / column data types
- 目标类别分布 / target class distribution
- 数值特征分布图 / numeric feature distribution charts

输出文件 / Output:

```text
outputs/data_report.json
```

## 自动模型比较 / Auto Model Comparison

自动比较模式会在同一次训练/测试划分上训练：

Auto comparison trains these models on the same train/test split:

- SVM
- RandomForest
- KNN

模型按 Macro F1 和 Accuracy 排名，并保存最佳模型。

Models are ranked by Macro F1 and Accuracy. The best model is saved and registered.

Iris 示例 / Iris example:

```text
1. svm: accuracy=0.947, macro_f1=0.949
2. knn: accuracy=0.921, macro_f1=0.922
3. random_forest: accuracy=0.895, macro_f1=0.897
```

## 模型注册表 / Model Registry

每次训练会创建版本化模型：

Each training run creates a versioned model:

```text
models/model_YYYYMMDD_HHMMSS.joblib
```

模型元数据记录在：

Model metadata is stored in:

```text
models/model_registry.json
```

记录内容包括模型名称、创建时间、算法、数据集、Accuracy、Precision、Recall 和 Macro F1。

It records model name, creation time, algorithm, dataset, Accuracy, Precision, Recall, and Macro F1.

当前版本是单用户本地项目，`model_registry.json` 不处理并发写入。

Current scope: this is a single-user local project. `model_registry.json` does not handle concurrent writes.

## 字段解释 / Column Descriptions

字段展示配置位于：

Field display configuration lives in:

```text
config/column_descriptions.json
```

它可以显示中文字段名和字段说明，但不会修改原始 CSV 字段名。

It provides friendly display names and descriptions without changing original CSV column names.

## 缺失值策略 / Missing Value Strategy

训练不会静默删除缺失特征行。预处理策略：

Training does not silently drop rows with missing feature values. Preprocessing uses:

- 数值特征：中位数填充 / numeric features: median imputation
- 类别特征：众数填充 / categorical features: most frequent value imputation

如果目标列本身缺失，程序会报错。

If the target label itself is missing, the program raises an error.

## 预测字段检查 / Prediction Field Check

预测前会检查上传 CSV 是否包含训练模型需要的全部字段。

Before prediction, the app checks whether the uploaded CSV contains all required feature columns.

示例 / Example:

```text
Prediction CSV is missing required columns: petal width (cm)
```

## Docker

```powershell
docker build -t universal-classifier:latest .
docker run --rm -p 8502:8501 universal-classifier:latest
```

打开 / Open:

```text
http://localhost:8502
```

## 技术栈 / Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- Streamlit
- joblib
- Docker

## 项目状态 / Status

当前版本可作为第一个机器学习工程作品的 `v1.0`。

Current version can be treated as `v1.0` of a first machine learning engineering portfolio project.
