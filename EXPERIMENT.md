# Experiment Record

## Goal

将原始 Iris 分类脚本升级为一个可复用的表格分类平台。

Upgrade the original Iris classifier script into a reusable tabular classification platform.

The experiment focuses on moving from a simple demo script to a structured machine learning workflow with data loading, analysis, preprocessing, model comparison, prediction, reporting, and web interaction.

本实验的重点是从简单示例脚本走向结构化机器学习流程，覆盖数据读取、数据分析、预处理、模型对比、预测、报告和网页交互。

## Main Upgrade

原始版本使用：

The original version used:

```python
load_iris()
```

当前版本使用真实 CSV 文件：

The current version uses a real CSV file:

```text
data/iris.csv
```

This means the model workflow is no longer tied to one built-in dataset and can be reused for other tabular classification tasks.

这意味着模型流程不再绑定某个内置数据集，而是可以迁移到其他表格分类任务。

## Method

- 数据格式：CSV
- 目标列：`label`
- 模型：SVM、Random Forest、KNN
- 数值特征预处理：中位数填充 + 标准化
- 类别特征预处理：众数填充 + One-Hot 编码
- 模型保存：`joblib`
- Web 界面：Streamlit
- 部署支持：Docker

English summary:

- Data format: CSV
- Target column: `label`
- Models: SVM, Random Forest, KNN
- Numeric preprocessing: median imputation + scaling
- Categorical preprocessing: most frequent imputation + one-hot encoding
- Model persistence: `joblib`
- Web UI: Streamlit
- Deployment support: Docker

## Result

```text
Model type: svm
Accuracy: 0.947
Macro precision: 0.949
Macro recall: 0.949
Macro F1: 0.949
```

预测示例：

Prediction demo:

```text
5.1,3.5,1.4,0.2 -> setosa
6.0,2.9,4.5,1.5 -> versicolor
6.5,3.0,5.8,2.2 -> virginica
```

## Data Quality Analysis

Iris 数据分析结果：

Iris data analysis result:

```text
rows: 150
columns: 5
missing values: 0
duplicate rows: 1
target classes: 3
```

Data analysis matters because model performance is often limited by data quality. Even for a small and familiar dataset, recording missing values, duplicates, and target classes makes the workflow more transparent.

数据分析很重要，因为模型效果的上限往往由数据质量决定。即使是 Iris 这样的小型经典数据集，记录缺失值、重复行和目标类别数量，也能让建模流程更透明。

## Auto Model Comparison

Model comparison on the same train/test split:

```text
1. svm: accuracy=0.947, macro_f1=0.949
2. knn: accuracy=0.921, macro_f1=0.922
3. random_forest: accuracy=0.895, macro_f1=0.897
```

The best model is saved as a versioned model and recorded in the model registry.

最佳模型会被保存为版本化模型，并写入模型注册表，便于后续追踪、复现和替换。

## Engineering Notes

- Training and prediction are separated.
- UI and business logic are separated.
- `sklearn Pipeline` keeps preprocessing consistent between training and prediction.
- Model registry records model versions and metrics.
- Prediction validates required feature columns before inference.
- Docker image excludes outputs and model history artifacts.

中文说明：

- 训练和预测流程分离
- UI 层与业务逻辑分离
- `sklearn Pipeline` 保证训练和预测阶段预处理一致
- 模型注册表记录版本和指标
- 预测前检查必要字段是否完整
- Docker 镜像排除输出文件和模型历史产物

## What To Learn

- A machine learning project is not just model training; it also includes data analysis, preprocessing, evaluation, persistence, prediction, and presentation.
- A tabular classification task can be abstracted into a reusable engineering framework.
- Model registry and versioning are basic MLOps concepts.
- The web layer should call existing services instead of duplicating training logic.

你应该理解：

- 机器学习项目不只是训练模型，还包括数据分析、预处理、评估、保存、预测和展示
- 表格分类任务可以抽象成可复用的工程框架
- 模型注册和版本管理是 MLOps 的基础思想
- Web 应用层不应该复制训练代码，而应该调用已有服务层
