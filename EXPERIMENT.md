# Experiment Record / 实验记录

## Goal / 目标

将原始 Iris 分类脚本升级为一个可复用的表格分类平台。

Upgrade the original Iris classifier script into a reusable tabular classification platform.

## Main Upgrade / 核心升级

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

这意味着模型流程不再绑定某个内置数据集，而是可以迁移到其他表格分类任务。

This means the model workflow is no longer tied to one built-in dataset and can be reused for other tabular classification tasks.

## Method / 方法

- 数据格式 / Data format: CSV
- 目标列 / Target column: `label`
- 模型 / Models: SVM, RandomForest, KNN
- 预处理 / Preprocessing:
  - 数值特征：中位数填充 + 标准化 / numeric features: median imputation + scaling
  - 类别特征：众数填充 + One-Hot 编码 / categorical features: most frequent imputation + one-hot encoding
- 模型保存 / Model persistence: joblib
- Web 界面 / Web UI: Streamlit
- Docker 支持 / Docker support

## Result / 结果

```text
Model type: svm
Accuracy: 0.947
Macro precision: 0.949
Macro recall: 0.949
Macro F1: 0.949
```

预测示例 / Prediction demo:

```text
5.1,3.5,1.4,0.2 -> setosa
6.0,2.9,4.5,1.5 -> versicolor
6.5,3.0,5.8,2.2 -> virginica
```

## Data Quality Analysis / 数据质量分析

Iris 数据分析结果：

Iris data analysis result:

```text
rows: 150
columns: 5
missing values: 0
duplicate rows: 1
target classes: 3
```

数据分析的意义在于：模型效果的上限往往由数据质量决定。

Data analysis matters because model performance is often limited by data quality.

## Auto Model Comparison / 自动模型比较

同一次训练/测试划分下的模型比较结果：

Model comparison on the same train/test split:

```text
1. svm: accuracy=0.947, macro_f1=0.949
2. knn: accuracy=0.921, macro_f1=0.922
3. random_forest: accuracy=0.895, macro_f1=0.897
```

最佳模型会被保存为版本化模型，并写入模型注册表。

The best model is saved as a versioned model and recorded in the model registry.

## Engineering Notes / 工程说明

- 训练和预测分离 / training and prediction are separated
- UI 和业务逻辑分离 / UI and business logic are separated
- sklearn Pipeline 保证训练和预测预处理一致 / sklearn Pipeline keeps preprocessing consistent between training and prediction
- 模型注册表记录版本和指标 / model registry records model versions and metrics
- 预测前检查字段完整性 / prediction validates required feature columns
- Docker 镜像排除输出文件和模型历史 / Docker image excludes outputs and model history artifacts

## What To Learn / 你应该理解什么

- 机器学习项目不只是训练模型，还包括数据分析、预处理、评估、保存、预测和展示。
- 表格分类项目可以抽象成可复用工程框架。
- 模型注册和版本管理是 MLOps 的基础思想。
- Web 应用层不应该复制训练代码，而应该调用已有服务层。

- A machine learning project is not just model training; it also includes data analysis, preprocessing, evaluation, persistence, prediction, and presentation.
- A tabular classification task can be abstracted into a reusable engineering framework.
- Model registry and versioning are basic MLOps concepts.
- The Web layer should call existing services instead of duplicating training logic.
