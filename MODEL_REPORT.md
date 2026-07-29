# Model Report

## Scope

Universal Classifier packages a local tabular classification workflow with CSV ingestion, automated preprocessing, model comparison, model registry, prediction, reporting, and Streamlit interaction.

Universal Classifier 封装了一个本地表格分类流程，覆盖 CSV 导入、自动预处理、模型对比、模型注册、预测、报告生成和 Streamlit 交互。

## Dataset

The current demonstration uses `data/iris.csv` with `label` as the target column. The project structure is not tied to the Iris loader and can be adapted to other CSV-based classification datasets.

当前演示使用 `data/iris.csv`，目标列为 `label`。项目结构不依赖 Iris 内置加载器，可迁移到其他基于 CSV 的分类数据集。

## Pipeline

- CSV data loading
- Target-column validation
- Data quality analysis
- Numeric preprocessing with median imputation and scaling
- Categorical preprocessing with most-frequent imputation and one-hot encoding
- SVM, Random Forest, and KNN training
- Model comparison with macro metrics
- Versioned model registry
- Prediction schema validation
- HTML report and Streamlit display

中文说明：

- 读取 CSV 数据
- 校验目标列
- 执行数据质量分析
- 对数值特征进行中位数填充和标准化
- 对类别特征进行众数填充和 One-Hot 编码
- 训练 SVM、Random Forest 和 KNN
- 使用宏平均指标进行模型对比
- 记录版本化模型注册表
- 预测前校验输入字段结构
- 输出 HTML 报告并提供 Streamlit 展示

## Evaluation

```text
Model type: svm
Accuracy: 0.947
Macro precision: 0.949
Macro recall: 0.949
Macro F1: 0.949
```

Model comparison on the same train/test split:

```text
1. svm: accuracy=0.947, macro_f1=0.949
2. knn: accuracy=0.921, macro_f1=0.922
3. random_forest: accuracy=0.895, macro_f1=0.897
```

The best model is saved as a versioned artifact and recorded in `models/model_registry.json`.

最佳模型会保存为版本化文件，并记录在 `models/model_registry.json` 中。

## Prediction Example

```text
5.1,3.5,1.4,0.2 -> setosa
6.0,2.9,4.5,1.5 -> versicolor
6.5,3.0,5.8,2.2 -> virginica
```

## Constraints

- The current demo targets tabular classification only.
- The implementation is local-first and single-user.
- The model registry is file-based rather than database-backed.
- Hyperparameter search and cross-validation are not enabled in the current version.

中文说明：

- 当前演示仅面向表格分类
- 当前实现以本地单用户使用为主
- 模型注册表基于文件保存，不是数据库系统
- 当前版本未启用超参数搜索和交叉验证
