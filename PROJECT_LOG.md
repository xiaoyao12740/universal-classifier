# Project Log

## v0.1 - From Iris Script To CSV Workflow

Completed:

- Replaced direct `load_iris()` usage with a CSV-based workflow.
- Added configurable data paths and target-column settings.
- Added reusable data loading and preprocessing modules.
- Added training and prediction commands.
- Saved trained models with `joblib`.

中文说明：

- 将直接调用 `load_iris()` 的示例脚本改为 CSV 工作流
- 增加可配置的数据路径和目标列设置
- 增加可复用的数据读取与预处理模块
- 增加训练和预测命令
- 使用 `joblib` 保存训练后的模型

## v0.2 - Model Comparison And Registry

Completed:

- Added SVM, Random Forest, and KNN model options.
- Added automatic model comparison on the same train/test split.
- Added accuracy, macro precision, macro recall, and macro F1 metrics.
- Added a file-based model registry for version tracking.
- Added prediction schema validation before inference.

中文说明：

- 增加 SVM、Random Forest 和 KNN 模型选项
- 在同一训练/测试划分下自动对比模型
- 增加 accuracy、macro precision、macro recall 和 macro F1 指标
- 增加基于文件的模型注册表，用于记录版本
- 在预测前校验输入字段结构

## v0.3 - Streamlit App And Reports

Completed:

- Added a Streamlit interface for local interaction.
- Kept UI code separate from training and prediction services.
- Added data quality analysis before model training.
- Added HTML report generation and confusion matrix output.
- Added Docker configuration for repeatable local deployment.

中文说明：

- 增加 Streamlit 本地交互页面
- 将 UI 代码与训练、预测服务分离
- 在训练前加入数据质量分析
- 增加 HTML 报告和混淆矩阵输出
- 增加 Docker 配置，便于复现本地部署

## v1.0 - Documentation Polish

Work note:

This version completes the main classification demo as a local application: CSV ingestion, automated preprocessing, model comparison, model registry, train/predict separation, Streamlit interaction, HTML reports, and Docker support.

工作记录：

当前版本完成了表格分类演示项目的主要闭环：CSV 导入、自动预处理、模型对比、模型注册、训练与预测流程分离、Streamlit 交互页面、HTML 报告和 Docker 支持。
