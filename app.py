from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.services import (
    column_descriptions,
    load_config,
    model_history,
    run_data_analysis,
    run_model_comparison,
    run_prediction,
    run_training,
    save_uploaded_csv,
    update_training_config,
)


ZH = "\u4e2d\u6587"
MODEL_OPTIONS = {"SVM": "svm", "RandomForest": "random_forest", "KNN": "knn"}


TEXT = {
    ZH: {
        "title": "\u901a\u7528\u8868\u683c\u5206\u7c7b\u5668",
        "intro_title": "\u9879\u76ee\u7b80\u4ecb",
        "intro": "\u8fd9\u662f\u4e00\u4e2a\u672c\u5730\u8fd0\u884c\u7684\u8f7b\u91cf AutoML \u8868\u683c\u5206\u7c7b\u5e94\u7528\u3002\u4f60\u53ef\u4ee5\u4e0a\u4f20 CSV\uff0c\u5148\u505a\u6570\u636e\u5206\u6790\uff0c\u518d\u9009\u62e9\u6807\u7b7e\u5217\u548c\u6a21\u578b\u8fdb\u884c\u8bad\u7ec3\u4e0e\u9884\u6d4b\u3002",
        "how_title": "\u4f7f\u7528\u65b9\u6cd5",
        "how": [
            "\u6570\u636e\u5206\u6790\uff1a\u4e0a\u4f20 CSV\uff0c\u67e5\u770b\u6837\u672c\u6570\u3001\u7f3a\u5931\u503c\u3001\u91cd\u590d\u884c\u3001\u7c7b\u522b\u5206\u5e03\u548c\u6570\u503c\u7279\u5f81\u5206\u5e03\u3002",
            "\u6a21\u578b\u8bad\u7ec3\uff1a\u4e0a\u4f20\u5e26\u6807\u7b7e\u5217\u7684 CSV\uff0c\u9009\u62e9\u76ee\u6807\u5217\u548c\u6a21\u578b\uff0c\u70b9\u51fb\u5f00\u59cb\u8bad\u7ec3\u3002",
            "\u81ea\u52a8\u6bd4\u8f83\uff1a\u4e00\u6b21\u8bad\u7ec3 SVM\u3001RandomForest\u3001KNN\uff0c\u67e5\u770b\u6392\u884c\u699c\u5e76\u4fdd\u5b58\u6700\u4f73\u6a21\u578b\u3002",
            "\u6a21\u578b\u9884\u6d4b\uff1a\u4e0a\u4f20\u7279\u5f81\u5217\u4e00\u81f4\u7684\u65b0 CSV\uff0c\u67e5\u770b\u5e76\u4e0b\u8f7d\u9884\u6d4b\u7ed3\u679c\u3002",
            "\u6a21\u578b\u5386\u53f2\uff1a\u67e5\u770b\u6bcf\u6b21\u8bad\u7ec3\u4fdd\u5b58\u7684\u6a21\u578b\u7248\u672c\u548c\u6307\u6807\u3002",
        ],
        "analysis_tab": "\u6570\u636e\u5206\u6790",
        "train_tab": "\u6a21\u578b\u8bad\u7ec3",
        "predict_tab": "\u6a21\u578b\u9884\u6d4b",
        "history_tab": "\u6a21\u578b\u5386\u53f2",
        "analysis_header": "\u6570\u636e\u8d28\u91cf\u5206\u6790",
        "analysis_upload": "\u4e0a\u4f20\u9700\u8981\u5206\u6790\u7684 CSV",
        "target_optional": "\u76ee\u6807\u5217\uff08\u53ef\u9009\uff09",
        "run_analysis": "\u751f\u6210\u6570\u636e\u5206\u6790",
        "rows": "\u6837\u672c\u6570",
        "columns": "\u5217\u6570",
        "features": "\u7279\u5f81\u6570",
        "missing": "\u7f3a\u5931\u503c",
        "duplicates": "\u91cd\u590d\u884c",
        "dtypes": "\u5b57\u6bb5\u7c7b\u578b",
        "target_summary": "\u76ee\u6807\u5217\u7c7b\u522b\u5206\u5e03",
        "target_plot": "\u7c7b\u522b\u5206\u5e03\u56fe",
        "feature_plots": "\u6570\u503c\u7279\u5f81\u5206\u5e03",
        "data_report": "\u6570\u636e\u5206\u6790\u62a5\u544a",
        "missing_rate": "\u7f3a\u5931\u7387",
        "duplicate_rate": "\u91cd\u590d\u7387",
        "scale": "\u6570\u636e\u89c4\u6a21\u8bc4\u4ef7",
        "display_name": "\u663e\u793a\u540d\u79f0",
        "description": "\u5b57\u6bb5\u8bf4\u660e",
        "prediction_error": "\u9884\u6d4b\u5931\u8d25",
        "train_header": "\u8bad\u7ec3\u8868\u683c\u5206\u7c7b\u6a21\u578b",
        "train_upload": "\u4e0a\u4f20\u8bad\u7ec3 CSV",
        "target_column": "\u76ee\u6807\u5217",
        "feature_columns": "\u7279\u5f81\u5217",
        "model": "\u6a21\u578b",
        "start_training": "\u5f00\u59cb\u8bad\u7ec3",
        "training_mode": "\u8bad\u7ec3\u6a21\u5f0f",
        "single_model": "\u5355\u6a21\u578b\u8bad\u7ec3",
        "auto_compare": "\u81ea\u52a8\u6a21\u578b\u6bd4\u8f83",
        "start_compare": "\u5f00\u59cb\u81ea\u52a8\u6bd4\u8f83",
        "comparing": "\u6b63\u5728\u8bad\u7ec3\u5e76\u6bd4\u8f83\u591a\u4e2a\u6a21\u578b...",
        "leaderboard": "\u6a21\u578b\u6392\u884c\u699c",
        "best_model": "\u6700\u4f73\u6a21\u578b",
        "comparison_saved": "\u6bd4\u8f83\u7ed3\u679c\u5df2\u4fdd\u5b58",
        "no_features": "\u8bf7\u9009\u62e9\u4e00\u4e2a\u76ee\u6807\u5217\uff0c\u5e76\u786e\u4fdd\u81f3\u5c11\u5269\u4e0b\u4e00\u4e2a\u7279\u5f81\u5217\u3002",
        "training": "\u6b63\u5728\u8bad\u7ec3\u6a21\u578b\u5e76\u751f\u6210\u62a5\u544a...",
        "accuracy": "\u51c6\u786e\u7387",
        "precision": "\u7cbe\u786e\u7387",
        "recall": "\u53ec\u56de\u7387",
        "macro_f1": "Macro F1",
        "confusion_matrix": "\u6df7\u6dc6\u77e9\u9635",
        "model_saved": "\u6a21\u578b\u5df2\u4fdd\u5b58",
        "registered": "\u6a21\u578b\u5df2\u767b\u8bb0",
        "html_report": "HTML \u62a5\u544a",
        "sample_train": "\u8bad\u7ec3\u6837\u4f8b\u6570\u636e\u9884\u89c8",
        "predict_header": "\u4f7f\u7528\u5df2\u4fdd\u5b58\u6a21\u578b\u9884\u6d4b",
        "predict_upload": "\u4e0a\u4f20\u9884\u6d4b CSV",
        "predicting": "\u6b63\u5728\u52a0\u8f7d\u6700\u65b0\u6a21\u578b\u5e76\u9884\u6d4b...",
        "predictions_saved": "\u9884\u6d4b\u7ed3\u679c\u5df2\u4fdd\u5b58",
        "download_predictions": "\u4e0b\u8f7d\u9884\u6d4b CSV",
        "sample_predict": "\u9884\u6d4b\u6837\u4f8b\u6570\u636e\u9884\u89c8",
        "history_header": "\u6a21\u578b\u7248\u672c\u5386\u53f2",
        "no_history": "\u8fd8\u6ca1\u6709\u6a21\u578b\u5386\u53f2\u3002\u8bf7\u5148\u8bad\u7ec3\u4e00\u4e2a\u6a21\u578b\u3002",
    },
    "English": {
        "title": "Universal Classifier",
        "intro_title": "About This App",
        "intro": "A local AutoML-style app for tabular classification. Upload a CSV, analyze data quality, choose the label column and model, then train and predict with saved models.",
        "how_title": "How To Use",
        "how": [
            "Data Analysis: upload a CSV and review rows, missing values, duplicates, class distribution, and numeric feature distributions.",
            "Train: upload a labeled CSV, select the target column and model, then start training.",
            "Auto Compare: train SVM, RandomForest, and KNN in one run, view the leaderboard, and save the best model.",
            "Predict: upload a new CSV with the same feature columns and download predictions.",
            "Model History: review saved model versions and metrics.",
        ],
        "analysis_tab": "Data Analysis",
        "train_tab": "Train",
        "predict_tab": "Predict",
        "history_tab": "Model History",
        "analysis_header": "Data Quality Analysis",
        "analysis_upload": "Upload CSV for analysis",
        "target_optional": "Target column (optional)",
        "run_analysis": "Generate Data Analysis",
        "rows": "Rows",
        "columns": "Columns",
        "features": "Features",
        "missing": "Missing values",
        "duplicates": "Duplicate rows",
        "dtypes": "Column types",
        "target_summary": "Target class distribution",
        "target_plot": "Class distribution chart",
        "feature_plots": "Numeric feature distributions",
        "data_report": "Data analysis report",
        "missing_rate": "Missing rate",
        "duplicate_rate": "Duplicate rate",
        "scale": "Data scale evaluation",
        "display_name": "Display name",
        "description": "Description",
        "prediction_error": "Prediction failed",
        "train_header": "Train A Tabular Classifier",
        "train_upload": "Upload training CSV",
        "target_column": "Target column",
        "feature_columns": "Feature columns",
        "model": "Model",
        "start_training": "Start Training",
        "training_mode": "Training mode",
        "single_model": "Single model",
        "auto_compare": "Auto model comparison",
        "start_compare": "Start Auto Compare",
        "comparing": "Training and comparing multiple models...",
        "leaderboard": "Model leaderboard",
        "best_model": "Best model",
        "comparison_saved": "Comparison saved",
        "no_features": "Please choose a target column that leaves at least one feature column.",
        "training": "Training model and generating report...",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "macro_f1": "Macro F1",
        "confusion_matrix": "Confusion Matrix",
        "model_saved": "Model saved",
        "registered": "Model registered",
        "html_report": "HTML report",
        "sample_train": "Sample training data preview",
        "predict_header": "Predict With Saved Model",
        "predict_upload": "Upload prediction CSV",
        "predicting": "Loading the latest model and predicting...",
        "predictions_saved": "Predictions saved",
        "download_predictions": "Download predictions CSV",
        "sample_predict": "Sample prediction data preview",
        "history_header": "Model Version History",
        "no_history": "No model history yet. Train a model first.",
    },
}


def describe_column_for_table(column, descriptions):
    info = descriptions.get(column, {})
    zh_name = info.get("zh_name", column)
    description = info.get("description", "")
    display_name = f"{zh_name} ({column})" if zh_name != column else column
    return display_name, description


def show_analysis_result(result, labels, descriptions):
    report = result["report"]
    cols = st.columns(5)
    cols[0].metric(labels["rows"], report["rows"])
    cols[1].metric(labels["columns"], report["columns"])
    cols[2].metric(labels["features"], report["feature_count"])
    cols[3].metric(labels["missing"], report["missing_values_total"])
    cols[4].metric(labels["duplicates"], report["duplicate_rows"])

    rate_cols = st.columns(3)
    rate_cols[0].metric(labels["missing_rate"], f"{report['missing_rate']:.2%}")
    rate_cols[1].metric(labels["duplicate_rate"], f"{report['duplicate_rate']:.2%}")
    rate_cols[2].metric(labels["scale"], report["scale_evaluation"]["size"])
    st.caption(report["scale_evaluation"]["advice"])

    st.markdown(f"**{labels['dtypes']}**")
    dtype_rows = []
    for column, dtype in report["dtypes"].items():
        display_name, description = describe_column_for_table(column, descriptions)
        dtype_rows.append(
            {
                "column": column,
                labels["display_name"]: display_name,
                "dtype": dtype,
                labels["description"]: description,
            }
        )
    dtype_table = pd.DataFrame(
        dtype_rows
    )
    st.dataframe(dtype_table, use_container_width=True)

    if report["target_summary"]:
        st.markdown(f"**{labels['target_summary']}**")
        summary = report["target_summary"]
        class_table = pd.DataFrame(
            [
                {"class": key, "count": value, "ratio": summary["class_ratios"][key]}
                for key, value in summary["class_counts"].items()
            ]
        )
        st.dataframe(class_table, use_container_width=True)
        if result["target_plot"]:
            st.image(str(result["target_plot"]), caption=labels["target_plot"])

    if result["feature_plots"]:
        st.markdown(f"**{labels['feature_plots']}**")
        for path in result["feature_plots"]:
            st.image(str(path))

    st.info(f"{labels['data_report']}: {result['report_path']}")


st.set_page_config(page_title="Universal Classifier", layout="wide")
top_left, top_right = st.columns([4, 1])
with top_right:
    language = st.selectbox("Language / \u8bed\u8a00", [ZH, "English"], label_visibility="collapsed")

t = TEXT[language]
top_left.title(t["title"])

with st.expander(t["intro_title"], expanded=True):
    st.write(t["intro"])
    st.markdown(f"**{t['how_title']}**")
    for item in t["how"]:
        st.markdown(f"- {item}")

config = load_config(PROJECT_ROOT)
descriptions = column_descriptions(PROJECT_ROOT)
tabs = st.tabs([t["analysis_tab"], t["train_tab"], t["predict_tab"], t["history_tab"]])
tab_analysis, tab_train, tab_predict, tab_history = tabs

with tab_analysis:
    st.subheader(t["analysis_header"])
    uploaded_analysis = st.file_uploader(t["analysis_upload"], type=["csv"], key="analysis_csv")
    if uploaded_analysis:
        analysis_path = PROJECT_ROOT / "data" / "uploaded_analysis.csv"
        analysis_data = save_uploaded_csv(uploaded_analysis, analysis_path)
        st.dataframe(analysis_data.head(20), use_container_width=True)
        target_choices = [""] + list(analysis_data.columns)
        selected_target = st.selectbox(t["target_optional"], target_choices)
        if st.button(t["run_analysis"], type="primary"):
            result = run_data_analysis(PROJECT_ROOT, config, analysis_data, selected_target or None)
            show_analysis_result(result, t, descriptions)
    else:
        sample_path = PROJECT_ROOT / config["data"]["csv_path"]
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            st.caption(t["sample_train"])
            st.dataframe(sample.head(10), use_container_width=True)

with tab_train:
    st.subheader(t["train_header"])
    uploaded_train = st.file_uploader(t["train_upload"], type=["csv"], key="train_csv")
    if uploaded_train:
        train_path = PROJECT_ROOT / "data" / "uploaded_train.csv"
        preview = save_uploaded_csv(uploaded_train, train_path)
        st.dataframe(preview.head(20), use_container_width=True)
        target_column = st.selectbox(t["target_column"], preview.columns)
        feature_columns = [column for column in preview.columns if column != target_column]
        st.caption(f"{t['feature_columns']}: {', '.join(feature_columns)}")
        model_label = st.selectbox(t["model"], list(MODEL_OPTIONS.keys()))
        mode = st.radio(t["training_mode"], [t["single_model"], t["auto_compare"]], horizontal=True)
        button_label = t["start_compare"] if mode == t["auto_compare"] else t["start_training"]
        if st.button(button_label, type="primary"):
            if not feature_columns:
                st.error(t["no_features"])
            else:
                active_config = update_training_config(
                    config,
                    csv_path="data/uploaded_train.csv",
                    target_column=target_column,
                    model_type=MODEL_OPTIONS[model_label],
                )
                if mode == t["auto_compare"]:
                    with st.spinner(t["comparing"]):
                        result = run_model_comparison(PROJECT_ROOT, active_config)
                    best = result["best_record"]
                    st.markdown(f"**{t['leaderboard']}**")
                    st.dataframe(result["comparison_table"], use_container_width=True)
                    cols = st.columns(4)
                    cols[0].metric(t["best_model"], best["model_type"])
                    cols[1].metric(t["accuracy"], f"{best['accuracy']:.3f}")
                    cols[2].metric(t["precision"], f"{best['macro_precision']:.3f}")
                    cols[3].metric(t["macro_f1"], f"{best['macro_f1']:.3f}")
                    st.success(f"{t['model_saved']}: {result['model_path']}")
                    st.info(f"{t['registered']}: {result['registry_record']['model_name']}")
                    st.info(f"{t['comparison_saved']}: {result['comparison_path']}")
                else:
                    with st.spinner(t["training"]):
                        result = run_training(PROJECT_ROOT, active_config)
                    metrics = result["metrics"]
                    cols = st.columns(4)
                    cols[0].metric(t["accuracy"], f"{metrics['accuracy']:.3f}")
                    cols[1].metric(t["precision"], f"{metrics['macro_precision']:.3f}")
                    cols[2].metric(t["recall"], f"{metrics['macro_recall']:.3f}")
                    cols[3].metric(t["macro_f1"], f"{metrics['macro_f1']:.3f}")
                    st.image(str(result["confusion_matrix_path"]), caption=t["confusion_matrix"])
                    st.success(f"{t['model_saved']}: {result['model_path']}")
                    st.info(f"{t['registered']}: {result['registry_record']['model_name']}")
                    st.info(f"{t['html_report']}: {result['report_path']}")
    else:
        sample_path = PROJECT_ROOT / config["data"]["csv_path"]
        if sample_path.exists():
            st.caption(t["sample_train"])
            st.dataframe(pd.read_csv(sample_path).head(10), use_container_width=True)

with tab_predict:
    st.subheader(t["predict_header"])
    uploaded_predict = st.file_uploader(t["predict_upload"], type=["csv"], key="predict_csv")
    if uploaded_predict:
        try:
            with st.spinner(t["predicting"]):
                prediction_table, prediction_path = run_prediction(PROJECT_ROOT, config, uploaded_predict)
            st.dataframe(prediction_table, use_container_width=True)
            st.success(f"{t['predictions_saved']}: {prediction_path}")
            csv_bytes = prediction_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                t["download_predictions"],
                data=csv_bytes,
                file_name="new_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"{t['prediction_error']}: {exc}")
    else:
        sample_path = PROJECT_ROOT / "data" / "new_samples.csv"
        if sample_path.exists():
            st.caption(t["sample_predict"])
            st.dataframe(pd.read_csv(sample_path), use_container_width=True)

with tab_history:
    st.subheader(t["history_header"])
    records = model_history(PROJECT_ROOT)
    if records:
        history = pd.DataFrame(records)
        st.dataframe(history.iloc[::-1], use_container_width=True)
    else:
        st.info(t["no_history"])
