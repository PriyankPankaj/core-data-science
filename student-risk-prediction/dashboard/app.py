"""Student Risk Prediction Dashboard — all data from actual saved results."""
import streamlit as st
import pandas as pd
import json

RESULTS_DIR = "../results"

st.set_page_config(page_title="Student Risk Prediction", layout="wide")

PAGES = ["Overview", "Dataset", "Data Quality", "EDA", "Statistical Tests",
         "Feature Engineering", "Model Comparison", "SVM Analysis",
         "Feature Selection", "Explainability", "Student Risk Prediction",
         "Student Segmentation"]

page = st.sidebar.radio("Navigate", PAGES)


def load_json(name):
    with open(f"{RESULTS_DIR}/{name}") as f:
        return json.load(f)


if page == "Overview":
    st.title("Student Performance & At-Risk Prediction")
    st.markdown("Predicting academic risk from UCI Student Performance data (649 students)")
    comp = pd.read_csv(f"{RESULTS_DIR}/model_comparison.csv")
    col1, col2, col3 = st.columns(3)
    col1.metric("Students", 649)
    col2.metric("At-Risk Rate", "15.4%")
    col3.metric("Best F1", f"{comp['test_f1'].max():.3f}")

elif page == "Dataset":
    st.title("Dataset")
    st.write("UCI Student Performance (Portuguese course), 649 students, 33 original columns.")
    st.write("G1, G2, G3 excluded from features (leakage prevention).")

elif page == "Data Quality":
    st.title("Data Quality")
    st.markdown("No missing values, duplicates, or invalid values found — see DATA_QUALITY_REPORT.md")

elif page == "EDA":
    st.title("EDA")
    for img in ["target_distribution.png", "numerical_distributions.png", "boxplots.png",
                "categorical_risk_rates.png", "correlation_heatmap.png"]:
        st.image(f"{RESULTS_DIR}/{img}")

elif page == "Statistical Tests":
    st.title("Statistical Tests")
    tests = load_json("statistical_tests.json")
    st.dataframe(pd.DataFrame(tests)[["feature", "test", "p_value", "significant_bonferroni", "effect_size"]])

elif page == "Feature Engineering":
    st.title("Feature Engineering")
    st.markdown("6 engineered features — see FEATURE_ENGINEERING_REPORT.md for rationale")

elif page == "Model Comparison":
    st.title("Model Comparison")
    comp = pd.read_csv(f"{RESULTS_DIR}/model_comparison.csv")
    st.dataframe(comp[["model", "kernel_or_variant", "test_f1", "test_recall", "test_roc_auc", "training_time_s"]])

elif page == "SVM Analysis":
    st.title("SVM: Linear vs RBF")
    svm = load_json("svm_results.json")
    st.json(svm)

elif page == "Feature Selection":
    st.title("Feature Selection")
    fs = pd.read_csv(f"{RESULTS_DIR}/feature_selection_results.csv")
    st.dataframe(fs)

elif page == "Explainability":
    st.title("Explainability")
    st.image(f"{RESULTS_DIR}/shap_summary_rf.png")
    exp = load_json("explainability_results.json")
    st.subheader("Random Forest Permutation Importance")
    st.dataframe(pd.DataFrame(exp["rf_permutation_importance_top10"]))
    st.warning(exp["causal_interpretation_caveat"])

elif page == "Student Risk Prediction":
    st.title("Student Risk Prediction")
    pred = load_json("example_prediction.json")
    st.metric("Risk Level", pred["risk_level"])
    st.metric("Risk Probability", f"{pred['risk_probability']*100:.1f}%")
    for f in pred["top_contributing_factors"]:
        st.write(f"- **{f['feature']}** ({f['direction']}): {f['analytical_suggestion']}")
    st.caption(pred["disclaimer"])

elif page == "Student Segmentation":
    st.title("Student Segmentation")
    st.info("Not measured — segmentation was optional per spec and not run for this project.")