"""
Phase 15: Student risk prediction interface.

Takes a single student's features, returns predicted risk, a calibrated
probability (since we use SVC(probability=True) / RF's native
predict_proba, both genuine probability estimates, not raw decision
scores), and the top contributing features for that specific student.

Recommendations are framed as analytical suggestions only, never as
guaranteed interventions, medical, psychological, or disciplinary
conclusions — per spec.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_SUGGESTIONS = {
    "failures": "Past class failures — consider academic support/tutoring resources.",
    "failure_risk_flag": "History of failing a class — consider academic support/tutoring resources.",
    "studytime": "Low weekly study time — study skills or time-management support may help.",
    "study_efficiency": "Study time not translating into avoided failures — consider a different study approach or targeted tutoring.",
    "absences": "Elevated absences — attendance support/counseling may be worth exploring.",
    "higher": "Limited stated interest in higher education — motivational/career counseling may help.",
    "wants_higher_no_support": "Motivated toward higher education but lacking formal school support — a strong candidate for targeted support.",
    "total_alcohol_consumption": "Elevated reported alcohol consumption — general wellbeing check-in may be worth considering.",
    "goout": "High frequency of going out socially — time-management balance may be worth discussing.",
    "Medu": "Lower parental education level — family engagement resources may help, though this is a background factor, not something to intervene on directly.",
    "Fedu": "Lower parental education level — family engagement resources may help, though this is a background factor, not something to intervene on directly.",
}


def predict_student_risk(student_features: dict, model_path=f"{RESULTS_DIR}/random_forest.pkl"):
    pipeline = joblib.load(model_path)

    X = pd.DataFrame([student_features])

    proba = pipeline.predict_proba(X)[0][1]
    predicted_class = int(pipeline.predict(X)[0])

    if proba >= 0.5:
        risk_label = "HIGH"
    elif proba >= 0.3:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    # Top contributing features via SHAP on this single prediction
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_row = shap_values[0]
    top_indices = np.argsort(np.abs(shap_row))[::-1][:3]

    top_factors = []
    for idx in top_indices:
        full_name = feature_names[idx].split("__")[-1]
        raw_feature_name = next(
            (key for key in FEATURE_SUGGESTIONS if key in full_name),
            full_name)
        suggestion = FEATURE_SUGGESTIONS.get(raw_feature_name, "Contributing factor — see full feature list for detail.")
        top_factors.append({
            "feature": feature_names[idx],
            "shap_value": float(shap_row[idx]),
            "direction": "increases risk" if shap_row[idx] > 0 else "decreases risk",
            "analytical_suggestion": suggestion,
        })

    return {
        "predicted_risk": "At Risk" if predicted_class == 1 else "Not At Risk",
        "risk_level": risk_label,
        "risk_probability": round(float(proba), 4),
        "top_contributing_factors": top_factors,
        "disclaimer": "These are analytical suggestions based on statistical patterns, "
                      "not guaranteed interventions, medical advice, or disciplinary conclusions.",
    }


if __name__ == "__main__":
    # Example: a genuinely at-risk-profile student, using real feature names
    # from the dataset (values chosen to represent a plausible high-risk case)
    example_student = {
        "school": "MS", "sex": "M", "age": 17, "address": "R", "famsize": "GT3",
        "Pstatus": "T", "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "other",
        "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 1,
        "failures": 2, "schoolsup": "no", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "no", "higher": "no", "internet": "no",
        "romantic": "no", "famrel": 3, "freetime": 4, "goout": 4, "Dalc": 3,
        "Walc": 4, "health": 3, "absences": 15,
        "parental_education_avg": 1.0, "failure_risk_flag": 1,
        "total_alcohol_consumption": 7, "study_efficiency": 1 / 3,
        "social_engagement": 8, "wants_higher_no_support": 0,
    }

    print("=== Example Student Risk Prediction ===\n")
    result = predict_student_risk(example_student)
    print(f"Predicted Risk: {result['predicted_risk']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Probability: {result['risk_probability']*100:.1f}%")
    print(f"\nTop Contributing Factors:")
    for factor in result["top_contributing_factors"]:
        print(f"  - {factor['feature']} ({factor['direction']}): {factor['analytical_suggestion']}")
    print(f"\n{result['disclaimer']}")

    with open(f"{RESULTS_DIR}/example_prediction.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/example_prediction.json")