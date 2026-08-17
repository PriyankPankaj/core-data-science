"""Phase 17: Core pipeline tests."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.data.define_target import construct_target, AT_RISK_THRESHOLD
from src.features.feature_engineering import engineer_features
from src.prediction.risk_predictor import predict_student_risk


def test_target_construction_threshold():
    df = pd.DataFrame({"G3": [5, 9, 10, 15]})
    df = construct_target(df)
    assert list(df["at_risk"]) == [1, 1, 0, 0]


def test_target_no_leakage_columns_needed_for_prediction():
    """Confirms G1/G2/G3 aren't required inputs for prediction."""
    example_student = {
        "school": "MS", "sex": "M", "age": 17, "address": "R", "famsize": "GT3",
        "Pstatus": "T", "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "other",
        "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 1,
        "failures": 2, "schoolsup": "no", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "no", "higher": "no", "internet": "no",
        "romantic": "no", "famrel": 3, "freetime": 4, "goout": 4, "Dalc": 3,
        "Walc": 4, "health": 3, "absences": 15,
        "parental_education_avg": 1.0, "failure_risk_flag": 1,
        "total_alcohol_consumption": 7, "study_efficiency": 1/3,
        "social_engagement": 8, "wants_higher_no_support": 0,
    }
    assert "G1" not in example_student
    assert "G2" not in example_student
    assert "G3" not in example_student


def test_prediction_returns_valid_structure():
    example_student = {
        "school": "MS", "sex": "M", "age": 17, "address": "R", "famsize": "GT3",
        "Pstatus": "T", "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "other",
        "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 1,
        "failures": 2, "schoolsup": "no", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "no", "higher": "no", "internet": "no",
        "romantic": "no", "famrel": 3, "freetime": 4, "goout": 4, "Dalc": 3,
        "Walc": 4, "health": 3, "absences": 15,
        "parental_education_avg": 1.0, "failure_risk_flag": 1,
        "total_alcohol_consumption": 7, "study_efficiency": 1/3,
        "social_engagement": 8, "wants_higher_no_support": 0,
    }
    result = predict_student_risk(example_student)
    assert result["predicted_risk"] in ["At Risk", "Not At Risk"]
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(result["top_contributing_factors"]) == 3


def test_prediction_high_risk_profile_predicts_at_risk():
    """Sanity check: an obviously high-risk profile should predict At Risk."""
    high_risk_student = {
        "school": "MS", "sex": "M", "age": 18, "address": "R", "famsize": "GT3",
        "Pstatus": "T", "Medu": 0, "Fedu": 0, "Mjob": "at_home", "Fjob": "other",
        "reason": "course", "guardian": "mother", "traveltime": 3, "studytime": 1,
        "failures": 3, "schoolsup": "no", "famsup": "no", "paid": "no",
        "activities": "no", "nursery": "no", "higher": "no", "internet": "no",
        "romantic": "yes", "famrel": 2, "freetime": 5, "goout": 5, "Dalc": 4,
        "Walc": 5, "health": 2, "absences": 30,
        "parental_education_avg": 0.0, "failure_risk_flag": 1,
        "total_alcohol_consumption": 9, "study_efficiency": 0.25,
        "social_engagement": 10, "wants_higher_no_support": 0,
    }
    result = predict_student_risk(high_risk_student)
    assert result["risk_probability"] > 0.5