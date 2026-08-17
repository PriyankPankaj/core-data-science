"""Phase 16: FastAPI endpoints for student risk prediction."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prediction.risk_predictor import predict_student_risk
import json

app = FastAPI(title="Student Risk Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": ["logistic_regression", "svm_linear", "svm_rbf", "random_forest", "xgboost"],
            "deployed_model": "random_forest"}


@app.get("/model-metrics")
def model_metrics():
    with open("results/model_comparison.csv") as f:
        import csv
        reader = csv.DictReader(f)
        return {"metrics": list(reader)}


@app.get("/feature-importance")
def feature_importance():
    with open("results/explainability_results.json") as f:
        return json.load(f)


class StudentFeatures(BaseModel):
    school: str; sex: str; age: int; address: str; famsize: str; Pstatus: str
    Medu: int; Fedu: int; Mjob: str; Fjob: str; reason: str; guardian: str
    traveltime: int; studytime: int; failures: int; schoolsup: str; famsup: str
    paid: str; activities: str; nursery: str; higher: str; internet: str
    romantic: str; famrel: int; freetime: int; goout: int; Dalc: int
    Walc: int; health: int; absences: int


@app.post("/predict")
def predict(student: StudentFeatures):
    features = student.model_dump()
    features["parental_education_avg"] = (features["Medu"] + features["Fedu"]) / 2
    features["failure_risk_flag"] = int(features["failures"] > 0)
    features["total_alcohol_consumption"] = features["Dalc"] + features["Walc"]
    features["study_efficiency"] = features["studytime"] / (features["failures"] + 1)
    features["social_engagement"] = features["goout"] + features["freetime"]
    features["wants_higher_no_support"] = int(features["higher"] == "yes" and features["schoolsup"] == "no")

    try:
        return predict_student_risk(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/student-risk")
def student_risk(student: StudentFeatures):
    return predict(student)