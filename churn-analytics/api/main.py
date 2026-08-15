"""Phase 8: FastAPI prediction endpoint."""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")
model = joblib.load("../results/best_model.pkl")


class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.post("/predict")
def predict(customer: CustomerInput):
    data = customer.model_dump()
    # Reconstruct engineered features expected by the pipeline
    data["tenure_bucket"] = pd.cut([data["tenure"]], bins=[-1,12,24,48,72],
        labels=["New (0-12mo)","Growing (12-24mo)","Established (24-48mo)","Loyal (48-72mo)"])[0]
    data["avg_monthly_spend"] = data["TotalCharges"] / data["tenure"] if data["tenure"] > 0 else data["MonthlyCharges"]
    service_flags = [data[c] == "Yes" for c in ["MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]]
    data["service_count"] = sum(service_flags)
    data["has_internet_addons"] = int(any(data[c]=="Yes" for c in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"]))
    data["is_month_to_month"] = int(data["Contract"] == "Month-to-month")
    data["payment_delay_risk"] = int(data["PaymentMethod"] == "Electronic check")
    data["charges_per_service"] = data["MonthlyCharges"] / (data["service_count"] + 1)

    X = pd.DataFrame([data])
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.30)  # using our F1-optimal threshold from Phase 5

    return {
        "churn_probability": round(float(proba), 4),
        "predicted_churn": bool(pred),
        "model_version": "logistic_regression_v1",
        "threshold_used": 0.30,
    }