"""FastAPI inference service for the churn champion model.

Loads the DVC-tracked champion pipeline (preprocessing + model travel together)
and exposes:
    GET  /health   -> liveness check
    POST /predict  -> churn probability for one customer

Because the saved object is a full sklearn Pipeline, raw customer fields can be
posted as-is: encoding and imputation happen inside the model.

Run locally:
    uvicorn app:app --reload --port 7860
Interactive docs at http://localhost:7860/docs
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "champion/model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 5,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 351.75,
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    return Prediction(churn_probability=proba, churn=proba >= 0.5)
