"""Train a churn classifier and log the run to MLflow.

Loads a batch CSV, preprocesses it, trains a RandomForest inside an sklearn
Pipeline (so preprocessing travels with the model), evaluates on a holdout,
writes metrics/candidate.json, saves the fitted model, and logs params +
metrics + model to MLflow.

MLflow tracking is configured via environment variables (loaded from .env
locally, or set as secrets in CI):
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
If MLFLOW_TRACKING_URI is unset, MLflow falls back to a local ./mlruns dir.

Usage:
    python train.py
    python train.py --data data/raw/batch_1.csv --n-estimators 200
"""
import argparse
import json
import os
import sys

# MLflow prints unicode (e.g. emoji) when logging run URLs; force UTF-8 so the
# default Windows cp1252 console doesn't crash on it.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

load_dotenv()

TARGET = "Churn"
DROP_COLS = ["customerID"]


def load_and_clean(path):
    df = pd.read_csv(path)
    # TotalCharges has blank strings (spaces) for tenure-0 customers; coerce
    # those to NaN so they can be imputed numerically.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    # Binary target -> 1/0
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_pipeline(X, n_estimators, max_depth, random_state):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)]), numeric, categorical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/batch_0.csv")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--metrics-out", default="metrics/candidate.json")
    ap.add_argument("--model-out", default="models/candidate.joblib")
    args = ap.parse_args()

    X, y = load_and_clean(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )

    pipe, numeric, categorical = build_pipeline(
        X, args.n_estimators, args.max_depth, args.random_state
    )

    mlflow.set_experiment("churn")
    with mlflow.start_run() as run:
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_te)[:, 1]
        preds = pipe.predict(X_te)
        auc = roc_auc_score(y_te, proba)
        f1 = f1_score(y_te, preds)

        params = {
            "model": "random_forest",
            "n_estimators": args.n_estimators,
            "max_depth": str(args.max_depth),
            "data": args.data,
            "n_numeric_features": len(numeric),
            "n_categorical_features": len(categorical),
        }
        mlflow.log_params(params)
        mlflow.log_metrics({"roc_auc": auc, "f1": f1})
        mlflow.sklearn.log_model(
            pipe, name="model", serialization_format="cloudpickle"
        )

        # Persist artifacts for the eval gate (Week 3) and serving (Week 5).
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump({"roc_auc": auc, "f1": f1}, f, indent=2)
        joblib.dump(pipe, args.model_out)

        print(f"run_id={run.info.run_id}")
        print(f"roc_auc={auc:.4f}  f1={f1:.4f}")
        print(f"wrote {args.metrics_out} and {args.model_out}")


if __name__ == "__main__":
    main()
