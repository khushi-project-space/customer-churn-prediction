"""Reusable data loading, preprocessing, and model evaluation utilities."""
from pathlib import Path
import json
import warnings
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, precision_score,
    f1_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"
TARGET = "Churn"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Read data and normalize common churn target encodings to 0/1."""
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' target column.")
    data = data.drop_duplicates().copy()
    y_raw = data[TARGET]
    if pd.api.types.is_string_dtype(y_raw.dtype) or y_raw.dtype == object:
        normalized = y_raw.astype("string").str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        y = normalized.map(mapping)
        if y.isna().any():
            raise ValueError("Churn target must contain Yes/No or 1/0 values.")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
    valid = y.notna()
    data, y = data.loc[valid].copy(), y.loc[valid].astype(int)
    X = data.drop(columns=[TARGET])
    # Customer IDs are identifiers, not meaningful predictive features.
    id_cols = [c for c in X.columns if c.lower() in {"customerid", "customer_id", "id"}]
    X = X.drop(columns=id_cols, errors="ignore")
    return data, X, y


def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    # XGBoost is opt-in because it is a heavier optional dependency.
    # Set CHURN_INCLUDE_XGBOOST=1 to include it in model comparison.
    if os.getenv("CHURN_INCLUDE_XGBOOST", "0") == "1":
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"
            )
        except ImportError:
            warnings.warn("XGBoost is not installed; skipping XGBoost model.")
    return models


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = pred
    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "ROC-AUC": float(auc),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, pred, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_test, pred).tolist(),
        "Classification Report": classification_report(
            y_test, pred, target_names=["No Churn", "Churn"], zero_division=0,
            output_dict=True
        ),
    }


def train_and_compare(data_path=DATA_PATH, save=True):
    """Train all available models on a stratified holdout split."""
    from joblib import dump
    data, X, y = load_data(data_path)
    if y.nunique() < 2:
        raise ValueError("Target must contain both churn and non-churn examples.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    results, pipelines = {}, {}
    for name, estimator in build_models().items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        results[name] = evaluate_model(pipe, X_test, y_test)
        pipelines[name] = pipe

    # Select the model with the highest ROC-AUC, a ranking metric less tied to
    # a single threshold; report all metrics rather than claiming one is best for every use.
    best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dump(pipelines[best_name], MODEL_DIR / "best_model.joblib")
        with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"best_model": best_name, "results": results}, f, indent=2)
        # Save the raw column schema so the app can build a valid input row.
        with open(MODEL_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump({
                "features": X.columns.tolist(),
                "dtypes": {c: str(X[c].dtype) for c in X.columns},
                "categorical_values": {
                    c: sorted(X[c].dropna().astype(str).unique().tolist())
                    for c in X.select_dtypes(exclude=["number"]).columns
                },
                "numeric_defaults": {
                    c: float(X[c].median()) for c in X.select_dtypes(include=["number"]).columns
                },
            }, f, indent=2)
    return data, X_train, X_test, y_train, y_test, results, pipelines, best_name
