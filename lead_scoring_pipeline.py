#!/usr/bin/env python3
"""
lead_scoring_pipeline.py

Train a lead scoring pipeline using 'lead_scoring_dataset.csv' (in same dir).
This version auto-detects the correct OneHotEncoder argument for different sklearn versions.

Run:
    python lead_scoring_pipeline.py
"""

import os
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

# -------------------------
# CONFIG
# -------------------------
DATA_CSV = "lead_scoring_dataset.csv"
MODEL_PKL = "lead_scoring_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20

REQUIRED_COLUMNS = [
    "lead_id",
    "source",
    "course_interest",
    "email_opened",
    "clicked_ad",
    "income_level1",
    "education_level",
    "location_tier",
    "days_to_followup",
    "Summaryofconversation",
    "converted",
]

FEATURE_COLUMNS = [
    "source",
    "course_interest",
    "email_opened",
    "clicked_ad",
    "income_level1",
    "education_level",
    "location_tier",
    "days_to_followup",
    "Summaryofconversation",
]

CATEGORICAL_COLS = [
    "source",
    "course_interest",
    "income_level1",
    "education_level",
    "location_tier",
    "Summaryofconversation",
]

NUMERIC_COLS = ["email_opened", "clicked_ad", "days_to_followup"]

# -------------------------
# UTILITIES
# -------------------------
def init_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

def build_onehot_encoder():
    """
    Build OneHotEncoder in a way that's compatible across scikit-learn versions.
    Tries sparse_output=False (newer sklearn), falls back to sparse=False (older sklearn).
    """
    # Preferred for sklearn >= 1.2: sparse_output=False
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return ohe
    except TypeError:
        # Older sklearn versions use `sparse=False`
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            return ohe
        except TypeError:
            # As a last resort, call with default and rely on transformers to produce sparse matrices.
            # We'll accept sparse matrix output if neither arg is accepted.
            return OneHotEncoder(handle_unknown="ignore")

def build_pipeline():
    ohe = build_onehot_encoder()
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, CATEGORICAL_COLS),
            ("num", scaler, NUMERIC_COLS),
        ],
        remainder="drop",
    )

    base_rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    calibrated = CalibratedClassifierCV(estimator=base_rf, cv=3)
    pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", calibrated)])
    return pipeline

# -------------------------
# TRAINING / EVALUATION
# -------------------------
def train_and_save(data_csv: str = DATA_CSV, model_pkl: str = MODEL_PKL, test_size: float = TEST_SIZE):
    logging.info(f"Loading dataset from '{data_csv}'")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Dataset file not found: {data_csv}")

    df = pd.read_csv(data_csv)
    check_columns(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df["converted"].astype(int)

    logging.info("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    pipeline = build_pipeline()
    logging.info("Training pipeline (this may take a moment)...")
    pipeline.fit(X_train, y_train)

    logging.info("Predicting on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = float(roc_auc_score(y_test, y_proba))

    eval_summary = {"classification_report": report, "roc_auc": roc_auc}

    logging.info(f"Saving trained pipeline to '{model_pkl}'")
    joblib.dump(pipeline, model_pkl)

    # Permutation importance
    logging.info("Computing permutation importances...")
    try:
        X_test_trans = pipeline.named_steps["pre"].transform(X_test)
        perm = permutation_importance(pipeline.named_steps["clf"], X_test_trans, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)

        # get ohe feature names if available (OneHotEncoder might be sparse or dense)
        try:
            ohe = pipeline.named_steps["pre"].named_transformers_["cat"]
            ohe_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
        except Exception:
            # fallback: create generic names for categorical columns
            ohe_names = []
            # if transform returned sparse matrix, we can't easily map names — skip mapping
        trans_names = ohe_names + NUMERIC_COLS
        if len(trans_names) != len(perm.importances_mean):
            # fallback: enumerate feature names
            trans_names = [f"f_{i}" for i in range(len(perm.importances_mean))]

        importances_df = (
            pd.DataFrame({
                "feature": trans_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        logging.warning("Could not compute permutation importances: %s", e)
        importances_df = None

    logging.info("Training complete.")
    return pipeline, eval_summary, importances_df

# -------------------------
# SCORING
# -------------------------
def score_new(model_pkl: str, new_csv: str, out_csv: Optional[str] = None):
    if not os.path.exists(model_pkl):
        raise FileNotFoundError(f"Model file not found: {model_pkl}")
    if not os.path.exists(new_csv):
        raise FileNotFoundError(f"New leads file not found: {new_csv}")

    pipeline = joblib.load(model_pkl)
    df_new = pd.read_csv(new_csv)

    missing = [c for c in FEATURE_COLUMNS if c not in df_new.columns]
    if missing:
        raise ValueError(f"New leads CSV missing required feature columns: {missing}")

    X_new = df_new[FEATURE_COLUMNS].copy()
    logging.info("Predicting probabilities for new leads...")
    proba = pipeline.predict_proba(X_new)[:, 1]
    df_new["lead_probability"] = np.round(proba, 4)
    df_new["lead_score"] = np.round(proba * 100, 2)

    if out_csv:
        df_new.to_csv(out_csv, index=False)
        logging.info(f"Scored leads saved to '{out_csv}'")

    return df_new

# -------------------------
# MAIN
# -------------------------
def main():
    init_logger()
    logging.info("Lead scoring trainer starting...")

    try:
        pipeline, eval_summary, importances = train_and_save()
    except Exception as e:
        logging.exception("Training failed: %s", e)
        return

    print("\n=== Evaluation Summary ===")
    print(f"ROC AUC: {eval_summary['roc_auc']:.4f}\n")
    print("Classification report (test):")
    cr_df = pd.DataFrame(eval_summary["classification_report"]).transpose()
    print(cr_df.to_string(float_format=lambda x: f"{x:.3f}"))

    if importances is not None:
        print("\nTop permutation importances (transformed features):")
        print(importances.head(20).to_string(index=False))

    print(f"\nTrained model saved to: {MODEL_PKL}")
    print("\nTo score new leads later, call score_new('lead_scoring_model.pkl', 'new_leads.csv', out_csv='scored.csv')")

if __name__ == "__main__":
    main()
