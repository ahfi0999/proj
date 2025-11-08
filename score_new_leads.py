import joblib
import pandas as pd
import numpy as np

# File names (make sure they exist)
MODEL_FILE = "lead_scoring_model.pkl"
INPUT_FILE = "new_leads.csv"
OUTPUT_FILE = "scored_leads.csv"

# Columns used for scoring
FEATURE_COLUMNS = [
    "source", "course_interest", "email_opened", "clicked_ad",
    "income_level1", "education_level", "location_tier",
    "days_to_followup", "Summaryofconversation"
]

print("Loading model...")
model = joblib.load(MODEL_FILE)

print("Loading new leads data...")
df = pd.read_csv(INPUT_FILE)

missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input file: {missing}")

print("Scoring leads...")
X = df[FEATURE_COLUMNS]
proba = model.predict_proba(X)[:, 1]
df["lead_probability"] = np.round(proba, 4)
df["lead_score"] = np.round(proba * 100, 2)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Scoring complete. Results saved to {OUTPUT_FILE}")
print(df[["lead_id", "lead_probability", "lead_score"]].head())
