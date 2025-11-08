from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "lead_scoring_model.pkl"

FEATURE_COLUMNS = [
    "source", "course_interest", "email_opened", "clicked_ad",
    "income_level1", "education_level", "location_tier",
    "days_to_followup", "Summaryofconversation"
]

# Dropdown options (edit to match your data)
SOURCES = ["Facebook", "Google Ads", "Referral", "Website", "Email Campaign"]
COURSES = ["Data Science", "AI", "Web Development", "MBA", "Cloud Computing"]
INCOME = ["Low", "Medium", "High"]
EDUCATION = ["High School", "Bachelors", "Masters", "PhD"]
LOCATIONS = ["Tier 1", "Tier 2", "Tier 3"]
SUMMARIES = ["Very interested", "Mildly interested", "Just exploring", "Not interested"]

# Load model once
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sources": SOURCES,
        "courses": COURSES,
        "income": INCOME,
        "education": EDUCATION,
        "locations": LOCATIONS,
        "summaries": SUMMARIES
    })


@app.post("/api/score")
async def api_score(payload: dict):
    """
    Returns one simple explanation (text) about what affects the score most.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    df = pd.DataFrame([payload])
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    try:
        proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1][0]
        score = round(10 + 90 * proba, 2)  # preferred scaling version
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


    # 💡 Simple rule-based explanation logic
    reasons = []
    improvements = []

    # Behavior-based reasoning
    if df["email_opened"].iloc[0] == 1:
        reasons.append("opened your email")
    else:
        improvements.append("encourage them to open emails")

    if df["clicked_ad"].iloc[0] == 1:
        reasons.append("clicked your ad")
    else:
        improvements.append("make ads more engaging or follow up personally")

    days = df["days_to_followup"].iloc[0]
    if days <= 2:
        reasons.append("follow-up was quick")
    elif days >= 7:
        improvements.append("follow up sooner (ideally within 2–3 days)")

    summary = df["Summaryofconversation"].iloc[0].lower()
    if "very" in summary:
        reasons.append("showed strong interest")
    elif "mildly" in summary or "exploring" in summary:
        improvements.append("nurture the lead to build more interest")
    elif "not" in summary:
        improvements.append("re-engage disinterested leads with better messaging")

    income = df["income_level1"].iloc[0]
    if income.lower() == "low":
        improvements.append("emphasize scholarships or EMI options")
    elif income.lower() == "high":
        reasons.append("has higher income potential")

    # Combine explanation text
    if score >= 70:
        text = f"Lead score is high mainly because they {' and '.join(reasons)}. Keep engaging to convert soon."
    elif 40 <= score < 70:
        text = f"Lead score is moderate. They {' and '.join(reasons) if reasons else 'show some interest'}. To improve: {', '.join(improvements)}."
    else:
        text = f"Lead score is low. To improve: {', '.join(improvements)}."

    return JSONResponse({
        "lead_score": score,
        "lead_probability": round(proba, 4),
        "explanation": text
    })
