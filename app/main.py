# app/main.py
from fastapi import FastAPI
from app.schemas import HeartInput
import joblib
import json
import numpy as np
import os

# Determine project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute paths for model & meta
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model", "heart_model.joblib"))
META_PATH = os.getenv("META_PATH", os.path.join(BASE_DIR, "model", "meta.json"))

app = FastAPI(title="Heart Disease Prediction API")

# Load model & metadata
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"model_type": meta.get("model_type", "unknown"), "features": meta.get("features", [])}

@app.post("/predict")
def predict(payload: HeartInput):
    features = meta.get("features")
    x = [getattr(payload, feat) for feat in features]
    arr = np.array(x).reshape(1, -1)
    proba = None
    try:
        proba = float(model.predict_proba(arr)[0][1])
    except Exception:
        proba = None
    pred = int(model.predict(arr)[0])
    return {"heart_disease": bool(pred), "probability": proba}
