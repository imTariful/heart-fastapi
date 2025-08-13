import streamlit as st
import joblib
import json
import numpy as np
import os

# --- Load model & metadata ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "meta.json")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()
features = meta.get("features", [])

# --- Streamlit UI ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details to predict the risk of heart disease.")

# Create input fields for each feature dynamically
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(
        f"{feature.replace('_', ' ').title()}",
        value=0.0
    )

if st.button("Predict"):
    x = np.array([user_input[feat] for feat in features]).reshape(1, -1)
    try:
        proba = float(model.predict_proba(x)[0][1])
    except Exception:
        proba = None

    pred = int(model.predict(x)[0])
    if pred == 1:
        st.error(f"⚠️ High risk of heart disease! (Probability: {proba:.2f})" if proba else "⚠️ High risk of heart disease!")
    else:
        st.success(f"✅ Low risk of heart disease (Probability: {proba:.2f})" if proba else "✅ Low risk of heart disease")
