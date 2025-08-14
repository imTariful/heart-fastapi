import streamlit as st
import joblib
import numpy as np
import json
import os

# ====== Load model and metadata ======
MODEL_PATH = os.path.join("model", "heart_model.joblib")
META_PATH = os.path.join("model", "meta.json")

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    meta = json.load(f)

features = meta["features"]

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the **likelihood of heart disease** based on medical parameters.
Each field includes a brief explanation.
""")

# ====== Feature Inputs ======
st.header("Enter Patient Data")

age = st.number_input(
    "Age",
    min_value=1, max_value=120, value=50,
    help="Patient's age in years"
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Male" if x == 1 else "Female",
    help="0 = Female, 1 = Male"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }[x],
    help="Type of chest pain experienced"
)

trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=50, max_value=250, value=120,
    help="Resting blood pressure in mm Hg"
)

chol = st.number_input(
    "Serum Cholesterol (chol)",
    min_value=100, max_value=600, value=200,
    help="Serum cholesterol in mg/dl"
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    options=[0, 1],
    format_func=lambda x: "True" if x == 1 else "False",
    help="1 = True, 0 = False"
)

restecg = st.selectbox(
    "Resting Electrocardiographic Results (restecg)",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }[x],
    help="ECG results at rest"
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved (thalach)",
    min_value=50, max_value=250, value=150,
    help="Maximum heart rate achieved during test"
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="1 = Yes, 0 = No"
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0, max_value=10.0, value=1.0, step=0.1,
    help="ST depression induced by exercise relative to rest"
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment (slope)",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    }[x],
    help="Slope of the peak exercise ST segment"
)

ca = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy (ca)",
    options=[0, 1, 2, 3],
    help="Number of major vessels (0–3) colored by fluoroscopy"
)

thal = st.selectbox(
    "Thalassemia Type (thal)",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Unknown",
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }[x],
    help="Type of thalassemia detected"
)

# ====== Prediction ======
if st.button("🔍 Predict Heart Disease"):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
    
    arr = np.array(input_data).reshape(1, -1)

    try:
        proba = float(model.predict_proba(arr)[0][1])
    except:
        proba = None

    pred = int(model.predict(arr)[0])

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"🚨 The model predicts **Heart Disease** risk. Probability: {proba:.2f}" if proba else "🚨 The model predicts **Heart Disease** risk.")
    else:
        st.success(f"✅ The model predicts **No Heart Disease**. Probability: {proba:.2f}" if proba else "✅ The model predicts **No Heart Disease**.")
