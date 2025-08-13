# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os

os.makedirs("model", exist_ok=True)

# Load data - make sure you've placed the kaggle csv at data/heart.csv
df = pd.read_csv('Data/heart.csv')

# Inspect columns quickly - adjust if the dataset has different names
# Common columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
TARGET = "target"  # adjust if your file uses 'target' or 'heart_disease'

# Some datasets may call target differently; raise if not present
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found. Columns: {df.columns.tolist()}")

X = df[FEATURES]
y = df[TARGET].apply(lambda v: 1 if v>0 else 0)  # normalize to binary 0/1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/heart_model.joblib")
# Save features list & model meta
meta = {"model_type": "RandomForestClassifier", "features": FEATURES}
with open("model/meta.json", "w") as f:
    json.dump(meta, f)

# Print simple score
print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))
print("Saved model to model/heart_model.joblib")
