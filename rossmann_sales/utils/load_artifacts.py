# rossmann_sales/utils/load_artifacts.py

import joblib
import json
from rossmann_sales.config import MODEL_DIR, FEATURE_NAMES_PATH

def load_model():
    return joblib.load(MODEL_DIR / "random_forest.pkl")

def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")

def load_feature_names():
    with open(FEATURE_NAMES_PATH, "r") as f:
        return json.load(f)
