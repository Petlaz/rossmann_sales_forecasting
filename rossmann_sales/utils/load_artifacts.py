# rossmann_sales/utils/load_artifacts.py
import joblib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

def load_model():
    model_path = MODEL_DIR / "random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def load_scaler():
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    return joblib.load(scaler_path)

def load_feature_names():
    features_path = MODEL_DIR / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")
    with open(features_path, "r") as f:
        features = json.load(f)
    return features

