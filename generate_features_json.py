# scripts/generate_features_json.py

import json
import joblib
import pandas as pd
from rossmann_sales.config import FEATURE_NAMES_PATH

def main():
    X = joblib.load("models/X_train.pkl")  # or wherever your full training data is
    feature_names = list(X.columns)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"✅ Saved {len(feature_names)} features to {FEATURE_NAMES_PATH}")

if __name__ == "__main__":
    main()
    
#Run this script to generate the features.json file: python generate_features_json.py
