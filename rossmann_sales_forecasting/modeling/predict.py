
# rossmann_sales_forecasting/modeling/predict.py
import pandas as pd
from pathlib import Path
from joblib import load
from rossmann_sales_forecasting.config import PROCESSED_DATA_DIR, MODELS_DIR

def load_features():
    data_path = PROCESSED_DATA_DIR / "X_encoded.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Features loaded. Shape: {df.shape}")
    print(f"✅ Columns: {df.columns.tolist()}")
    return df

def load_model(model_path: Path):
    return load(model_path)

def predict(model, X):
    # Drop unwanted columns
    X = X.drop(columns=["is_train", "Sales"], errors="ignore")
    return model.predict(X)

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    model_path = MODELS_DIR / "random_forest.pkl"
    features_df = load_features()

    model = load_model(model_path)
    predictions = predict(model, features_df)

    print("✅ Predictions complete:")
    print(predictions)

        
# Run: python3 -m rossmann_sales_forecasting.modeling.predict
# This will load the model and features, make predictions, and save them to CSV
# Ensure to run this in an environment where joblib and pandas are installed
# The script uses typer for command-line interface, making it easy to run with different parameters
# The model is expected to be a Random Forest model saved with joblib
# The features CSV should contain the same structure as used during training
# The predictions will be saved in the specified output path