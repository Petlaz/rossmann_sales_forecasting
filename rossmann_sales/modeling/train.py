# rossmann_sales/modeling/train.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from rossmann_sales.config import (
    PROCESSED_DIR,
    MODEL_DIR,
    FEATURE_NAMES_PATH
)

def load_data():
    """Load processed feature and target datasets."""
    print("📦 Loading data...")
    X = pd.read_csv(PROCESSED_DIR / "X_encoded.csv")
    y = pd.read_csv(PROCESSED_DIR / "y.csv").values.ravel()
    print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")
    return X, y

def build_pipeline():
    """Build a training pipeline with scaling and random forest."""
    print("🧱 Building pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    return pipeline

def train_and_save_model():
    """Train the model and save the pipeline artifacts."""
    try:
        X, y = load_data()
        pipeline = build_pipeline()

        print("🏋️ Training model...")
        pipeline.fit(X, y)

        print("📈 Evaluating model...")
        preds = pipeline.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        print(f"✅ Training RMSE: {rmse:.2f}")

        print("💾 Saving model and scaler...")
        joblib.dump(pipeline.named_steps['model'], MODEL_DIR / "random_forest.pkl")
        joblib.dump(pipeline.named_steps['scaler'], MODEL_DIR / "scaler.pkl")

        print("📝 Saving feature names...")
        with open(FEATURE_NAMES_PATH, "w") as f:
            for col in X.columns:
                f.write(col + "\n")

        print("✅ Training complete. Model artifacts saved.")
    
    except Exception as e:
        print("❌ An error occurred during training:", e)

if __name__ == "__main__":
    print("🚀 Starting training script...")
    train_and_save_model()
