# rossmann_sales_forecasting/config.py
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Path to processed features
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "X_encoded.csv"

# Path to save predictions
PREDICTIONS_FILE = PROJ_ROOT / "predictions.csv"
