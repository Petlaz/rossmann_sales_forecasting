# rossmann_sales/config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

FEATURE_NAMES_PATH = MODEL_DIR / "features.json"  # ✅ updated from feature_names.txt
