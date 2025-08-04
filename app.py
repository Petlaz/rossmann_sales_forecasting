# app.py (in project root)
import os
import gdown
from rossmann_sales.app_gradio import launch_app

MODEL_PATH = "models/random_forest.pkl"
GDRIVE_ID = "1zVNpAIysnyQ_tmHIQjRTt2y2mnI6CMpV"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

if __name__ == "__main__":
    launch_app()
