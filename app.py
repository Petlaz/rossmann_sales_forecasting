import os
import gdown
from rossmann_sales.app_gradio import launch_app
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

files_to_download = {
    "random_forest.pkl": "1zVNpAIysnyQ_tmHIQjRTt2y2mnI6CMpV",
    "scaler.pkl": "1PpkFIj879FbYsOY2tXp9_F3fcqh2RiDn",
    "features.json": "1A6i5_05MFzo5iJvDywGX2LBjpaC-SmfJ",
}

for filename, gdrive_id in files_to_download.items():
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(filepath), quiet=False)

if __name__ == "__main__":
    launch_app()
