# rossmann_sales_forecasting/modeling/train.py
# rossmann_sales_forecasting/modeling/train.py
import joblib
from pathlib import Path
from rossmann_sales_forecasting.config import MODELS_DIR

if __name__ == "__main__":
    model_path = MODELS_DIR / "random_forest.pkl"
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
    
# Run: python3 -m rossmann_sales_forecasting.modeling.train
# This will load the trained model and print a success message
# Ensure to run this in an environment where joblib is installed
# This script is designed to be run directly, not as a module import
# It will not be imported by other modules, so no need for __all__ or imports
# The model is expected to be in the MODELS_DIR defined in config.py
# The MODELS_DIR should be set to the directory where the model is saved
# The model is expected to be a Random Forest model saved with joblib
# The script will print the path from which the model was loaded
# This script is designed for training and loading models, not for feature generation or data loading
# It is a standalone script that can be run independently to load the model
# The model can then be used for predictions or further analysis
# The script does not perform any predictions or evaluations, it only loads the model
# The model loading is done using joblib, which is efficient for large models
# The script is intended for use in a production environment where the model needs to be loaded for inference
# It does not include any data processing or feature generation steps
# The model is expected to be a pre-trained Random Forest model, ready for use
# The script is designed to be simple and focused on the task of loading the model
