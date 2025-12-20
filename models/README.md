# Models Directory

## 📂 Directory Structure

This directory contains trained model artifacts and related files from the Rossmann Sales Forecasting project.

```
models/
├── preprocessing/          # Data preprocessing artifacts
│   ├── feature_types.json     # Feature type classifications
│   └── preprocessing_report.json  # Preprocessing metadata
├── training_run_20251213_001055/  # Latest training run (Phase 3 & 4)
│   ├── *.csv                  # Feature importance and evaluation results  
│   ├── *.json                 # Training metadata and performance metrics
│   └── *.md                   # Summary reports
└── training_run_baseline_backup/  # Backup training results
    ├── *.csv                  # Feature importance files
    └── *.json                 # Training metadata
```

## 🚨 Large Files Not in Git

**Model pickle files (*.pkl) are excluded from git due to size constraints:**

- `xgboost_model.pkl` - Production champion model (74.1% R²)
- `random_forest_model.pkl` - Challenger model (59.5% R²) 
- `decision_tree_model.pkl` - Interpretable model (49.9% R²)
- Preprocessing artifacts: scalers, encoders, etc.

## 🔄 Reproducing Models

To regenerate all model files locally:

```bash
# Ensure data is available in data/ directory
python scripts/create_features.py    # Generate features
python scripts/preprocess_data.py    # Create preprocessing artifacts  
python scripts/train_models.py       # Train all models and save .pkl files
```

## 📊 Model Performance Summary

| Model | R² Score | RMSE | Annual Impact | Status |
|-------|----------|------|---------------|--------|
| **XGBoost** | **74.1%** | **€7,908** | **€37.97M** | 🟢 **CHAMPION** |
| Random Forest | 59.5% | €9,948 | €26.85M | 🟡 **CHALLENGER** |  
| Decision Tree | 49.9% | €11,084 | €22.41M | 🔵 **INTERPRETABLE** |

## 🎯 Files Included in Git

- **JSON files**: Training results, metadata, evaluation metrics
- **CSV files**: Feature importance, model comparisons, business analysis  
- **Markdown files**: Summary reports and documentation
- **Configuration**: Preprocessing settings and feature classifications

All model performance data and business analysis results are preserved in these lightweight files.