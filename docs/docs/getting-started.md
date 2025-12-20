# Getting Started with Rossmann Sales Forecasting

## 🚀 Quick Setup Guide

This guide will help you set up the Rossmann Sales Forecasting project on a clean environment and reproduce the complete analysis pipeline.

## 📋 Prerequisites

- **Python**: 3.11+ (3.13 recommended)
- **Git**: For version control
- **Storage**: ~500MB for datasets and outputs

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Petlaz/rossmann_sales_forecasting.git
cd rossmann_sales_forecasting
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Project Dependencies
The project uses the following core libraries:
```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualization
plotly>=5.15.0         # Interactive plots
jupyter>=1.0.0         # Notebook environment
```

## 📊 Data Preparation

### 1. Extract Raw Data
The Rossmann dataset is provided in `data/raw/archive.zip`:
```bash
cd data/raw
unzip archive.zip
```

**Expected files after extraction:**
- `train.csv` (1,017,209 records) - Historical sales data
- `test.csv` (41,088 records) - Test set for predictions  
- `store.csv` (1,115 records) - Store metadata and features
- `sample_submission.csv` - Submission format reference

### 2. Verify Data Structure
```bash
ls -la data/raw/
# Should show: train.csv, test.csv, store.csv, sample_submission.csv
```

## 🧪 Running the Analysis Pipeline

### Phase 1: Exploratory Data Analysis ✅
```bash
# Open the EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

**Key EDA Outputs:**
- Statistical analysis across 9 data categories
- Sales-customer correlation insights (R² = 0.801)
- Promotion impact analysis (+38.8% sales lift)
- Store performance variance (275% range)
- Temporal pattern identification

### Phase 2: Feature Engineering & Preprocessing ✅
```bash
# Step 1: Run comprehensive feature engineering
python scripts/create_features.py

# Step 2: Run data preprocessing for modeling
python scripts/preprocess_data.py
```

**Generated Datasets:**
- `data/processed/train_modeling.csv` (675,958 × 79) - Feature-engineered training set
- `data/processed/val_modeling.csv` (168,380 × 79) - Feature-engineered validation set  
- `data/processed/test_processed.csv` (41,088 × 57) - Feature-engineered test set
- `data/processed/X_train.csv` (675,958 × 70) - Preprocessed training features
- `data/processed/X_val.csv` (168,380 × 70) - Preprocessed validation features
- `data/processed/X_test.csv` (41,088 × 70) - Preprocessed test features
- `data/processed/y_train.csv` / `y_val.csv` - Target variables
- `models/preprocessing/` - Serialized preprocessing artifacts (scalers, encoders)

**Feature Categories Created:**
- ⏰ **26 Temporal Features**: Cyclical encoding, seasonality
- 📊 **11 Lag/Rolling Features**: Historical patterns (7d/14d/30d)
- 🏪 **9 Competition Features**: Market intelligence
- 📢 **12 Promotion Features**: Campaign analysis
- 🏬 **5 Store Features**: Performance tiers
- 🎄 **6 Holiday Features**: Calendar effects

### Phase 3: Model Training 🔄 (Next Phase)
```bash
# Coming next: Multi-model training pipeline
python rossmann_sales_forecasting/modeling/train.py
```

## 🔍 Exploring Results

### 1. Review EDA Insights
- Open `notebooks/01_eda.ipynb` for comprehensive analysis
- Key business insights documented with visualizations
- Statistical validation and correlation studies

### 2. Feature Engineering Analysis  
- Review `docs/docs/FEATURE_ENGINEERING_REPORT.md` for detailed feature analysis
- Examine `data/processed/feature_engineering_report.json` for technical metadata
- Validate feature distributions and engineering logic

### 3. Data Validation
```python
import pandas as pd

# Quick validation of processed datasets
train_df = pd.read_csv('data/processed/train_modeling.csv')
print(f"Training set: {train_df.shape}")
print(f"Features: {len(train_df.columns)}")
print(f"Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
```

## 🛠️ Development Workflow

### Project Structure
```
rossmann_sales_forecasting/
├── data/                      # Data storage
│   ├── raw/                   # Original datasets  
│   ├── processed/             # Engineered features
│   └── external/              # Reference data
├── notebooks/                 # Jupyter analysis
├── rossmann_sales_forecasting/  # Core modules
│   ├── dataset.py             # Data processing
│   ├── features.py            # Feature engineering
│   └── modeling/              # ML implementations
├── scripts/                   # Automation scripts
├── docs/                      # Documentation
└── reports/                   # Analysis outputs
```

### Code Quality Standards
- **Modular Design**: Separate modules for data, features, and models
- **Documentation**: Comprehensive docstrings and comments
- **Validation**: Data quality checks and error handling
- **Reproducibility**: Consistent random seeds and version control

## 🎯 Next Steps

1. **Complete Phase 3**: Model training with 5 ML algorithms
2. **Model Evaluation**: Performance benchmarking and comparison
3. **Deployment**: Gradio web application for predictions
4. **Documentation**: Final project presentation materials

## 🆘 Troubleshooting

### Common Issues
- **Memory Issues**: Use chunked processing for large datasets
- **Missing Dependencies**: Ensure all packages in requirements.txt are installed
- **Data Path Errors**: Verify working directory and file paths
- **Version Conflicts**: Use virtual environment for clean dependencies

### Getting Help
- Review project documentation in `docs/`
- Check issue tracking for known problems
- Examine notebook outputs for expected results
- Validate environment setup and dependencies
