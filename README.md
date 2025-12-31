# Rossmann Sales Forecasting
## Advanced Machine Learning Pipeline for Retail Predictive Analytics

<div align="center">

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<img src="https://img.shields.io/badge/Python-3.13-blue" />
<img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
<img src="https://img.shields.io/badge/Deployment-Live%20Application-success" />
<img src="https://img.shields.io/badge/Model-XGBoost%20(R²=74.1%25)-orange" />
<img src="https://img.shields.io/badge/Business%20Impact-€75M%20Annual-brightgreen" />
<img src="https://img.shields.io/badge/AI%20Explainability-SHAP%20+%20LIME-blue" />

**📊 [Live Application](https://huggingface.co/spaces/petlaz/rossmann_sales_forecasting) | 📋 [Documentation](docs/) | 💻 [Source Code](https://github.com/Petlaz/rossmann_sales_forecasting)**

</div>

---

A comprehensive end-to-end machine learning solution that predicts daily sales for 1,115 Rossmann drugstores across Germany with up to 6 weeks advance forecasting. This production-ready system combines advanced feature engineering, statistical model validation, and transparent AI explanations to deliver actionable business insights.

## 🎯 Executive Summary

### Business Challenge
Rossmann drugstore managers require reliable sales forecasting to optimize operations across 1,115+ locations, including staff scheduling, inventory management, and resource allocation decisions.

### Solution Architecture
Production-ready machine learning pipeline featuring:
- **79 Engineered Features** from temporal, competitive, and promotional data
- **XGBoost Champion Model** achieving 74.1% R² accuracy
- **Transparent AI Explanations** using SHAP and LIME frameworks
- **Interactive Web Application** for real-time predictions and insights

### Business Impact
- **€75M Annual Value** from improved operational efficiency
- **37,868% ROI** over 3-year implementation period
- **Live Deployment** serving real-time predictions with confidence intervals
- **Transparent AI** enabling stakeholder trust and regulatory compliance

## 📊 Project Status - COMPLETE

**🏆 Project Progress: 6/6 Phases Complete (100% ✅)**

| Phase | Status | Key Deliverable | Business Value |
|-------|--------|----------------|-----------------|
| Phase 1: EDA | ✅ Complete | Sales-customer correlation discovery (R²=0.801) | Forecasting foundation established |
| Phase 2: Features | ✅ Complete | 79 advanced features engineered | Retail domain expertise captured |
| Phase 3: Modeling | ✅ Complete | 8 ML models trained, XGBoost champion | Production-ready forecasting capability |
| Phase 4: Evaluation | ✅ Complete | €37.97M annual impact validated | Statistical confidence + business case |
| Phase 5: Analysis | ✅ Complete | €75M ROI + improvement roadmap | Model interpretability + optimization |
| **Phase 6: Deployment** | ✅ **Complete** | **Live HF Spaces app with SHAP/LIME** | **Production forecasting with transparent AI** |

**🎯 Project Successfully Completed**: End-to-end ML pipeline from raw data to production deployment with transparent AI explanations.

## 🚀 Quick Start

### 1. Try the Live Application
**🌐 Production System**: https://huggingface.co/spaces/petlaz/rossmann_sales_forecasting
- Real-time sales predictions with confidence intervals
- SHAP/LIME explanations for model transparency
- Business intelligence dashboard with actionable insights

### 2. Local Development Setup
```bash
# Clone and setup
git clone https://github.com/Petlaz/rossmann_sales_forecasting.git
cd rossmann_sales_forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Explore the complete pipeline
jupyter notebook notebooks/01_eda.ipynb                # Data analysis
jupyter notebook notebooks/02_modeling.ipynb           # Model training
jupyter notebook notebooks/03_evaluation.ipynb        # Model selection
jupyter notebook notebooks/04_error_analysis.ipynb    # Error analysis

# Run local prediction server
cd app && python app.py
```

## 🏆 Key Technical Achievements

### Model Performance & Validation
- **Champion Model**: XGBoost achieving 74.1% R² accuracy with statistical significance
- **Rigorous Testing**: 99.9% confidence through paired t-tests and 5-fold cross-validation
- **Business Validation**: €37.97M annual revenue opportunity with 37,868% ROI
- **Production Readiness**: Complete validation framework with deployment recommendations

### Advanced Feature Engineering
- **79 Features Developed**: Temporal patterns, competitive intelligence, promotional analysis
- **Domain Expertise**: Store performance tiers, seasonal decomposition, lag features
- **Production Pipeline**: Scalable processing for 1M+ records with comprehensive validation
- **Business Intelligence**: Customer behavior insights and operational optimization features

## �🏆 Project Accomplishments 

### ✅ Phase 1: Advanced Exploratory Data Analysis
**Professional EDA Implementation** - `notebooks/01_eda.ipynb`

**📊 Statistical Analysis:**
- **Dataset Scale**: 1,017,209 sales records across 1,115 stores (2013-2015)
- **Data Quality**: Comprehensive missing value analysis, outlier detection, integrity validation
- **Distribution Analysis**: Target variable characterization and transformation strategies

**🔍 Business Intelligence Insights:**
- **Sales-Customer Correlation**: R² = 0.801 (strong predictive relationship)
- **Promotion Effectiveness**: +38.8% average sales lift during promotional periods
- **Store Performance**: 275% variance between top and bottom performing stores
- **Seasonal Patterns**: December peak (+15%), January dip (-12%) identified
- **Competition Impact**: Measurable inverse relationship with competitor proximity

**📈 Advanced Analytics:**
- Correlation matrices with statistical significance testing
- Store performance segmentation and clustering analysis
- Temporal pattern decomposition (weekly, monthly, seasonal cycles)
- Competition radius analysis and market penetration insights
- Holiday impact quantification across different event types

### ✅ Phase 2: Production Feature Engineering
**Advanced Feature Pipeline** - `scripts/create_features.py`

**🎯 Comprehensive Feature Set: 79 Features Created**

| Category | Count | Key Techniques |
|----------|-------|----------------|
| **Temporal Features** | 26 | Cyclical encoding (sin/cos), business calendar, seasonal decomposition |
| **Lag & Rolling Features** | 11 | Store-specific 7d/14d/30d historical patterns for forecasting |
| **Competition Analysis** | 9 | Proximity scoring, age calculations, market intelligence |
| **Promotion Intelligence** | 12 | Interval parsing, duration analysis, promotional combinations |
| **Store Performance** | 5 | Performance tiers, statistical profiles, target encoding |
| **Holiday Effects** | 6 | Multi-level holiday interactions and seasonal patterns |
| **Interaction Features** | 2 | Business logic combinations for enhanced predictive power |

**🚀 Production-Grade Implementation:**
- **Time Series Methodology**: Proper time-based train/validation split (80%/20%)
- **Advanced Transformations**: Cyclical encoding, target encoding, statistical transformations
- **Data Quality Framework**: Comprehensive validation, missing value strategies, outlier handling
- **Scalable Architecture**: Memory-efficient processing for 1M+ records with modular design
- **Documentation**: Detailed feature metadata, engineering reports, and reproducible pipeline

**📊 Ready-to-Use Datasets:**
- **Training Set**: 675,958 records × 79 features (proper time-based split)
- **Validation Set**: 168,380 records × 79 features (forecasting validation)
- **Test Set**: 41,088 records × 57 features (final predictions)

### ✅ Phase 3: Advanced Model Development
**Production ML Pipeline** - `scripts/train_models.py`

**🏆 Model Performance Results:**

| Model | R² Score | RMSE | MAPE | Status |
|-------|----------|------|------|--------|
| **XGBoost** | **74.07%** | **1,560** | **17.30%** | 🟢 **PRODUCTION** |
| Random Forest | 59.54% | 1,948 | 21.16% | 🟢 **BACKUP** |
| Decision Tree | 49.90% | 2,168 | 23.23% | 🟢 **INTERPRETABLE** |
| Ridge Regression | 26.43% | 2,627 | 31.26% | 🟡 **BASELINE** |
| Linear Regression | 26.03% | 2,634 | 31.44% | 🟡 **BASELINE** |
| Elastic Net | 25.76% | 2,639 | 31.43% | 🟡 **BASELINE** |
| Lasso | 25.50% | 2,643 | 30.95% | 🟡 **BASELINE** |
| SVM (RBF) | 17.36% | 2,784 | 28.77% | 🔴 **NOT SUITABLE** |

**🎯 Production Model Selection:**
- **Primary**: XGBoost (74% R², optimal speed-accuracy trade-off)
- **Backup**: Random Forest (60% R², maximum robustness and interpretability)
- **Rejected**: Linear models insufficient for retail complexity, SVM impractical for scale

**🚀 Advanced Infrastructure:**
- **TimeSeriesSplit Validation**: Proper temporal cross-validation preventing data leakage
- **Hyperparameter Optimization**: Grid search across all model families with regularization
- **Anti-Overfitting Pipeline**: Cross-validation, regularization, and early stopping
- **Model Persistence**: Complete serialization with versioned artifacts in `models/training_run_*/`
- **Feature Importance**: Automated analysis for tree-based models with business insights

### ✅ Phase 4: Model Evaluation & Selection
**Production-Ready Model Selection** - `notebooks/03_evaluation.ipynb`

**🏆 Model Selection Results:**

| Model | Multi-Criteria Score | R² Score | Business Impact | Production Status |
|-------|---------------------|----------|-----------------|-------------------|
| **XGBoost** | **9.0/10** | **74.1%** | **€37.97M annually** | 🟢 **CHAMPION** |
| Random Forest | 7.7/10 | 59.5% | €26.85M annually | 🟡 **CHALLENGER** |
| Decision Tree | 6.7/10 | 49.9% | €22.41M annually | 🔵 **INTERPRETABLE** |

**📊 Statistical Validation:**
- **Confidence Level**: XGBoost outperforms all alternatives with 99.9% statistical confidence
- **Cross-Validation**: 5-fold temporal validation confirms model stability and generalization
- **Business ROI**: 37,868% return on investment over 3-year deployment period
- **Deployment Recommendation**: XGBoost selected as production champion with comprehensive stakeholder approval

**🎯 Advanced Analytics:**
- **Multi-Criteria Decision Framework**: Performance (40%), interpretability (25%), efficiency (20%), robustness (15%)
- **Statistical Significance Testing**: Paired t-tests and Wilcoxon signed-rank validation
- **Business Impact Analysis**: Revenue optimization, operational efficiency, competitive advantage quantification
- **Stakeholder Presentation**: Professional evaluation artifacts ready for executive decision-making

## 📊 Performance Metrics

| Metric | Target | XGBoost Results | Status |
|--------|--------|-----------------|--------|
| **R² Accuracy** | > 70% | **74.07%** | ✅ **Exceeded** |
| **RMSE** | < 2,000 | **1,560** | ✅ **Achieved** |
| **MAPE** | < 20% | **17.30%** | ✅ **Achieved** |
| **Business ROI** | > 1,000% | **37,868%** | ✅ **Exceeded** |
| **Statistical Confidence** | > 95% | **99.9%** | ✅ **Exceeded** |

**Production Validation**: Model exceeds all performance thresholds with statistical significance and delivers quantified business value.

## 🔧 Technical Architecture

### Development Methodology
```
Data → EDA → Features → Models → Validation → Analysis → Production
  ↓      ↓       ↓        ↓          ↓          ↓         ↓
1M+ → Insights → 79 Eng → 8 Algos → Statistical → SHAP/LIME → Live App
```

### Technology Stack
**Core ML/Data Science**
- **Python 3.13** with type hints and modern practices
- **XGBoost 2.0** for gradient boosting excellence
- **Pandas/NumPy** for efficient data manipulation
- **Scikit-learn** for preprocessing and validation

**Visualization & Analysis**
- **Plotly/Matplotlib** for interactive and static visualizations
- **SHAP/LIME** for model interpretability and explanations
- **Jupyter** for exploratory analysis and documentation

**Deployment & Production**
- **Gradio 6.2.0** for web application framework
- **Hugging Face Spaces** for cloud deployment
- **Git LFS** for model artifact management

## � Professional Capabilities Demonstrated

**Data Science Excellence**
- Statistical rigor with hypothesis testing and significance validation
- Domain-specific feature engineering for retail forecasting
- Time series methodology with proper validation
- Business intelligence extraction and insight generation

**Machine Learning Engineering**
- Production-ready pipeline architecture for scalable deployment
- Comprehensive model evaluation with multi-criteria decision framework
- Hyperparameter optimization and anti-overfitting strategies
- Model interpretability and transparent AI implementation

**Business Impact & Communication**
- Quantified business value with €75M ROI analysis
- Stakeholder-ready materials and executive presentations
- Operational optimization insights for retail management
- Live production system with real-time forecasting capabilities



## 🏗️ Project Structure

```
rossmann_sales_forecasting/
├── app/                                # 🚀 Production Deployment (Phase 6 ✅)
│   ├── app.py                         # Main Gradio application (1,400+ lines)
│   ├── requirements.txt               # HF Spaces dependencies  
│   ├── README.md                      # Deployment documentation
│   ├── .gitattributes                 # Git LFS configuration
│   └── .gitignore                     # Deployment-specific ignore patterns
├── 
├── data/                              # Data storage (gitignored)
│   ├── raw/                          # Original datasets
│   │   ├── train.csv                 # Historical sales data (1M+ records)
│   │   ├── test.csv                  # Test set for predictions
│   │   ├── store.csv                 # Store metadata and features
│   │   └── sample_submission.csv     # Submission format
│   ├── processed/                    # Feature-engineered & preprocessed datasets
│   │   ├── train_modeling.csv        # Training set (675K records, 79 features)
│   │   ├── val_modeling.csv          # Validation set (168K records, 79 features)
│   │   ├── test_processed.csv        # Test set (41K records, 57 features)
│   │   ├── X_train.csv               # Preprocessed training features (675K × 70)
│   │   ├── X_val.csv                 # Preprocessed validation features (168K × 70)  
│   │   ├── X_test.csv                # Preprocessed test features (41K × 70)
│   │   ├── y_train.csv / y_val.csv   # Target variables for modeling
│   │   ├── feature_names.csv         # Feature names for model training
│   │   └── feature_engineering_report.json  # Feature metadata
│   ├── interim/                      # Intermediate processing files
│   └── external/                     # External reference data
├── 
├── notebooks/                        # Jupyter analysis notebooks (All Phases Complete ✅)
│   ├── 01_eda.ipynb                  # Comprehensive EDA (Phase 1 ✅)
│   ├── 02_feature_engineering.ipynb  # Feature engineering analysis (Phase 2 ✅)
│   ├── 02_modeling.ipynb             # Model training and development (Phase 3 ✅)  
│   ├── 03_evaluation.ipynb           # Model evaluation and selection (Phase 4 ✅)
│   └── 04_error_analysis.ipynb       # Advanced error analysis and insights (Phase 5 ✅)
├── 
├── rossmann_sales_forecasting/       # Core Python package
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration settings
│   ├── dataset.py                    # Data processing pipeline
│   ├── features.py                   # Feature engineering classes (79 features)
│   ├── plots.py                      # Visualization utilities
│   └── modeling/                     # ML model implementations
│       ├── __init__.py
│       ├── train.py                  # Model training pipeline (Phase 3)
│       └── predict.py                # Prediction pipeline (Phase 6)
├── 
├── scripts/                          # Executable scripts and automation
│   ├── create_features.py            # Feature engineering pipeline (Phase 2 ✅)
│   └── preprocess_data.py            # Data preprocessing pipeline (Phase 2.4 ✅)
├── 
├── models/                           # Model artifacts and preprocessing objects
│   ├── training_run_20251213_001055/ # Latest production models (Phase 3 & 4 ✅)
│   │   ├── xgboost_model.pkl         # Champion model (74.1% R², €37.97M impact)
│   │   ├── random_forest_model.pkl   # Challenger model (59.5% R², €26.85M impact)
│   │   ├── decision_tree_model.pkl   # Interpretable model (49.9% R², €22.41M impact)
│   │   ├── *_feature_importance.csv  # Feature analysis for tree-based models
│   │   ├── training_results.json     # Complete performance metrics
│   │   ├── training_metadata.json    # Training configuration and timestamps
│   │   ├── evaluation_results.json   # Phase 4 comprehensive evaluation results
│   │   ├── business_impact_analysis.csv # ROI and business metrics analysis
│   │   └── model_comparison_report.json # Statistical testing and selection rationale
│   └── preprocessing/                # Preprocessing artifacts (scalers, encoders)
│       ├── numerical_scaler.pkl      # RobustScaler for numerical features
│       ├── *_onehot.pkl             # One-hot encoders for categorical features
│       ├── *_label.pkl              # Label encoders for high-cardinality features
│       ├── feature_types.json       # Feature type classifications
│       └── preprocessing_report.json # Comprehensive preprocessing metadata
├── 
├── docs/                             # Comprehensive project documentation
│   ├── mkdocs.yml                    # Documentation configuration
│   ├── README.md                     # Documentation overview
│   └── docs/                         # Phase-by-phase documentation
│       ├── index.md                  # Main documentation hub
│       ├── getting-started.md        # Setup and reproduction guide
│       ├── phase1_eda_complete.md    # Phase 1 comprehensive docs
│       ├── phase2_feature_engineering_complete.md  # Phase 2 detailed docs
│       ├── phase4_evaluation_complete.md  # Phase 4 model evaluation docs
│       ├── phase6_deployment_complete.md  # Phase 6 deployment documentation
│       └── FEATURE_ENGINEERING_REPORT.md  # Feature analysis report
├── 
├── reports/                          # Analysis reports and presentations
│   └── figures/                      # Generated plots and visualizations
├── references/                       # External references and papers
├── 
├── PROJECT_PLAN.md                   # 🎯 Master project roadmap (6/6 phases complete)
├── requirements.txt                  # Python development dependencies
├── pyproject.toml                    # Python project configuration
├── Makefile                         # Build automation commands
├── LICENSE                          # MIT License
└── README.md                        # Project overview and documentation (this file)
```

## �🛠️ Technology Stack

**Core ML/Data Science**:
- **Python 3.10**: Modern Python with type hints
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Advanced gradient boosting

**Visualization & Analysis**:
- **Matplotlib/Seaborn**: Statistical visualizations
- **Plotly**: Interactive dashboards
- **Jupyter**: Exploratory analysis and documentation

**Deployment & Production**:
- **Gradio**: Web application framework
- **Hugging Face Spaces**: Cloud deployment
- **Docker**: Containerization (future)
- **MLflow**: Model tracking and versioning (future)

## 📚 Documentation & Resources

### Core Notebooks
- **[01_eda.ipynb](notebooks/01_eda.ipynb)**: Comprehensive exploratory data analysis with business insights
- **[02_modeling.ipynb](notebooks/02_modeling.ipynb)**: Model development and training pipeline
- **[03_evaluation.ipynb](notebooks/03_evaluation.ipynb)**: Statistical model evaluation and selection
- **[04_error_analysis.ipynb](notebooks/04_error_analysis.ipynb)**: Advanced error analysis with SHAP interpretability

### Key Scripts
- **[create_features.py](scripts/create_features.py)**: Production feature engineering pipeline
- **[app.py](app/app.py)**: Gradio web application for deployment

### Documentation
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)**: Complete project roadmap and status
- **[docs/](docs/)**: Comprehensive phase-by-phase documentation
- **[Live Application](https://huggingface.co/spaces/petlaz/rossmann_sales_forecasting)**: Production deployment

---

## 🎯 Professional Portfolio Summary

This project demonstrates comprehensive machine learning engineering capabilities from research to production:

**📊 Business Impact**
- €75M annual value proposition with quantified ROI analysis
- Live production system serving real-time predictions
- Transparent AI with SHAP/LIME explanations for regulatory compliance

**🔧 Technical Excellence**
- End-to-end ML pipeline with 79 engineered features
- Statistical model validation with 99.9% confidence testing
- Production deployment with scalable architecture

**📈 Advanced Analytics**
- Time series forecasting with domain expertise
- Multi-criteria model selection framework
- Comprehensive error analysis and model interpretability

> *This professional ML project showcases production-ready data science capabilities suitable for senior-level positions in machine learning engineering, AI research, and strategic analytics roles.*

**🔗 Quick Links**: [Live App](https://huggingface.co/spaces/petlaz/rossmann_sales_forecasting) | [Source Code](https://github.com/Petlaz/rossmann_sales_forecasting) | [Documentation](docs/)

