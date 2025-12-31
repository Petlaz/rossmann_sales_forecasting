title: 🏪 Rossmann Sales Forecasting
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
header: mini
---

# 🏪 Rossmann Sales Forecasting

Advanced ML-powered sales prediction with SHAP & LIME explanations for transparent AI decision-making.

## 🚀 Live Demo

Experience real-time sales forecasting with:
- **XGBoost Champion Model** (74.1% R² accuracy)
- **SHAP Analysis** for global feature importance
- **LIME Explanations** for local interpretability
- **Business Intelligence** insights and risk assessment

## 📱 Features

### 🧠 ML Predictions
- **Advanced Modeling**: XGBoost with 70 engineered features
- **Confidence Intervals**: ±€2,535 statistical bounds (90% confidence)
- **Risk Assessment**: Automated business risk scoring
- **Real-time Inference**: Sub-second prediction response

### 📊 Model Interpretability
- **SHAP (SHapley Additive exPlanations)**: Global feature importance across all predictions
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations for specific predictions
- **Visual Analytics**: Interactive Plotly charts with business-friendly insights

### 💼 Business Intelligence
- **Performance Categorization**: High/Medium/Low performing store classification
- **Risk Assessment**: Confidence scoring with actionable recommendations
- **Annual Impact Projections**: Extrapolated business impact analysis

## 🏗️ Architecture

- **Framework**: Gradio 4.0+ for interactive web interface
- **ML Stack**: XGBoost + Scikit-learn + SHAP + LIME
- **Data Processing**: 1M+ transactions, 70 engineered features
- **Deployment**: Hugging Face Spaces with public sharing

## 🚀 Usage

### Input Parameters
- **Store ID**: 1-1115 (Rossmann store identifier)
- **Promotion Status**: Active promotional campaigns
- **School Holiday**: Local school holiday indicator
- **Competition Distance**: Distance to nearest competitor (meters)
- **Prediction Date**: Target date for sales forecast
- **Expected Customers**: Estimated daily customer count (optional)

### Output Information
- **Sales Prediction**: Point estimate with confidence range
- **Risk Assessment**: Business confidence scoring
- **Feature Explanations**: SHAP & LIME interpretability
- **Business Recommendations**: Actionable insights

## 🎯 Model Performance

- **Algorithm**: XGBoost Champion (Phase 4 winner)
- **Accuracy**: R² = 74.1% on validation set
- **Error Rate**: RMSE = €1,560 average error
- **Business ROI**: €75M annual validated benefits
- **Training Data**: 1M+ historical transactions

## 🔬 Interpretability Methods

### SHAP (SHapley Additive exPlanations)
- **Purpose**: Global feature importance across all predictions
- **Method**: Game theory-based fair attribution
- **Visualization**: Horizontal bar chart with positive/negative impacts
- **Use Case**: Understanding overall model behavior

### LIME (Local Interpretable Model-agnostic Explanations)
- **Purpose**: Local explanation for specific prediction
- **Method**: Approximation around individual data point
- **Visualization**: Feature contribution chart for this prediction
- **Use Case**: Understanding individual decision reasoning
- **Model Path**: `../models/training_run_20251213_001055/`
- **Preprocessing**: `../models/preprocessing/`
- **Feature Engineering**: Automated pipeline
- **Mock Mode**: Fallback when models unavailable

### Configuration
- **Port**: 7860 (configurable in app.py)
- **Host**: 0.0.0.0 for external access
- **Share**: Public link generation available
- **Debug**: Enabled for development

## 📊 Input Parameters

### Store Configuration
- **Store ID**: 1-1115 (Rossmann store identifier)
- **Promotion**: Active/Inactive promotional status
- **School Holiday**: Holiday period flag
- **Competition Distance**: Distance to nearest competitor (meters)

### Prediction Settings  
- **Date**: Target prediction date
- **Customers**: Expected daily customer count (optional)

## 📈 Output Analysis

### Sales Forecast
- **Point Estimate**: Daily sales prediction in Euros
- **Confidence Range**: Statistical confidence bounds
- **Annual Projection**: 365-day extrapolation
- **Performance Tier**: High/Medium/Low categorization

### Risk Assessment
- **Confidence Score**: Prediction reliability (0.5-0.95)
- **Risk Level**: Low/Medium/High business risk
- **Monitoring Priority**: Attention level required
- **Business Impact**: Revenue significance

### SHAP Interpretation
- **Feature Ranking**: Top 5 influential factors
- **Impact Direction**: Positive/negative effects
- **Magnitude**: Quantified feature contribution
- **Visual Chart**: Interactive bar plot

## 🎯 Business Applications

### Inventory Management
- **Stock Planning**: Optimize inventory based on predictions
- **Demand Forecasting**: Anticipate product requirements
- **Supply Chain**: Coordinate deliveries with forecasts

### Staff Scheduling
- **Workforce Planning**: Align staffing with predicted sales
- **Peak Period Preparation**: Handle high-volume days
- **Cost Optimization**: Reduce labor costs during low periods

### Strategic Planning
- **Promotional Timing**: Optimize campaign scheduling
- **Performance Monitoring**: Track store efficiency
- **Market Analysis**: Understand competitive positioning

## 🔍 Troubleshooting

### Model Loading Issues
```bash
# Check model files exist
ls -la ../models/training_run_20251213_001055/
ls -la ../models/preprocessing/

# Verify permissions
chmod 644 ../models/**/*.pkl
```

### Import Errors
```bash
# Install/update dependencies
pip install -r ../requirements.txt --upgrade

# Check Python version
python --version  # Requires 3.10+
```

### Port Conflicts
```bash
# Edit app.py line ~660
app.launch(server_port=8080)  # Change port
```

### Performance Issues
- Reduce SHAP sample size in app.py (line ~180)
- Enable caching for repeated predictions
- Monitor memory usage during operation

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access: http://localhost:7860
```

### Hugging Face Spaces
```bash
# Upload folder to HF Spaces
# config.yaml contains Space configuration
```

### Custom Server
```bash
# Modify launch parameters in app.py
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
## 🛠️ Technical Stack

- **Framework**: Gradio 4.0+ (Hugging Face Spaces compatible)
- **ML Pipeline**: XGBoost + Scikit-learn preprocessing
- **Interpretability**: SHAP + LIME explanations
- **Visualization**: Plotly interactive charts
- **Deployment**: Hugging Face Spaces with public sharing

## 📁 Project Context

This application is part of a comprehensive ML engineering project:
- **Phase 1-3**: Data preprocessing, EDA, feature engineering
- **Phase 4**: Model training and selection (XGBoost champion)
- **Phase 5**: Error analysis and business impact assessment  
- **Phase 6**: Production deployment with interpretability (current)

## 🎯 Business Value

- **Automated Forecasting**: Replace manual sales estimation
- **Risk Management**: Confidence-based decision making
- **Resource Optimization**: Data-driven inventory and staffing
- **Transparent AI**: SHAP & LIME explanations build trust

## 🔮 Demo Instructions

1. **Select Store**: Choose store ID (1-1115)
2. **Set Conditions**: Configure promotions and holidays
3. **Pick Date**: Select prediction target date
4. **View Results**: Analyze prediction + explanations
5. **Business Action**: Use recommendations for decisions

---

## 🏆 Acknowledgments

**Project**: Advanced ML Engineering Portfolio  
**Deployment**: Hugging Face Spaces  
**Date**: December 31, 2025  

*Professional sales forecasting with transparent AI and business intelligence* 🚀
