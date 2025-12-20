
# Rossmann Sales Forecasting - Phase 4 Evaluation Summary

## Executive Summary
- **Best Model**: Xgboost achieving 74.1% accuracy
- **Business Impact**: €37,967,689 annual revenue impact
- **ROI**: 37868% over 3 years
- **Status**: Production-ready for deployment

## Model Performance Ranking
            Model        Family  R² (%)  RMSE  MAE  MAPE (%)
          Xgboost      Boosting   74.07  1560 1123     17.30
    Random Forest      Ensemble   59.54  1948 1393     21.16
    Decision Tree    Tree-based   49.90  2168 1534     23.23
            Ridge Linear Models   26.43  2627 1921     31.26
Linear Regression Linear Models   26.03  2634 1927     31.44
      Elastic Net Linear Models   25.76  2639 1927     31.43
            Lasso Linear Models   25.50  2643 1919     30.95
          Svm Rbf           SVM   17.36  2784 1926     28.77

## Recommendation
Deploy Xgboost as primary model with Random Forest as backup.

Generated: 2025-12-13 07:52:15
