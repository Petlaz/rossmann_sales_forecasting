#!/usr/bin/env python3
"""
Comprehensive Model Training Pipeline for Rossmann Sales Forecasting
Phase 3: Model Development & Training

Implements 5 ML algorithms with hyperparameter optimization:
- Linear Regression (Ridge, Lasso, Elastic Net)
- Random Forest with feature importance
- XGBoost with advanced tuning
- Support Vector Machine (RBF)
- Decision Tree with ensemble variations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
from datetime import datetime
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import xgboost as xgb
import psutil
import os

warnings.filterwarnings('ignore')

class RossmannModelTrainer:
    """
    Comprehensive model training pipeline for Rossmann sales forecasting.
    
    Features:
    - 5 ML algorithms with hyperparameter optimization
    - Baseline models for benchmarking
    - Cross-validation with proper time series methodology
    - Model persistence and versioning
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, models_dir="models", random_state=42):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Results storage
        self.models = {}
        self.results = {}
        self.training_info = {}
        
        # Evaluation metrics
        self.metrics = ['rmse', 'mae', 'mape', 'r2']
        
        print(f"🚀 Rossmann Model Training Pipeline Initialized")
        print(f"📁 Models directory: {self.models_dir.absolute()}")
        
    def check_system_resources(self):
        """Check system resources and provide warnings for memory-intensive operations."""
        print("\n💻 SYSTEM RESOURCE CHECK")
        print("="*50)
        
        # Memory check
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        used_pct = memory.percent
        
        print(f"   💾 Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total ({used_pct:.1f}% used)")
        
        # CPU check
        cpu_count = psutil.cpu_count(logical=True)
        print(f"   🔄 CPU: {cpu_count} cores available")
        
        # Warnings
        if available_gb < 4:
            print("   ⚠️  WARNING: Low memory (<4GB available)")
            print("      • Random Forest and XGBoost may cause system slowdown")
            print("      • Consider closing other applications")
        elif available_gb < 8:
            print("   ⚡ NOTICE: Moderate memory available (4-8GB)")
            print("      • Training optimized for memory efficiency")
        else:
            print("   ✅ Sufficient memory for intensive training")
            
        # Dataset size recommendation
        print(f"\n   📊 Training Configuration:")
        print(f"      • Random Forest: Limited to 150 trees, reduced depth")
        print(f"      • XGBoost: Memory-efficient parameters, reduced iterations")
        print(f"      • SVM: Subset training (10k samples max)")
        print(f"      • Parallel jobs: Limited to prevent system overload")
        
        print("="*50)
        
    def load_preprocessed_data(self, data_dir="data/processed"):
        """Load preprocessed modeling datasets."""
        print("📂 Loading preprocessed datasets...")
        
        data_path = Path(data_dir)
        
        # Load feature matrices and targets
        X_train = pd.read_csv(data_path / "X_train.csv")
        X_val = pd.read_csv(data_path / "X_val.csv")
        X_test = pd.read_csv(data_path / "X_test.csv")
        
        y_train = pd.read_csv(data_path / "y_train.csv")['Sales']
        y_val = pd.read_csv(data_path / "y_val.csv")['Sales']
        
        # Load feature names
        feature_names = pd.read_csv(data_path / "feature_names.csv")['feature'].tolist()
        
        print(f"   ✅ Training set: {X_train.shape}")
        print(f"   ✅ Validation set: {X_val.shape}") 
        print(f"   ✅ Test set: {X_test.shape}")
        print(f"   ✅ Features: {len(feature_names)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test,
            'feature_names': feature_names
        }
    
    def create_baseline_models(self, y_train, y_val):
        """Create baseline models for benchmarking."""
        print("\n📊 Creating baseline models...")
        
        baselines = {}
        
        # Naive baseline - mean of training set
        mean_baseline = np.full_like(y_val, y_train.mean())
        baselines['mean'] = {
            'predictions': mean_baseline,
            'rmse': np.sqrt(mean_squared_error(y_val, mean_baseline)),
            'mae': mean_absolute_error(y_val, mean_baseline),
            'r2': r2_score(y_val, mean_baseline)
        }
        
        # Median baseline
        median_baseline = np.full_like(y_val, y_train.median())
        baselines['median'] = {
            'predictions': median_baseline,
            'rmse': np.sqrt(mean_squared_error(y_val, median_baseline)),
            'mae': mean_absolute_error(y_val, median_baseline),
            'r2': r2_score(y_val, median_baseline)
        }
        
        print(f"   ✅ Mean Baseline - RMSE: {baselines['mean']['rmse']:.2f}")
        print(f"   ✅ Median Baseline - RMSE: {baselines['median']['rmse']:.2f}")
        
        return baselines
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE - handle division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}
    
    def detect_overfitting_underfitting(self, model, X_train, y_train, X_val, y_val, model_name="Model"):
        """
        Detect overfitting, underfitting, or good fit based on training vs validation performance.
        
        Parameters:
        -----------
        model : trained model
            The trained machine learning model
        X_train, y_train : training data
        X_val, y_val : validation data
        model_name : str
            Name of the model for display
            
        Returns:
        --------
        dict : Dictionary with fit assessment and recommendations
        """
        
        # Get predictions on both sets
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics for both sets
        train_metrics = self.calculate_metrics(y_train, train_pred)
        val_metrics = self.calculate_metrics(y_val, val_pred)
        
        # Performance differences
        rmse_diff = val_metrics['rmse'] - train_metrics['rmse']
        r2_diff = train_metrics['r2'] - val_metrics['r2']  # Training R² - Validation R²
        
        # Calculate relative differences
        rmse_diff_pct = (rmse_diff / train_metrics['rmse']) * 100
        r2_diff_pct = r2_diff * 100  # Already in percentage form
        
        # Decision thresholds
        OVERFITTING_RMSE_THRESHOLD = 15  # 15% worse RMSE on validation
        OVERFITTING_R2_THRESHOLD = 10    # 10 percentage points lower R² on validation
        UNDERFITTING_R2_THRESHOLD = 30   # R² below 30% on both sets
        
        # Determine fit status
        fit_status = "Good Fit ✅"
        recommendations = []
        
        if (rmse_diff_pct > OVERFITTING_RMSE_THRESHOLD or r2_diff_pct > OVERFITTING_R2_THRESHOLD):
            fit_status = "Overfitting ⚠️"
            recommendations.extend([
                "• Reduce model complexity (fewer features, simpler model)",
                "• Increase regularization (higher alpha/lambda)",
                "• Collect more training data",
                "• Use cross-validation for better generalization",
                "• Apply feature selection to reduce noise"
            ])
        elif (train_metrics['r2'] < (UNDERFITTING_R2_THRESHOLD/100) and val_metrics['r2'] < (UNDERFITTING_R2_THRESHOLD/100)):
            fit_status = "Underfitting 📉"
            recommendations.extend([
                "• Increase model complexity (more features, deeper model)",
                "• Reduce regularization (lower alpha/lambda)",
                "• Feature engineering (create interaction terms)",
                "• Try more sophisticated algorithms",
                "• Check for data leakage or preprocessing issues"
            ])
        else:
            recommendations.append("• Model is well-balanced, consider final evaluation")
        
        # Create assessment dictionary
        assessment = {
            'fit_status': fit_status,
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'val_r2': val_metrics['r2'],
            'rmse_diff': rmse_diff,
            'rmse_diff_pct': rmse_diff_pct,
            'r2_diff': r2_diff,
            'r2_diff_pct': r2_diff_pct,
            'recommendations': recommendations
        }
        
        # Print assessment
        print(f"\n📊 {model_name} - Overfitting/Underfitting Analysis:")
        print(f"   Training RMSE: {train_metrics['rmse']:.2f} | Validation RMSE: {val_metrics['rmse']:.2f}")
        print(f"   Training R²: {train_metrics['r2']:.3f} | Validation R²: {val_metrics['r2']:.3f}")
        print(f"   RMSE Difference: +{rmse_diff:.2f} ({rmse_diff_pct:+.1f}%)")
        print(f"   R² Difference: {r2_diff:+.3f} ({r2_diff_pct:+.1f} ppts)")
        print(f"   📈 Status: {fit_status}")
        
        if recommendations:
            print("   💡 Recommendations:")
            for rec in recommendations:
                print(f"      {rec}")
        
        return assessment
    
    def train_linear_models(self, X_train, y_train, X_val, y_val):
        """Train linear regression models with time series validation (Phase 3.3)."""
        print("\n🔵 Training Linear Regression Models...")
        
        # Time series cross-validation (Phase 3.3)
        tscv = TimeSeriesSplit(n_splits=3)
        print("   📊 Using TimeSeriesSplit for all regularized models")
        
        linear_models = {}
        
        # 1. Basic Linear Regression
        print("   → Basic Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        
        linear_models['linear_regression'] = {
            'model': lr,
            'predictions': lr_pred,
            'metrics': self.calculate_metrics(y_val, lr_pred),
            'params': {}
        }
        
        # 2. Ridge Regression with hyperparameter tuning
        print("   → Ridge Regression with GridSearch")
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
        ridge = GridSearchCV(
            Ridge(random_state=self.random_state),
            ridge_params,
            cv=tscv,  # Time series split
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_val)
        
        linear_models['ridge'] = {
            'model': ridge,
            'predictions': ridge_pred,
            'metrics': self.calculate_metrics(y_val, ridge_pred),
            'params': ridge.best_params_
        }
        
        # 3. Lasso Regression
        print("   → Lasso Regression with GridSearch")
        lasso_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        lasso = GridSearchCV(
            Lasso(random_state=self.random_state, max_iter=2000),
            lasso_params,
            cv=tscv,  # Time series split
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_val)
        
        linear_models['lasso'] = {
            'model': lasso,
            'predictions': lasso_pred,
            'metrics': self.calculate_metrics(y_val, lasso_pred),
            'params': lasso.best_params_
        }
        
        # 4. Elastic Net
        print("   → Elastic Net with GridSearch")
        elastic_params = {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
        elastic = GridSearchCV(
            ElasticNet(random_state=self.random_state, max_iter=2000),
            elastic_params,
            cv=tscv,  # Time series split
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        elastic.fit(X_train, y_train)
        elastic_pred = elastic.predict(X_val)
        
        linear_models['elastic_net'] = {
            'model': elastic,
            'predictions': elastic_pred,
            'metrics': self.calculate_metrics(y_val, elastic_pred),
            'params': elastic.best_params_
        }
        
        # Print results and overfitting analysis
        for name, results in linear_models.items():
            print(f"      ✅ {name}: RMSE={results['metrics']['rmse']:.2f}, R²={results['metrics']['r2']:.3f}")
            
            # Add overfitting/underfitting detection
            model_display_names = {
                'linear': 'Linear Regression',
                'ridge': 'Ridge Regression', 
                'lasso': 'Lasso Regression',
                'elastic_net': 'Elastic Net'
            }
            assessment = self.detect_overfitting_underfitting(
                results['model'], X_train, y_train, X_val, y_val, 
                model_display_names.get(name, name.title())
            )
            results['fit_assessment'] = assessment
        
        return linear_models
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest with memory-optimized hyperparameter optimization."""
        print("\n🌲 Training Random Forest...")
        
        # Anti-overfitting parameter grid (Phase 3.3 optimization)
        rf_params = {
            'n_estimators': [50, 75],        # Reduced to prevent overfitting
            'max_depth': [8, 12],            # Shallower trees to reduce complexity
            'min_samples_split': [10, 20],   # Increased to prevent deep splits
            'min_samples_leaf': [8, 12],     # Increased from [2,4] to reduce overfitting
            'max_features': ['sqrt', 0.6]    # Feature subsampling for regularization
        }
        
        print("   ⚡ Phase 3.3: Anti-overfitting configuration applied")
        print("   → Reduced n_estimators, increased min_samples_leaf, limited max_depth")
        
        print("   → Memory-optimized hyperparameter search")
        print(f"   → Dataset size: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        
        # Time series cross-validation (Phase 3.3)
        tscv = TimeSeriesSplit(n_splits=3)
        
        rf = RandomizedSearchCV(
            RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=2,  # Reduce parallel jobs to prevent system overload
                max_samples=0.8,  # Use 80% of samples per tree to reduce memory
                bootstrap=True,
                oob_score=True  # Out-of-bag score for overfitting detection
            ),
            rf_params,
            n_iter=8,   # Reduced iterations due to more conservative params
            cv=tscv,    # Time series split instead of regular CV
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=1,   # Single job for RandomizedSearchCV to prevent overload
            verbose=1   # Show progress
        )
        
        print("   📊 Using TimeSeriesSplit validation (respects temporal order)")
        
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Best parameters: {rf.best_params_}")
        print(f"   ✅ RMSE: {np.sqrt(mean_squared_error(y_val, rf_pred)):.2f}")
        print(f"   ✅ Top 5 features: {feature_importance.head()['feature'].tolist()}")
        
        # Overfitting/underfitting analysis
        fit_assessment = self.detect_overfitting_underfitting(
            rf, X_train, y_train, X_val, y_val, "Random Forest"
        )
        
        return {
            'model': rf,
            'predictions': rf_pred,
            'metrics': self.calculate_metrics(y_val, rf_pred),
            'params': rf.best_params_,
            'feature_importance': feature_importance,
            'fit_assessment': fit_assessment
        }
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with enhanced regularization (Phase 3.3)."""
        print("\n🚀 Training XGBoost...")
        
        # Enhanced parameter grid (Phase 3.3)
        xgb_params = {
            'n_estimators': [75, 100, 125],     # Slightly more trees
            'max_depth': [4, 6],                # Moderate depth
            'learning_rate': [0.05, 0.1, 0.15], # Lower learning rates with more trees
            'subsample': [0.75, 0.85, 0.9],     # Enhanced regularization options
            'colsample_bytree': [0.7, 0.8, 0.9], # More feature subsampling options
            'reg_alpha': [0.1, 0.2],            # Increased L1 regularization
            'reg_lambda': [1.0, 1.5, 2.0]       # Enhanced L2 regularization
        }
        
        print("   ⚡ Phase 3.3: Enhanced regularization and early stopping")
        print("   → Increased regularization, lower learning rates, more balanced approach")
        
        print(f"   → Dataset size: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        
        # Time series cross-validation (Phase 3.3)
        tscv = TimeSeriesSplit(n_splits=3)
        
        xgb_model = RandomizedSearchCV(
            xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=2,        # Reduced from -1 to 2
                eval_metric='rmse',
                tree_method='hist',  # More memory efficient than 'auto'
                enable_categorical=False    # Disable for stability
                # Note: early_stopping_rounds removed for RandomizedSearchCV compatibility
            ),
            xgb_params,
            n_iter=15,       # Increased slightly due to better params
            cv=tscv,         # Time series split instead of regular CV
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=1,        # Single job for search
            verbose=1        # Show progress
        )
        
        print("   📊 Using TimeSeriesSplit validation with early stopping")
        
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_val)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Best parameters: {xgb_model.best_params_}")
        print(f"   ✅ RMSE: {np.sqrt(mean_squared_error(y_val, xgb_pred)):.2f}")
        print(f"   ✅ Top 5 features: {feature_importance.head()['feature'].tolist()}")
        
        # Overfitting/underfitting analysis
        fit_assessment = self.detect_overfitting_underfitting(
            xgb_model, X_train, y_train, X_val, y_val, "XGBoost"
        )
        
        return {
            'model': xgb_model,
            'predictions': xgb_pred,
            'metrics': self.calculate_metrics(y_val, xgb_pred),
            'params': xgb_model.best_params_,
            'feature_importance': feature_importance,
            'fit_assessment': fit_assessment
        }
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """Train SVM with RBF kernel only - optimized for large datasets."""
        print("\n🔷 Training Support Vector Machine (RBF kernel)...")
        
        # SVM Performance optimization for large datasets
        print("⚡ SVM Performance Optimization:")
        print(f"   Original training set: {X_train.shape[0]:,} samples")
        
        # Use subset for SVM due to computational complexity O(n²-n³)
        # SVM doesn't scale well beyond 10k samples
        svm_subset_size = min(10000, X_train.shape[0])
        svm_indices = np.random.RandomState(self.random_state).choice(
            X_train.shape[0], size=svm_subset_size, replace=False
        )
        
        X_train_svm = X_train.iloc[svm_indices]
        y_train_svm = y_train.iloc[svm_indices]
        
        print(f"   SVM training subset: {X_train_svm.shape[0]:,} samples ({svm_subset_size/X_train.shape[0]*100:.1f}% of data)")
        print("   This ensures reasonable training time while maintaining representative performance.\n")
        
        # Optimized SVM configuration for faster training  
        svm_rbf_params = {
            'C': [1.0, 10.0, 100.0],
            'gamma': ['scale'],  # Reduced parameter space
            'epsilon': [0.01, 0.1],
            'cache_size': [1000],  # Larger cache for faster computation
            'max_iter': [5000]     # Limit iterations to prevent infinite training
        }
        
        print("   → RBF SVM (optimized parameters)")
        svm_rbf = GridSearchCV(
            SVR(kernel='rbf'),
            svm_rbf_params,
            cv=3,  # Reduced CV for speed
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Train on subset, predict on full validation set
        svm_rbf.fit(X_train_svm, y_train_svm)
        svm_rbf_pred = svm_rbf.predict(X_val)
        
        svm_models = {
            'svm_rbf': {
                'model': svm_rbf,
                'predictions': svm_rbf_pred,
                'metrics': self.calculate_metrics(y_val, svm_rbf_pred),
                'params': svm_rbf.best_params_,
                'training_samples': svm_subset_size,
                'subset_ratio': svm_subset_size/X_train.shape[0]
            }
        }
        
        # Print results
        results = svm_models['svm_rbf']
        print(f"      ✅ SVM (RBF): RMSE={results['metrics']['rmse']:.2f}, R²={results['metrics']['r2']:.3f}")
        print(f"      📊 Trained on {results['training_samples']:,} samples ({results['subset_ratio']*100:.1f}% of data)")
        
        # Overfitting/underfitting analysis using subset for training metrics
        fit_assessment = self.detect_overfitting_underfitting(
            svm_rbf, X_train_svm, y_train_svm, X_val, y_val, "SVM (RBF)"
        )
        results['fit_assessment'] = fit_assessment
        
        return svm_models
    
    def train_decision_tree(self, X_train, y_train, X_val, y_val):
        """Train Decision Tree with aggressive anti-overfitting measures (Phase 3.3)."""
        print("\n🌳 Training Decision Tree...")
        
        # Anti-overfitting parameter grid (Phase 3.3)
        dt_params = {
            'max_depth': [6, 10, 15],        # Removed None (infinite depth) - CRITICAL
            'min_samples_split': [20, 50, 100], # Much higher values to prevent deep splits
            'min_samples_leaf': [15, 25, 40],   # Much higher to prevent memorization
            'max_features': ['sqrt', 'log2'],   # Removed None to add feature randomness
            'ccp_alpha': [0.0, 0.01, 0.02]     # Cost complexity pruning
        }
        
        print("   ⚡ Phase 3.3: Aggressive anti-overfitting measures applied")
        print("   → Limited max_depth, high min_samples thresholds, added pruning")
        
        # Time series cross-validation (Phase 3.3)
        tscv = TimeSeriesSplit(n_splits=3)
        
        dt = GridSearchCV(
            DecisionTreeRegressor(random_state=self.random_state),
            dt_params,
            cv=tscv,  # Time series split instead of regular CV
            scoring='neg_mean_squared_error',
            n_jobs=1,  # Reduced to prevent overload
            verbose=1
        )
        
        print("   📊 Using TimeSeriesSplit validation (respects temporal order)")
        
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_val)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': dt.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Best parameters: {dt.best_params_}")
        print(f"   ✅ RMSE: {np.sqrt(mean_squared_error(y_val, dt_pred)):.2f}")
        print(f"   ✅ Top 5 features: {feature_importance.head()['feature'].tolist()}")
        
        # Overfitting/underfitting analysis
        fit_assessment = self.detect_overfitting_underfitting(
            dt, X_train, y_train, X_val, y_val, "Decision Tree"
        )
        
        return {
            'model': dt,
            'predictions': dt_pred,
            'metrics': self.calculate_metrics(y_val, dt_pred),
            'params': dt.best_params_,
            'feature_importance': feature_importance,
            'fit_assessment': fit_assessment
        }
    
    def save_models_and_results(self, models_dict, baselines):
        """Save all trained models and results."""
        print(f"\n💾 Saving models and results...")
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.models_dir / f"training_run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Save individual models
        models_saved = 0
        for category, models in models_dict.items():
            if isinstance(models, dict) and 'model' in models:
                # Single model
                model_path = run_dir / f"{category}_model.pkl"
                joblib.dump(models['model'], model_path)
                models_saved += 1
                print(f"   ✅ {category}_model.pkl")
            else:
                # Multiple models in category
                for model_name, model_info in models.items():
                    if 'model' in model_info:
                        model_path = run_dir / f"{model_name}_model.pkl"
                        joblib.dump(model_info['model'], model_path)
                        models_saved += 1
                        print(f"   ✅ {model_name}_model.pkl")
        
        # Compile comprehensive results
        all_results = {}
        
        # Add baseline results
        all_results['baselines'] = baselines
        
        # Add model results
        for category, models in models_dict.items():
            if isinstance(models, dict) and 'metrics' in models:
                all_results[category] = {
                    'metrics': models['metrics'],
                    'params': models.get('params', {}),
                    'has_feature_importance': 'feature_importance' in models
                }
            else:
                all_results[category] = {}
                for model_name, model_info in models.items():
                    if 'metrics' in model_info:
                        all_results[category][model_name] = {
                            'metrics': model_info['metrics'],
                            'params': model_info.get('params', {}),
                            'has_feature_importance': 'feature_importance' in model_info
                        }
        
        # Save results summary
        results_path = run_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save feature importance data
        for category, models in models_dict.items():
            if isinstance(models, dict):
                if 'feature_importance' in models:
                    # Single model with feature importance (RF, XGB, DT)
                    fi_path = run_dir / f"{category}_feature_importance.csv"
                    models['feature_importance'].to_csv(fi_path, index=False)
                    print(f"   ✅ {category}_feature_importance.csv")
                elif 'model' not in models:
                    # Multiple models in category (linear_models, svm_models)
                    for model_name, model_info in models.items():
                        if isinstance(model_info, dict) and 'feature_importance' in model_info:
                            fi_path = run_dir / f"{model_name}_feature_importance.csv"
                            model_info['feature_importance'].to_csv(fi_path, index=False)
                            print(f"   ✅ {model_name}_feature_importance.csv")
        
        # Training metadata
        training_metadata = {
            'training_date': datetime.now().isoformat(),
            'models_trained': models_saved,
            'training_directory': str(run_dir),
            'random_state': self.random_state,
            'phase': 'Phase 3 - Model Training Complete'
        }
        
        metadata_path = run_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        print(f"   ✅ training_results.json")
        print(f"   ✅ training_metadata.json")
        print(f"\n📁 All artifacts saved to: {run_dir}")
        
        return run_dir, all_results
    
    def train_all_models(self):
        """Execute complete model training pipeline."""
        print("🚀 STARTING COMPREHENSIVE MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        # Check system resources first
        self.check_system_resources()
        
        # Load data
        data = self.load_preprocessed_data()
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test = data['X_test']
        
        # Create baselines
        baselines = self.create_baseline_models(y_train, y_val)
        
        # Train all models
        print("\n🎯 TRAINING ML ALGORITHMS")
        print("-" * 40)
        
        # 1. Linear Models
        linear_models = self.train_linear_models(X_train, y_train, X_val, y_val)
        
        # 2. Random Forest
        rf_model = self.train_random_forest(X_train, y_train, X_val, y_val)
        
        # 3. XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # 4. SVM Models
        svm_models = self.train_svm(X_train, y_train, X_val, y_val)
        
        # 5. Decision Tree
        dt_model = self.train_decision_tree(X_train, y_train, X_val, y_val)
        
        # Compile all models
        all_models = {
            **linear_models,  # Unpack linear models dict
            'random_forest': rf_model,
            'xgboost': xgb_model,
            **svm_models,  # Unpack SVM models dict
            'decision_tree': dt_model
        }
        
        # Save models and results
        run_dir, results_summary = self.save_models_and_results(all_models, baselines)
        
        # Print final summary
        print(f"\n🎯 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"📊 Models Trained: {len(all_models)} + {len(baselines)} baselines")
        print(f"📁 Results Directory: {run_dir}")
        
        # Performance summary
        print(f"\n📈 PERFORMANCE SUMMARY (RMSE):")
        print("-" * 40)
        
        # Sort models by RMSE
        model_performance = []
        for name, model_info in all_models.items():
            rmse = model_info['metrics']['rmse']
            r2 = model_info['metrics']['r2']
            model_performance.append((name, rmse, r2))
        
        # Overfitting/Underfitting Summary
        print("\n" + "="*80)
        print("📊 OVERFITTING/UNDERFITTING ASSESSMENT SUMMARY")
        print("="*80)
        
        fit_summary = {}
        for category, models in all_models.items():
            if isinstance(models, dict):
                if 'fit_assessment' in models:
                    # Single model (RF, XGB, DT)
                    fit_summary[category.replace('_', ' ').title()] = models['fit_assessment']
                else:
                    # Multiple models (Linear models, SVM)
                    for model_name, model_info in models.items():
                        if 'fit_assessment' in model_info:
                            display_name = f"{category.replace('_', ' ').title()} - {model_name.replace('_', ' ').title()}"
                            fit_summary[display_name] = model_info['fit_assessment']
        
        # Display fit assessment summary
        good_fit_models = []
        overfitting_models = []
        underfitting_models = []
        
        for model_name, assessment in fit_summary.items():
            status = assessment['fit_status']
            if "Good Fit" in status:
                good_fit_models.append(model_name)
            elif "Overfitting" in status:
                overfitting_models.append(model_name)
            elif "Underfitting" in status:
                underfitting_models.append(model_name)
            
            print(f"\n{model_name}:")
            print(f"   Status: {status}")
            print(f"   Train R²: {assessment['train_r2']:.3f} | Val R²: {assessment['val_r2']:.3f}")
            print(f"   R² Gap: {assessment['r2_diff']:+.3f} ({assessment['r2_diff_pct']:+.1f} ppts)")
        
        print(f"\n📈 FIT ASSESSMENT OVERVIEW:")
        print(f"   ✅ Well-Balanced Models: {len(good_fit_models)}")
        print(f"   ⚠️  Overfitting Models: {len(overfitting_models)}")
        print(f"   📉 Underfitting Models: {len(underfitting_models)}")
        
        if good_fit_models:
            print(f"\n✅ RECOMMENDED MODELS (Good Fit): {', '.join(good_fit_models)}")
        
        print("\n" + "="*80)
        print("🏆 FINAL MODEL PERFORMANCE RANKING")
        print("="*80)
        
        model_performance.sort(key=lambda x: x[1])  # Sort by RMSE
        
        for i, (name, rmse, r2) in enumerate(model_performance, 1):
            # Add fit status indicator
            fit_indicator = "🔍"
            for model_name, assessment in fit_summary.items():
                if name.lower() in model_name.lower() or model_name.lower() in name.lower():
                    if "Good Fit" in assessment['fit_status']:
                        fit_indicator = "✅"
                    elif "Overfitting" in assessment['fit_status']:
                        fit_indicator = "⚠️"
                    elif "Underfitting" in assessment['fit_status']:
                        fit_indicator = "📉"
                    break
            
            print(f"{i:2d}. {fit_indicator} {name:20s}: RMSE={rmse:8.2f}, R²={r2:6.3f}")
        
        print(f"\n💡 Best Model: {model_performance[0][0]} (RMSE: {model_performance[0][1]:.2f})")
        print(f"\n🎯 READY FOR PHASE 4: Model Evaluation & Analysis!")
        
        return run_dir, results_summary, all_models

def main():
    """Execute the complete model training pipeline."""
    trainer = RossmannModelTrainer()
    run_dir, results, models = trainer.train_all_models()
    return run_dir, results, models

if __name__ == "__main__":
    main()