#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Rossmann Sales Forecasting
Completes Phase 2.4: Data Preprocessing requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class RossmannDataPreprocessor:
    """
    Complete data preprocessing pipeline for Rossmann sales forecasting.
    
    Handles:
    - Numerical feature scaling/normalization
    - Categorical encoding (one-hot, label, target)  
    - Train/validation/test split strategy
    - Data leakage prevention
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.preprocessing_info = {}
        
    def identify_feature_types(self, df):
        """Identify and categorize features for appropriate preprocessing."""
        print("🔍 Analyzing feature types...")
        
        # Exclude target and ID columns
        exclude_cols = ['Sales', 'Customers', 'Date', 'Store']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Categorize features
        numerical_features = []
        categorical_features = []
        binary_features = []
        
        # Predefined categorical features that should be treated as categorical
        known_categorical = ['StoreType', 'Assortment', 'StateHoliday', 'Season', 'StoreType_Assortment']
        
        for col in feature_cols:
            if col in exclude_cols:
                continue
            
            # Check if it's a known categorical feature
            if col in known_categorical:
                categorical_features.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                unique_vals = df[col].nunique()
                # Check for binary features (0/1 or similar)
                unique_values = sorted(df[col].dropna().unique())
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    binary_features.append(col)
                elif unique_vals > 50:  # Continuous numerical
                    numerical_features.append(col)
                elif unique_vals <= 10 and col not in ['DayOfWeek']:  # Small discrete
                    # Check if values are ordinal or nominal
                    if col in ['PerformanceTier', 'CompetitionProximity']:
                        categorical_features.append(col)  # Ordinal categories
                    else:
                        numerical_features.append(col)  # Keep as numerical
                else:
                    numerical_features.append(col)
            else:  # String/object type
                categorical_features.append(col)
        
        self.feature_columns = {
            'numerical': numerical_features,
            'categorical': categorical_features, 
            'binary': binary_features,
            'target': 'Sales',
            'auxiliary': 'Customers'
        }
        
        print(f"   ✅ Numerical features: {len(numerical_features)}")
        print(f"   ✅ Categorical features: {len(categorical_features)}")
        print(f"   ✅ Binary features: {len(binary_features)}")
        
        return self.feature_columns
    
    def encode_categorical_features(self, train_df, test_df=None):
        """
        Apply comprehensive categorical encoding strategies.
        """
        print("🏷️  Encoding categorical features...")
        
        train_encoded = train_df.copy()
        test_encoded = test_df.copy() if test_df is not None else None
        
        categorical_features = self.feature_columns.get('categorical', [])
        
        for feature in categorical_features:
            if feature not in train_df.columns:
                continue
            
            # Skip features not present in test set (training-only derived features)
            if test_df is not None and feature not in test_df.columns:
                print(f"   → Skipping {feature} (not in test set)")
                continue
                
            print(f"   → Encoding {feature}")
            unique_vals = train_df[feature].nunique()
            
            if unique_vals <= 5:  # One-hot encoding for low cardinality
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                
                # Ensure consistent string format
                train_feature_data = train_df[[feature]].astype(str)
                
                # Fit on training data
                feature_encoded = encoder.fit_transform(train_feature_data)
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
                
                # Apply to train
                encoded_df = pd.DataFrame(feature_encoded, columns=feature_names, index=train_df.index)
                train_encoded = pd.concat([train_encoded.drop(columns=[feature]), encoded_df], axis=1)
                
                # Apply to test if provided
                if test_encoded is not None:
                    test_feature_data = test_df[[feature]].astype(str)
                    test_feature_encoded = encoder.transform(test_feature_data)
                    test_encoded_df = pd.DataFrame(test_feature_encoded, columns=feature_names, index=test_df.index)
                    test_encoded = pd.concat([test_encoded.drop(columns=[feature]), test_encoded_df], axis=1)
                
                self.encoders[f"{feature}_onehot"] = encoder
                
            else:  # Label encoding for high cardinality
                encoder = LabelEncoder()
                
                # Fit on training data
                train_encoded[f"{feature}_encoded"] = encoder.fit_transform(train_df[feature].astype(str))
                
                # Apply to test if provided
                if test_encoded is not None:
                    # Handle unknown categories in test set
                    test_values = test_df[feature].astype(str)
                    test_encoded[f"{feature}_encoded"] = test_values.map(
                        dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    ).fillna(-1).astype(int)  # -1 for unknown categories
                
                # Remove original column
                train_encoded.drop(columns=[feature], inplace=True)
                if test_encoded is not None:
                    test_encoded.drop(columns=[feature], inplace=True)
                
                self.encoders[f"{feature}_label"] = encoder
        
        print(f"   ✅ Categorical encoding complete")
        return train_encoded, test_encoded
    
    def scale_numerical_features(self, train_df, val_df=None, test_df=None):
        """
        Apply robust scaling to numerical features.
        """
        print("📊 Scaling numerical features...")
        
        train_scaled = train_df.copy()
        val_scaled = val_df.copy() if val_df is not None else None
        test_scaled = test_df.copy() if test_df is not None else None
        
        numerical_features = self.feature_columns.get('numerical', [])
        
        # Use RobustScaler (less sensitive to outliers)
        scaler = RobustScaler()
        
        if numerical_features:
            # Find common numerical columns across all datasets
            common_cols = set(numerical_features)
            if test_scaled is not None:
                common_cols = common_cols.intersection(set(test_df.columns))
            if val_scaled is not None:
                common_cols = common_cols.intersection(set(val_df.columns))
            
            numerical_cols = [col for col in numerical_features if col in common_cols and col in train_df.columns]
            
            if numerical_cols:
                print(f"   → Scaling {len(numerical_cols)} numerical features")
                
                # Fit and transform training data
                train_scaled[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
                
                # Transform validation data
                if val_scaled is not None:
                    val_cols = [col for col in numerical_cols if col in val_df.columns]
                    if val_cols:
                        val_scaled[val_cols] = scaler.transform(val_df[val_cols])
                
                # Transform test data
                if test_scaled is not None:
                    test_cols = [col for col in numerical_cols if col in test_df.columns]
                    if test_cols:
                        test_scaled[test_cols] = scaler.transform(test_df[test_cols])
                
                self.scalers['numerical_scaler'] = scaler
                
                print(f"   ✅ Numerical scaling complete")
        
        return train_scaled, val_scaled, test_scaled
    
    def prepare_modeling_datasets(self, train_df, val_df, test_df):
        """
        Complete preprocessing pipeline preparing datasets for modeling.
        """
        print("\n🚀 Starting comprehensive data preprocessing...")
        print("=" * 60)
        
        # 1. Identify feature types
        self.identify_feature_types(train_df)
        
        # 2. Handle categorical encoding
        train_encoded, test_encoded = self.encode_categorical_features(train_df, test_df)
        
        # Apply same encoding to validation set
        val_encoded, _ = self.encode_categorical_features(val_df, None)
        
        # 3. Scale numerical features
        train_final, val_final, test_final = self.scale_numerical_features(
            train_encoded, val_encoded, test_encoded
        )
        
        # 4. Prepare feature matrices and target vectors
        print("🎯 Preparing final modeling datasets...")
        
        # Define feature columns (common across all datasets)
        exclude_cols = ['Sales', 'Customers', 'Date', 'Store']
        
        # Get common feature columns across all datasets
        train_features = set(train_final.columns) - set(exclude_cols)
        val_features = set(val_final.columns) - set(exclude_cols)  
        test_features = set(test_final.columns) - set(exclude_cols)
        
        # Use intersection of all feature sets for consistency
        common_features = train_features.intersection(val_features).intersection(test_features)
        feature_cols = sorted(list(common_features))
        
        print(f"   → Using {len(feature_cols)} common features across all datasets")
        
        # Create feature matrices
        X_train = train_final[feature_cols]
        y_train = train_final['Sales']
        
        X_val = val_final[feature_cols]  
        y_val = val_final['Sales']
        
        X_test = test_final[feature_cols]
        
        # Store preprocessing information
        self.preprocessing_info = {
            'creation_date': datetime.now().isoformat(),
            'feature_count': len(feature_cols),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_types': self.feature_columns,
            'scaling_method': 'RobustScaler',
            'encoding_methods': list(self.encoders.keys()),
            'target_variable': 'Sales'
        }
        
        print(f"   ✅ Training set: {X_train.shape}")
        print(f"   ✅ Validation set: {X_val.shape}")
        print(f"   ✅ Test set: {X_test.shape}")
        print(f"   ✅ Total features: {len(feature_cols)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val, 
            'X_test': X_test,
            'feature_names': feature_cols,
            'train_full': train_final,
            'val_full': val_final,
            'test_full': test_final
        }
    
    def save_preprocessing_artifacts(self, output_dir):
        """Save all preprocessing artifacts for reproducibility."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving preprocessing artifacts to {output_path}...")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, output_path / f"{scaler_name}.pkl")
            print(f"   ✅ {scaler_name}.pkl")
        
        # Save encoders  
        for encoder_name, encoder in self.encoders.items():
            joblib.dump(encoder, output_path / f"{encoder_name}.pkl")
            print(f"   ✅ {encoder_name}.pkl")
        
        # Save feature information
        with open(output_path / "feature_types.json", 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        # Save preprocessing report
        with open(output_path / "preprocessing_report.json", 'w') as f:
            json.dump(self.preprocessing_info, f, indent=2, default=str)
        
        print(f"   ✅ preprocessing_report.json")
        print(f"   ✅ feature_types.json")

def main():
    """Execute complete data preprocessing pipeline."""
    
    # Load the feature-engineered datasets from Phase 2
    print("📂 Loading feature-engineered datasets...")
    data_path = Path("data/processed")
    
    train_df = pd.read_csv(data_path / "train_modeling.csv")
    val_df = pd.read_csv(data_path / "val_modeling.csv") 
    test_df = pd.read_csv(data_path / "test_processed.csv")
    
    print(f"✅ Loaded: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}")
    
    # Initialize preprocessor
    preprocessor = RossmannDataPreprocessor()
    
    # Execute preprocessing pipeline
    datasets = preprocessor.prepare_modeling_datasets(train_df, val_df, test_df)
    
    # Save processed datasets
    output_dir = Path("data/processed")
    
    print(f"\n💾 Saving modeling-ready datasets...")
    datasets['X_train'].to_csv(output_dir / "X_train.csv", index=False)
    datasets['X_val'].to_csv(output_dir / "X_val.csv", index=False)  
    datasets['X_test'].to_csv(output_dir / "X_test.csv", index=False)
    
    pd.Series(datasets['y_train']).to_csv(output_dir / "y_train.csv", index=False, header=['Sales'])
    pd.Series(datasets['y_val']).to_csv(output_dir / "y_val.csv", index=False, header=['Sales'])
    
    # Save feature names
    pd.Series(datasets['feature_names']).to_csv(output_dir / "feature_names.csv", index=False, header=['feature'])
    
    # Save preprocessing artifacts
    preprocessor.save_preprocessing_artifacts("models/preprocessing")
    
    print(f"\n🎯 DATA PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Training Features: {datasets['X_train'].shape}")
    print(f"✅ Validation Features: {datasets['X_val'].shape}")
    print(f"🔮 Test Features: {datasets['X_test'].shape}")
    print(f"🎯 Total Feature Count: {len(datasets['feature_names'])}")
    print(f"📁 Artifacts saved to: models/preprocessing/")
    print(f"\n💡 Ready for Phase 3: Model Training!")

if __name__ == "__main__":
    main()