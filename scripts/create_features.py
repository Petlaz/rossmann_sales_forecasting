#!/usr/bin/env python3
"""
Quick Feature Engineering Pipeline for Rossmann Sales Forecasting
Professional-grade feature engineering demonstrating advanced ML skills
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

def load_and_merge_data():
    """Load and merge all datasets."""
    print("📂 Loading datasets...")
    
    # Define paths
    data_path = Path("data/raw")
    
    # Load datasets
    train_df = pd.read_csv(data_path / "train.csv")
    store_df = pd.read_csv(data_path / "store.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    # Merge with store data
    train_merged = train_df.merge(store_df, on='Store', how='left')
    test_merged = test_df.merge(store_df, on='Store', how='left')
    
    print(f"✅ Data loaded: Train {train_merged.shape}, Test {test_merged.shape}")
    return train_merged, test_merged

def engineer_comprehensive_features(df, is_train=True):
    """
    Create comprehensive feature set demonstrating advanced feature engineering.
    
    This function showcases:
    - Temporal feature engineering with cyclical encoding
    - Business domain knowledge application
    - Advanced statistical transformations  
    - Missing value handling strategies
    - Lag and rolling window features
    """
    print(f"🔧 Engineering features for {'training' if is_train else 'test'} set...")
    df = df.copy()
    
    # ==================== DATA CLEANING ====================
    print("   → Cleaning and imputing missing values...")
    
    # Smart missing value imputation based on EDA insights
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)
    
    # Missing value indicators (important for model interpretability)
    df['HasCompetitionInfo'] = ((df['CompetitionOpenSinceMonth'] > 0) & 
                               (df['CompetitionOpenSinceYear'] > 0)).astype(int)
    df['HasPromo2Info'] = ((df['Promo2SinceWeek'] > 0) & 
                          (df['Promo2SinceYear'] > 0)).astype(int)
    
    # ==================== TEMPORAL FEATURES ====================
    print("   → Creating advanced temporal features...")
    
    # Convert to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Basic temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Advanced temporal features
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
    df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding (crucial for ML models to understand temporal cycles)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['WeekOfYear_sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
    df['WeekOfYear_cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # Season mapping
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring', 
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['Season'] = df['Month'].map(season_map)
    
    # Time-based features
    df['DaysSince2013'] = (df['Date'] - pd.to_datetime('2013-01-01')).dt.days
    
    # ==================== COMPETITION FEATURES ====================
    print("   → Engineering competition features...")
    
    # Competition age calculation
    df['CompetitionAge_Months'] = np.where(
        (df['CompetitionOpenSinceYear'] > 0) & (df['CompetitionOpenSinceMonth'] > 0),
        (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + 
        (df['Month'] - df['CompetitionOpenSinceMonth']),
        0
    )
    df['CompetitionAge_Months'] = np.maximum(df['CompetitionAge_Months'], 0)
    
    # Competition distance transformations
    df['CompetitionDistance_Log'] = np.log1p(df['CompetitionDistance'])
    df['CompetitionDistance_Sqrt'] = np.sqrt(df['CompetitionDistance'])
    
    # Competition proximity categories (based on EDA insights)
    df['CompetitionProximity'] = pd.cut(
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].max()),
        bins=[0, 500, 1000, 2000, 5000, float('inf')],
        labels=[4, 3, 2, 1, 0]  # Higher score = closer competition
    ).astype(int)
    
    # ==================== PROMOTION FEATURES ====================
    print("   → Creating promotion features...")
    
    # Promo2 age calculation
    df['Promo2Age_Weeks'] = np.where(
        (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0),
        (df['Year'] - df['Promo2SinceYear']) * 52 + 
        (df['WeekOfYear'] - df['Promo2SinceWeek']),
        0
    )
    df['Promo2Age_Weeks'] = np.maximum(df['Promo2Age_Weeks'], 0)
    
    # PromoInterval parsing
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    df['InPromoInterval'] = 0
    for idx, row in df.iterrows():
        if row['PromoInterval'] != 'None' and pd.notna(row['PromoInterval']):
            try:
                promo_months = [month_map.get(month.strip(), 0) 
                               for month in row['PromoInterval'].split(',')]
                if row['Month'] in promo_months:
                    df.at[idx, 'InPromoInterval'] = 1
            except:
                pass
    
    # Combined promotion features
    df['AnyPromo'] = ((df['Promo'] == 1) | (df['InPromoInterval'] == 1)).astype(int)
    df['PromoIntensity'] = df['Promo'] + df['InPromoInterval']
    
    # ==================== STORE FEATURES ====================
    print("   → Developing store-specific features...")
    
    # Store type combinations
    df['StoreType_Assortment'] = df['StoreType'].astype(str) + '_' + df['Assortment'].astype(str)
    
    # Store performance tiers (only for training data with Sales)
    if is_train and 'Sales' in df.columns:
        store_performance = df[df['Open'] == 1].groupby('Store').agg({
            'Sales': ['mean', 'std', 'count'],
            'Customers': ['mean', 'std']
        }).round(2)
        
        store_performance.columns = ['Sales_Mean', 'Sales_Std', 'Sales_Count', 
                                   'Customers_Mean', 'Customers_Std']
        store_performance['Sales_CV'] = store_performance['Sales_Std'] / store_performance['Sales_Mean']
        store_performance['SalesPerCustomer'] = store_performance['Sales_Mean'] / store_performance['Customers_Mean']
        
        # Performance categories
        quantiles = store_performance['Sales_Mean'].quantile([0.33, 0.67])
        store_performance['PerformanceTier'] = pd.cut(
            store_performance['Sales_Mean'],
            bins=[0, quantiles.iloc[0], quantiles.iloc[1], float('inf')],
            labels=[0, 1, 2]  # Low, Medium, High
        ).astype(int)
        
        # Merge back
        df = df.merge(store_performance.reset_index(), on='Store', how='left')
        
        # Create lag features (demonstrating time series expertise)
        print("   → Computing lag and rolling window features...")
        df_sorted = df.sort_values(['Store', 'Date'])
        
        # Sales lag features
        for lag_days in [7, 14, 30]:
            df_sorted[f'Sales_Lag_{lag_days}d'] = df_sorted.groupby('Store')['Sales'].shift(lag_days)
        
        # Rolling window features
        for window in [7, 14, 30]:
            df_sorted[f'Sales_Roll_Mean_{window}d'] = (df_sorted.groupby('Store')['Sales']
                                                      .rolling(window=window, min_periods=1)
                                                      .mean().reset_index(0, drop=True))
            df_sorted[f'Sales_Roll_Std_{window}d'] = (df_sorted.groupby('Store')['Sales']
                                                     .rolling(window=window, min_periods=1)
                                                     .std().reset_index(0, drop=True))
        
        # Customer lag features
        for lag_days in [7, 14]:
            df_sorted[f'Customers_Lag_{lag_days}d'] = df_sorted.groupby('Store')['Customers'].shift(lag_days)
        
        df = df_sorted.copy()
        
        # Fill lag feature NaNs with medians
        lag_columns = [col for col in df.columns if 'Lag_' in col or 'Roll_' in col]
        for col in lag_columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # ==================== HOLIDAY FEATURES ====================
    print("   → Processing holiday effects...")
    
    # Enhanced holiday features
    df['StateHoliday_Binary'] = (df['StateHoliday'] != '0').astype(int)
    df['AnyHoliday'] = ((df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1)).astype(int)
    df['HolidayIntensity'] = df['StateHoliday_Binary'] + df['SchoolHoliday']
    
    # Special events (German calendar knowledge)
    df['IsDecember'] = (df['Month'] == 12).astype(int)  # Christmas season
    df['IsJanuary'] = (df['Month'] == 1).astype(int)    # Post-holiday period
    
    # ==================== ADVANCED TRANSFORMATIONS ====================
    print("   → Applying advanced transformations...")
    
    # Target encoding for categorical variables (if training data)
    if is_train and 'Sales' in df.columns:
        # Store type target encoding
        store_type_encoding = df[df['Sales'] > 0].groupby('StoreType')['Sales'].mean()
        df['StoreType_TargetEnc'] = df['StoreType'].map(store_type_encoding)
        
        # Assortment target encoding  
        assortment_encoding = df[df['Sales'] > 0].groupby('Assortment')['Sales'].mean()
        df['Assortment_TargetEnc'] = df['Assortment'].map(assortment_encoding)
    
    # Interaction features (demonstrating feature engineering creativity)
    df['Promo_Weekend'] = df['Promo'] * df['IsWeekend']
    df['Promo_Holiday'] = df['Promo'] * df['AnyHoliday']
    df['Competition_Store'] = df['CompetitionProximity'] * df['StoreType'].astype('category').cat.codes
    
    print(f"✅ Feature engineering complete! Created {len(df.columns)} total features")
    return df

def create_final_datasets():
    """Create production-ready datasets with comprehensive features."""
    
    print("🚀 Starting comprehensive feature engineering pipeline...")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_and_merge_data()
    
    # Engineer features
    train_processed = engineer_comprehensive_features(train_df, is_train=True)
    test_processed = engineer_comprehensive_features(test_df, is_train=False)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare modeling datasets
    print("\n📊 Preparing modeling datasets...")
    
    # Remove zero sales for training (stores were closed)
    train_modeling = train_processed[train_processed['Sales'] > 0].copy()
    
    # Time-based train/validation split for proper forecasting validation
    train_modeling = train_modeling.sort_values('Date')
    split_date = train_modeling['Date'].quantile(0.8)
    
    train_final = train_modeling[train_modeling['Date'] <= split_date]
    val_final = train_modeling[train_modeling['Date'] > split_date]
    
    # Save datasets
    print("\n💾 Saving processed datasets...")
    
    datasets = {
        'train_processed_full': train_processed,
        'test_processed': test_processed, 
        'train_modeling': train_final,
        'val_modeling': val_final
    }
    
    for name, dataset in datasets.items():
        filepath = output_dir / f"{name}.csv"
        dataset.to_csv(filepath, index=False)
        print(f"   ✅ {name}: {dataset.shape} → {filepath}")
    
    # Create feature documentation
    feature_info = {
        'creation_date': datetime.now().isoformat(),
        'total_features': len(train_processed.columns),
        'feature_categories': {
            'temporal': len([col for col in train_processed.columns if any(x in col for x in ['Month', 'Week', 'Day', 'Year', 'sin', 'cos', 'Season'])]),
            'competition': len([col for col in train_processed.columns if 'Competition' in col]),
            'promotion': len([col for col in train_processed.columns if 'Promo' in col]),
            'store': len([col for col in train_processed.columns if 'Store' in col]),
            'holiday': len([col for col in train_processed.columns if 'Holiday' in col]),
            'lag_features': len([col for col in train_processed.columns if 'Lag_' in col or 'Roll_' in col]),
            'interactions': len([col for col in train_processed.columns if '_' in col and 'Enc' in col])
        },
        'modeling_info': {
            'train_records': len(train_final),
            'val_records': len(val_final),
            'test_records': len(test_processed),
            'split_date': str(split_date),
            'target_variable': 'Sales'
        },
        'key_engineered_features': [
            'Cyclical temporal encoding (sin/cos transformations)',
            'Competition age and proximity features',
            'Promotion interval parsing and encoding', 
            'Store performance tiers and statistics',
            'Lag and rolling window features (7d, 14d, 30d)',
            'Target encoding for categorical variables',
            'Interaction features for business insights',
            'Advanced missing value handling'
        ]
    }
    
    with open(output_dir / "feature_engineering_report.json", 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    
    print(f"\n🎯 FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)
    print(f"📊 Total Features Created: {len(train_processed.columns)}")
    print(f"🎯 Training Records: {len(train_final):,}")
    print(f"✅ Validation Records: {len(val_final):,}")
    print(f"🔮 Test Records: {len(test_processed):,}")
    print(f"📁 Output Directory: {output_dir.absolute()}")
    print("\n💡 Ready for Phase 3: Model Training!")
    
    return train_processed, test_processed, train_final, val_final

if __name__ == "__main__":
    # Execute comprehensive feature engineering
    train_processed, test_processed, train_final, val_final = create_final_datasets()