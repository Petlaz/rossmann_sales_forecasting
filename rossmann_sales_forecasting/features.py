from pathlib import Path
from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from rossmann_sales_forecasting.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

warnings.filterwarnings('ignore')

app = typer.Typer()


class RossmannFeatureEngineering:
    """
    Comprehensive feature engineering pipeline for Rossmann sales forecasting.
    Based on insights from exploratory data analysis.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.logger = logger
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and merge training, store, and test data."""
        self.logger.info("Loading raw datasets...")
        
        # Load datasets
        train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
        store_df = pd.read_csv(RAW_DATA_DIR / "store.csv") 
        test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
        
        # Merge with store information
        train_merged = train_df.merge(store_df, on='Store', how='left')
        test_merged = test_df.merge(store_df, on='Store', how='left')
        
        self.logger.success(f"Data loaded: Train {train_merged.shape}, Test {test_merged.shape}")
        return train_merged, test_merged, store_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle missing values based on EDA insights."""
        self.logger.info("Cleaning data and handling missing values...")
        df = df.copy()
        
        # Handle missing competition data
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        
        # Handle missing Promo2 data
        df['Promo2SinceWeek'].fillna(0, inplace=True)
        df['Promo2SinceYear'].fillna(0, inplace=True)
        df['PromoInterval'].fillna('None', inplace=True)
        
        # Create missing value indicators (important for model interpretability)
        df['CompetitionDistance_Missing'] = pd.isna(df['CompetitionDistance']).astype(int)
        df['CompetitionInfo_Missing'] = ((df['CompetitionOpenSinceMonth'] == 0) | 
                                        (df['CompetitionOpenSinceYear'] == 0)).astype(int)
        df['Promo2Info_Missing'] = ((df['Promo2SinceWeek'] == 0) | 
                                   (df['Promo2SinceYear'] == 0)).astype(int)
        
        self.logger.success("Data cleaning completed")
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features based on EDA insights."""
        self.logger.info("Creating temporal features...")
        df = df.copy()
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        # Advanced temporal features
        df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
        df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for temporal features (important for ML models)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['WeekOfYear_sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
        df['WeekOfYear_cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)
        
        # Season encoding
        df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Days since important events
        df['DaysSince2013'] = (df['Date'] - pd.to_datetime('2013-01-01')).dt.days
        
        self.logger.success("Temporal features created")
        return df
    
    def create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competition-related features."""
        self.logger.info("Creating competition features...")
        df = df.copy()
        
        # Competition age calculation
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].replace(0, np.nan)
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].replace(0, np.nan)
        
        # Calculate competition age in months
        df['CompetitionAge'] = ((df['Year'] - df['CompetitionOpenSinceYear']) * 12 + 
                               (df['Month'] - df['CompetitionOpenSinceMonth']))
        df['CompetitionAge'] = df['CompetitionAge'].fillna(0).clip(lower=0)
        
        # Competition distance features
        df['CompetitionDistance_log'] = np.log1p(df['CompetitionDistance'])
        df['HasCompetition'] = (df['CompetitionDistance'].notna()).astype(int)
        
        # Competition distance categories (based on EDA insights)
        df['CompetitionDistanceCategory'] = pd.cut(
            df['CompetitionDistance'].fillna(df['CompetitionDistance'].max()),
            bins=[0, 500, 1000, 2000, 5000, float('inf')],
            labels=['VeryClose', 'Close', 'Medium', 'Far', 'VeryFar']
        )
        
        self.logger.success("Competition features created")
        return df
        
    def create_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create promotion-related features based on EDA insights."""
        self.logger.info("Creating promotion features...")
        df = df.copy()
        
        # Promo2 features
        df['Promo2Active'] = df['Promo2']
        
        # Promo2 age calculation
        df['Promo2Age'] = 0
        mask = (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0)
        df.loc[mask, 'Promo2Age'] = ((df.loc[mask, 'Year'] - df.loc[mask, 'Promo2SinceYear']) * 52 + 
                                     (df.loc[mask, 'WeekOfYear'] - df.loc[mask, 'Promo2SinceWeek']))
        df['Promo2Age'] = df['Promo2Age'].clip(lower=0)
        
        # PromoInterval features
        df['HasPromoInterval'] = (df['PromoInterval'] != 'None').astype(int)
        
        # Check if current month is in promo interval
        df['InPromoInterval'] = 0
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        for idx, row in df.iterrows():
            if row['PromoInterval'] != 'None' and pd.notna(row['PromoInterval']):
                promo_months = [month_map.get(month.strip(), 0) 
                               for month in row['PromoInterval'].split(',')]
                if row['Month'] in promo_months:
                    df.at[idx, 'InPromoInterval'] = 1
        
        # Combined promotion features
        df['AnyPromo'] = ((df['Promo'] == 1) | (df['InPromoInterval'] == 1)).astype(int)
        df['PromoIntensity'] = df['Promo'] + df['InPromoInterval']
        
        self.logger.success("Promotion features created")
        return df
    
    def create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-specific features."""
        self.logger.info("Creating store features...")
        df = df.copy()
        
        # Store type and assortment combinations (based on EDA insights)
        df['StoreType_Assortment'] = df['StoreType'] + '_' + df['Assortment']
        
        # Store performance categories (if Sales column exists)
        if 'Sales' in df.columns:
            store_performance = df[df['Open'] == 1].groupby('Store')['Sales'].agg(['mean', 'std']).reset_index()
            store_performance['CV'] = store_performance['std'] / store_performance['mean']
            
            # Categorize stores by performance
            performance_quantiles = store_performance['mean'].quantile([0.33, 0.67])
            store_performance['PerformanceCategory'] = pd.cut(
                store_performance['mean'],
                bins=[0, performance_quantiles.iloc[0], performance_quantiles.iloc[1], float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            
            # Merge back to main dataframe
            df = df.merge(store_performance[['Store', 'PerformanceCategory', 'CV']], 
                         on='Store', how='left')
            df['PerformanceCategory'] = df['PerformanceCategory'].fillna('Medium')
            df['CV'] = df['CV'].fillna(df['CV'].median())
        
        self.logger.success("Store features created")
        return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday-related features."""
        self.logger.info("Creating holiday features...")
        df = df.copy()
        
        # State holiday encoding
        df['StateHoliday_Binary'] = (df['StateHoliday'] != '0').astype(int)
        
        # Combined holiday effect
        df['AnyHoliday'] = ((df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1)).astype(int)
        
        # Holiday intensity
        df['HolidayIntensity'] = df['StateHoliday_Binary'] + df['SchoolHoliday']
        
        self.logger.success("Holiday features created")
        return df
    
    def encode_categorical_features(self, train_df: pd.DataFrame, 
                                   test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features using various strategies."""
        self.logger.info("Encoding categorical features...")
        
        # Features for label encoding
        label_encode_features = ['StoreType', 'Assortment', 'StateHoliday', 'Season', 
                               'CompetitionDistanceCategory', 'StoreType_Assortment']
        
        # Features for one-hot encoding
        onehot_features = ['PerformanceCategory'] if 'PerformanceCategory' in train_df.columns else []
        
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        # Label encoding
        for feature in label_encode_features:
            if feature in train_df.columns:
                self.label_encoders[feature] = LabelEncoder()
                train_encoded[feature] = self.label_encoders[feature].fit_transform(train_df[feature].astype(str))
                test_encoded[feature] = self.label_encoders[feature].transform(test_df[feature].astype(str))
        
        # One-hot encoding
        for feature in onehot_features:
            if feature in train_df.columns:
                train_dummies = pd.get_dummies(train_df[feature], prefix=feature)
                test_dummies = pd.get_dummies(test_df[feature], prefix=feature)
                
                # Ensure same columns in both sets
                all_columns = set(train_dummies.columns) | set(test_dummies.columns)
                for col in all_columns:
                    if col not in train_dummies.columns:
                        train_dummies[col] = 0
                    if col not in test_dummies.columns:
                        test_dummies[col] = 0
                
                train_encoded = pd.concat([train_encoded.drop(feature, axis=1), train_dummies], axis=1)
                test_encoded = pd.concat([test_encoded.drop(feature, axis=1), test_dummies], axis=1)
        
        self.logger.success("Categorical encoding completed")
        return train_encoded, test_encoded
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Sales') -> pd.DataFrame:
        """Create lag and rolling window features (only for training data)."""
        if target_col not in df.columns:
            self.logger.info(f"Skipping lag features - {target_col} not found")
            return df
            
        self.logger.info("Creating lag and rolling window features...")
        df = df.copy()
        df = df.sort_values(['Store', 'Date'])
        
        # Create lag features (7, 14, 30 days)
        for lag in [7, 14, 30]:
            df[f'{target_col}_lag_{lag}'] = df.groupby('Store')[target_col].shift(lag)
        
        # Rolling window features
        for window in [7, 14, 30]:
            df[f'{target_col}_rolling_mean_{window}'] = (df.groupby('Store')[target_col]
                                                        .rolling(window=window, min_periods=1)
                                                        .mean().reset_index(0, drop=True))
            df[f'{target_col}_rolling_std_{window}'] = (df.groupby('Store')[target_col]
                                                       .rolling(window=window, min_periods=1)
                                                       .std().reset_index(0, drop=True))
        
        # Fill missing lag values with median
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        for col in lag_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        self.logger.success("Lag features created")
        return df
    
    def process_pipeline(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the complete feature engineering pipeline."""
        self.logger.info("Starting comprehensive feature engineering pipeline...")
        
        # Apply all feature engineering steps
        train_processed = self.clean_data(train_df)
        test_processed = self.clean_data(test_df)
        
        train_processed = self.create_temporal_features(train_processed)
        test_processed = self.create_temporal_features(test_processed)
        
        train_processed = self.create_competition_features(train_processed)
        test_processed = self.create_competition_features(test_processed)
        
        train_processed = self.create_promotion_features(train_processed)
        test_processed = self.create_promotion_features(test_processed)
        
        train_processed = self.create_store_features(train_processed)
        test_processed = self.create_store_features(test_processed)
        
        train_processed = self.create_holiday_features(train_processed)
        test_processed = self.create_holiday_features(test_processed)
        
        # Create lag features only for training data
        train_processed = self.create_lag_features(train_processed)
        
        # Encode categorical features
        train_processed, test_processed = self.encode_categorical_features(train_processed, test_processed)
        
        self.logger.success("Feature engineering pipeline completed!")
        return train_processed, test_processed


@app.command()
def main(
    output_dir: Path = PROCESSED_DATA_DIR,
    create_splits: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Main function to execute feature engineering pipeline.
    
    Args:
        output_dir: Directory to save processed datasets
        create_splits: Whether to create train/validation splits
        test_size: Size of validation set (if create_splits=True)
        random_state: Random seed for reproducibility
    """
    # Initialize feature engineering
    feature_engineer = RossmannFeatureEngineering()
    
    # Load data
    train_df, test_df, store_df = feature_engineer.load_data()
    
    # Process features
    train_processed, test_processed = feature_engineer.process_pipeline(train_df, test_df)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed datasets
    train_processed.to_csv(output_dir / "train_processed.csv", index=False)
    test_processed.to_csv(output_dir / "test_processed.csv", index=False)
    logger.success(f"Processed datasets saved to {output_dir}")
    
    # Create train/validation splits if requested
    if create_splits:
        # Remove rows where Sales = 0 for training
        train_for_modeling = train_processed[train_processed['Sales'] > 0].copy()
        
        # Time-based split (recommended for forecasting)
        train_for_modeling = train_for_modeling.sort_values('Date')
        split_date = train_for_modeling['Date'].quantile(0.8)
        
        train_split = train_for_modeling[train_for_modeling['Date'] <= split_date]
        val_split = train_for_modeling[train_for_modeling['Date'] > split_date]
        
        # Save splits
        train_split.to_csv(output_dir / "train_split.csv", index=False)
        val_split.to_csv(output_dir / "val_split.csv", index=False)
        
        logger.success(f"Train/validation splits created:")
        logger.info(f"  - Train split: {len(train_split):,} records")
        logger.info(f"  - Validation split: {len(val_split):,} records")
        logger.info(f"  - Split date: {split_date}")
    
    # Save feature information
    feature_info = {
        'total_features': len(train_processed.columns),
        'feature_names': list(train_processed.columns),
        'categorical_features': list(feature_engineer.label_encoders.keys()),
        'target_column': 'Sales',
        'processed_date': pd.Timestamp.now().isoformat()
    }
    
    import json
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.success(f"Feature engineering completed! Total features: {len(train_processed.columns)}")


if __name__ == "__main__":
    app()
