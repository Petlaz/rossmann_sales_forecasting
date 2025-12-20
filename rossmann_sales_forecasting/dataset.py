from pathlib import Path
from typing import Tuple, Optional, List
import warnings

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from rossmann_sales_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

warnings.filterwarnings('ignore')

app = typer.Typer()


class RossmannDataProcessor:
    """
    Data processing and validation pipeline for Rossmann sales forecasting.
    Handles data loading, cleaning, validation, and preparation for modeling.
    """
    
    def __init__(self):
        self.logger = logger
        self.data_quality_report = {}
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all raw datasets with validation."""
        self.logger.info("Loading raw datasets...")
        
        try:
            train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
            store_df = pd.read_csv(RAW_DATA_DIR / "store.csv")
            test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
            sample_submission = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
            
            self.logger.success(f"Raw data loaded successfully:")
            self.logger.info(f"  - Train: {train_df.shape}")
            self.logger.info(f"  - Store: {store_df.shape}")
            self.logger.info(f"  - Test: {test_df.shape}")
            self.logger.info(f"  - Sample submission: {sample_submission.shape}")
            
            return train_df, store_df, test_df, sample_submission
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            raise
    
    def validate_data_integrity(self, train_df: pd.DataFrame, store_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> dict:
        """Perform comprehensive data validation and quality checks."""
        self.logger.info("Validating data integrity...")
        
        validation_report = {
            'train_shape': train_df.shape,
            'store_shape': store_df.shape,
            'test_shape': test_df.shape,
            'issues': []
        }
        
        # Check for required columns
        required_train_cols = ['Store', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 
                              'StateHoliday', 'SchoolHoliday']
        required_store_cols = ['Store', 'StoreType', 'Assortment']
        
        missing_train_cols = set(required_train_cols) - set(train_df.columns)
        missing_store_cols = set(required_store_cols) - set(store_df.columns)
        
        if missing_train_cols:
            validation_report['issues'].append(f"Missing train columns: {missing_train_cols}")
        if missing_store_cols:
            validation_report['issues'].append(f"Missing store columns: {missing_store_cols}")
        
        # Check for data consistency
        train_stores = set(train_df['Store'].unique())
        store_stores = set(store_df['Store'].unique())
        test_stores = set(test_df['Store'].unique())
        
        if not train_stores.issubset(store_stores):
            validation_report['issues'].append("Some train stores not in store data")
        if not test_stores.issubset(store_stores):
            validation_report['issues'].append("Some test stores not in store data")
        
        # Check for negative sales or customers
        if (train_df['Sales'] < 0).any():
            validation_report['issues'].append("Negative sales values found")
        if (train_df['Customers'] < 0).any():
            validation_report['issues'].append("Negative customer values found")
        
        # Check for impossible combinations
        closed_with_sales = train_df[(train_df['Open'] == 0) & (train_df['Sales'] > 0)]
        if len(closed_with_sales) > 0:
            validation_report['issues'].append(f"Found {len(closed_with_sales)} closed stores with sales")
        
        self.data_quality_report = validation_report
        
        if validation_report['issues']:
            self.logger.warning(f"Data validation found {len(validation_report['issues'])} issues:")
            for issue in validation_report['issues']:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.success("Data validation passed - no issues found")
        
        return validation_report
    
    def create_data_summary(self, train_df: pd.DataFrame, store_df: pd.DataFrame) -> dict:
        """Create comprehensive data summary for documentation."""
        self.logger.info("Creating data summary report...")
        
        # Merge for complete analysis
        merged_df = train_df.merge(store_df, on='Store', how='left')
        
        summary = {
            'dataset_overview': {
                'total_records': len(train_df),
                'unique_stores': train_df['Store'].nunique(),
                'date_range': {
                    'start': train_df['Date'].min(),
                    'end': train_df['Date'].max(),
                    'total_days': (pd.to_datetime(train_df['Date'].max()) - 
                                 pd.to_datetime(train_df['Date'].min())).days
                },
                'zero_sales_percentage': (train_df['Sales'] == 0).mean() * 100
            },
            
            'sales_statistics': {
                'mean_daily_sales': train_df[train_df['Sales'] > 0]['Sales'].mean(),
                'median_daily_sales': train_df[train_df['Sales'] > 0]['Sales'].median(),
                'max_daily_sales': train_df['Sales'].max(),
                'sales_std': train_df[train_df['Sales'] > 0]['Sales'].std()
            },
            
            'store_characteristics': {
                'store_types': store_df['StoreType'].value_counts().to_dict(),
                'assortment_types': store_df['Assortment'].value_counts().to_dict(),
                'stores_with_competition': store_df['CompetitionDistance'].notna().sum(),
                'stores_with_promo2': store_df['Promo2'].sum()
            },
            
            'temporal_patterns': {
                'records_by_day_of_week': train_df['DayOfWeek'].value_counts().sort_index().to_dict(),
                'records_by_year': pd.to_datetime(train_df['Date']).dt.year.value_counts().sort_index().to_dict(),
                'school_holiday_percentage': train_df['SchoolHoliday'].mean() * 100,
                'state_holiday_percentage': (train_df['StateHoliday'] != '0').mean() * 100
            },
            
            'missing_data': {
                'train_missing': train_df.isnull().sum().to_dict(),
                'store_missing': store_df.isnull().sum().to_dict()
            }
        }
        
        self.logger.success("Data summary report created")
        return summary
    
    def prepare_modeling_dataset(self, processed_train: pd.DataFrame, 
                                target_col: str = 'Sales') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare final dataset for modeling with proper train/test split."""
        self.logger.info("Preparing modeling dataset...")
        
        # Remove closed stores (Sales = 0) for training
        modeling_data = processed_train[processed_train[target_col] > 0].copy()
        
        # Sort by date for time-based splitting
        modeling_data = modeling_data.sort_values('Date')
        
        # Time-based split (80% train, 20% validation)
        split_date = modeling_data['Date'].quantile(0.8)
        
        train_final = modeling_data[modeling_data['Date'] <= split_date].copy()
        val_final = modeling_data[modeling_data['Date'] > split_date].copy()
        
        # Remove non-modeling columns
        columns_to_remove = ['Date']  # Keep Date for reference but don't use in modeling
        
        self.logger.success(f"Modeling dataset prepared:")
        self.logger.info(f"  - Training records: {len(train_final):,}")
        self.logger.info(f"  - Validation records: {len(val_final):,}")
        self.logger.info(f"  - Split date: {split_date}")
        self.logger.info(f"  - Features: {len([col for col in train_final.columns if col not in columns_to_remove + [target_col]])}")
        
        return train_final, val_final
    
    def save_processed_data(self, datasets: dict, output_dir: Path):
        """Save all processed datasets with metadata."""
        self.logger.info(f"Saving processed datasets to {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        for name, df in datasets.items():
            filepath = output_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            self.logger.info(f"  - Saved {name}: {df.shape} -> {filepath}")
        
        # Save metadata
        metadata = {
            'processing_date': pd.Timestamp.now().isoformat(),
            'datasets': {name: {'shape': df.shape, 'columns': list(df.columns)} 
                        for name, df in datasets.items()},
            'data_quality_report': self.data_quality_report,
            'processing_notes': [
                "Removed records with Sales = 0 for modeling datasets",
                "Used time-based split for train/validation",
                "Applied comprehensive feature engineering pipeline",
                "All categorical variables encoded appropriately"
            ]
        }
        
        import json
        with open(output_dir / "processing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.success("All processed datasets and metadata saved")


@app.command()
def main(
    output_dir: Path = PROCESSED_DATA_DIR,
    run_feature_engineering: bool = True,
    create_modeling_splits: bool = True,
    validation_size: float = 0.2
):
    """
    Main data processing pipeline.
    
    Args:
        output_dir: Output directory for processed data
        run_feature_engineering: Whether to run feature engineering
        create_modeling_splits: Whether to create train/val splits for modeling
        validation_size: Size of validation set (0.0 to 1.0)
    """
    processor = RossmannDataProcessor()
    
    # Load raw data
    train_df, store_df, test_df, sample_submission = processor.load_raw_data()
    
    # Validate data integrity
    validation_report = processor.validate_data_integrity(train_df, store_df, test_df)
    
    # Create data summary
    data_summary = processor.create_data_summary(train_df, store_df)
    
    # Save interim data and summary
    interim_dir = INTERIM_DATA_DIR
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data summary
    import json
    with open(interim_dir / "data_summary.json", 'w') as f:
        json.dump(data_summary, f, indent=2, default=str)
    
    if run_feature_engineering:
        logger.info("Running feature engineering pipeline...")
        
        # Import and run feature engineering
        from rossmann_sales_forecasting.features import RossmannFeatureEngineering
        
        feature_engineer = RossmannFeatureEngineering()
        train_processed, test_processed = feature_engineer.process_pipeline(train_df, test_df)
        
        datasets_to_save = {
            'train_processed': train_processed,
            'test_processed': test_processed
        }
        
        # Create modeling splits if requested
        if create_modeling_splits:
            train_final, val_final = processor.prepare_modeling_dataset(train_processed)
            datasets_to_save.update({
                'train_modeling': train_final,
                'val_modeling': val_final
            })
        
        # Save all datasets
        processor.save_processed_data(datasets_to_save, output_dir)
        
        logger.success("Complete data processing pipeline finished!")
        logger.info(f"Check {output_dir} for all processed datasets and metadata")
        
    else:
        logger.info("Skipping feature engineering - saving raw merged data only")
        
        # Simple merge and save
        train_merged = train_df.merge(store_df, on='Store', how='left')
        test_merged = test_df.merge(store_df, on='Store', how='left')
        
        datasets_to_save = {
            'train_merged': train_merged,
            'test_merged': test_merged,
            'sample_submission': sample_submission
        }
        
        processor.save_processed_data(datasets_to_save, output_dir)


@app.command()
def validate_only():
    """Run data validation only without processing."""
    processor = RossmannDataProcessor()
    
    # Load and validate data
    train_df, store_df, test_df, _ = processor.load_raw_data()
    validation_report = processor.validate_data_integrity(train_df, store_df, test_df)
    data_summary = processor.create_data_summary(train_df, store_df)
    
    # Print summary
    logger.info("=== DATA VALIDATION SUMMARY ===")
    logger.info(f"Training records: {len(train_df):,}")
    logger.info(f"Unique stores: {train_df['Store'].nunique():,}")
    logger.info(f"Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    logger.info(f"Zero sales percentage: {(train_df['Sales'] == 0).mean()*100:.1f}%")
    
    if validation_report['issues']:
        logger.warning(f"Found {len(validation_report['issues'])} data quality issues")
    else:
        logger.success("Data validation passed - ready for processing")


if __name__ == "__main__":
    app()
