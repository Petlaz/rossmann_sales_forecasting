# rossmann_sales_forecasting/dataset.py
import pandas as pd
from pathlib import Path

class RossmannDataLoader:
    """Production-ready data loader"""
    
    def __init__(self):
        # Adjusted path to account for package structure
        self.PROJ_ROOT = Path(__file__).parent.parent
        self.DATA = {
            'raw': self.PROJ_ROOT / 'data' / 'raw' / 'data_df.csv',
            'processed': self.PROJ_ROOT / 'data' / 'processed' / 'data_feat.csv'
        }
        self._init_folders()
    
    def _init_folders(self):
        """Ensure folder structure exists"""
        for path in self.DATA.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_raw(self):
        """Load raw data with validation"""
        if not self.DATA['raw'].exists():
            raise FileNotFoundError(
                f"❌ Missing raw data at: {self.DATA['raw']}\n"
                "1. Download from: https://www.kaggle.com/c/rossmann-store-sales/data\n"
                f"2. Save as: {self.DATA['raw']}"
            )
        return pd.read_csv(self.DATA['raw'])

if __name__ == "__main__":
    try:
        loader = RossmannDataLoader()
        df = loader.load_raw()
        print(f"✅ Success! Loaded {len(df):,} rows")
        print(df.head(2))
        
    except Exception as e:
        print(f"\n🚨 Critical Error: {str(e)}")
        print("💡 Solution steps:")
        print("1. Download data from Kaggle")
        print(f"2. Create directory: mkdir -p {Path(__file__).parent.parent / 'data' / 'raw'}")
        print(f"3. Save file as: {Path(__file__).parent.parent / 'data' / 'raw' / 'data_df.csv'}")
        
# Run: python3 rossmann_sales_forecasting/dataset.py
# This will load the raw data and print the first two rows
# Ensure to run this in an environment where pandas is installed
# If the file is missing, it will raise an error with instructions to fix it
# This script is designed to be run directly, not as a module import
# It will not be imported by other modules, so no need for __all__ or imports
# The class is designed to be used in production, with clear error handling and folder management