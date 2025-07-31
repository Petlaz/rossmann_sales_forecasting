# rossmann_sales_forecasting/tests/test_data.py

import pandas as pd
from pathlib import Path

def test_data_file_exists():
    path = Path("data/processed/data_encoded.csv")
    assert path.exists(), f"{path} does not exist"

def test_data_is_not_empty():
    df = pd.read_csv("data/processed/data_encoded.csv")
    assert not df.empty, "Data file is empty"
