import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional features from raw data."""
    
    # Ensure 'Date' is in datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Drop rows with invalid or missing dates
        df = df.dropna(subset=['Date'])

        # Add time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    else:
        raise ValueError("Missing required column: 'Date'")

    # Convert 'StateHoliday' to categorical numeric if it exists
    if 'StateHoliday' in df.columns:
        df['StateHoliday'] = df['StateHoliday'].replace(0, '0')  # sometimes mixed types
        df['StateHoliday'] = df['StateHoliday'].astype('category').cat.codes

    # Ensure correct dtypes or fill missing important flags
    for col in ['Promo', 'SchoolHoliday', 'Open']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df

# Example usage
if __name__ == "__main__":
    # Test with a sample DataFrame
    sample_data = {
        'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'Store': [1, 2, 3],
        'Sales': [1000, 1500, 2000],
        'StateHoliday': [0, 1, 0],
        'Promo': [1, 0, 1],
        'SchoolHoliday': [0, 1, 0],
        'Open': [1, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    processed_df = generate_features(df)
    print(processed_df.head())

# Run: python3 rossmann_sales_forecasting/features.py
# This will print the DataFrame with new features added
# Ensure to run this in an environment where pandas is installed
