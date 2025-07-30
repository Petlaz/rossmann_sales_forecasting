import pandas as pd
from rossmann_sales_forecasting.config import PROCESSED_DATA_FILE, INTERIM_DATA_DIR
from loguru import logger


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based and store-related features from processed data."""
    logger.info("Creating features...")

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort to ensure consistent lag/rolling calculations
    df = df.sort_values(["Store", "Date"]).copy()

    # Date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"] >= 5
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    # Lag features
    df["Sales_lag_1"] = df.groupby("Store")["Sales"].shift(1)
    df["Sales_lag_7"] = df.groupby("Store")["Sales"].shift(7)

    # Rolling features
    df["Sales_roll_mean_7"] = df.groupby("Store")["Sales"].shift(1).rolling(window=7).mean()
    df["Sales_roll_std_7"] = df.groupby("Store")["Sales"].shift(1).rolling(window=7).std()

    logger.info("Feature creation complete.")
    return df


def main():
    logger.info("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df_features = create_features(df)

    # Output path
    output_path = INTERIM_DATA_DIR / "data_features.csv"
    logger.info(f"Saving features to: {output_path}")
    df_features.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

# Run: python3 features.py

# Ensure the script is run from the project root directory.
