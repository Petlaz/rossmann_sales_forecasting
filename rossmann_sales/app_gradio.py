import os
import gdown
import gradio as gr
import pandas as pd
from pathlib import Path

from rossmann_sales.utils.load_artifacts import load_model, load_scaler, load_feature_names

# 🔽 Download model from Google Drive if missing
MODEL_PATH = Path("models/random_forest.pkl")
GDRIVE_FILE_ID = "1zVNpAIysnyQ_tmHIQjRTt2y2mnI6CMpV"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
os.makedirs("models", exist_ok=True)
if not MODEL_PATH.exists():
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, str(MODEL_PATH), quiet=False)

# 🔁 Load model artifacts
model = load_model()
scaler = load_scaler()
features = load_feature_names()

# 🔢 Categorical options
store_types = ["a", "b", "c", "d"]
state_holiday_options = ["0", "a", "b", "c"]
assortment_types = ["a", "b", "c"]
promo_interval_options = ["Jan,Apr,Jul,Oct", "Mar,Jun,Sept,Dec", "None"]

def create_input_dict(inputs):
    input_dict = {feat: 0 for feat in features}
    numeric_feats = [
        "Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
        "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "CompetitionOpen",
        "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
        "Sales_lag_1", "Sales_lag_7", "Sales_lag_14",
        "Sales_roll_mean_7", "Sales_roll_mean_14"
    ]
    for feat in numeric_feats:
        if feat in inputs:
            input_dict[feat] = inputs[feat]

    input_dict[f"StoreType_{inputs.get('StoreType', 'a')}"] = 1
    input_dict[f"StateHoliday_{inputs.get('StateHoliday', '0')}"] = 1
    if f"Assortment_{inputs['Assortment']}" in input_dict:
        input_dict[f"Assortment_{inputs['Assortment']}"] = 1
    if f"PromoInterval_{inputs['PromoInterval']}" in input_dict:
        input_dict[f"PromoInterval_{inputs['PromoInterval']}"] = 1

    return input_dict

def predict_sales(
    Store, DayOfWeek, Open, Promo, SchoolHoliday,
    StoreType, Assortment, StateHoliday, PromoInterval,
    CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
    Promo2, Promo2SinceWeek, Promo2SinceYear,
    CompetitionOpen, Year, Month, Day, WeekOfYear, IsWeekend,
    Sales_lag_1, Sales_lag_7, Sales_lag_14,
    Sales_roll_mean_7, Sales_roll_mean_14
):
    input_data = {
        "Store": Store,
        "DayOfWeek": DayOfWeek,
        "Open": Open,
        "Promo": Promo,
        "SchoolHoliday": SchoolHoliday,
        "StoreType": StoreType,
        "Assortment": Assortment,
        "StateHoliday": StateHoliday,
        "PromoInterval": PromoInterval,
        "CompetitionDistance": CompetitionDistance,
        "CompetitionOpenSinceMonth": CompetitionOpenSinceMonth,
        "CompetitionOpenSinceYear": CompetitionOpenSinceYear,
        "Promo2": Promo2,
        "Promo2SinceWeek": Promo2SinceWeek,
        "Promo2SinceYear": Promo2SinceYear,
        "CompetitionOpen": CompetitionOpen,
        "Year": Year,
        "Month": Month,
        "Day": Day,
        "WeekOfYear": WeekOfYear,
        "IsWeekend": IsWeekend,
        "Sales_lag_1": Sales_lag_1,
        "Sales_lag_7": Sales_lag_7,
        "Sales_lag_14": Sales_lag_14,
        "Sales_roll_mean_7": Sales_roll_mean_7,
        "Sales_roll_mean_14": Sales_roll_mean_14
    }

    X = pd.DataFrame([create_input_dict(input_data)])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return f"Predicted Sales: €{prediction:,.2f}"

example_presets = {
    "Store": 1,
    "DayOfWeek": 5,
    "Open": 1,
    "Promo": 1,
    "SchoolHoliday": 0,
    "StoreType": "a",
    "Assortment": "a",
    "StateHoliday": "0",
    "PromoInterval": "Jan,Apr,Jul,Oct",
    "CompetitionDistance": 200.0,
    "CompetitionOpenSinceMonth": 9,
    "CompetitionOpenSinceYear": 2008,
    "Promo2": 1,
    "Promo2SinceWeek": 14,
    "Promo2SinceYear": 2011,
    "CompetitionOpen": 24,
    "Year": 2015,
    "Month": 6,
    "Day": 19,
    "WeekOfYear": 25,
    "IsWeekend": 0,
    "Sales_lag_1": 5500,
    "Sales_lag_7": 5400,
    "Sales_lag_14": 5300,
    "Sales_roll_mean_7": 5200,
    "Sales_roll_mean_14": 5100
}

def load_preset_values():
    return tuple(example_presets[key] for key in example_presets)

with gr.Blocks(title="Rossmann Sales Forecasting") as demo:
    gr.Markdown("## 🏪 Rossmann Sales Forecasting App")
    gr.Markdown("Use this app to predict future sales for a Rossmann store based on store data and promotional events.")

    with gr.Row():
        with gr.Column():
            Store = gr.Number(value=1, label="Store ID", info="Unique ID for the store")
            DayOfWeek = gr.Slider(1, 7, value=5, step=1, label="Day of Week", info="1=Monday, 7=Sunday")
            Open = gr.Radio([0, 1], value=1, label="Open")
            Promo = gr.Radio([0, 1], value=1, label="Promo")
            SchoolHoliday = gr.Radio([0, 1], value=0, label="School Holiday")
            StoreType = gr.Dropdown(store_types, value="a", label="Store Type")
            Assortment = gr.Dropdown(assortment_types, value="a", label="Assortment Type")
            StateHoliday = gr.Dropdown(state_holiday_options, value="0", label="State Holiday")
            PromoInterval = gr.Dropdown(promo_interval_options, value="Jan,Apr,Jul,Oct", label="Promo Interval")

        with gr.Column():
            CompetitionDistance = gr.Number(value=200.0, label="Competition Distance (m)")
            CompetitionOpenSinceMonth = gr.Slider(1, 12, value=9, step=1, label="Competition Open Month")
            CompetitionOpenSinceYear = gr.Slider(2000, 2015, value=2008, step=1, label="Competition Open Year")
            Promo2 = gr.Radio([0, 1], value=1, label="Promo2")
            Promo2SinceWeek = gr.Slider(1, 52, value=14, step=1, label="Promo2 Since Week")
            Promo2SinceYear = gr.Slider(2000, 2015, value=2011, step=1, label="Promo2 Since Year")
            CompetitionOpen = gr.Number(value=24, label="Competition Open (Months)")
            Year = gr.Slider(2013, 2015, value=2015, step=1, label="Year")
            Month = gr.Slider(1, 12, value=6, step=1, label="Month")
            Day = gr.Slider(1, 31, value=19, step=1, label="Day")
            WeekOfYear = gr.Slider(1, 52, value=25, step=1, label="Week of Year")
            IsWeekend = gr.Radio([0, 1], value=0, label="Is Weekend")
            Sales_lag_1 = gr.Number(value=5500, label="Sales Lag 1 Day")
            Sales_lag_7 = gr.Number(value=5400, label="Sales Lag 7 Days")
            Sales_lag_14 = gr.Number(value=5300, label="Sales Lag 14 Days")
            Sales_roll_mean_7 = gr.Number(value=5200, label="7-Day Rolling Sales Mean")
            Sales_roll_mean_14 = gr.Number(value=5100, label="14-Day Rolling Sales Mean")

    with gr.Row():
        predict_btn = gr.Button("Predict Sales")
        result_output = gr.Textbox(label="Prediction", lines=1)

    predict_btn.click(
        predict_sales,
        inputs=[
            Store, DayOfWeek, Open, Promo, SchoolHoliday,
            StoreType, Assortment, StateHoliday, PromoInterval,
            CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
            Promo2, Promo2SinceWeek, Promo2SinceYear,
            CompetitionOpen, Year, Month, Day, WeekOfYear, IsWeekend,
            Sales_lag_1, Sales_lag_7, Sales_lag_14,
            Sales_roll_mean_7, Sales_roll_mean_14
        ],
        outputs=result_output
    )

    gr.Button("Use Example Data").click(
        lambda: load_preset_values(), outputs=[
            Store, DayOfWeek, Open, Promo, SchoolHoliday,
            StoreType, Assortment, StateHoliday, PromoInterval,
            CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
            Promo2, Promo2SinceWeek, Promo2SinceYear,
            CompetitionOpen, Year, Month, Day, WeekOfYear, IsWeekend,
            Sales_lag_1, Sales_lag_7, Sales_lag_14,
            Sales_roll_mean_7, Sales_roll_mean_14
        ]
    )

def launch_app():
    demo.launch()

if __name__ == "__main__":
    launch_app()
