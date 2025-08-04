# rossmann_sales/app_gradio.py
import gradio as gr
import pandas as pd
from rossmann_sales.utils.load_artifacts import load_model, load_scaler, load_feature_names

model = load_model()
scaler = load_scaler()
features = load_feature_names()

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

    store_type = inputs.get("StoreType", "a")
    input_dict[f"StoreType_{store_type}"] = 1

    state_holiday = inputs.get("StateHoliday", "0")
    input_dict[f"StateHoliday_{state_holiday}"] = 1

    assortment = inputs.get("Assortment", "a")
    if f"Assortment_{assortment}" in input_dict:
        input_dict[f"Assortment_{assortment}"] = 1

    promo_interval = inputs.get("PromoInterval", "None")
    if f"PromoInterval_{promo_interval}" in input_dict:
        input_dict[f"PromoInterval_{promo_interval}"] = 1

    return input_dict

def predict_sales(
    Store, DayOfWeek, Open, Promo, SchoolHoliday,
    CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
    Promo2, Promo2SinceWeek, Promo2SinceYear, CompetitionOpen,
    Year, Month, Day, WeekOfYear, IsWeekend,
    Sales_lag_1, Sales_lag_7, Sales_lag_14,
    Sales_roll_mean_7, Sales_roll_mean_14,
    StoreType, StateHoliday, Assortment, PromoInterval
):
    try:
        inputs = dict(
            Store=Store,
            DayOfWeek=DayOfWeek,
            Open=Open,
            Promo=Promo,
            SchoolHoliday=SchoolHoliday,
            CompetitionDistance=CompetitionDistance,
            CompetitionOpenSinceMonth=CompetitionOpenSinceMonth,
            CompetitionOpenSinceYear=CompetitionOpenSinceYear,
            Promo2=Promo2,
            Promo2SinceWeek=Promo2SinceWeek,
            Promo2SinceYear=Promo2SinceYear,
            CompetitionOpen=CompetitionOpen,
            Year=Year,
            Month=Month,
            Day=Day,
            WeekOfYear=WeekOfYear,
            IsWeekend=IsWeekend,
            Sales_lag_1=Sales_lag_1,
            Sales_lag_7=Sales_lag_7,
            Sales_lag_14=Sales_lag_14,
            Sales_roll_mean_7=Sales_roll_mean_7,
            Sales_roll_mean_14=Sales_roll_mean_14,
            StoreType=StoreType,
            StateHoliday=StateHoliday,
            Assortment=Assortment,
            PromoInterval=PromoInterval,
        )

        input_dict = create_input_dict(inputs)
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=features, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        return f"💰 Predicted Sales: €{prediction[0]:,.2f}"
    except Exception as e:
        return f"❌ Error: {e}"

example_presets = {
    "Typical Store": {
        "Store": 1,
        "DayOfWeek": 4,
        "Open": 1,
        "Promo": 1,
        "SchoolHoliday": 0,
        "CompetitionDistance": 500,
        "CompetitionOpenSinceMonth": 9,
        "CompetitionOpenSinceYear": 2010,
        "Promo2": 1,
        "Promo2SinceWeek": 20,
        "Promo2SinceYear": 2011,
        "CompetitionOpen": 1000,
        "Year": 2015,
        "Month": 6,
        "Day": 15,
        "WeekOfYear": 24,
        "IsWeekend": 0,
        "Sales_lag_1": 6000,
        "Sales_lag_7": 6200,
        "Sales_lag_14": 6300,
        "Sales_roll_mean_7": 6100,
        "Sales_roll_mean_14": 6150,
        "StoreType": "c",
        "StateHoliday": "0",
        "Assortment": "c",
        "PromoInterval": "None",
    },
    "Holiday Season": {
        "Store": 5,
        "DayOfWeek": 6,
        "Open": 1,
        "Promo": 0,
        "SchoolHoliday": 1,
        "CompetitionDistance": 300,
        "CompetitionOpenSinceMonth": 5,
        "CompetitionOpenSinceYear": 2012,
        "Promo2": 0,
        "Promo2SinceWeek": 0,
        "Promo2SinceYear": 0,
        "CompetitionOpen": 800,
        "Year": 2015,
        "Month": 12,
        "Day": 24,
        "WeekOfYear": 52,
        "IsWeekend": 1,
        "Sales_lag_1": 8000,
        "Sales_lag_7": 8500,
        "Sales_lag_14": 8700,
        "Sales_roll_mean_7": 8300,
        "Sales_roll_mean_14": 8400,
        "StoreType": "b",
        "StateHoliday": "a",
        "Assortment": "b",
        "PromoInterval": "Jan,Apr,Jul,Oct",
    }
}

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Rossmann Sales Forecaster")

    with gr.Row():
        with gr.Column():
            Store = gr.Number(label="Store", value=1, precision=0, info="Store ID")
            DayOfWeek = gr.Slider(1, 7, step=1, label="Day of Week", value=4, info="Day of week (1=Mon, 7=Sun)")
            Open = gr.Radio([0,1], label="Open", value=1, info="Is store open (0=closed, 1=open)")
            Promo = gr.Radio([0,1], label="Promo", value=1, info="Is a promotion active?")
            SchoolHoliday = gr.Radio([0,1], label="School Holiday", value=0, info="Is it a school holiday?")
            CompetitionDistance = gr.Number(label="Competition Distance", value=500, info="Distance to nearest competitor (meters)")
            CompetitionOpenSinceMonth = gr.Slider(1, 12, step=1, label="Competition Open Since Month", value=9, info="Month when competitor opened")
            CompetitionOpenSinceYear = gr.Number(label="Competition Open Since Year", value=2010, precision=0, info="Year when competitor opened")
            Promo2 = gr.Radio([0,1], label="Promo2", value=1, info="Is store participating in Promo2?")
            Promo2SinceWeek = gr.Slider(1, 52, step=1, label="Promo2 Since Week", value=20, info="Week when Promo2 started")
            Promo2SinceYear = gr.Number(label="Promo2 Since Year", value=2011, precision=0, info="Year when Promo2 started")
            CompetitionOpen = gr.Number(label="Competition Open", value=1000, info="Days competitor has been open")

        with gr.Column():
            Year = gr.Number(label="Year", value=2015, precision=0, info="Year of prediction")
            Month = gr.Slider(1, 12, step=1, label="Month", value=6, info="Month of prediction")
            Day = gr.Slider(1, 31, step=1, label="Day", value=15, info="Day of prediction")
            WeekOfYear = gr.Slider(1, 52, step=1, label="Week Of Year", value=24, info="Week number of year")
            IsWeekend = gr.Radio([0,1], label="Is Weekend", value=0, info="Is it a weekend day?")
            Sales_lag_1 = gr.Number(label="Sales lag 1", value=6000, info="Sales 1 day ago")
            Sales_lag_7 = gr.Number(label="Sales lag 7", value=6200, info="Sales 7 days ago")
            Sales_lag_14 = gr.Number(label="Sales lag 14", value=6300, info="Sales 14 days ago")
            Sales_roll_mean_7 = gr.Number(label="Sales roll mean 7", value=6100, info="7-day rolling mean sales")
            Sales_roll_mean_14 = gr.Number(label="Sales roll mean 14", value=6150, info="14-day rolling mean sales")
            StoreType = gr.Dropdown(store_types, label="Store Type", value="c", info="Store type category")
            StateHoliday = gr.Dropdown(state_holiday_options, label="State Holiday", value="0", info="Type of state holiday")
            Assortment = gr.Dropdown(assortment_types, label="Assortment", value="c", info="Store assortment type")
            PromoInterval = gr.Dropdown(promo_interval_options, label="Promo Interval", value="None", info="Promotion interval")

    with gr.Row():
        preset = gr.Dropdown(list(example_presets.keys()), label="Load Example Preset", value=None)
        load_btn = gr.Button("Load Preset")

    predict_btn = gr.Button("Predict Sales")
    output = gr.Textbox(label="Prediction")

    def load_preset_values(preset_name):
        if preset_name and preset_name in example_presets:
            preset = example_presets[preset_name]
            return [
                preset.get("Store", 0),
                preset.get("DayOfWeek", 0),
                preset.get("Open", 0),
                preset.get("Promo", 0),
                preset.get("SchoolHoliday", 0),
                preset.get("CompetitionDistance", 0),
                preset.get("CompetitionOpenSinceMonth", 0),
                preset.get("CompetitionOpenSinceYear", 0),
                preset.get("Promo2", 0),
                preset.get("Promo2SinceWeek", 0),
                preset.get("Promo2SinceYear", 0),
                preset.get("CompetitionOpen", 0),
                preset.get("Year", 0),
                preset.get("Month", 0),
                preset.get("Day", 0),
                preset.get("WeekOfYear", 0),
                preset.get("IsWeekend", 0),
                preset.get("Sales_lag_1", 0),
                preset.get("Sales_lag_7", 0),
                preset.get("Sales_lag_14", 0),
                preset.get("Sales_roll_mean_7", 0),
                preset.get("Sales_roll_mean_14", 0),
                str(preset.get("StoreType", "a")),
                str(preset.get("StateHoliday", "0")),
                str(preset.get("Assortment", "a")),
                str(preset.get("PromoInterval", "None")),
            ]
        else:
            return [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                "a", "0", "a", "None"
            ]

    load_btn.click(
        fn=load_preset_values,
        inputs=preset,
        outputs=[
            Store, DayOfWeek, Open, Promo, SchoolHoliday,
            CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
            Promo2, Promo2SinceWeek, Promo2SinceYear, CompetitionOpen,
            Year, Month, Day, WeekOfYear, IsWeekend,
            Sales_lag_1, Sales_lag_7, Sales_lag_14,
            Sales_roll_mean_7, Sales_roll_mean_14,
            StoreType, StateHoliday, Assortment, PromoInterval
        ]
    )

    predict_btn.click(
        fn=predict_sales,
        inputs=[
            Store, DayOfWeek, Open, Promo, SchoolHoliday,
            CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
            Promo2, Promo2SinceWeek, Promo2SinceYear, CompetitionOpen,
            Year, Month, Day, WeekOfYear, IsWeekend,
            Sales_lag_1, Sales_lag_7, Sales_lag_14,
            Sales_roll_mean_7, Sales_roll_mean_14,
            StoreType, StateHoliday, Assortment, PromoInterval
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()

# Run this script to start the Gradio app: python -m rossmann_sales.app_gradio
