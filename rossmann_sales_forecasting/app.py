# app.py
import gradio as gr
import pandas as pd
import joblib
import os

# Load model
MODEL_PATH = os.path.join("models", "random_forest.pkl")
model = joblib.load(MODEL_PATH)

# Load feature names from processed CSV
FEATURES_PATH = "/Users/peter/rossmann_sales_forecasting/data/processed/X_encoded.csv"
feature_names = pd.read_csv(FEATURES_PATH, nrows=0).columns.tolist()

def predict_sales(store, promo, school_holiday, day_of_week, month, year):
    # Create a dictionary for all features, fill unused with 0
    input_dict = {feat: 0 for feat in feature_names}
    input_dict.update({
        "Store": store,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "DayOfWeek": day_of_week,
        "Month": month,
        "Year": year
    })
    input_data = pd.DataFrame([input_dict])
    prediction = model.predict(input_data)
    return f"Predicted Sales: €{prediction[0]:,.2f}"

iface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Store"),
        gr.Radio([0, 1], label="Promo"),
        gr.Radio([0, 1], label="School Holiday"),
        gr.Slider(1, 7, step=1, label="Day of Week"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(2013, 2015, step=1, label="Year"),
    ],
    outputs=gr.Text(label="Sales Forecast"),
    title="Rossmann Sales Forecasting",
    description="Enter store and date details to predict sales using Random Forest."
)

if __name__ == "__main__":
    iface.launch()
    
# To run the app, use the command: python3 rossmann_sales_forecasting/app.py