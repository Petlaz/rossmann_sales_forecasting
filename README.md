# 🏪 Rossmann Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine-Learning-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff9a00?logo=gradio&logoColor=white)

Rossmann Sales Forecasting is a machine learning project that predicts daily sales for over 1,100 Rossmann drug stores across Europe. Leveraging historical sales data, promotions, holidays, and store-specific features, the project applies feature engineering and advanced regression models to deliver accurate forecasts with real business value.

Key Features

- **Time Series Forecasting**: Predicts daily sales using historical patterns
- **Feature Engineering**: Creates meaningful temporal and promotional features
- **Multiple Model Evaluation**: Tests various regression approaches
- **Production-Ready**: Includes Gradio web interface for easy predictions
- **Interpretable Results**: Provides feature importance and error analysis

Model Performance

Final Model Evaluation
| Metric | Value |
|--------|-------|
| RMSE   | 847.47 |
| MAE    | 568.17 |
| MAPE   | 8.88% |

Best Performing Model
After evaluating several regression models, **Random Forest Regressor** emerged as the top performer:

| Model              | RMSE    | MAE     | MAPE   |
|--------------------|---------|---------|--------|
| Random Forest      | 597.35  | 339.47  | 6.33%  |
| Gradient Boosting  | 1042.84 | 677.56  | 12.42% |
| Ridge Regression   | 1566.57 | 1084.93 | 17.99% |
| XGBoost            | 6721.08 | 5674.55 | 98.08% |

Insights

Top Predictive Features

1. Promo
2. Store
3. DayOfWeek
4. CompetitionDistance
5. SchoolHoliday

These insights inform store-level strategies for staffing, inventory, and promotions.


Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         rossmann_sales and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── rossmann_sales   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes rossmann_sales a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

Quick start

**Installation**:  
1. Clone the repository:
```bash
git clone https://github.com/Petlaz/rossmann_sales_forecasting.git
cd rossmann_sales_forecasting

2. Install dependencies:

pip install -r requirements.txt

3. Run the Gradio web app:

python -m rossmann_sales.app_gradio

 
 Dataset

Source: Kaggle - Rossmann Store Sales

Key Features:

* Store Info: Store, StoreType, Assortment, CompetitionDistance

* Temporal Features: Date, DayOfWeek, PromoInterval

* Promotions: Promo, Promo2, Promo2SinceWeek

* Calendar Events: SchoolHoliday, StateHoliday

* Engineered Features: Lag features, time components

* Target Variable: Sales

Roadmap

* EDA and preprocessing

* Feature engineering

* Model selection and tuning

* Evaluation and diagnostics

* Gradio-based prediction UI

* Model monitoring with real-time data


Contacts

👤 Peter Ugonna Obi
📧 Email: peter.obi96@yahoo.com
🔗 LinkedIn: linkedin.com/in/peter-obi-15a424161


License

This project is licensed under the MIT License. See the LICENSE file for details.cccccc




