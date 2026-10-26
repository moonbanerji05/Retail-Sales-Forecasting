# Retail Sales Forecasting

## Project Overview
Retail Sales Forecasting is a time-series machine learning project that predicts future retail sales using historical sales data, store information, and external factors such as holidays, temperature, fuel prices, CPI, and unemployment.

The project compares XGBoost and SARIMA models to understand their performance in sales forecasting.

## Objectives
- Analyze historical retail sales trends.
- Perform Exploratory Data Analysis (EDA).
- Create time-based, lag, and rolling statistical features.
- Train and evaluate forecasting models.
- Generate a four-week sales forecast.
- Compare model performance using MAE and RMSE.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** XGBoost, Scikit-learn
- **Time-Series Analysis:** Statsmodels (SARIMA/SARIMAX)
- **Development Environment:** VS Code, Jupyter Notebook

## Dataset
The project uses retail sales data containing:
- Historical weekly sales
- Store and department information
- Holiday indicators
- Weather and economic features

## Methodology
1. Data loading, cleaning, and preprocessing.
2. Exploratory Data Analysis and sales trend visualization.
3. Feature engineering using date features, lag values, and rolling statistics.
4. Model training using XGBoost and SARIMA.
5. Model evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
6. Forecasting future weekly sales and comparing model performance.

## Project Structure
- `data/` — Dataset files
- `notebooks/` — Jupyter Notebook and analysis
- `models/` — Saved trained models
- `outputs/` — Forecasts, comparison reports, and visualizations
- `README.md` — Project documentation

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Results
The project generates model evaluation reports, feature importance visualizations, SARIMA tuning results, and a four-week sales forecast. Refer to the files in the `outputs/` directory for the generated results.

## Future Improvements
- Deploy the forecasting model using Streamlit.
- Improve hyperparameter tuning.
- Add interactive sales visualizations.
- Explore additional time-series forecasting techniques.

## Author
B.Tech CSE (Artificial Intelligence) Student
