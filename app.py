import streamlit as st
import pandas as pd
import os
import joblib 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# File paths
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
FORECAST_PATH = BASE_DIR / "outputs" / "four_week_forecast.csv"
COMPARISON_PATH = BASE_DIR / "outputs" / "final_model_comparison.csv"
IMPORTANCE_PATH = BASE_DIR / "outputs" / "feature_importance.png"
SARIMA_PATH = BASE_DIR / "models" / "sarima_model.pkl"
# Load trained XGBoost model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = None

st.sidebar.title("🤖 Model Status")

if MODEL_PATH.exists():
    try:
        model = load_model()
        st.sidebar.success("XGBoost model loaded successfully!")
    except Exception as e:
        st.sidebar.error("Model loading failed")
        st.sidebar.code(str(e))
else:
    st.sidebar.error("XGBoost model file not found")

# Dashboard heading
st.title("📈 Retail Sales Forecasting")
st.write(
    "Analyze historical retail sales and explore future "
    "sales forecasts using machine learning and time-series analysis."
)

st.divider()

# Project overview
st.subheader("📊 Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Machine Learning", "XGBoost")
col2.metric("Time-Series Model", "SARIMA")
col3.metric("Forecast Horizon", "4 Weeks")

st.divider()

# Four-week forecast
st.subheader("🔮 Four-Week Sales Forecast")

if  FORECAST_PATH.exists():
    try:
        forecast_df = pd.read_csv(FORECAST_PATH)
        st.dataframe(forecast_df, use_container_width=True)

        if "Predicted_Weekly_Sales" in forecast_df.columns:
            chart_df = forecast_df.copy()
            chart_df["Predicted_Weekly_Sales"] = pd.to_numeric(
                chart_df["Predicted_Weekly_Sales"],
                errors="coerce"
            )
            st.line_chart(
                chart_df.set_index(chart_df.columns[0])[
                    "Predicted_Weekly_Sales"
                ]
            )

        st.download_button(
            label="Download Forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name="four_week_forecast.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error("Could not read the forecast file.")
        st.code(str(e))
else:
    st.warning("Forecast file not found in outputs folder.")

st.divider()

# Model comparison
st.subheader("📉 Model Performance Comparison")

if COMPARISON_PATH.exists():
    try:
        comparison_df = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison_df, use_container_width=True)
    except Exception as e:
        st.error("Could not read the comparison report.")
        st.code(str(e))
else:
    st.info("Model comparison report is not available.")

st.divider()

# Feature importance
st.subheader("🌟 XGBoost Feature Importance")

if IMPORTANCE_PATH.exists():
    st.image(
        IMPORTANCE_PATH,
        caption="XGBoost Feature Importance",
        use_container_width=True
    )
else:
    st.info("Feature importance image is not available.")

st.divider()

# Check saved model files
st.subheader("📁 Saved Model Files")

if MODEL_PATH.exists():
    st.success("XGBoost model file found.")
else:
    st.error(f"XGBoost model not found: {MODEL_PATH}")


if SARIMA_PATH.exists():
    st.success("SARIMA model file found.")
else:
    st.info(
        "SARIMA model file is not included in the deployed app. "
        "The saved forecast and model comparison are available above."
    )
