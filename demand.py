import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Load the saved model
model = joblib.load("demand_forecast_model.pkl")

st.title("📈 Demand Forecast Web App")
st.write("تطبيق لتوقع الطلب اليومي باستخدام Prophet")

# Input: number of future days
periods = st.slider("اختر عدد الأيام للتوقع:", min_value=7, max_value=60, value=30)

# Create future dataframe
future = model.make_future_dataframe(periods=periods)

# Forecast
forecast = model.predict(future)

# Plot forecast
st.subheader("🔮 توقع الطلب")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Show forecast table
st.subheader("📋 جدول التوقعات")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))
