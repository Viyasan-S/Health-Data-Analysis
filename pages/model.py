import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load example dataset (you'll need actual disease data)
# For demonstration purposes, let's assume a CSV file with columns: Date, Cases
data = pd.read_csv("disease.csv")

# Sidebar for user input
st.sidebar.header("User Input")

# Allow users to select forecasting method
forecast_method = st.sidebar.selectbox("Select Forecasting Method", ["Random Forest", "ARIMA", "Prophet"])

# Depending on the selected method, provide relevant input options
if forecast_method == "Random Forest":
    forecast_days = st.sidebar.slider("Number of Days to Forecast", 1, 30, 7)

# Depending on the selected method, provide relevant input options
if forecast_method == "ARIMA":
    order_p = st.sidebar.slider("Order (p) for ARIMA", 0, 5, 2)
    order_d = st.sidebar.slider("Order (d) for ARIMA", 0, 2, 1)
    order_q = st.sidebar.slider("Order (q) for ARIMA", 0, 5, 2)

# Depending on the selected method, provide relevant input options
if forecast_method == "Prophet":
    growth = st.sidebar.selectbox("Growth", ["linear", "logistic"])
    n_changepoints = st.sidebar.slider("Number of Changepoints", 1, 20, 5)

# Data preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Feature engineering (you may need more sophisticated features)
data['Day'] = (data.index - data.index.min()).days

# Train-test split
train_data, test_data = train_test_split(data, test_size=forecast_days, shuffle=False)

# Train a model based on the selected method
if forecast_method == "Random Forest":
    model = RandomForestRegressor()
    model.fit(train_data[['Day']], train_data['Cases'])

# Add more forecasting methods here (ARIMA, Prophet, etc.)

# Make predictions for the next forecast_days
if forecast_method == "Random Forest":
    forecast_range = pd.date_range(data.index.max() + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_data = pd.DataFrame(index=forecast_range, columns=['Cases'])
    forecast_data['Day'] = (forecast_data.index - data.index.min()).days
    forecast_data['Predicted Cases'] = model.predict(forecast_data[['Day']])

# Visualize data and predictions
st.title("Disease Outbreak Forecasting")
st.subheader("Advanced Forecasting Program")

# Display raw data
st.write("Raw Data:")
st.write(data)

# Display forecast
st.write("Forecast:")
st.write(forecast_data)

# Plot raw data and predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.index, data['Cases'], label='Actual Cases', color='blue')
ax.plot(forecast_data.index, forecast_data['Predicted Cases'], label='Predicted Cases', linestyle='--', color='orange')
ax.set_title("Disease Outbreak Forecasting")
ax.set_xlabel("Date")
ax.set_ylabel("Cases")
ax.legend()
st.pyplot(fig)

# Additional insights or analytics can be added based on the specific disease and available data.
# For example, you can calculate growth rates, visualize trends, etc.

# Calculate Growth Rate
data['Growth Rate'] = data['Cases'].pct_change() * 100
st.subheader("Growth Rate Over Time:")
st.line_chart(data['Growth Rate'])

# Visualize Trends Over Time
st.subheader("Cases Trends Over Time:")
plt.figure(figsize=(12, 6))
sns.lineplot(x=data.index, y='Cases', data=data)
plt.title("Disease Outbreak Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.xticks(rotation=45)
st.pyplot()
