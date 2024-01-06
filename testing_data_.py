# testing_data.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate synthetic vaccination data
states = ["State1", "State2", "State3", "State4", "State5"]
vaccine_data = pd.DataFrame({
    "State/UT": np.random.choice(states, size=100),
    "1st Dose - HCWs": np.random.randint(1000, 5000, size=100),
    "1st Dose - FLWs": np.random.randint(5000, 10000, size=100),
    "1st Dose - 45+ years": np.random.randint(20000, 50000, size=100),
    "1st Dose - 18-44 years": np.random.randint(10000, 30000, size=100)
})

# Generate synthetic primary health data for children
children_health_data = pd.DataFrame({
    "State/UT": np.random.choice(states, size=50),
    "Underweight Percentage": np.random.uniform(5, 20, size=50),
    "Immunization Coverage": np.random.uniform(70, 95, size=50),
    "Child Mortality Rate": np.random.uniform(2, 8, size=50)
})

# Generate synthetic disease detection rates
districts = ["District1", "District2", "District3", "District4", "District5"]
disease_data = pd.DataFrame({
    "State": np.random.choice(states, size=200),
    "District": np.random.choice(districts, size=200),
    "Disease Type": np.random.choice(["Dengue", "Malaria", "COVID-19"], size=200),
    "Detection Rate": np.random.uniform(0.1, 10, size=200)
})

# Generate synthetic date range for data
date_range = [datetime.now() - timedelta(days=x) for x in range(200)]

# Generate synthetic ML forecast for disease outbreaks
ml_forecast_data = pd.DataFrame({
    "Date": random.sample(date_range, 50),
    "Forecasted Cases": np.random.randint(50, 500, size=50),
    "Disease Type": np.random.choice(["Dengue", "Malaria", "COVID-19"], size=50)
})

# Save synthetic data to CSV files
vaccine_data.to_csv("synthetic_vaccine_data.csv", index=False)
children_health_data.to_csv("synthetic_children_health_data.csv", index=False)
disease_data.to_csv("synthetic_disease_data.csv", index=False)
ml_forecast_data.to_csv("synthetic_ml_forecast_data.csv", index=False)
