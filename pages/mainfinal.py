# Advanced Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
# Load the trained ML model
model = joblib.load('advanced_vaccine_dose_model.pkl')

# Function to load synthetic data for testing
def load_synthetic_data():
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

    # Save synthetic data to CSV files
    vaccine_data.to_csv("synthetic_vaccine_data.csv", index=False)
    children_health_data.to_csv("synthetic_children_health_data.csv", index=False)
    disease_data.to_csv("synthetic_disease_data.csv", index=False)

    return vaccine_data, children_health_data, disease_data

states = ["State1", "State2", "State3", "State4", "State5"]
districts = ["District1", "District2", "District3", "District4", "District5"]

# Load synthetic data for testing
vaccine_data, children_health_data, disease_data = load_synthetic_data()

positivity_data = pd.DataFrame({
    "State": np.random.choice(states, size=50),
    "District": np.random.choice(districts, size=50),
    "Positivity": np.random.uniform(0, 10, size=50)
})

# Data Preprocessing (Similar to model_creation.py)
numeric_columns = vaccine_data.select_dtypes(include=[np.number]).columns
imputer_numeric = SimpleImputer(strategy="mean")
vaccine_data_imputed_numeric = pd.DataFrame(imputer_numeric.fit_transform(vaccine_data[numeric_columns]), columns=numeric_columns)
vaccine_data_imputed = pd.concat([vaccine_data_imputed_numeric, vaccine_data[vaccine_data.select_dtypes(exclude=[np.number]).columns]], axis=1)
vaccine_data_imputed['Total Doses'] = vaccine_data_imputed['1st Dose - HCWs'] + vaccine_data_imputed['1st Dose - FLWs'] + vaccine_data_imputed['1st Dose - 45+ years'] + vaccine_data_imputed['1st Dose - 18-44 years']

# Streamlit app
st.title("Health Data Analysis App")

# Allow users to select visualization options
visualization_option = st.selectbox("Select Visualization Option:", ["Vaccine Doses", "Positivity Rates", "Testing Data"])
selected_state = st.selectbox("Select State:", vaccine_data["State/UT"].unique())

# Display selected data
if visualization_option == "Vaccine Doses":
    st.subheader("Vaccine Data:")
    st.dataframe(vaccine_data_imputed[vaccine_data_imputed["State/UT"] == selected_state])

    # Visualize vaccine doses using bar charts
    st.subheader("Vaccine Doses - 18-44 years:")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="State/UT", y="1st Dose - 18-44 years", data=vaccine_data_imputed)
    plt.title("1st Dose Administered - 18-44 years")
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())

    # Provide additional insights
    total_doses = vaccine_data_imputed["1st Dose - HCWs"] + vaccine_data_imputed["1st Dose - FLWs"] + vaccine_data_imputed["1st Dose - 45+ years"] + vaccine_data_imputed["1st Dose - 18-44 years"]
    st.write(f"Total Doses Administered: {total_doses.sum():,.0f}")

elif visualization_option == "Positivity Rates":
    st.subheader("Positivity Data:")
    st.dataframe(positivity_data[positivity_data["State"] == selected_state])

    # Visualize positivity rates using line charts
    st.subheader("Positivity Rates by District:")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="District", y="Positivity", data=positivity_data)
    plt.title("Positivity Rates by District")
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())

    # Provide additional insights
    avg_positivity_rate = positivity_data["Positivity"].mean()
    st.write(f"Average Positivity Rate: {avg_positivity_rate:.2f}%")

elif visualization_option == "Testing Data":
    st.subheader("Testing Data:")
    st.dataframe(vaccine_data)  # Display synthetic testing data

# Allow users to interact with the ML model
st.subheader("ML Model Prediction:")
hcw_dose = st.slider("Healthcare Workers Dose:", min_value=0, max_value=100000, step=1000)
flw_dose = st.slider("Frontline Workers Dose:", min_value=0, max_value=100000, step=1000)
age45_dose = st.slider("45+ Years Dose:", min_value=0, max_value=1000000, step=10000)

# Placeholder for the missing feature
missing_feature_value = 0
# Predict using the trained model
prediction = model.predict([[float(hcw_dose), float(flw_dose), float(age45_dose), missing_feature_value]])
st.write(f"Predicted Dose for 18-44 years: {prediction[0]:,.0f}")

# Dynamic Visualization based on ML prediction
st.subheader("Dynamic Visualization:")
if st.button("Visualize Prediction Impact"):
    # Create a hypothetical scenario dataframe
    scenario_df = pd.DataFrame({
        "Healthcare Workers Dose": [hcw_dose],
        "Frontline Workers Dose": [flw_dose],
        "45+ Years Dose": [age45_dose]
    })

    # Predict the impact on total doses
    scenario_df["Total Doses"] = scenario_df.sum(axis=1)

    # Visualize the impact using a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=scenario_df.columns[:-1], y=scenario_df.iloc[0, :-1])
    plt.title("Impact on Total Doses")

    # Display the chart using st.plotly_chart
    st.plotly_chart(fig)
    
    # Show the new total doses prediction
    st.write(f"Predicted Total Doses for the scenario: {scenario_df['Total Doses'].values[0]:,.0f}")
