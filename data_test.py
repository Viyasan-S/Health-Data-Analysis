# data_test.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the trained ML model
model = joblib.load('advanced_vaccine_dose_model.pkl')

def test_data_visualization(vaccine_data, positivity_data):
    # Visualize vaccine doses using bar charts
    plt.figure(figsize=(12, 6))
    sns.barplot(x="State/UT", y="1st Dose - 18-44 years", data=vaccine_data)
    plt.title("1st Dose Administered - 18-44 years")
    plt.xticks(rotation=90)
    st.pyplot()

    # Visualize positivity rates using line charts
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="District", y="Positivity", data=positivity_data)
    plt.title("Positivity Rates by District")
    plt.xticks(rotation=90)
    st.pyplot()

def forecast_disease_outbreak(hcw_dose, flw_dose, age45_dose):
    # Make predictions using the trained model
    missing_feature_value = 0
    prediction = model.predict([[float(hcw_dose), float(flw_dose), float(age45_dose), missing_feature_value]])
    return prediction[0]

# data_test.py (continued)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def visualize_disease_forecast(hcw_dose, flw_dose, age45_dose):
    # Create a hypothetical scenario dataframe
    scenario_df = pd.DataFrame({
        "Healthcare Workers Dose": [hcw_dose],
        "Frontline Workers Dose": [flw_dose],
        "45+ Years Dose": [age45_dose]
    })

    # Predict the impact on total doses
    scenario_df["Total Doses"] = scenario_df.sum(axis=1)

    # Visualize the impact using a bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(x=scenario_df.columns[:-1], y=scenario_df.iloc[0, :-1])
    plt.title("Impact on Total Doses")
    st.pyplot()

    # Show the new total doses prediction
    st.write(f"Predicted Total Doses for the scenario: {scenario_df['Total Doses'].values[0]:,.0f}")

def main():
    st.title("Health Data Testing App")

    # Load datasets
    vaccine_data = pd.read_csv("RS_Session_254_AU_878_1.csv")
    positivity_data = pd.read_csv("WeeklyDistrictPositivityRateGT10_10-16November2021.csv")

    st.subheader("Test Data Visualization:")
    test_data_visualization(vaccine_data, positivity_data)

    st.subheader("Disease Forecasting:")
    hcw_dose = st.slider("Healthcare Workers Dose:", min_value=0, max_value=100000, step=1000)
    flw_dose = st.slider("Frontline Workers Dose:", min_value=0, max_value=100000, step=1000)
    age45_dose = st.slider("45+ Years Dose:", min_value=0, max_value=1000000, step=10000)

    visualize_disease_forecast(hcw_dose, flw_dose, age45_dose)

if __name__ == "__main__":
    main()
