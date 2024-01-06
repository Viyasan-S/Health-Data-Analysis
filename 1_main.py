# Advanced Streamlit App
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the trained ML model
model = joblib.load('advanced_vaccine_dose_model.pkl')

# Load datasets
vaccine_data = pd.read_csv("RS_Session_254_AU_878_1.csv")
positivity_data = pd.read_csv("WeeklyDistrictPositivityRateGT10_10-16November2021.csv")

# Streamlit app
st.title("Health Data Analysis App")

# Allow users to select visualization options
visualization_option = st.selectbox("Select Visualization Option:", ["Vaccine Doses", "Positivity Rates"])
selected_state = st.selectbox("Select State:", vaccine_data["State/UT"].unique())

# Display selected data
if visualization_option == "Vaccine Doses":
    st.subheader("Vaccine Data:")
    st.dataframe(vaccine_data[vaccine_data["State/UT"] == selected_state])

    # Visualize vaccine doses using bar charts
    st.subheader("Vaccine Doses - 18-44 years:")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="State/UT", y="1st Dose - 18-44 years", data=vaccine_data)
    plt.title("1st Dose Administered - 18-44 years")
    plt.xticks(rotation=90)
    st.pyplot()

    # Provide additional insights
    total_doses = vaccine_data["1st Dose - HCWs"] + vaccine_data["1st Dose - FLWs"] + vaccine_data["1st Dose - 45+ years"] + vaccine_data["1st Dose - 18-44 years"]
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
    st.pyplot()

    # Provide additional insights
    avg_positivity_rate = positivity_data["Positivity"].mean()
    st.write(f"Average Positivity Rate: {avg_positivity_rate:.2f}%")

# Allow users to interact with the ML model
st.subheader("ML Model Prediction:")
hcw_dose = st.slider("Healthcare Workers Dose:", min_value=0, max_value=100000, step=1000)
flw_dose = st.slider("Frontline Workers Dose:", min_value=0, max_value=100000, step=1000)
age45_dose = st.slider("45+ Years Dose:", min_value=0, max_value=1000000, step=10000)

# Assuming your model was trained with 4 features, add a placeholder for the missing feature
missing_feature_value = 0  # Replace with an appropriate value
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
    plt.figure(figsize=(8, 4))
    sns.barplot(x=scenario_df.columns[:-1], y=scenario_df.iloc[0, :-1])
    plt.title("Impact on Total Doses")
    st.pyplot(plt.gcf())

    # Show the new total doses prediction
    st.write(f"Predicted Total Doses for the scenario: {scenario_df['Total Doses'].values[0]:,.0f}")



