import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained models
rf = joblib.load("random_forest.pkl")
svm = joblib.load("svm.pkl")
lr = joblib.load("linear_regression.pkl")

# Title
st.title("🌍 Earthquake Magnitude Prediction")

# Sidebar: Select model
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "SVM", "Linear Regression"])

# User input
st.subheader("Enter Earthquake Features")
col1, col2 = st.columns(2)
with col1:
    depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0)
    latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
with col2:
    longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
    time = st.number_input("Time (e.g. hour in 24h)", 0, 23, 12)

# Create feature vector
features = np.array([[depth, latitude, longitude, time]])

# Predict button
if st.button("Predict Magnitude"):
    if model_choice == "Random Forest":
        prediction = rf.predict(features)[0]
    elif model_choice == "SVM":
        prediction = svm.predict(features)[0]
    else:
        prediction = lr.predict(features)[0]

    st.success(f"Predicted Magnitude: {prediction:.2f}")

# Model Comparison Table
st.subheader("📊 Model Comparison")
comparison_data = {
    'Model Name': ['Linear Regression', 'SVM', 'Random Forest'],
    'MSE': [0.187942, 0.172375, 0.166287],
    'R² Score': [0.023514, 0.104396, 0.136024]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df)
