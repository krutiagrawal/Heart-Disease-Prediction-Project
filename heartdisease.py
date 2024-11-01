import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Loading the trained model
model = load_model('heart_disease_model.h5')
scaler = joblib.load('scaler.pkl')

# Setting up Streamlit structure
st.title("Heart Disease Prediction App")
st.write("Enter the patient details to predict heart disease")

# Collecting user input
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=["Female", "Male"], index=1)

# Chest Pain Type Selection
cp_option = st.selectbox("Chest Pain Type",
                         options=[
                             (1, "Typical Angina"),
                             (2, "Atypical Angina"),
                             (3, "Non-Anginal Pain"),
                             (4, "Asymptomatic")
                         ],
                         format_func=lambda x: x[1])
cp = cp_option[0]

# Resting Blood Pressure
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

# Serum Cholesterol
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

# Fasting Blood Sugar Selection
fbs_option = st.selectbox("Fasting Blood Sugar (mg/dL)",
                          options=[
                              (0, "Sugar < 120 mg/dL"),
                              (1, "Sugar > 120 mg/dL")
                          ],
                          format_func=lambda x: x[1])
fbs = fbs_option[0]

# Resting ECG Results Selection
restecg_option = st.selectbox("Resting ECG Results",
                              options=[
                                  (0, "Normal"),
                                  (1, "ST-T Wave Abnormality"),
                                  (2, "Left Ventricular Hypertrophy")
                              ],
                              format_func=lambda x: x[1])
restecg = restecg_option[0]

# Max Heart Rate Achieved
thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=70, max_value=210, value=150)

# Exercise Induced Angina Selection
exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"], index=0)
exang = 1 if exang == "Yes" else 0

# ST Depression
oldpeak = st.number_input("ST Depression (mm)", min_value=0.0, max_value=10.0, value=1.0)

# Slope of ST Segment Selection
slope_option = st.selectbox("Slope of ST Segment",
                            options=[
                                (1, "Upward"),
                                (2, "Flat"),
                                (3, "Downward")
                            ],
                            format_func=lambda x: x[1])
slope = slope_option[0]

# Prediction
if st.button("Predict"):
    # Create features array for prediction
    features = np.array([[float(age),
                          1 if sex == "Male" else 0,
                          float(cp),
                          float(trestbps),
                          float(chol),
                          float(fbs),
                          float(restecg),
                          float(thalach),
                          float(exang),
                          float(oldpeak),
                          float(slope)]])

    # Scale the features
    features = scaler.transform(features)

    # Make the prediction
    prediction = model.predict(features)

    # Determine output message
    output_message = "No heart disease" if prediction[0][0] <= 0.5 else "Chances of heart disease"

    # Display prediction result
    st.write("Prediction: ", output_message)
