# heart_disease_app.py
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# App Configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="💖", layout="wide")

# Banner and Title
col1, col2 = st.columns([1, 2])

with col1:
    st.image("data:image/jpeg;base64,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", width=150)

with col2:
    st.title("💖 Heart Disease Prediction App")
    st.markdown("Get a quick prediction of heart disease risk based on medical inputs.")

st.divider()  # Horizontal line

# Form Layout
with st.form("user_input_form"):
    st.header("Enter Your Medical Details")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        sex = st.radio("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])

    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

    with col3:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 5.0, 1.0)

    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert inputs to numerical values
    sex = 1 if sex == "Male" else 0
    cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

    # Predict Button
    submitted = st.form_submit_button("💡 Predict")

# Prediction Logic
if submitted:
    # Prepare input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    # Display Result
    st.divider()
    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
