import streamlit as st
import pandas as pd
import joblib

# Load model and tools
model = joblib.load("sleep_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Sleep Quality Predictor", layout="centered")
st.title("😴 Sleep Smart - Predict Your Sleep Quality")

st.markdown("### Enter your daily habits:")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 80)
occupation = st.selectbox("Occupation", ["Student", "Doctor", "Engineer", "Teacher", "Artist", "Manager"])
stress = st.slider("Stress Level (1-5)", 1, 5)
bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
bp = st.selectbox("Blood Pressure", ["120/80", "130/85", "110/75", "140/90", "115/78", "125/80"])
hr = st.slider("Heart Rate", 50, 120)
steps = st.slider("Daily Steps", 1000, 20000, step=500)
activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
sleep = st.slider("Sleep Duration (in hours)", 1, 12)

# Predict button
if st.button("🔍 Predict Sleep Quality"):
    input_df = pd.DataFrame([[gender, age, occupation, stress, bmi, bp, hr, steps, activity, sleep]],
                            columns=['Gender', 'Age', 'Occupation', 'Stress_Level', 'BMI_Category',
                                     'Blood_Pressure', 'Heart_Rate', 'Daily_Steps', 'Physical_Activity_Level', 'Sleep_Duration'])

    # Encode using saved encoders
    for col in input_df.select_dtypes(include='object').columns:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    sleep_quality = label_encoders["Sleep_Quality"].inverse_transform([prediction])[0]

    st.success(f"🛌 Predicted Sleep Quality: **{sleep_quality}**")


