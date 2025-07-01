import joblib
import numpy as np 
import streamlit as st

model = joblib.load("gbr.pkl")
scaler = joblib.load("scaler.pkl")

### Give Title
st.title("Your Insurance Charge Predictor ⚖")

# collect input
age = st.number_input("AGE")
bmi = st.number_input("BMI")
children = st.number_input("No.of Children")
smoker = st.selectbox("Do you smoke?",["yes","no"])

smoker_val =1 if smoker =="yes" else 0

# convert into numpy array
user_input = np.array([[age,bmi,children,smoker_val]])

# scale the user input
user_input_scaled = scaler.transform(user_input)

if st.button("Click to Predict"):
    prediction = model.predict(user_input_scaled)
    st.success(f"estimated insurance charge:{prediction}")