import streamlit as st
import numpy as np
import joblib


model = joblib.load("knn_model.pkl")

st.title("Room Occupancy Prediction App")

st.header("Enter Input Features")
user_input = []
for column in ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light',
        'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
        'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR'
    ]:
    val = st.number_input(f"{column}", format="%.2f")
    user_input.append(val)

if st.button("Predict"):
    user_array = np.array([user_input])
    prediction = model.predict(user_array)[0]
    st.success(f"Room Occupancy Count: {prediction}")
