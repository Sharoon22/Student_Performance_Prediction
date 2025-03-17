import streamlit as st
import pandas as pd

# st.title("Students Performance Prediction")
# name=st.text_input("Enter Your name")


# # data=pd.read_csv("Student_Performance.csv")
# # st.dataframe(data)

# if st.button("Predict"):
#     if name=="Rashid":
#         st.success(f"Your name is {name}")
#     else :
#         st.error("user not found")


import joblib

model = joblib.load("LinearRegression.pkl")
scaler = joblib.load("Scaler.pkl")
encoder = joblib.load("encoder.pkl")

hour_studied = st.number_input("Hour studied")
prev_score = st.number_input("Previous Score")
sleep_hours = st.number_input("Sleep Hours")
paper = st.number_input("Sample Question paper practiced")
eca = st.selectbox("Extracuricular Activity",("Yes","No"))

eca = encoder.transform([eca])
# st.write(eca)

dff=pd.DataFrame({"Hours Studied":hour_studied,"Previous Scores":prev_score,"Sleep Hours":sleep_hours,"Sample Question Papers Practiced":paper,'ECA':eca	})
# st.write(dff)

scaled_data = scaler.transform(dff)
# st.write(scaled_data)


if st.button("Predict"):
    prediction = model.predict(scaled_data)[0]
    st.write(prediction)

