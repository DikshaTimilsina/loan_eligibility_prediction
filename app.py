import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Load the trained model
with open('rf_model (1).pkl', 'rb') as file:
    model = joblib.load(file)
    
#title
st.title("Loan Approval Prediction App")

#enter details 
st.write("Enter applicant details to predict loan approval")

#Input Fields

#Categorical inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

#Numerical inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
Loan_Amount = st.number_input("Loan Amount", min_value=0)
Loan_Term_Years = st.number_input("Loan Term (Years)", min_value=1)

#feature engineering
Total_Income = Applicant_Income + Coapplicant_Income
EMI = Loan_Amount / (Loan_Term_Years * 12)
Income_Loan_Ratio = Total_Income / Loan_Amount if Loan_Amount != 0 else 0


#Encode categorical values
Gender = 1 if Gender == "Male" else 0
Married = 1 if Married == "Yes" else 0
Education = 1 if Education == "Graduate" else 0
Self_Employed = 1 if Self_Employed == "Yes" else 0

Dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
Dependents = Dependents_map[Dependents]

Property_Area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
Property_Area = Property_Area_map[Property_Area]


#Create Input DataFrame
input_data = pd.DataFrame([[
    Gender, Married, Dependents, Education, Self_Employed,
    Applicant_Income, Coapplicant_Income, Loan_Amount,
    Credit_History, Property_Area,
    Total_Income, EMI, Income_Loan_Ratio, Loan_Term_Years
]], columns=[
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
    "Credit_History", "Property_Area",
    "Total_Income", "EMI", "Income_Loan_Ratio", "Loan_Term_Years"
])





#predict and display result

if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
            st.success(f" Loan Approved\n\nApproval Chance: {probability*100:.2f}%")
    else:
            st.error(f"Loan Rejected\n\nApproval Chance: {probability*100:.2f}%")

        

        
    
