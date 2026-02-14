import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('rf_model (1).pkl')

# Page title
st.set_page_config(page_title="Loan Eligibility Checker")
st.title("Loan Eligibility Prediction App")
st.write("Fill the form below to check if you are eligible for loan")

# Create input fields
st.subheader("Personal Information")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

st.subheader("Loan Information")

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

applicant_income = st.number_input("Applicant Income (per month)", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income (per month)", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
loan_term = st.number_input("Loan Term (in years)", min_value=1, max_value=40, value=30)

# Calculate new features
total_income = applicant_income + coapplicant_income
monthly_emi = loan_amount / (loan_term * 12)
income_to_loan_ratio = total_income / loan_amount

# Convert text inputs to numbers that model understands
if gender == "Male":
    gender_num = 1
else:
    gender_num = 0

if married == "Yes":
    married_num = 1
else:
    married_num = 0

if dependents == "0":
    dependents_num = 0
elif dependents == "1":
    dependents_num = 1
elif dependents == "2":
    dependents_num = 2
else:
    dependents_num = 3

if education == "Graduate":
    education_num = 1
else:
    education_num = 0

if self_employed == "Yes":
    self_employed_num = 1
else:
    self_employed_num = 0

if property_area == "Urban":
    property_area_num = 2
elif property_area == "Semiurban":
    property_area_num = 1
else:
    property_area_num = 0

if credit_history == "Good (1)":
    credit_history_num = 1
else:
    credit_history_num = 0

# Create a single row of data for prediction
input_features = [[
    gender_num,
    married_num,
    dependents_num,
    education_num,
    self_employed_num,
    applicant_income,
    coapplicant_income,
    loan_amount,
    credit_history_num,
    property_area_num,
    total_income,
    monthly_emi,
    income_to_loan_ratio,
    loan_term
]]

# Define column names
column_names = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
    "Credit_History", "Property_Area",
    "Total_Income", "EMI", "Income_Loan_Ratio", "Loan_Term_Years"
]

# Convert to DataFrame
input_df = pd.DataFrame(input_features, columns=column_names)

# Add a predict button
st.markdown("---")
predict_button = st.button("Check Loan Eligibility")

# Make prediction when button is clicked
if predict_button:
    # Show what was calculated
    st.write("---")
    st.subheader("Calculated Values")
    st.write(f"Total Income: ₹{total_income}")
    st.write(f"Monthly EMI: ₹{monthly_emi:.2f}")
    st.write(f"Income to Loan Ratio: {income_to_loan_ratio:.2f}")
    
    # Get prediction
    result = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.write("---")
    st.subheader("Prediction Result")
    
    if result == 1:
        st.success("✅ Congratulations! Your loan is APPROVED")
        st.write(f"Approval Probability: {probability[1]*100:.2f}%")
    else:
        st.error("❌ Sorry! Your loan is REJECTED")
        st.write(f"Approval Probability: {probability[1]*100:.2f}%")
        
        # Simple explanation for rejection
        st.write("---")
        st.write("Possible reasons for rejection:")
        if credit_history_num == 0:
            st.write("• Bad credit history")
        if income_to_loan_ratio < 0.3:
            st.write("• Income is too low for this loan amount")
        if loan_term > 30:
            st.write("• Loan term is too long")

# Add note at bottom
st.write("---")
st.write("Note: This is a prediction based on data. Final decision depends on bank's policies.")