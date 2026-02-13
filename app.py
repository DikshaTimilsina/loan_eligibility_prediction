import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
@st.cache_resource
def load_model():
    with open('rf_model (1).pkl', 'rb') as file:
        return joblib.load(file)

model = load_model()

# Get the feature names the model expects
expected_features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
if expected_features:
    st.sidebar.write("Model expects these features:", expected_features)

#title with styling
st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="💰")
st.title("💰 Loan Eligibility Prediction App")
st.markdown("---")

#enter details 
st.write("Please enter the applicant details below to predict loan approval status")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    #Categorical inputs
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.subheader("Loan Details")
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0], 
                                 format_func=lambda x: "Yes" if x == 1.0 else "No")
    
    #Numerical inputs
    Applicant_Income = st.number_input("Applicant Income (₹)", min_value=0, step=1000, value=5000)
    Coapplicant_Income = st.number_input("Coapplicant Income (₹)", min_value=0, step=1000, value=0)
    Loan_Amount = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=100000)
    
    # IMPORTANT: Use Years instead of Months to match the model
    Loan_Term_Years = st.number_input("Loan Term (Years)", min_value=1, max_value=40, value=30)

st.markdown("---")

#feature engineering
Total_Income = Applicant_Income + Coapplicant_Income
# Calculate EMI based on Years (convert to months for EMI calculation)
EMI = Loan_Amount / (Loan_Term_Years * 12) if Loan_Term_Years > 0 else 0
Income_Loan_Ratio = Total_Income / Loan_Amount if Loan_Amount > 0 else 0

# Display calculated features
with st.expander("View Calculated Features"):
    col3, col4, col5 = st.columns(3)
    col3.metric("Total Income", f"₹{Total_Income:,.2f}")
    col4.metric("Monthly EMI", f"₹{EMI:,.2f}")
    col5.metric("Income to Loan Ratio", f"{Income_Loan_Ratio:.2f}")

#Encode categorical values
Gender_encoded = 1 if Gender == "Male" else 0
Married_encoded = 1 if Married == "Yes" else 0
Education_encoded = 1 if Education == "Graduate" else 0
Self_Employed_encoded = 1 if Self_Employed == "Yes" else 0

Dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
Dependents_encoded = Dependents_map[Dependents]

Property_Area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
Property_Area_encoded = Property_Area_map[Property_Area]

#Create Input DataFrame with EXACT column names the model expects
input_data = pd.DataFrame([[
    Gender_encoded, Married_encoded, Dependents_encoded, Education_encoded, Self_Employed_encoded,
    Applicant_Income, Coapplicant_Income, Loan_Amount,
    Credit_History, Property_Area_encoded,
    Total_Income, EMI, Income_Loan_Ratio, Loan_Term_Years  # Using Loan_Term_Years here
]], columns=[
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
    "Credit_History", "Property_Area",
    "Total_Income", "EMI", "Income_Loan_Ratio", "Loan_Term_Years"  # Column name matches model
])

# Debug: Show the input data shape and columns
if st.checkbox("Show technical details (for debugging)"):
    st.write("Input data shape:", input_data.shape)
    st.write("Input columns:", input_data.columns.tolist())
    if expected_features:
        st.write("Model expects:", expected_features)
        st.write("Columns match:", input_data.columns.tolist() == expected_features)

#predict and display result
if st.button("🔮 Predict Loan Status", type="primary"):
    try:
        with st.spinner("Analyzing application..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # Create a nice layout for results
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col6, col7 = st.columns(2)
            
            with col6:
                if prediction == 1:
                    st.success(f"### ✅ LOAN APPROVED")
                    st.metric("Approval Probability", f"{probability*100:.2f}%")
                    
                    # Add confidence indicator
                    if probability > 0.8:
                        st.info("📈 High confidence approval")
                    elif probability > 0.6:
                        st.info("📊 Moderate confidence approval")
                    else:
                        st.warning("📉 Borderline approval")
                        
                else:
                    st.error(f"### ❌ LOAN REJECTED")
                    st.metric("Approval Probability", f"{probability*100:.2f}%")
                    
                    # Add rejection reason hints
                    rejection_reasons = []
                    if Credit_History == 0:
                        rejection_reasons.append("❌ Poor credit history")
                    if Income_Loan_Ratio < 0.3:
                        rejection_reasons.append("❌ Low income relative to loan amount")
                    if Loan_Term_Years > 30:
                        rejection_reasons.append("❌ Very long loan term")
                    
                    if rejection_reasons:
                        st.warning("Potential reasons:")
                        for reason in rejection_reasons:
                            st.write(reason)
                    else:
                        st.info("Application doesn't meet all criteria")
            
            with col7:
                # Create a simple gauge for probability
                st.subheader("Probability Gauge")
                prob_display = probability * 100
                st.progress(prob_display/100)
                st.caption(f"{prob_display:.1f}% chance of approval")
                
                # Add threshold indicator
                if prob_display >= 70:
                    st.success("✨ Above approval threshold")
                elif prob_display >= 50:
                    st.warning("⚖️ Near threshold")
                else:
                    st.error("📉 Below approval threshold")
                    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check that all inputs are valid and try again.")

# Add footer with disclaimer
st.markdown("---")
st.caption("⚠️ Disclaimer: This is a predictive model-based tool. Final approval depends on bank's verification and policies.")