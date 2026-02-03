import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("src/loan_model.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("### 📊 Model Information")
st.write("Algorithm: Decision Tree Classifier")
st.write("Accuracy: ~80%")
st.write("Enter applicant details to predict loan approval status.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", value=360)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding (same logic as training)
def encode_input():
    return pd.DataFrame({
        "Loan_ID": [0],
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "Dependents": [3 if dependents == "3+" else int(dependents)],
        "Education": [1 if education == "Graduate" else 0],
        "Self_Employed": [1 if self_employed == "Yes" else 0],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [{"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]]
    })

# Predict Button
if st.button("Predict Loan Status"):
    input_df = encode_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()
    
    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {probability:.2f})")

    