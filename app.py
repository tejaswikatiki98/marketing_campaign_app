import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("marketing_response_model.pkl")

st.title("🎯 Marketing Campaign Response Predictor")
st.write("Enter customer details to predict if they will respond positively to the campaign.")

# Input fields (matching your dataset)
gender = st.selectbox("Gender", ["M", "F"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Student", "Retired", "Unemployed"])
age = st.number_input("Age", min_value=18, max_value=80, value=30)
income_inr = st.number_input("Annual Income (INR)", min_value=100000, max_value=3000000, value=500000, step=10000)
spending_score = st.slider("Spending Score", 1, 100, 50)
campaign_contacts = st.number_input("Number of Campaign Contacts", min_value=0, max_value=20, value=3)
product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Groceries", "Furniture", "Cosmetics"])
preferred_channel = st.selectbox("Preferred Communication Channel", ["Email", "SMS", "Phone", "In-person"])

# Create DataFrame for model
input_data = pd.DataFrame({
    'Gender': [gender],
    'Marital_Status': [marital_status],
    'Occupation': [occupation],
    'Age': [age],
    'Income_INR': [income_inr],
    'Spending_Score': [spending_score],
    'Campaign_Contacts': [campaign_contacts],
    'Product_Category': [product_category],
    'Preferred_Channel': [preferred_channel]
})

# Predict
if st.button("Predict Response"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.success(f"✅ Customer is likely to respond positively! (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Customer is unlikely to respond. (Confidence: {1 - probability:.2f})")
