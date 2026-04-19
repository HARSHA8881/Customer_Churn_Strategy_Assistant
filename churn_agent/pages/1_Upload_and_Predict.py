import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Ensure imports work from within pages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Milestone 1: Upload & Predict", layout="wide")

st.markdown('<div style="color:#1a1a2e;font-size:28px;font-weight:600;">Milestone 1: ML-Based Customer Churn Prediction</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trained_models.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.markdown('<div style="color:#e9c46a;font-weight:bold;">Model not found. Please train a model first or check if models/trained_models.pkl exists.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color:#1a1a2e;font-size:20px;font-weight:600;margin-top:20px;margin-bottom:10px;">Predict Churn using Trained ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 40)
        
    with col2:
        tenure = st.number_input("Tenure", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
        num_products = st.selectbox("Num Of Products", [1, 2, 3, 4])
        
    with col3:
        has_crcard = st.selectbox("Has Credit Card (1=Yes, 0=No)", [1, 0])
        is_active = st.selectbox("Is Active Member (1=Yes, 0=No)", [1, 0])
        salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)
        
    if st.button("Predict Risk", type="primary"):
        input_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_crcard,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary
        }
        
        df = pd.DataFrame([input_data])
        prob = model.predict_proba(df)[0][1]
        
        st.metric("Churn Risk", f"{prob:.1%}")
        if prob > 0.5:
            st.markdown('<div style="color:#e63946;font-weight:bold;">High Risk of Churn</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#2a9d8f;font-weight:bold;">Customer Retained</div>', unsafe_allow_html=True)
