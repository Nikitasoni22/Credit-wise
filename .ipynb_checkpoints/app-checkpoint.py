import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading Models
    models = {
        "Logistic Regression": pickle.load(open('logistic_model.pkl', 'rb')),
        "KNN": pickle.load(open('knn_model.pkl', 'rb')),
        "Naive Bayes": pickle.load(open('nb_model.pkl', 'rb'))
    }
    # Loading Transformers
    ohe = pickle.load(open('ohe.pkl', 'rb'))
    le_edu = pickle.load(open('le_education.pkl', 'rb'))
    # Note: le_loan is loaded but usually only needed if inverse_transforming results
    le_loan = pickle.load(open('le_loan_status.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    
    return models, ohe, le_edu, le_loan, scaler

# Initialize assets
try:
    models, ohe, le_edu, le_loan, scaler = load_assets()
except Exception as e:
    st.error(f"Error: Missing or incompatible pickle files! {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Model Settings")
selected_algo = st.sidebar.selectbox("Choose Algorithm", list(models.keys()))
current_model = models[selected_algo]

# --- MAIN UI ---
st.title("🏦 AI Loan Eligibility System")
st.write(f"Currently using: **{selected_algo}**")
st.divider()

# Organize Inputs into 3 Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital_Status", ["Single", "Married"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education_Level", ["Graduate", "Not Graduate"])

with col2:
    st.subheader("💼 Employment & Income")
    emp_status = st.selectbox("Employment_Status", ["Salaried", "Self-employed", "Unemployed"])
    emp_cat = st.selectbox("Employer_Category", ["Government", "MNC", "Private", "Unemployed", "Business"])
    app_income = st.number_input("Applicant_Income", min_value=0, value=50000)
    co_income = st.number_input("Coapplicant_Income", min_value=0, value=0)
    savings = st.number_input("Savings", min_value=0, value=10000)

with col3:
    st.subheader("💰 Loan Requirements")
    loan_purpose = st.selectbox("Loan_Purpose", ["Personal", "Car", "Home", "Education"])
    prop_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])
    loan_amt = st.number_input("Loan_Amount", min_value=0, value=5000)
    loan_term = st.number_input("Loan_Term", min_value=1, value=360)
    existing_loans = st.number_input("Existing_Loans", min_value=0, value=0)
    collateral = st.number_input("Collateral_Value", min_value=0, value=0)
    credit_score = st.slider("Credit_Score", 300, 900, 750)
    dti_ratio = st.number_input("DTI_Ratio (e.g., 0.35)", value=0.30, format="%.2f")

# --- PREDICTION LOGIC ---
if st.button("Analyze Loan Risk", type="primary", use_container_width=True):
    
    # 1. Label Encoding (Education)
    edu_encoded = le_edu.transform([education])[0]

    # 2. One-Hot Encoding (Categorical Data)
    # Order must match the 'cols' list used during ohe.fit()
    ohe_input = pd.DataFrame([[
        emp_status, married, loan_purpose, prop_area, gender, emp_cat
    ]], columns=["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"])
    
    ohe_transformed = ohe.transform(ohe_input)
    ohe_cols = ohe.get_feature_names_out()
    ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_cols)

    # 3. Feature Engineering (The Squared Features)
    # We create these but do NOT include the raw score/dti as per your notebook's X_train.head()
    cs_sq = float(credit_score ** 2)
    dti_sq = float(dti_ratio ** 2)

    # 4. Construct Final Feature Vector (Exactly 27 Columns)
    # Order: [Numericals/Label -> 15 OHE Columns -> 2 Squared Columns]
    
    # Start with the first 10 columns
    final_features_dict = {
        'Applicant_Income': app_income,
        'Coapplicant_Income': co_income,
        'Age': age,
        'Dependents': dependents,
        'Existing_Loans': existing_loans,
        'Savings': savings,
        'Collateral_Value': collateral,
        'Loan_Amount': loan_amt,
        'Loan_Term': loan_term,
        'Education_Level': edu_encoded
    }
    
    # Add the 15 OHE columns dynamically from your ohe_df
    for col in ohe_cols:
        final_features_dict[col] = ohe_df[col].values[0]
        
    # Add the remaining engineered features at the very end
    final_features_dict['Credit_Score_sq'] = cs_sq
    final_features_dict['DTI_Ratio_sq'] = dti_sq

    # Convert to DataFrame (This ensures we have 27 columns)
    final_df = pd.DataFrame([final_features_dict])

    # 5. Scaling
    # Crucial: final_df must have 27 columns matching what scaler was fitted on
    final_scaled = scaler.transform(final_df)

    # 6. Make Prediction
    prediction = current_model.predict(final_scaled)
    probability = current_model.predict_proba(final_scaled)

    # --- DISPLAY RESULTS ---
    st.divider()
    confidence = np.max(probability) * 100
    
    if prediction[0] == 1:
        st.balloons()
        st.success(f"### 🎉 Result: Loan Approved!")
        st.metric("Approval Confidence", f"{confidence:.2f}%")
        st.write("The applicant meets the eligibility criteria based on the selected model.")
    else:
        st.error(f"### ❌ Result: Loan Rejected")
        st.metric("Rejection Confidence", f"{confidence:.2f}%")
        st.write("The applicant does not meet the necessary risk threshold.")

# Footer info
st.sidebar.divider()
st.sidebar.info("This deployment uses a 27-feature vector including squared Credit Score and DTI metrics.")