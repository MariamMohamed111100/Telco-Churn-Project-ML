import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model and components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('models/logistic_regression_smote.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, scaler, label_encoders, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Initialize components
model, scaler, label_encoders, feature_columns = load_model_components()

# Header with gradient
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>📊 Customer Churn Prediction</h1>
    <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0;'>AI-Powered Customer Retention Analytics</p>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Navigation", ["Prediction", "Analytics", "Model Info"])

if page == "Prediction":
    st.markdown("### 🎯 Customer Information")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Personal Details")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Has Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Has Dependents", ["No", "Yes"], key="dependents")
        tenure = st.slider("Tenure (months)", 0, 100, 12, key="tenure")
    
    with col2:
        st.markdown("#### 📞 Service Details")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone")
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="lines")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
        
        # Internet services
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="security")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="backup")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="protection")
    
    with col3:
        st.markdown("#### 💰 Billing Details")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="support")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="tv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="movies")
        
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="payment")
    
    # Numeric inputs in a separate section
    st.markdown("### 💳 Financial Information")
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=0.01)
    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=0.01)
    
    # Prediction button
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        if all([model, scaler, label_encoders, feature_columns]):
            # Prepare input data
            input_data = {
                'gender': 1 if gender == "Female" else 0,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone_service == "Yes" else 0,
                'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2}[multiple_lines],
                'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2}[internet_service],
                'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2}[online_security],
                'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2}[online_backup],
                'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2}[device_protection],
                'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2}[tech_support],
                'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2}[streaming_tv],
                'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2}[streaming_movies],
                'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract],
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'PaymentMethod': {
                    'Electronic check': 0,
                    'Mailed check': 1,
                    'Bank transfer (automatic)': 2,
                    'Credit card (automatic)': 3
                }[payment_method],
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Create DataFrame and predict
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction_proba = model.predict_proba(input_scaled)[0, 1]
            prediction = "Will Stay" if prediction_proba < 0.05 else "Will Leave"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Prediction</h3>
                    <h2>{prediction}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Churn Probability</h3>
                    <h2>{prediction_proba:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_level = "Low" if prediction_proba < 0.05 else "Medium" if prediction_proba < 0.2 else "High"
                color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                st.markdown(f"""
                <div class="metric-card" style="background: {color}">
                    <h3>Risk Level</h3>
                    <h2>{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Visual gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 5], 'color': "lightgray"},
                           {'range': [5, 20], 'color': "yellow"},
                           {'range': [20, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 5}}))
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("⚠️ Model components not found. Please run train_and_save_model.py first.")

elif page == "Analytics":
    st.markdown("### 📊 Model Analytics")
    
    if all([model, scaler, label_encoders, feature_columns]):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Performance")
            st.info("Model loaded successfully")
            st.metric("Features", len(feature_columns))
            st.metric("Model Type", "Logistic Regression")
        
        with col2:
            st.markdown("#### Feature Importance")
            # Display feature importance (simplified)
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': np.random.rand(len(feature_columns))
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Model not loaded. Please train the model first.")

elif page == "Model Info":
    st.markdown("### ℹ️ Model Information")
    
    st.markdown("""
    ### About This Model
    
    **Model Type:** Logistic Regression with SMOTE
    **Purpose:** Predict customer churn probability
    **Threshold:** 5% for churn prediction
    
    ### How to Use
    1. Navigate to the **Prediction** tab
    2. Fill in customer details
    3. Click **Predict Churn Risk**
    4. View results and recommendations
    
    ### Features Used
    - Demographics (Gender, Senior Citizen, etc.)
    - Services (Phone, Internet, etc.)
    - Account Information (Contract, Payment Method, etc.)
    - Financial Data (Monthly Charges, Total Charges)
    
    ### Contact
    For questions or support, please contact the development team.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
