# Customer Churn Prediction Cases - Streamlit App

This document contains specific input combinations for the churn prediction app that will show different prediction outcomes.

## "Will Leave" Cases (Churn Probability ≥ 5%)

### Case 1: High-Risk New Customer
- **Gender**: Male
- **Senior Citizen**: No
- **Partner**: No
- **Dependents**: No
- **Tenure**: 2 months
- **Phone Service**: Yes
- **Multiple Lines**: Yes
- **Internet Service**: Fiber optic
- **Online Security**: No
- **Online Backup**: No
- **Device Protection**: No
- **Tech Support**: No
- **Streaming TV**: Yes
- **Streaming Movies**: Yes
- **Contract**: Month-to-month
- **Paperless Billing**: Yes
- **Payment Method**: Electronic check
- **Monthly Charges**: $95.00
- **Total Charges**: $190.00

### Case 2: Senior Citizen Premium Services
- **Gender**: Female
- **Senior Citizen**: Yes
- **Partner**: No
- **Dependents**: No
- **Tenure**: 6 months
- **Phone Service**: Yes
- **Multiple Lines**: Yes
- **Internet Service**: Fiber optic
- **Online Security**: No
- **Online Backup**: No
- **Device Protection**: No
- **Tech Support**: No
- **Streaming TV**: Yes
- **Streaming Movies**: Yes
- **Contract**: Month-to-month
- **Paperless Billing**: Yes
- **Payment Method**: Electronic check
- **Monthly Charges**: $105.00
- **Total Charges**: $630.00

### Case 3: Short-term Basic Services
- **Gender**: Male
- **Senior Citizen**: No
- **Partner**: No
- **Dependents**: No
- **Tenure**: 3 months
- **Phone Service**: No
- **Multiple Lines**: No phone service
- **Internet Service**: DSL
- **Online Security**: No
- **Online Backup**: No
- **Device Protection**: No
- **Tech Support**: No
- **Streaming TV**: No
- **Streaming Movies**: No
- **Contract**: Month-to-month
- **Paperless Billing**: Yes
- **Payment Method**: Electronic check
- **Monthly Charges**: $55.00
- **Total Charges**: $165.00

## "Will Stay" Cases (Churn Probability < 5%)

### Case 1: Loyal Long-term Customer
- **Gender**: Female
- **Senior Citizen**: No
- **Partner**: Yes
- **Dependents**: Yes
- **Tenure**: 36 months
- **Phone Service**: Yes
- **Multiple Lines**: Yes
- **Internet Service**: DSL
- **Online Security**: Yes
- **Online Backup**: Yes
- **Device Protection**: Yes
- **Tech Support**: Yes
- **Streaming TV**: Yes
- **Streaming Movies**: Yes
- **Contract**: Two year
- **Paperless Billing**: No
- **Payment Method**: Credit card (automatic)
- **Monthly Charges**: $55.00
- **Total Charges**: $1980.00

### Case 2: Stable Low-risk Customer
- **Gender**: Male
- **Senior Citizen**: No
- **Partner**: Yes
- **Dependents**: No
- **Tenure**: 24 months
- **Phone Service**: Yes
- **Multiple Lines**: No
- **Internet Service**: DSL
- **Online Security**: Yes
- **Online Backup**: No
- **Device Protection**: Yes
- **Tech Support**: Yes
- **Streaming TV**: No
- **Streaming Movies**: No
- **Contract**: One year
- **Paperless Billing**: Yes
- **Payment Method**: Bank transfer (automatic)
- **Monthly Charges**: $65.00
- **Total Charges**: $1560.00

### Case 3: Committed Auto-pay Customer
- **Gender**: Female
- **Senior Citizen**: Yes
- **Partner**: No
- **Dependents**: No
- **Tenure**: 48 months
- **Phone Service**: Yes
- **Multiple Lines**: Yes
- **Internet Service**: DSL
- **Online Security**: Yes
- **Online Backup**: Yes
- **Device Protection**: Yes
- **Tech Support**: Yes
- **Streaming TV**: Yes
- **Streaming Movies**: Yes
- **Contract**: Two year
- **Paperless Billing**: Yes
- **Payment Method**: Credit card (automatic)
- **Monthly Charges**: $75.00
- **Total Charges**: $3600.00

## Key Risk Factors Summary

**High Risk (Will Leave):**
- Tenure < 12 months
- Month-to-month contract
- Electronic check payment
- Monthly charges > $70
- Missing premium services

**Low Risk (Will Stay):**
- Tenure > 12 months
- One-year or two-year contract
- Automatic payment methods
- Monthly charges $30-80
- Long-term commitment indicators
