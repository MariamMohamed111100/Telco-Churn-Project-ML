import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Configure paths
DATA_PATH = os.path.join('..', 'data', 'cleaned_telco_data.csv')
MODEL_PATH = '../models/logistic_regression_smote.pkl'
SCALER_PATH = '../models/scaler.pkl'
ENCODER_PATH = '../models/label_encoders.pkl'

def train_and_save_model():
    """Train model and save all components"""
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Basic preprocessing
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Prepare features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].astype(int)
    
    # Drop customerID
    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    model = LogisticRegression(
        class_weight='balanced', 
        max_iter=5000, 
        random_state=42
    )
    model.fit(X_train_res, y_train_res)
    
    # Evaluate model
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_class = (y_pred_prob >= 0.05).astype(int)
    
    print("Model Performance:")
    print(classification_report(y_test, y_pred_class))
    
    # Save all components
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    
    # Save feature columns
    joblib.dump(X.columns.tolist(), '../models/feature_columns.pkl')
    
    print("✅ Model and components saved successfully!")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Scaler: {SCALER_PATH}")
    print(f"   Encoders: {ENCODER_PATH}")
    
    return model, scaler, label_encoders, X.columns.tolist()

if __name__ == "__main__":
    train_and_save_model()
