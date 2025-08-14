import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_regression_smote.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoders.pkl')

# Global variables for model components
model = None
scaler = None
label_encoders = {}
feature_columns = None

def load_model_components():
    """Load all model components safely"""
    global model, scaler, label_encoders, feature_columns
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        logger.info("Model components loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Handle missing values
        for col in df.columns:
            if col in ['customerID', 'Churn']:
                continue
                
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        # Apply label encoding
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Drop customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Ensure all expected columns are present
        expected_features = feature_columns or model.n_features_in_
        if len(df.columns) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(df.columns)}")
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        return scaled_features
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess and predict
        processed_data = preprocess_input(data)
        prediction_proba = model.predict_proba(processed_data)[0, 1]
        prediction = int(prediction_proba >= 0.05)  # Using your threshold
        
        return jsonify({
            'prediction': prediction,
            'probability': float(prediction_proba),
            'threshold': 0.05
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        results = []
        for item in data['data']:
            processed = preprocess_input(item)
            proba = model.predict_proba(processed)[0, 1]
            pred = int(proba >= 0.05)
            results.append({
                'prediction': pred,
                'probability': float(proba)
            })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model components
    load_model_components()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
