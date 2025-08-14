# Telco Customer Churn Prediction App

A comprehensive machine learning application for predicting customer churn in the telecommunications industry.

## 🚀 Features

- **Real-time Predictions**: Interactive web interface for predicting customer churn probability
- **Comprehensive Analysis**: Detailed customer information input with visual results
- **Model Insights**: Feature importance and model performance metrics
- **User-Friendly**: Clean, intuitive interface built with Streamlit

## 📊 Technology Stack

- **Backend**: Python, Scikit-learn, Pandas
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Logistic Regression with SMOTE
- **Data Processing**: Pandas, NumPy

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telco-churn-prediction-app.git
cd telco-churn-prediction-app
```

2. Install dependencies:
```bash
pip install -r requirements/requirements.txt
```

3. Run the application:
```bash
streamlit run deployment/streamlit_app.py
```

## 📁 Project Structure

```
telco-churn-prediction-app/
├── data/                     # Dataset files
├── notebooks/                # Jupyter notebooks for EDA
├── src/                      # Source code
├── models/                   # Trained model artifacts
├── deployment/               # Streamlit application
├── docs/                     # Documentation
├── requirements/             # Dependencies
├── preprocessing/            # Preprocess
├── README.md                 # Project documentation
└── .gitignore                # Git ignore file
```

## 🎯 Usage

1. **Training the Model**: Run `src/train_model.py` to train and save the model
2. **Making Predictions**: Use the Streamlit app to input customer details and get predictions
3. **Testing Cases**: Refer to `docs/churn_prediction_cases.md` for specific test cases

## 🔧 Model Details

- **Algorithm**: Logistic Regression with SMOTE
- **Accuracy**: Optimized for churn prediction
- **Threshold**: 5% for churn classification
- **Features**: 20+ customer attributes including demographics, services, and billing

## 📊 Prediction Categories

- **Will Leave**: Churn probability ≥ 5%
- **Will Stay**: Churn probability < 5%
