import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoders.pkl"

def preprocess_data(df, fit=False):
    # 1. Strip spaces from column names
    df.columns = df.columns.str.strip()

    # 2. Convert target if exists
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 3. Drop duplicates
    df = df.drop_duplicates()

    # 4. Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # 5. Separate X and y
    y = None
    if "Churn" in df.columns:
        y = df["Churn"].astype(int)
        X = df.drop("Churn", axis=1)
    else:
        X = df.copy()

    # 6. Drop customerID if exists
    if "customerID" in X.columns:
        X = X.drop("customerID", axis=1)

    # 7. Encode categorical variables
    if fit:
        encoders = {}
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)
    else:
        encoders = joblib.load(ENCODER_PATH)
        for col, le in encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))

    # 8. Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X = scaler.transform(X)

    return X, y
