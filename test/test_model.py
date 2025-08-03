import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Get the current script directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Going up one level from 'test' folder

# Define paths
model_path = os.path.join(project_root, 'model', 'gradient_boosting_regressor_model.pkl')
data_path = os.path.join(project_root, 'data', 'Advertising.csv')

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def clean_advertising_data(df: pd.DataFrame) -> pd.DataFrame:
    # This is a simplified version of the cleaning function for testing purposes
    # In a real scenario, you would import the exact cleaning function from your data processing module
    df = df.dropna()
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    numerical_cols = ['tv', 'radio', 'newspaper', 'sales']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def preprocess_data(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    # Feature Engineering Steps
    df_cleaned['total_advertising_spend'] = df_cleaned['tv'] + df_cleaned['radio'] + df_cleaned['newspaper']
    df_cleaned['tv_radio_interaction'] = df_cleaned['tv'] * df_cleaned['radio']
    df_cleaned['tv_newspaper_interaction'] = df_cleaned['tv'] * df_cleaned['newspaper']
    df_cleaned['radio_newspaper_interaction'] = df_cleaned['radio'] * df_cleaned['newspaper']
    df_cleaned['tv^2'] = df_cleaned['tv']**2
    df_cleaned['radio^2'] = df_cleaned['radio']**2
    df_cleaned['newspaper^2'] = df_cleaned['newspaper']**2

    # Feature Scaling
    numerical_cols_to_scale = ['tv', 'radio', 'newspaper', 'total_advertising_spend',
                               'tv_radio_interaction', 'tv_newspaper_interaction',
                               'radio_newspaper_interaction', 'tv^2', 'radio^2', 'newspaper^2']
    scaler = StandardScaler()
    df_cleaned[numerical_cols_to_scale] = scaler.fit_transform(df_cleaned[numerical_cols_to_scale])
    return df_cleaned

def test_model_inference():
    print("\n--- Running Model Inference Test ---")
    # Load the trained model
    model = load_model(model_path)
    assert model is not None, "Model not loaded successfully."
    print("Model loaded successfully.")

    # Load a small sample of data for inference
    df = pd.read_csv(data_path)
    df_cleaned = clean_advertising_data(df.copy())
    df_processed = preprocess_data(df_cleaned.copy())

    # Prepare features for inference
    X_inference = df_processed.drop(columns=['unnamed:_0', 'sales'])

    print("Input features for inference (first 5 rows):")
    print(X_inference.head(5))

    # Get actual sales values for comparison
    actual_sales = df_processed['sales'].head(5).values

    # Make a prediction
    predictions = model.predict(X_inference.head(5))
    assert len(predictions) == 5, "Predictions not generated correctly."
    print(f"Actual Sales (first 5): {actual_sales}")
    print(f"Generated Predictions (first 5): {predictions}")
    print("--- Model Inference Test Completed Successfully ---")

if __name__ == "__main__":
    test_model_inference()