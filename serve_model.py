import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Get the current script directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
model_path = os.path.join(current_dir, 'model', 'gradient_boosting_regressor_model.pkl')

# Load the trained model
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model(model_path)

# Define the input data model for FastAPI
class PredictionInput(BaseModel:
    tv: float
    radio: float
    newspaper: float

# Data preprocessing functions (simplified, should ideally be imported from a shared module)
def clean_advertising_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    numerical_cols = ['tv', 'radio', 'newspaper', 'sales'] # 'sales' will be dropped later for prediction
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def preprocess_data(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    df_cleaned['total_advertising_spend'] = df_cleaned['tv'] + df_cleaned['radio'] + df_cleaned['newspaper']
    df_cleaned['tv_radio_interaction'] = df_cleaned['tv'] * df_cleaned['radio']
    df_cleaned['tv_newspaper_interaction'] = df_cleaned['tv'] * df_cleaned['newspaper']
    df_cleaned['radio_newspaper_interaction'] = df_cleaned['radio'] * df_cleaned['newspaper']
    df_cleaned['tv^2'] = df_cleaned['tv']**2
    df_cleaned['radio^2'] = df_cleaned['radio']**2
    df_cleaned['newspaper^2'] = df_cleaned['newspaper']**2

    numerical_cols_to_scale = ['tv', 'radio', 'newspaper', 'total_advertising_spend',
                               'tv_radio_interaction', 'tv_newspaper_interaction',
                               'radio_newspaper_interaction', 'tv^2', 'radio^2', 'newspaper^2']
    scaler = StandardScaler()
    df_cleaned[numerical_cols_to_scale] = scaler.fit_transform(df_cleaned[numerical_cols_to_scale])
    return df_cleaned

@app.post("/predict/")
async def predict(data: PredictionInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Apply preprocessing steps
    # Note: 'sales' column is not present in input_df, so clean_advertising_data needs adjustment
    # For simplicity, we'll assume the input data is already clean enough for these features
    # In a real scenario, you'd have a dedicated inference preprocessing pipeline
    processed_df = preprocess_data(input_df.copy())

    # Drop any columns not used for prediction (e.g., 'unnamed:_0' if it existed)
    # Ensure the order of columns matches the training data
    # For this model, the features are the ones created in preprocess_data
    X_inference = processed_df[['tv', 'radio', 'newspaper', 'total_advertising_spend',
                                'tv_radio_interaction', 'tv_newspaper_interaction',
                                'radio_newspaper_interaction', 'tv^2', 'radio^2', 'newspaper^2']]

    prediction = model.predict(X_inference)[0]
    return {"prediction": prediction}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)