import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# Get the current script directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Going up one level from 'train' folder

# Construct paths relative to the project root
data_path = os.path.join(project_root, 'data', 'Advertising.csv')

# Load the data
df = pd.read_csv(data_path)


# Data Cleaning Function (from previous steps)
def clean_advertising_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Advertising dataset.

    Args:
        df: The input pandas DataFrame containing the advertising data.

    Returns:
        The cleaned pandas DataFrame.
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Dropping rows with missing values.")
        # Handle missing values by dropping rows
        df = df.dropna()
        print("Rows with missing values dropped.")

    # Check for duplicate rows
    if df.duplicated().sum() > 0:
        print("Duplicate rows found. Removing duplicate rows.")
        # Remove duplicate rows
        df = df.drop_duplicates()
        print("Duplicate rows removed.")
    else:
        print("No duplicate rows found.")

    # Standardize column names
    print("Standardizing column names.")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print("Column names standardized.")

    # Examine and ensure correct data types
    print("Examining and ensuring correct data types.")
    # Based on the data, 'unnamed:_0' is an identifier and can be an integer,
    # while 'tv', 'radio', 'newspaper', and 'sales' are numerical and should be float.
    # We will explicitly convert them to ensure consistency.
    df['unnamed:_0'] = df['unnamed:_0'].astype(int)
    df['tv'] = df['tv'].astype(float)
    df['radio'] = df['radio'].astype(float)
    df['newspaper'] = df['newspaper'].astype(float)
    df['sales'] = df['sales'].astype(float)

    print("Data types ensured.")

    # Handle outliers
    print("Handling outliers using IQR capping.")
    numerical_cols = ['tv', 'radio', 'newspaper', 'sales']

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25) # Step 2: Calculate Q1
        Q3 = df[col].quantile(0.75) # Step 2: Calculate Q3
        IQR = Q3 - Q1             # Step 2: Calculate IQR

        lower_bound = Q1 - 1.5 * IQR # Step 3: Define lower bound
        upper_bound = Q3 + 1.5 * IQR # Step 3: Define upper bound

        # Step 5 & 6: Cap outliers
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    print("Outliers capped using IQR.")

    return df

# Apply cleaning function
df_cleaned = clean_advertising_data(df.copy())

# Feature Engineering Steps (from previous steps)
# Create interaction features
df_cleaned['total_advertising_spend'] = df_cleaned['tv'] + df_cleaned['radio'] + df_cleaned['newspaper']
df_cleaned['tv_radio_interaction'] = df_cleaned['tv'] * df_cleaned['radio']
df_cleaned['tv_newspaper_interaction'] = df_cleaned['tv'] * df_cleaned['newspaper']
df_cleaned['radio_newspaper_interaction'] = df_cleaned['radio'] * df_cleaned['newspaper']

# Polynomial features
# Using degree 2 for simplicity as explored before
df_cleaned['tv^2'] = df_cleaned['tv']**2
df_cleaned['radio^2'] = df_cleaned['radio']**2
df_cleaned['newspaper^2'] = df_cleaned['newspaper']**2

# Feature Scaling (from previous steps)
# Identify numerical columns for scaling, excluding identifiers and the target variable
numerical_cols_to_scale = ['tv', 'radio', 'newspaper', 'total_advertising_spend',
                           'tv_radio_interaction', 'tv_newspaper_interaction',
                           'radio_newspaper_interaction', 'tv^2', 'radio^2', 'newspaper^2']

# Instantiate the StandardScaler
scaler = StandardScaler()

# Fit and transform the data before splitting
df_cleaned[numerical_cols_to_scale] = scaler.fit_transform(df_cleaned[numerical_cols_to_scale])


# Define the features (X) and the target variable (y) using the cleaned data with engineered features
X = df_cleaned.drop(columns=['unnamed:_0', 'sales'])
y = df_cleaned['sales']

# Split the data into training and testing sets (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Gradient Boosting Regressor model
final_gbr_model = GradientBoostingRegressor(random_state=42)

# Train the model on the training data
final_gbr_model.fit(X_train, y_train)

# Predict on the test data
y_pred_test = final_gbr_model.predict(X_test)

# Calculate and print the R2 score on the test data
r2_test = r2_score(y_test, y_pred_test)
print(f"\nR2 score on the test data: {r2_test:.4f}")


# Save the trained model to a pkl file
model_dir = os.path.join(project_root, 'model')
os.makedirs(model_dir, exist_ok=True)  # Create model directory if it doesn't exist
model_path = os.path.join(model_dir, 'gradient_boosting_regressor_model.pkl')

pickle.dump(final_gbr_model, open(model_path, 'wb'))
print(f"Trained Gradient Boosting Regressor model saved to {model_path}")