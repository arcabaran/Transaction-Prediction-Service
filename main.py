#main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained LightGBM model
with open('lightgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class TransactionInput(BaseModel):
    transaction_date: str = Field(..., example="2024-07-01")
    merchant_id: int = Field(..., example=535)

# Ensure merchant_id is one of the allowed values
ALLOWED_MERCHANT_IDS = {535, 42616, 46774, 57192, 86302, 124381, 129316}

# List of all feature columns used during training
FEATURE_COLUMNS = [ 
'merchant_id',	'Price	month',	'day_of_month','day_of_year','week_of_year',	'is_wknd',	
'is_month_start',	'is_month_end',	'sales_lag_91',	'sales_lag_120',	'sales_lag_152',	
'sales_lag_182',	'sales_lag_242',	'sales_lag_402',	'sales_lag_542',	'sales_lag_722',
'sales_roll_mean_91',	'sales_roll_mean_120',	'sales_roll_mean_152',	'sales_roll_mean_182',
'sales_roll_mean_242',	'sales_roll_mean_402',	'sales_roll_mean_542',	'sales_roll_mean_722',
'sales_ewm_alpha_095_lag_91',	'sales_ewm_alpha_095_lag_120',	'sales_ewm_alpha_095_lag_152',	
'sales_ewm_alpha_095_lag_182',	'sales_ewm_alpha_095_lag_242',	'sales_ewm_alpha_095_lag_402',
'sales_ewm_alpha_095_lag_542',	'sales_ewm_alpha_095_lag_722',	'sales_ewm_alpha_09_lag_91',
'sales_ewm_alpha_09_lag_120',	'sales_ewm_alpha_09_lag_152',	'sales_ewm_alpha_09_lag_182',
'sales_ewm_alpha_09_lag_242',	'sales_ewm_alpha_09_lag_402',	'sales_ewm_alpha_09_lag_542',	
'sales_ewm_alpha_09_lag_722',	'sales_ewm_alpha_08_lag_91',	'sales_ewm_alpha_08_lag_120',	
'sales_ewm_alpha_08_lag_152',	'sales_ewm_alpha_08_lag_182',	'sales_ewm_alpha_08_lag_242',	
'sales_ewm_alpha_08_lag_402',	'sales_ewm_alpha_08_lag_542',	'sales_ewm_alpha_08_lag_722',
'sales_ewm_alpha_07_lag_91',	'sales_ewm_alpha_07_lag_120',	'sales_ewm_alpha_07_lag_152',
'sales_ewm_alpha_07_lag_182',	'sales_ewm_alpha_07_lag_242',	'sales_ewm_alpha_07_lag_402',	
'sales_ewm_alpha_07_lag_542',	'sales_ewm_alpha_07_lag_722',	'sales_ewm_alpha_05_lag_91',
'sales_ewm_alpha_05_lag_120',	'sales_ewm_alpha_05_lag_152',	'sales_ewm_alpha_05_lag_182',
'sales_ewm_alpha_05_lag_242',	'sales_ewm_alpha_05_lag_402',	'sales_ewm_alpha_05_lag_542',
'sales_ewm_alpha_05_lag_722',	'day_of_week_0',	'day_of_week_1',	'day_of_week_2',	
'day_of_week_3',	'day_of_week_4',	'day_of_week_5',	'day_of_week_6',	'year_2018',	
'year_2019',	'year_2020'
]

async def enforce_exact_merchant_ids(merchant_id: int = Header(..., description="Merchant ID")):
    if merchant_id not in ALLOWED_MERCHANT_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid merchant_id. Allowed IDs are: {ALLOWED_MERCHANT_IDS}")

@app.post("/predict")
async def predict(input_data: TransactionInput):
    # Validate merchant_id
    if input_data.merchant_id not in ALLOWED_MERCHANT_IDS:
        raise HTTPException(status_code=400, detail="Invalid merchant_id")

    # Convert transaction_date to datetime and extract features
    try:
        transaction_date = datetime.strptime(input_data.transaction_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Example feature extraction
    year = transaction_date.year
    month = transaction_date.month
    day = transaction_date.day
    weekday = transaction_date.weekday()

    # Prepare the input data for prediction with all required features
    input_features = {
        'merchant_id': input_data.merchant_id,
        'year': year,
        'month': month,
        'day': day,
        'weekday': weekday,
    }

    # Fill missing features with default values (e.g., zero)
    for feature in FEATURE_COLUMNS:
        if feature not in input_features:
            input_features[feature] = 0

    # Ensure the order of features matches the training data
    input_df = pd.DataFrame([input_features], columns=FEATURE_COLUMNS)

    # Make predictions
    predictions = model.predict(input_df, predict_disable_shape_check=True)

    # Print predictions for debugging
    print("Shape of predictions:", predictions.shape)
    print("Content of predictions:", predictions)

    # Convert predictions to required format
    total_paid_pred = np.expm1(predictions[0])  # Assuming predictions is a scalar or single-element array


    return {
        "Total_Paid": total_paid_pred
    }

    
# To run the app, use the command: python -m uvicorn main:app --reload
