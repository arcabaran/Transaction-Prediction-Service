# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Any
from datetime import date
import pandas as pd
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transaction Prediction API", description="API to predict total paid based on merchant ID and transaction date", version="1.0")

# Load the model
try:
    with open('transaction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Pickle file not found.")
    raise HTTPException(status_code=404, detail="Pickle file not found")
except Exception as e:
    logger.error(f"Error loading Pickle file: {e}")
    raise HTTPException(status_code=500, detail="Error loading Pickle file")

# Allowed merchant IDs
ALLOWED_MERCHANT_IDS = {535, 42616, 46774, 57192, 86302, 124381, 129316}

# Define the input data model
class PredictionInput(BaseModel):
    merchant_id: int
    transaction_date: date
    features: Dict[str, Any]

    @validator('merchant_id')
    def check_merchant_id(cls, v):
        if v not in ALLOWED_MERCHANT_IDS:
            raise ValueError(f'Merchant ID {v} is not allowed')
        return v

@app.post("/predict/", summary="Predict Total Paid", response_description="Predicted total paid for the given merchant ID and transaction date")
async def predict(input_data: PredictionInput):
    """
    Predict the total paid for the given merchant ID and transaction date.
    """
    try:
        # Convert the input features to a DataFrame
        input_df = pd.DataFrame([input_data.features])

        # Ensure the input data has the correct columns
        required_columns = model.get_params()['fit_intercept']
        if not all(column in input_df.columns for column in required_columns):
            missing_cols = list(set(required_columns) - set(input_df.columns))
            logger.warning(f"Missing columns in input data: {missing_cols}")
            raise HTTPException(status_code=400, detail=f"Missing columns in input data: {missing_cols}")

        # Add merchant_id and transaction_date to input_df for the prediction
        input_df['merchant_id'] = input_data.merchant_id
        input_df['transaction_date'] = input_data.transaction_date

        # Make predictions
        predictions = model.predict(input_df)
        
        # Return the prediction with merchant_id and transaction_date
        result = {
            "merchant_id": input_data.merchant_id,
            "transaction_date": input_data.transaction_date,
            "Total_Paid": predictions[0]
        }
        
        logger.info(f"Returning result for merchant ID {input_data.merchant_id} and transaction date {input_data.transaction_date}: {result}")
        return result
    
    except ValueError as ve:
        logger.error(f"Value error during processing: {ve}")
        raise HTTPException(status_code=400, detail=f"Value error during processing: {ve}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# To run the app, use the command: python -m uvicorn main:app --reload
