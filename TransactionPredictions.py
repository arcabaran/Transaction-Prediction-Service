from pydantic import BaseModel
class TrnsactionPredictions(BaseModel):
    merchant_id:int
    Total_Paid:float
    transaction_date:float
import pandas as pd

# Step 1: Read the CSV file
csv_file_path = 'submission_case.csv'
df = pd.read_csv(csv_file_path)

# Step 2: Convert the transaction_date column to datetime64
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Step 3: Convert the transaction_date column from datetime64 to float
df['transaction_date_float'] = df['transaction_date'].apply(lambda x: x.timestamp())

