# Transaction-Prediction-Service
a transaction prediction service using fastapi
The case' dataset consist of 4 features (transaction_date, merchant_id, Total_Transaction, Total_Paid) and 7667 entrys from 2018-01-01 to 2020-12-31(3 years)
what wanted was the last 3 months of the dataset Total_Transaction prediction value using a lightgbm model
After the construction and training of the model, using fastapi the model became a reusable ml service
