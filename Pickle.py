import pandas as pd

# Read the CSV file
csv_file_path = 'submission_case.csv'
df = pd.read_csv(csv_file_path)

# Save the DataFrame as a pickle file
pickle_file_path = 'submission_case.pkl'
df.to_pickle(pickle_file_path)

print(f"CSV file has been successfully converted to {pickle_file_path}")
