import pandas as pd

# Load the dataset
test_data_path = r'C:\Users\NGUYEN\Documents\TDTU\2023 - 2024\DoAn\PPM_hrv_analysis\data\raw\heart_rate_non_linear_features_test.csv'
train_data_path = r'C:\Users\NGUYEN\Documents\TDTU\2023 - 2024\DoAn\PPM_hrv_analysis\data\raw\heart_rate_non_linear_features_train.csv'

# Load the CSV files into DataFrames
test_df = pd.read_csv(test_data_path)
train_df = pd.read_csv(train_data_path)

# Inspect the first few rows of the data
print("Test Data:")
print(test_df.head())

print("\nTrain Data:")
print(train_df.head())

# Check the column names and types
print("\nTest Data Info:")
print(test_df.info())

print("\nTrain Data Info:")
print(train_df.info())
