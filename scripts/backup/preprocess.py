import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Exclude UUID column from numerical processing
    numerical_cols = ['SD1', 'SD2', 'sampen', 'higuci']
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    
    # Normalize the features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

if __name__ == "__main__":
    # train_data_path = r'C:\Users\NGUYEN\Documents\TDTU\2023 - 2024\DoAn\PPM_hrv_analysis\data\raw\heart_rate_non_linear_features_train.csv'
    # test_data_path = r'C:\Users\NGUYEN\Documents\TDTU\2023 - 2024\DoAn\PPM_hrv_analysis\data\raw\heart_rate_non_linear_features_test.csv'

    train_data_path = os.path.join('data', 'raw', 'heart_rate_non_linear_features_train.csv')
    test_data_path = os.path.join('data', 'raw', 'heart_rate_non_linear_features_test.csv')

    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_df.to_csv('data/processed/heart_rate_non_linear_features_train_processed.csv', index=False)
    test_df.to_csv('data/processed/heart_rate_non_linear_features_test_processed.csv', index=False)

    print("Data preprocessing completed.")
