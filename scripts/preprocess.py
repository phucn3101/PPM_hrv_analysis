# Run this file to preprocess dataset in data/raw if needed
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, dataset_type):
    if dataset_type == 'non_linear':
        numerical_cols = ['SD1', 'SD2', 'sampen', 'higuci']
    elif dataset_type == 'frequency':
        numerical_cols = ['VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF']
    elif dataset_type == 'time':
        numerical_cols = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'pNN25', 'pNN50', 'KURT', 'SKEW',
                          'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR',
                          'KURT_REL_RR', 'SKEW_REL_RR']
    else:
        raise ValueError("Unknown dataset type")
    
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    df.fillna(method='ffill', inplace=True)
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

if __name__ == "__main__":
    train_data_path = os.path.join('data', 'raw', 'heart_rate_non_linear_features_train.csv')
    test_data_path = os.path.join('data', 'raw', 'heart_rate_non_linear_features_test.csv')

    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)

    train_df = preprocess_data(train_df, 'non_linear')
    test_df = preprocess_data(test_df, 'non_linear')

    train_df.to_csv('data/processed/heart_rate_non_linear_features_train_processed.csv', index=False)
    test_df.to_csv('data/processed/heart_rate_non_linear_features_test_processed.csv', index=False)

    print("Data preprocessing completed.")
