import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load HRV data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Clean the HRV data by removing or imputing missing values.
    """
    data = data.dropna()  # Simple drop NA for now; consider imputation for advanced preprocessing
    return data

def extract_rr_intervals(data):
    """
    Extract RR intervals from HRV data. Assume the column name is 'RR_intervals'.
    """
    rr_intervals = data['RR_intervals']
    return rr_intervals

def preprocess_rr_intervals(rr_intervals):
    """
    Preprocess RR intervals: e.g., normalization, outlier removal.
    """
    rr_intervals = rr_intervals[rr_intervals.between(rr_intervals.quantile(.05), rr_intervals.quantile(.95))]
    return rr_intervals

if __name__ == "__main__":
    # Example usage
    file_path = "../data/raw/hrv_data.csv"
    data = load_data(file_path)
    data = clean_data(data)
    rr_intervals = extract_rr_intervals(data)
    rr_intervals = preprocess_rr_intervals(rr_intervals)
    
    # Save preprocessed data
    preprocessed_data_path = "../data/processed/preprocessed_rr_intervals.csv"
    rr_intervals.to_csv(preprocessed_data_path, index=False)
