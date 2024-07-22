import pandas as pd
import numpy as np
from scipy.signal import welch

def calculate_psd(hrv_signal, fs=4):
    f, Pxx = welch(hrv_signal, fs=fs)
    return f, Pxx

def get_frequency_domain_features(df, fs=4):
    features = []
    for index, row in df.iterrows():
        hrv_signal = row['hrv_signal']
        f, Pxx = calculate_psd(hrv_signal, fs)
        features.append({
            'uuid': row['uuid'],
            'LF': np.sum(Pxx[(f >= 0.04) & (f <= 0.15)]),
            'HF': np.sum(Pxx[(f >= 0.15) & (f <= 0.4)]),
            'LF_HF_ratio': np.sum(Pxx[(f >= 0.04) & (f <= 0.15)]) / np.sum(Pxx[(f >= 0.15) & (f <= 0.4)])
        })
    return pd.DataFrame(features)

if __name__ == "__main__":
    df = pd.read_csv('data/raw/heart_rate_non_linear_features_train.csv')
    frequency_features = get_frequency_domain_features(df)
    print(frequency_features)
