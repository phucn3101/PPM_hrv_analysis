import pandas as pd
import numpy as np
from scipy.signal import welch

def calculate_psd(signal, fs=4):
    f, Pxx = welch(signal, fs=fs)
    return f, Pxx

def get_frequency_domain_features(df, fs=4):
    features = []
    for index, row in df.iterrows():
        print(f"Processing row {index} with UUID: {row['uuid']}")  # Debugging line
        combined_signal = np.array([row['SD1'], row['SD2']])
        f, Pxx = calculate_psd(combined_signal, fs)
        lf_power = np.sum(Pxx[(f >= 0.04) & (f <= 0.15)])
        hf_power = np.sum(Pxx[(f >= 0.15) & (f <= 0.4)])
        features.append({
            'uuid': row['uuid'],
            'LF': lf_power,
            'HF': hf_power,
            'LF_HF_ratio': lf_power / hf_power if hf_power != 0 else np.nan
        })
    print(features)  # Debugging line
    return pd.DataFrame(features)

if __name__ == "__main__":
    df = pd.read_csv('data/raw/heart_rate_non_linear_features_train.csv')
    frequency_features = get_frequency_domain_features(df)
    print(frequency_features)
