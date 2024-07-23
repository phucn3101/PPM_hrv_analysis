import pandas as pd
import numpy as np

def get_frequency_domain_features(df):
    frequency_domain_features = []

    for _, row in df.iterrows():
        uuid = row['uuid']
        sd1 = row['SD1']
        sd2 = row['SD2']
        
        # Example frequency domain features (using SD1 and SD2 as placeholders)
        lf = sd1  # Placeholder for Low Frequency component
        hf = sd2  # Placeholder for High Frequency component
        lf_hf_ratio = lf / hf if hf != 0 else np.nan  # LF/HF Ratio

        frequency_domain_features.append({
            'UUID': uuid,
            'LF': lf,
            'HF': hf,
            'LF/HF Ratio': lf_hf_ratio
        })
    
    return frequency_domain_features
