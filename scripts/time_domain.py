import pandas as pd
import numpy as np

def get_time_domain_features(df):
    time_domain_features = []

    for _, row in df.iterrows():
        uuid = row['uuid']
        sd1 = row['SD1']
        sd2 = row['SD2']
        
        # Example time domain features (using SD1 and SD2 as placeholders)
        mean_nn = sd1  # Placeholder for mean of NN intervals
        sdnn = sd2  # Placeholder for standard deviation of NN intervals
        rmssd = np.sqrt(np.mean(np.square(sd1 - sd2)))  # Placeholder for RMSSD
        pnn50 = np.mean(np.abs(sd1 - sd2) > 50) * 100  # Placeholder for pNN50

        time_domain_features.append({
            'UUID': uuid,
            'Mean NN': mean_nn,
            'SDNN': sdnn,
            'RMSSD': rmssd,
            'pNN50': pnn50
        })
    
    return time_domain_features
