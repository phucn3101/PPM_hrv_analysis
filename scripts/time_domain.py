import pandas as pd

def calculate_time_domain_features(df):
    features = []
    for index, row in df.iterrows():
        features.append({
            'uuid': row['uuid'],
            'mean_rr': row['SD1'],  # Placeholder for actual time domain calculation
            'sdnn': row['SD2']     # Placeholder for actual time domain calculation
        })
    return pd.DataFrame(features)

if __name__ == "__main__":
    df = pd.read_csv('data/raw/heart_rate_non_linear_features_train.csv')
    time_features = calculate_time_domain_features(df)
    print(time_features)
