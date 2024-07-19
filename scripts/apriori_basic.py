from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def binarize_data(df):
    return (df > df.mean()).astype(int)

def run_apriori(df):
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    
    # Generate the rules with a minimum confidence of 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    # Convert frozenset to list for JSON serialization
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return rules.to_dict(orient='records')

if __name__ == "__main__":
    train_data_path = r'C:\Users\NGUYEN\Documents\TDTU\2023 - 2024\DoAn\PPM_hrv_analysis\data\processed\heart_rate_non_linear_features_train_processed.csv'
    train_df = load_data(train_data_path)
    df_binarized = binarize_data(train_df[['SD1', 'SD2', 'sampen', 'higuci']])
    rules = run_apriori(df_binarized)
    print(rules)
