from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def binarize_data(df):
    return (df > df.mean()).astype(int)

def run_fp_growth(df):
    # Apply the FP-Growth algorithm
    frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)
    
    # Generate the rules with a minimum confidence of 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    # Convert frozenset to list for JSON serialization
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return rules.to_dict(orient='records')

if __name__ == "__main__":
    train_data_path = '../data/processed/heart_rate_non_linear_features_train_processed.csv'
    train_df = load_data(train_data_path)
    df_binarized = binarize_data(train_df[['SD1', 'SD2', 'sampen', 'higuci']])
    rules = run_fp_growth(df_binarized)
    print(rules)
