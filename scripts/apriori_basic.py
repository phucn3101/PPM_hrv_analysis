import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_data(file_path):
    """
    Load preprocessed HRV data.
    """
    data = pd.read_csv(file_path)
    return data

def apriori_basic(data, min_support=0.1, min_confidence=0.5):
    """
    Perform basic Apriori algorithm for frequent itemset mining.
    """
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

if __name__ == "__main__":
    # Example usage
    file_path = "../data/processed/preprocessed_rr_intervals.csv"
    data = load_data(file_path)
    
    # Convert data to one-hot encoding format if necessary
    # Assuming data is already in the right format for simplicity
    
    rules = apriori_basic(data)
    print(rules)
