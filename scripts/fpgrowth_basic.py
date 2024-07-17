import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def load_data(file_path):
    """
    Load preprocessed HRV data.
    """
    data = pd.read_csv(file_path)
    return data

def fpgrowth_basic(data, min_support=0.1, min_confidence=0.5):
    """
    Perform basic FP-Growth algorithm for frequent itemset mining.
    """
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

if __name__ == "__main__":
    # Example usage
    file_path = "../data/processed/preprocessed_rr_intervals.csv"
    data = load_data(file_path)
    
    # Convert data to one-hot encoding format if necessary
    # Assuming data is already in the right format for simplicity
    
    rules = fpgrowth_basic(data)
    print(rules)
