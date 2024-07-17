import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def load_data(file_path):
    """
    Load preprocessed HRV data.
    """
    data = pd.read_csv(file_path)
    return data

def fpgrowth_improved(data, min_support=0.1, min_confidence=0.5):
    """
    Perform optimized FP-Growth algorithm for frequent itemset mining.
    """
    # Example improvement: Optimize data storage
    data_sparse = data.astype(pd.SparseDtype("float", 0))
    frequent_itemsets = fpgrowth(data_sparse, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

if __name__ == "__main__":
    # Example usage
    file_path = "../data/processed/preprocessed_rr_intervals.csv"
    data = load_data(file_path)
    
    # Convert data to one-hot encoding format if necessary
    # Assuming data is already in the right format for simplicity
    
    rules = fpgrowth_improved(data)
    print(rules)
