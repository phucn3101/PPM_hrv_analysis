import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# not done yet