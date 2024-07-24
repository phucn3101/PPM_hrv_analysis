from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from scripts.visualize import plot_rules
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def binarize_data(df, dataset_type):
    if dataset_type == 'non_linear':
        cols = ['SD1', 'SD2', 'sampen', 'higuci']
    elif dataset_type == 'frequency':
        cols = ['VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF']
    elif dataset_type == 'time':
        cols = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'pNN25', 'pNN50', 'KURT', 'SKEW',
                'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR',
                'KURT_REL_RR', 'SKEW_REL_RR']
    else:
        raise ValueError("Unknown dataset type")
    
    return (df[cols] > df[cols].mean()).astype(int)

def run_apriori(df):
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    
    # Generate the rules with a minimum confidence of 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    # Convert frozenset to list for JSON serialization
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return rules.to_dict(orient='records')

def plot_rules(rules, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker="o", edgecolors="w", s=100)
    plt.title('Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()