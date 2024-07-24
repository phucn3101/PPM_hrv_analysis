from flask import Flask, request, render_template
import pandas as pd
import logging
from scripts.preprocess import preprocess_data
from scripts.apriori_basic import run_apriori, binarize_data
from scripts.fpgrowth_basic import run_fp_growth
from scripts.visualize import plot_rules
from scripts.frequency_domain import get_frequency_domain_features
from scripts.time_domain import get_time_domain_features
from scripts.visualize import plot_apriori_rules, plot_fp_growth_rules, plot_frequency_domain_features, plot_time_domain_features
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        try:
            start_time = time.time()
            df = pd.read_csv(file)
            logging.info("File loaded successfully.")
            
            # Determine dataset type from file name or user input (example assumes file name)
            if 'non_linear' in file.filename:
                dataset_type = 'non_linear'
            elif 'frequency' in file.filename:
                dataset_type = 'frequency'
            elif 'time' in file.filename:
                dataset_type = 'time'
            else:
                raise ValueError("Unknown dataset type in file name")

            df_processed = preprocess_data(df, dataset_type)
            logging.info("Data preprocessing completed.")

            df_binarized = binarize_data(df_processed, dataset_type)
            logging.info("Data binarization completed.")
            
            apriori_rules = run_apriori(df_binarized)
            plot_rules(apriori_rules, f'Apriori_{dataset_type}') # plot - created error - debug
            
            fp_growth_rules = run_fp_growth(df_binarized)
            plot_rules(fp_growth_rules, f'FP_Growth_{dataset_type}') # plot - created error - debug

            if dataset_type == 'non_linear':
                return render_template('results.html', apriori_rules=apriori_rules, fp_growth_rules=fp_growth_rules)
            elif dataset_type == 'frequency':
                frequency_features = get_frequency_domain_features(df_processed)
                logging.info("Frequency domain feature extraction completed.")
                # plot_frequency_domain_features(frequency_features) # plot - created error - debug
                return render_template('results.html', apriori_rules=apriori_rules, fp_growth_rules=fp_growth_rules, frequency_features=frequency_features)
            elif dataset_type == 'time':
                time_features = get_time_domain_features(df_processed)
                logging.info("Time domain feature extraction completed.")
                # plot_time_domain_features(time_features) # plot - created error - debug
                return render_template('results.html', apriori_rules=apriori_rules, fp_growth_rules=fp_growth_rules, time_features=time_features)
            else:
                df_binarized = binarize_data(df_processed)
                apriori_rules = run_apriori(df_binarized)
                plot_apriori_rules(apriori_rules) # plot - created error - debug
                fp_growth_rules = run_fp_growth(df_binarized)
                plot_fp_growth_rules(fp_growth_rules) # plot - created error - debug

            end_time = time.time()
            logging.info(f"Total processing time: {end_time - start_time} seconds.")

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return f"Error processing file: {e}"

if __name__ == '__main__':
    app.run(debug=True)
