from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import logging
from scripts.preprocess import preprocess_data
from scripts.apriori_basic import run_apriori
from scripts.fpgrowth_basic import run_fp_growth
from scripts.visualize import plot_rules
from scripts.frequency_domain import get_frequency_domain_features
from scripts.time_domain import get_time_domain_features
from scripts.periodic_patterns import get_periodic_patterns
import time
import json
import os

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

            df_processed = preprocess_data(df)
            logging.info("Data preprocessing completed.")

            # Ensure the DataFrame has the required columns for frequency domain feature extraction
            if 'SD1' not in df_processed.columns or 'SD2' not in df_processed.columns:
                raise ValueError("Input DataFrame must contain 'SD1' and 'SD2' columns")

            # Calculate periodic patterns
            periodic_patterns = get_periodic_patterns(df_processed)
            logging.info("Periodic patterns extraction completed.")

            # Calculate time domain features
            time_domain_features = get_time_domain_features(df_processed)
            logging.info("Time domain feature extraction completed.")

            # Calculate frequency domain features
            frequency_features = get_frequency_domain_features(df_processed)
            logging.info("Frequency domain feature extraction completed.")

            df_binarized = (df_processed[['SD1', 'SD2', 'sampen', 'higuci']] > df_processed[['SD1', 'SD2', 'sampen', 'higuci']].mean()).astype(int)
            logging.info("Data binarization completed.")
            
            apriori_start = time.time()
            apriori_rules = run_apriori(df_binarized)
            plot_rules(apriori_rules, 'Apriori')
            apriori_end = time.time()
            logging.info(f"Apriori algorithm completed in {apriori_end - apriori_start} seconds.")

            fp_growth_start = time.time()
            fp_growth_rules = run_fp_growth(df_binarized)
            plot_rules(fp_growth_rules, 'FP-Growth')
            fp_growth_end = time.time()
            logging.info(f"FP-Growth algorithm completed in {fp_growth_end - fp_growth_start} seconds.")

            end_time = time.time()
            logging.info(f"Total processing time: {end_time - start_time} seconds.")

            # return render_template('results.html', apriori_rules=apriori_rules, fp_growth_rules=fp_growth_rules)
            return render_template('results.html', 
                                   apriori_rules=apriori_rules, 
                                   fp_growth_rules=fp_growth_rules, 
                                   time_domain_features=time_domain_features,
                                   frequency_features=frequency_features,
                                   periodic_patterns=periodic_patterns
                                   )

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return f"Error processing file: {e}"

if __name__ == '__main__':
    app.run(debug=True)
