from flask import Flask, request, jsonify, render_template
import pandas as pd
import logging
from scripts.preprocess import preprocess_data, load_data
from scripts.apriori_basic import run_apriori
from scripts.fpgrowth_basic import run_fp_growth
import time
import json

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

            df_binarized = (df_processed.drop(columns=['uuid', 'condition', 'datasetId']) > df_processed.drop(columns=['uuid', 'condition', 'datasetId']).mean()).astype(int)
            logging.info("Data binarization completed.")

            apriori_start = time.time()
            apriori_rules = run_apriori(df_binarized)
            apriori_end = time.time()
            logging.info(f"Apriori algorithm completed in {apriori_end - apriori_start} seconds.")

            fp_growth_start = time.time()
            fp_growth_rules = run_fp_growth(df_binarized)
            fp_growth_end = time.time()
            logging.info(f"FP-Growth algorithm completed in {fp_growth_end - fp_growth_start} seconds.")

            end_time = time.time()
            logging.info(f"Total processing time: {end_time - start_time} seconds.")

            return render_template('results.html', apriori_rules=apriori_rules, fp_growth_rules=fp_growth_rules)

            # Print results
            print("Apriori Rules:")
            print(json.dumps(apriori_rules, indent=4))
            
            print("FP-Growth Rules:")
            print(json.dumps(fp_growth_rules, indent=4))

            # Save results to a file
            with open('apriori_rules.json', 'w') as f:
                json.dump(apriori_rules, f, indent=4)
            with open('fp_growth_rules.json', 'w') as f:
                json.dump(fp_growth_rules, f, indent=4)

            return "File processed and results saved."

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return f"Error processing file: {e}"

if __name__ == '__main__':
    app.run(debug=True)
