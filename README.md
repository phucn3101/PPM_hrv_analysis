# PPM HRV Analysis

## Project Description

The PPM HRV Analysis project is designed to analyze Heart Rate Variability (HRV) data. It includes preprocessing steps, feature extraction, and periodic pattern mining using the Apriori and FP-Growth algorithms. The project provides insights into HRV data through time domain, frequency domain, and non-linear feature analyses.

## Features

1. **Data Preprocessing**
   - Handles missing values.
   - Normalizes the data.

2. **Feature Extraction**
   - **Time Domain Features:** MEAN_RR, MEDIAN_RR, SDRR, RMSSD, SDSD, SDRR_RMSSD, pNN25, pNN50, KURT, SKEW, MEAN_REL_RR, MEDIAN_REL_RR, SDRR_REL_RR, RMSSD_REL_RR, SDSD_REL_RR, SDRR_RMSSD_REL_RR, KURT_REL_RR, SKEW_REL_RR.
   - **Frequency Domain Features:** VLF, VLF_PCT, LF, LF_PCT, LF_NU, HF, HF_PCT, HF_NU, TP, LF_HF, HF_LF.
   - **Non-Linear Features:** SD1, SD2, sampen, higuci.

3. **Periodic Pattern Mining**
   - Uses Apriori and FP-Growth algorithms to find frequent patterns in HRV data.

4. **Visualization**
   - Plots association rules for better understanding of periodic patterns.

## Getting Started

### Installation

1. **Extract the project:**

    - Extract the project to your desired location

2. **Navigate to the project directory:**

    - Navigate to .../PPM_hrv_analysis

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Run the Flask application:**

    ```sh
    python app.py
    ```

2. **Access the application:**
   - Open your web browser and go to `http://127.0.0.1:5000`.

3. **Upload your HRV dataset:**
   - Click on the "Choose File" button.
   - Select your HRV dataset (CSV format).
   - Click on the "Upload" button.

4. **View Results:**
   - The results will be displayed, including time domain features, frequency domain features, and periodic patterns.
   - Visualizations of the periodic patterns will be available on the results page.

## Project Structure

- `data/`: Contains sample HRV datasets.
  - `raw/`: Raw HRV datasets.
  - `processed/`: Preprocessed HRV datasets.
- `scripts/`: Contains the main scripts of the project.
  - `apriori_basic.py`: Contains the Apriori algorithm.
  - `fpgrowth_basic`: Contains the FP-Growth algorithm.
  - `frequency_domain.py`: Script for extracting frequency domain features.
  - `periodic_patterns`: Contains the Periodic Pattern Mining.
  - `preprocess.py`: Script for data preprocessing.
  - `time_domain.py`: Script for extracting time domain features.
  - `visualize.py`: Script for visualizing periodic patterns.
- `static/`: Contains static files like CSS and images.
- `templates/`: Contains HTML templates for rendering the web pages.
  - `index.html`: The main page for uploading files.
  - `results.html`: The results page displaying the analysis.
- `app.py`: The main Flask application file.
- `requirements.txt`: Contains the list of required Python packages.

## Dataset Source

Dataset was obtained from https://www.kaggle.com/datasets/saurav9786/heart-rate-prediction?resource=download&select=heart_rate_non_linear_features_test.csv
