# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
To produce modular, documented, and tested tested code for Customer Churn Prediction

## Project Description
To predict customer churn. The data can be found on [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers)

## Files in the Repo
The repo contains the following files:

- data
    - bank_data.csv
- images
    - eda
        - churn_distribution.png
        - customer_age_distribution.png
        - heatmap.png
        - marital_status_distribution.png
        - total_transaction_distribution.png
    - results
        - feature_importance.png
        - logistics_results.png
        - rf_results.png
        - roc_curve_result.png
- logs
    - churn_library.log
- models
    - logistic_model.pkl
    - rfc_model.pkl
- churn_library.py
- churn_notebook.ipynb
- churn_script_logging_and_tests.py

The churn_notebook.ipynb contains the base code and churn_library.py the modular code for production.
The unit tests are in churn_script_logging_and_tests.py.

Pickled models are stored in the models folder.

The logs are contained in logs/churn_library.log and graphs created during the EDA or depicting results are saved in images


## Running Files
The file can be run using:
`ipython churn_library.py`

## Unit Test
With pytest, unit tests and logs for churn_library.py can be obtained by:
`pytest churn_script_logging_and_tests.py`

## Pylint Tests
Pylint test scores can be obtained by running:
pylint churn_library.py
pylint churn_script_logging_and_tests.py

## Requirements
scikit-learn==0.22
shap==0.39.0
pylint==2.9.6
autopep8==1.5.7
seaborn==0.8.1
numpy==1.12.1
matplotlib==2.1.0
pandas==0.23.3