"""Rhys Jervis 08/16/21
Unit Tests for churn_library.py
"""

import os
import logging
from glob import glob
import pytest
import joblib
from churn_library import import_data, perform_eda,\
            encoder_helper, perform_feature_engineering, train_models

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name="dataframe")
def get_data():
    """
    read in data
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("pytest.fixture - dataframe creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("pytest.fixture - dataframe creation:FAILEd")
        raise err
    return dataframe


@pytest.fixture(name="dataframe_encoded")
def encoded_data(dataframe):
    """
    encoded data
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    try:
        dataframe_encoded = encoder_helper(dataframe, category_lst=cat_columns)
        logging.info("pytest.fixture - encoded creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("pytest.fixture - encoded creation:FAILEd")
        raise err
    return dataframe_encoded


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(dataframe):
    '''
    test perform eda function
    '''
    file_names = [
        "Heatmap_Corr.jpg",
        "Distplot_Total_Trans_Ct.jpg",
        "Marital_Status(normalize).jpg",
        "Histogram_of_Customer_Age.jpg",
        "Histogram_of_Churn.jpg"]
    root = "images/eda/"

    perform_eda(dataframe)

    for file in file_names:
        try:
            with open(root + file):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing iperform_eda: FAILED")
            raise err


def test_encoder_helper(dataframe_encoded):
    '''
    test encoder helper
    '''

    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns")
        raise err
        
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    
    
    try:
        for col in cat_columns:
            assert col + '_Churn' in dataframe_encoded
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have the encoded columns")
        raise err


def test_perform_feature_engineering(dataframe_encoded):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe_encoded)

    try:
        assert x_train.shape[0] > 0
        assert x_test.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering:\
                     The objects doesn't appear to have rows and columns")
        raise err

    try:
        assert x_train.shape[0] == len(y_train)
        assert x_test.shape[0] == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The objects have the wrong lengths")
        raise err


def test_train_models(dataframe_encoded):
    '''
    test train_models
    '''

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe_encoded)
        train_models(x_train, x_test, y_train, y_test)
        joblib.load('./models/rfc_model.pkl')
        joblib.load('./models/logistic_model.pkl')
        logging.info("Testing test_train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing test_train_models: FAILED")
        raise err


if __name__ == "__main__":
    pass
