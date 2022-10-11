'''
# test_model.py
unit tests for model.py

'''

import os
import logging
import pathlib
import pytest

import pandas as pd

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@pytest.fixture
def data():
    """
    fixture dataFrame from a csv file
    """
    cur_path = pathlib.Path(__file__).parent.resolve()
    dataframe = pd.read_csv(os.path.join(cur_path, "../../data/census_cleaned.csv"))

    return dataframe

@pytest.fixture
def train_test(data):
    """
    fixture for train and test data set from dataFrame
    """
    train, test = train_test_split(data, test_size=0.20, random_state=23)

    return train, test

@pytest.fixture
def train_test_data(train_test):
    '''
    fixture for segregated dataset.
    '''

    train, test = train_test

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
    )

    return X_train, y_train, X_test, y_test

@pytest.fixture
def model(train_test_data):
    '''
    fixture for a trained model
    '''

    X_train, y_train, _, _ = train_test_data
    model = train_model(X_train, y_train)
    return model

def test_train_model(model):
    """
    test for train_model()
    """
    try:
        assert hasattr(model, 'fit')
        logging.info("Testing training_model() : succeeded")
    except AssertionError as err:
        logging.error(
            "Testing train_models() : failed - model doesn't have fit method")
        raise err

@pytest.fixture
def preds(train_test_data, model):
    '''
    fixture for inference
    '''
    _, _, X_test, _ = train_test_data

    # Inference
    preds = inference(model, X_test)

    return preds

def test_inference(train_test_data, preds):
    '''
    test for inference()
    '''
    _, _, _, y_test = train_test_data

    try:
        assert y_test.size == preds.size
        logging.info("Testing test_inference() : succeeded")
    except AssertionError as err:
        logging.error(
            "Testing test_inference() : failed")
        raise err


def test_compute_model_metrics(train_test_data, preds):
    '''
    test for compute_model_metrics()
    '''
    _, _, _, y_test = train_test_data

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    #precisio : 0.7600896860986547, recall : 0.6223990208078335, fbeta : 0.6843876177658142
    try:
        assert 0.74 < precision < 0.78
        assert 0.61 < recall < 0.64
        assert 0.67 < fbeta  < 0.70

        logging.info("Testing compute_model_metrics() : succeeded")
    except AssertionError as err:
        logging.error(
            "Testing compute_model_metrics() : failed")
        raise err
