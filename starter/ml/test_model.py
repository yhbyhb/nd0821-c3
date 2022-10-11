import os
import logging
import numpy as np
import pandas as pd
import pathlib
import pytest

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@pytest.fixture
def data():
    """
    """
    cur_path = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(os.path.join(cur_path, "../../data/census_cleaned.csv"))

    return df


def test_train_model(data):
    """
    """
    np.random.seed(23)
    
    train, test = train_test_split(data, test_size=0.20, random_state=23)

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

    model = train_model(X_train, y_train)

    try:
        assert hasattr(model, 'fit')
        logging.info("Testing training_model() : succeeded")
    except AssertionError as err:
        logging.error(
            "Testing train_models() : failed - model doesn't have fit method")
        raise err


def test_inference(data):
    """
    """
    # Proces the test data with the process_data function.
    np.random.seed(23)
    
    train, test = train_test_split(data, test_size=0.20, random_state=23)

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
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)

    # Inference
    preds = inference(model, X_test)

    try:
        assert y_test.size == preds.size
        logging.info("Testing test_inference() : succeeded")
    except AssertionError as err:
        logging.error(
            "Testing test_inference() : failed")
        raise err


def test_compute_model_metrics(data):
    """
    """
    # Proces the test data with the process_data function.
    np.random.seed(23)
    
    train, test = train_test_split(data, test_size=0.20, random_state=23)

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
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)

    # Inference
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    #precisio : 0.7600896860986547, recall : 0.6223990208078335, fbeta : 0.6843876177658142
    try:
        assert 0.74 < precision < 0.78
        assert 0.61 < recall < 0.64  
        assert 0.67 < fbeta  < 0.69

        logging.info("Testing compute_model_metrics() : succeeded")
    except AssertionError as err:
        logging.error(
            "Testing compute_model_metrics() : failed")
        raise err