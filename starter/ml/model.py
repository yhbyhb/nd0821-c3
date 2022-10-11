from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_sliced_performace(df, model, encoder, lb, cat_features):
    """
    compute slicing data model performance

    Returns
    -------
    sliced_perf : List of dictionary includes performance metric and feature that using data slice

    """
    sliced_perf = []
    for feature in cat_features:
        for feature_value in df[feature].unique():
            # slicing dataframe with respect to feature and feature_value
            sliced_df = df[df[feature] == feature_value]
            X_slice, y_slice, _, _ = process_data(
                sliced_df,
                categorical_features=cat_features,
                label="salary", training=False,
                encoder=encoder, lb=lb)
            predictions_slice = inference(model, X_slice)
            precision, recall, f_beta = compute_model_metrics(y_slice, predictions_slice)
            slicing_perf_dict = {
                'feature' : feature,
                'feature_value' : feature_value,
                'precision' : precision,
                'recall' : recall,
                'f_beta' : f_beta,
            }
            sliced_perf.append(slicing_perf_dict)
    return sliced_perf
