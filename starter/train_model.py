# Script to train machine learning model.
import csv
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import yaml
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_sliced_performace


cur_path = pathlib.Path(__file__).parent.resolve()

data = pd.read_csv(os.path.join(cur_path, "../data/census_cleaned.csv"))

np.random.seed(23)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save a model.
model = train_model(X_train, y_train)

model_path = os.path.join(cur_path, "../model");

with open(os.path.join(model_path, "model.pkl"), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(model_path, "model_encoder.pkl"), 'wb') as f:
    pickle.dump(encoder, f)

with open(os.path.join(model_path, "model_lb.pkl"), 'wb') as f:
    pickle.dump(lb, f)        

# Inference and compute metrics
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f'precisio : {precision}, recall : {recall}, fbeta : {fbeta}')

# categorical data slicing
sliced_perf = compute_sliced_performace(data, model, encoder, lb, cat_features)

keys = sliced_perf[0].keys()

with open(os.path.join(cur_path, 'sliced_output.txt'), 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(sliced_perf)