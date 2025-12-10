import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)

# Helper constants
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Helper function
def train_test_subset():
    """
    # Loads a small subset of the census data for testing
    """
    data = pd.read_csv(DATA_PATH)

    # Get Subset
    data_sample = data.sample(n=1000, random_state=0)

    train, test = train_test_split(
        data_sample,
        test_size=0.2,
        random_state=42,
        stratify=data_sample["salary"],
    )

    # Run train/test splits
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, X_test, y_train, y_test


# implement the first test. Change the function name and input as needed
def test_return_random_forest():
    """
    # Checks that the train_model returns a RandomForestclassifier instance
    """
    X_train, X_test, y_train, y_test = train_test_subset()
    model = train_model(X_train, y_train)

    assert isinstance(
        model,
        RandomForestClassifier,
    ), "train_model should return a RandomForestClassifier."


# implement the second test. Change the function name and input as needed
def test_inference_output_shape():
    """
    # Test to see if Inference returns one prediction per input row
    """
    X_train, X_test, y_train, y_test = train_test_subset()
    model = train_model(X_train, y_train)

    n_samples = X_test.shape[0]
    preds = inference(model, X_test)

    assert (
        preds.shape[0] == n_samples
    ), "Number of predictions must match number of input samples."


# implement the third test. Change the function name and input as needed
def test_models_in_valid_range():
    """
    # compute_model_metrics should return precision, recall, and fbeta
    values between 0 and 1.
    """
    X_train, X_test, y_train, y_test = train_test_subset()
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    for metric, name in [
        (precision, "precision"),
        (recall, "recall"),
        (fbeta, "fbeta"),
    ]:
        assert 0.0 <= metric <= 1.0, f"{name} should be between 0 and 1."
