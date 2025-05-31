import os
import pandas as pd
import numpy as np

def load_data(data_path: str, as_numpy=False):
    """
    Load the dataset from the specified path.

    The dataset consists of training and test data for a classification task.

    Parameters:
    data_path (str): Path to the directory containing the dataset files.
    as_numpy (bool): If True, return the data as numpy arrays. If False, return as pandas DataFrames.
    Returns:
    A tuple containing:
        - X_train: Training data features.
        - y_train: Training data labels.
        - X_test: Test data features.
    """
    # Load the data
    X_train = pd.read_table(os.path.join(data_path, 'x_train.txt'), header=None, sep='\s+')
    X_test = pd.read_table(os.path.join(data_path, 'x_test.txt'), header=None, sep='\s+')
    y_train = pd.read_table(os.path.join(data_path, 'y_train.txt'), header=None, sep='\s+')

    # Convert to numpy arrays
    if as_numpy:
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values.ravel()

    return X_train, y_train, X_test

def cost_function(y_true: np.ndarray, y_proba: np.ndarray, n_features: int) -> float:
    """
    Calculate the cost function based on the true labels and predicted probabilities.
    The cost function is defined as:
    profit = income - cost
    where:
    - income is the number of correctly classified samples multiplied by 10
    - cost is the number of features multiplied by 200

    Only 1000 samples with the highest predicted probabilities are considered for the profit calculation.

    Parameters:
    y_true (np.ndarray): True labels of the samples.
    y_proba (np.ndarray): Predicted probabilities of the samples.
    Returns:
    float: The calculated profit.
    """

    indices = np.argsort(y_proba)[-1000:]
    y_true = y_true[indices]

    income = np.sum(y_true) * 10
    cost = 200 * n_features
    profit = income - cost

    return profit