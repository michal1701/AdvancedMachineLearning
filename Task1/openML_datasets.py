# This script evaluates OpenML datasets based on specific criteria - the number of variables should be at least 50% of the number of observations
# It also involses preprocessing - filling in missing values and removing collinear variables.
import numpy as np
import openml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

dataset_ids = [1137, 1158, 46611, 45088]

# Function to prepare and evaluate each dataset
def evaluate__preprocess_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    original_num_features = X.shape[1]

    # Fill missing values with the mean for numerical columns
    X = X.fillna(X.mean())
    
    # Encode target variable if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)
    
    # Remove collinear variables (correlation threshold: 0.9)
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    collinear_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
    X = X.drop(columns=collinear_features)
    
    # Evaluate dataset based on criteria
    num_observations = X.shape[0]
    num_features = X.shape[1]
    is_binary = pd.Series(y).nunique() == 2  # Fixed line
    are_features_numerical = X.select_dtypes(include='number').shape[1] == num_features
    feature_observation_ratio = num_features / num_observations
    meets_ratio_requirement = feature_observation_ratio >= 0.5
    
    # Print evaluation results
    print(f"Dataset ID: {dataset_id}")
    print(f"Name: {dataset.name}")
    print(f"Number of Observations: {num_observations}")
    print(f"Number of Features: {original_num_features}")
    print(f"Binary Classification: {'Yes' if is_binary else 'No'}")
    print(f"All Features Numerical: {'Yes' if are_features_numerical else 'No'}")
    print(f"Feature/Observation Ratio: {feature_observation_ratio:.2f} ({'Meets Requirement' if meets_ratio_requirement else 'Does Not Meet Requirement'})")
    print(f"Number of Features after preprocessing: {X.shape[1]}")
    print(f"Removed Collinear Features: {len(collinear_features)}")
    print("-" * 50)

# Evaluate each dataset
for dataset_id in dataset_ids:
    evaluate__preprocess_dataset(dataset_id)
