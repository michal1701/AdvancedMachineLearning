import os
import numpy as np
import pandas as pd

DATA_PATH = "../data"

def generate_synthetic_data(p, n, d, g):
    """
    Generate a synthetic dataset based on the given parameters.

    Parameters:
    - p: Class prior probability for Y=1 (Bernoulli distribution).
    - n: Number of observations.
    - d: Dimensionality of the feature vector.
    - g: Parameter controlling the covariance matrix.

    Returns:
    - X: Feature matrix (n x d).
    - y: Binary class labels (n,).
    """
    # Generate binary class labels Y from Bernoulli distribution
    ones_count = np.random.binomial(n, p)
    zeros_count = n - ones_count

    # Create covariance matrix S
    s = np.arange(0, d)
    ones_vector = np.ones((d,))
    S = np.zeros((d, d))
    for i in range(d):
        S[i, :] = g ** np.abs(s)
        s = s - ones_vector

    # Generate feature vectors X
    mean = np.array([1 / (j + 1) for j in range(d)])
    X_0 = np.random.multivariate_normal(mean=np.zeros(d), cov=S, size=zeros_count)
    X_1 = np.random.multivariate_normal(mean=mean, cov=S, size=ones_count)

    X = np.concat([X_0, X_1], axis=0)
    y = np.concat([np.zeros((zeros_count,)), np.ones((ones_count,))])

    # Shuffle randomly X and y
    idx = np.arange(n)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(d)]), pd.Series(y, name="Target")

if __name__ == "__main__":

    # set parameters
    p = 0.5
    n = 1000
    d = 10_000
    g = 0.8

    # Generate synthetic data
    X, y = generate_synthetic_data(p, n, d, g)
    synthetic_data = pd.concat([X, y], axis=1)

    # Save to CSV
    os.makedirs(DATA_PATH, exist_ok=True)
    # synthetic_data.to_csv(os.path.join(DATA_PATH, "synthetic_dataset.csv"), index=False)

    print("Synthetic dataset generated and saved to 'synthetic_dataset.csv'.")