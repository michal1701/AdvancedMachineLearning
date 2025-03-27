import numpy as np
import pandas as pd

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
    y = np.random.binomial(1, p, size=n)

    # Create covariance matrix S
    S = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            S[i, j] = g ** abs(i - j)

    # Generate feature vectors X
    X = np.zeros((n, d))
    for i in range(n):
        if y[i] == 0:
            X[i, :] = np.random.multivariate_normal(mean=np.zeros(d), cov=S)
        else:
            mean = np.array([1 / (j + 1) for j in range(d)])
            X[i, :] = np.random.multivariate_normal(mean=mean, cov=S)

    return pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(d)]), pd.Series(y, name="Target")

if __name__ == "__main__":

    # set parameters
    p = 0.5 
    n = 100  
    d = 5    
    g = 0.8  

    # Generate synthetic data
    X, y = generate_synthetic_data(p, n, d, g)
    synthetic_data = pd.concat([X, y], axis=1)

    # Save to CSV
    synthetic_data.to_csv("synthetic_dataset.csv", index=False)

    print("Synthetic dataset generated and saved to 'synthetic_dataset.csv'.")