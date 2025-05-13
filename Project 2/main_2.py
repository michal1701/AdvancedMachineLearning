import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def strategy_pca_logreg(x_train, y_train, n_components):
    x_train_scaled, _ = preprocess_data(x_train, x_train)
    x_train_sub, x_val, y_train_sub, y_val = train_test_split(
        x_train_scaled, y_train, test_size=0.2, random_state=42)

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_sub)
    x_val_pca = pca.transform(x_val)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_pca, y_train_sub)

    y_val_prob = clf.predict_proba(x_val_pca)[:, 1]
    k = min(1000, len(y_val))
    top_k_idx = np.argsort(y_val_prob)[-k:]
    true_positives = y_val[top_k_idx].sum()

    reward = true_positives * 10
    cost = n_components * 200
    score = reward - cost

    return {
        'strategy': 'PCA + Logistic Regression',
        'features_used': n_components,
        'true_positives': int(true_positives),
        'reward': reward,
        'cost': cost,
        'score': score
    }

def strategy_rf_importance(x_train, y_train, n_features):
    x_train_sub, x_val, y_train_sub, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train_sub, y_train_sub)

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:n_features]

    x_train_sel = x_train_sub[:, top_indices]
    x_val_sel = x_val[:, top_indices]

    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sel.fit(x_train_sel, y_train_sub)

    y_val_prob = rf_sel.predict_proba(x_val_sel)[:, 1]
    k = min(1000, len(y_val))
    top_k_idx = np.argsort(y_val_prob)[-k:]
    true_positives = y_val[top_k_idx].sum()

    reward = true_positives * 10
    cost = n_features * 200
    score = reward - cost

    return {
        'strategy': 'RF + Feature Importance',
        'features_used': n_features,
        'true_positives': int(true_positives),
        'reward': reward,
        'cost': cost,
        'score': score
    }


# Load data
x_train = np.loadtxt("Project 2/x_train.txt")
y_train = np.loadtxt("Project 2/y_train.txt")
x_test = np.loadtxt("Project 2/x_test.txt")  # used only for scaling consistency

# List of feature/component counts to evaluate
feature_counts = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60]

unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
from sklearn.metrics import roc_auc_score



# Store results
results = []
for n in feature_counts:
    results.append(strategy_pca_logreg(x_train, y_train, n_components=n))
    results.append(strategy_rf_importance(x_train, y_train, n_features=n))

    

# Save results
results_df = pd.DataFrame(results)
print(results_df.head())
results_df.to_csv("strategy_comparison.csv", index=False)


sns.lineplot(data=results_df, x="features_used", y="score", hue="strategy", marker='o')
plt.title("Validation Score vs Number of Features")
plt.grid(True)
plt.show()
