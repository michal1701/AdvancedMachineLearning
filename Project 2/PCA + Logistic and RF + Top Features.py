import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_full = pd.read_csv('x_train.txt', delim_whitespace=True, header=None).values
y_train_full = pd.read_csv('y_train.txt', delim_whitespace=True, header=None).values.ravel()
X_test = pd.read_csv('x_test.txt', delim_whitespace=True, header=None).values

y_train_full = np.where(y_train_full == 1, 1, 0) 


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.5, random_state=47, stratify=y_train_full
)

# Feature scaling for PCA+Logistic pipeline (we will keep an unscaled copy for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Determine feature importance ranking using a Random Forest on the full training set
full_rf = RandomForestClassifier(n_estimators=100, random_state=47)
full_rf.fit(X_train, y_train)
importances = full_rf.feature_importances_
feat_indices_sorted = np.argsort(importances)[::-1]  

# Define range of features to test
feature_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  25, 30, 35]
max_feat = X_train.shape[1]
feature_counts = [k for k in feature_counts if k <= max_feat]

# Lists to collect metrics for each approach
pca_acc = []      
pca_reward = []   
pca_cost = []     
pca_score = []    

rf_acc = []
rf_reward = []
rf_cost = []
rf_score = []

# Loop over each feature count
for k in feature_counts:
    # --- Strategy 1: PCA + Logistic Regression ---
    # Reduce to k principal components
    pca = PCA(n_components=k, random_state=47)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    
    # Train logistic regression on k components
    logreg = LogisticRegression(random_state=47, max_iter=500)
    logreg.fit(X_train_pca, y_train)
    
    # Validation predictions and accuracy
    val_preds = logreg.predict(X_val_pca)
    val_proba = logreg.predict_proba(X_val_pca)[:, 1]  
    accuracy = np.mean(val_preds == y_val)
    pca_acc.append(accuracy)
    
    # Determine reward on validation: pick top 1000 by probability
    top_N = min(1000, X_val.shape[0])
    # indices of top_N highest predicted probabilities
    top_idx = np.argsort(val_proba)[::-1][:top_N]
    # count how many of these are actually high usage (y_val == 1)
    true_positives = np.sum(y_val[top_idx] == 1)
    reward = 10 * true_positives
    cost = 200 * k
    score = reward - cost
    pca_reward.append(reward)
    pca_cost.append(cost)
    pca_score.append(score)
    
    test_proba = logreg.predict_proba(X_test_pca)[:, 1]
    
    # --- Strategy 2: Random Forest (top-k features) ---
    # Select top k important feature indices
    topk_idx = feat_indices_sorted[:k]
    X_train_topk = X_train[:, topk_idx]
    X_val_topk   = X_val[:, topk_idx]
    X_test_topk  = X_test[:, topk_idx]
    
    # Train RandomForest on these k features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_topk, y_train)
    
    # Validation predictions and accuracy
    val_preds_rf = rf.predict(X_val_topk)
    val_proba_rf = rf.predict_proba(X_val_topk)[:, 1] 
    accuracy_rf = np.mean(val_preds_rf == y_val)
    rf_acc.append(accuracy_rf)
    
    # Reward on validation for Random Forest approach
    top_idx_rf = np.argsort(val_proba_rf)[::-1][:top_N]
    true_positives_rf = np.sum(y_val[top_idx_rf] == 1)
    reward_rf = 10 * true_positives_rf
    cost_rf = 200 * k
    score_rf = reward_rf - cost_rf
    rf_reward.append(reward_rf)
    rf_cost.append(cost_rf)
    rf_score.append(score_rf)
    
    test_proba_rf = rf.predict_proba(X_test_topk)[:, 1]
    
    

best_pca_idx = int(np.argmax(pca_score))
best_rf_idx  = int(np.argmax(rf_score))

print(f"Best PCA+LogReg score: {pca_score[best_pca_idx]:.1f} EUR at k={feature_counts[best_pca_idx]} features")
print(f"Best RandomForest score: {rf_score[best_rf_idx]:.1f} EUR at k={feature_counts[best_rf_idx]} features")

# Choose the best model (based on final score)
if pca_score[best_pca_idx] > rf_score[best_rf_idx]:
    print("Using PCA + Logistic Regression as the final model.")
    best_k = feature_counts[best_pca_idx]
    # Use PCA + Logistic Regression
    pca = PCA(n_components=best_k, random_state=47)
    X_test_pca = pca.fit_transform(X_test_scaled)
    logreg = LogisticRegression(random_state=47, max_iter=500)
    logreg.fit(pca.fit_transform(X_train_scaled), y_train)
    test_proba = logreg.predict_proba(X_test_pca)[:, 1]
    selected_features = list(range(best_k))  
else:
    print("Using Random Forest as the final model.")
    best_k = feature_counts[best_rf_idx]
    # Use Random Forest
    topk_idx = feat_indices_sorted[:best_k]
    X_test_topk = X_test[:, topk_idx]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train[:, topk_idx], y_train)
    test_proba = rf.predict_proba(X_test_topk)[:, 1]
    selected_features = topk_idx.tolist()  

# Select top 1000 customers based on predicted probabilities
top_1000_idx = np.argsort(test_proba)[::-1][:1000]

# Save the results to files
np.savetxt("320618_obs.txt", top_1000_idx, fmt='%d')  # Indices of customers
np.savetxt("320618_vars.txt", selected_features, fmt='%d')  # Indices of variables

print("Results saved to 320618_obs.txt and 320618_vars.txt.")

import matplotlib.pyplot as plt

# Plot metrics vs number of features for both strategies
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle("Model Performance vs Number of Features", fontsize=16)

# Accuracy plot
axs[0, 0].plot(feature_counts, pca_acc, marker='o', label='PCA + Logistic')
axs[0, 0].plot(feature_counts, rf_acc, marker='s', label='RF (Top-k Features)')
axs[0, 0].set_xlabel("Number of Features")
axs[0, 0].set_ylabel("Validation Accuracy")
axs[0, 0].set_title("Accuracy on Validation Set")
axs[0, 0].legend(loc='lower right')
axs[0, 0].grid(True)

# Reward plot
axs[0, 1].plot(feature_counts, pca_reward, marker='o', label='PCA + Logistic')
axs[0, 1].plot(feature_counts, rf_reward, marker='s', label='RF (Top-k Features)')
axs[0, 1].set_xlabel("Number of Features")
axs[0, 1].set_ylabel("Reward (€)")
axs[0, 1].set_title("Reward (€/correct high usage in top 1000)")
axs[0, 1].legend(loc='best')
axs[0, 1].grid(True)

# Cost plot (same for both strategies, proportional to k)
axs[1, 0].plot(feature_counts, pca_cost, marker='o', label='PCA + Logistic')
axs[1, 0].plot(feature_counts, rf_cost, marker='s', label='RF (Top-k Features)', linestyle='--')
axs[1, 0].set_xlabel("Number of Features")
axs[1, 0].set_ylabel("Cost (€)")
axs[1, 0].set_title("Cost of Features")
axs[1, 0].legend(loc='best')
axs[1, 0].grid(True)

# Final Score plot
axs[1, 1].plot(feature_counts, pca_score, marker='o', label='PCA + Logistic')
axs[1, 1].plot(feature_counts, rf_score, marker='s', label='RF (Top-k Features)')
axs[1, 1].set_xlabel("Number of Features")
axs[1, 1].set_ylabel("Final Score (Reward - Cost, €)")
axs[1, 1].set_title("Final Profit Score")
axs[1, 1].legend(loc='best')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
