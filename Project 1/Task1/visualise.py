import pandas as pd
import matplotlib.pyplot as plt

def plot_synthetic_data(file_path):
    # Load the dataset
    synthetic_data = pd.read_csv(file_path)

    X = synthetic_data.iloc[:, :-1]  
    y = synthetic_data.iloc[:, -1] 

    # Plot the first two features with manual legend
    plt.figure(figsize=(8, 6))
    plt.scatter(X["Feature_1"][y == 0], X["Feature_2"][y == 0], color='blue', label='Class 0', edgecolor="k", alpha=0.7)
    plt.scatter(X["Feature_1"][y == 1], X["Feature_2"][y == 1], color='red', label='Class 1', edgecolor="k", alpha=0.7)
    plt.title("Scatter Plot of Synthetic Dataset")
    plt.xlabel("Feature_1")
    plt.ylabel("Feature_2")
    
    # Add manual legend
    plt.legend(title="Class (Target)", fontsize=10, title_fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the synthetic dataset CSV file
    file_path = "Task1/synthetic_dataset.csv"

    # Plot the synthetic dataset
    plot_synthetic_data(file_path)
