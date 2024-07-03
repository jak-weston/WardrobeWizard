#%%
import numpy as np
from scipy.io import loadmat

# Load the original and new .mat files
original_file_path = './data/encode_hn2_rnn_100_2_full_github.mat'
new_file_path = './data/encode_hn2_rnn_100_2_full_jupyterhub.mat'

original_data = loadmat(original_file_path)
new_data = loadmat(new_file_path)

# Check the keys to understand the structure of the loaded data
print("Original Data Keys:", original_data.keys())
print("New Data Keys:", new_data.keys())

# Extract the relevant data arrays
original_hn2 = original_data['hn2']
new_hn2 = new_data['hn2']

# Check the shapes to ensure they are comparable
print("Original hn2 Shape:", original_hn2.shape)
print("New hn2 Shape:", new_hn2.shape)

#%%
from sklearn.metrics import mean_squared_error

# Compute the mean squared error
mse = mean_squared_error(original_hn2, new_hn2)

print(f"Mean Squared Error between the original and new implementation: {mse}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Select a subset of the data for visualization
subset_size = 1000
original_subset = original_hn2[:subset_size].flatten()
new_subset = new_hn2[:subset_size].flatten()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(original_subset, new_subset, alpha=0.5, marker='o', color='b')
plt.xlabel('Original hn2 values')
plt.ylabel('New hn2 values')
plt.title('Scatter Plot of Original vs. New hn2 values')
plt.grid(True)
plt.show()

# Histogram
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(original_subset, bins=50, alpha=0.7, label='Original')
plt.hist(new_subset, bins=50, alpha=0.7, label='New')
plt.xlabel('hn2 values')
plt.ylabel('Frequency')
plt.title('Histogram of hn2 values')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(original_subset, bins=50, color='blue', label='Original', kde=True)
sns.histplot(new_subset, bins=50, color='red', label='New', kde=True)
plt.xlabel('hn2 values')
plt.ylabel('Density')
plt.title('Density Plot of hn2 values')
plt.legend()

plt.tight_layout()
plt.show()

# %%
