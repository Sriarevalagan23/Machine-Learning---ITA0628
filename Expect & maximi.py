import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic dataset with 2 Gaussian blobs
X, y_true = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)

# Visualize the generated data
plt.scatter(X[:, 0], X[:, 1], c='gray', s=30, label='Unlabeled Data')
plt.title("Original Unlabeled Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# Apply Gaussian Mixture Model (EM algorithm)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Get means and covariances
print("Estimated Means:\n", gmm.means_)
print("\nEstimated Covariances:\n", gmm.covariances_)

# Visualize the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title("Clusters Formed using EM Algorithm")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
