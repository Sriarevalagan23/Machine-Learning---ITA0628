import numpy as np
from collections import Counter

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

# Sample dataset: binary classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Use only 2 classes to make it binary classification
X = X[y != 2]
y = y[y != 2]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize and train the KNN classifier
clf = KNN(k=3)
clf.fit(X_train, y_train)

# Predict and evaluate
predictions = clf.predict(X_test)
acc = clf.accuracy(y_test, predictions)

print("Predictions:", predictions)
print("True Labels:", y_test)
print("Accuracy:", acc)
