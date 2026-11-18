"""k-Nearest Neighbors classifier based on the COMP 472 Lab"""

import numpy as np


class KNNClassifier:
    def __init__(self, k=3, distance="euclidean", weighted=False):
        self.k = k
        self.distance = distance
        self.weighted = weighted

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)

    def _predict_one(self, x):
        """Predict the label of one sample in the same steps as the notes."""
        x = np.asarray(x, dtype=np.float32)

        if self.distance == "euclidean":
            squared = np.sum(self.X_train ** 2, axis=1) + np.sum(x ** 2) - 2 * (self.X_train @ x)
            distances = np.sqrt(np.maximum(squared, 0))
        elif self.distance == "cosine":
            cosine_sim = np.dot(self.X_train, x) / (
            np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x)
            )

            distances = 1 - cosine_sim

        nearest = np.argpartition(distances, self.k - 1)[:self.k]
        labels = self.y_train[nearest]

        if not self.weighted:
            votes = np.bincount(labels)
        else:
            neighbor_distances = distances[nearest]
            weights = 1 / (neighbor_distances + 1e-10)
            votes = np.bincount(labels, weights=weights)

        return int(np.argmax(votes))

    def predict(self, X_test):
        """Predict one sample at a time, like the predict() method in the notes."""
        return np.array([self._predict_one(x) for x in X_test])

    def score(self, X_test, y_test):
        """Accuracy = fraction of predictions that match their true labels."""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))
