"""k-Nearest Neighbors classifier based on the COMP 472 lecture notes."""

import numpy as np


class KNNClassifier:
    def __init__(self, k=3, distance="euclidean", weighted=False):
        # k: number of neighbors; weighted: closer neighbors have stronger votes.
        self.k = k
        self.distance = distance
        self.weighted = weighted

    def fit(self, X_train, y_train):
        """kNN is a lazy learner: it simply stores the training data."""
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)

        # For Euclidean distance, save sum(x_i**2) for each training sample.
        # This speeds up distance calculations at prediction time.
        if self.distance == "euclidean":
            self.train_squared = np.sum(self.X_train ** 2, axis=1)

        elif self.distance == "cosine":
            self.train_norms = np.linalg.norm(self.X_train, axis=1)

    def _predict_one(self, x):
        """Predict the label of one sample in the same steps as the notes."""
        x = np.asarray(x, dtype=np.float32)

        # Step 1: distance from x to EVERY training sample.
        if self.distance == "euclidean":
            # sqrt(sum((a-b)**2)) rewritten using dot products for speed:
            # sqrt(sum(a**2) + sum(b**2) - 2*a.dot(b)).
            squared = self.train_squared + np.sum(x ** 2) - 2 * (self.X_train @ x)
            distances = np.sqrt(np.maximum(squared, 0))
        elif self.distance == "cosine":
            # Calculate cosine similarity.
            cosine_sim = np.dot(self.X_train, x) / (
            self.train_norms * np.linalg.norm(x)
            )

            # Convert cosine similarity into cosine distance.
            distances = 1 - cosine_sim

        # Step 2: take the k nearest training samples.
        # argpartition finds the k smallest distances without sorting everything.
        nearest = np.argpartition(distances, self.k - 1)[:self.k]
        labels = self.y_train[nearest]

        # Step 3: vote on the class label.
        if not self.weighted:
            # Each neighbor gets one vote (the majority vote in the notes).
            votes = np.bincount(labels)
        else:
            # Closer neighbors have more influence: weight = 1 / distance.
            neighbor_distances = distances[nearest]
            weights = 1 / neighbor_distances
            votes = np.bincount(labels, weights=weights)

        # The class with the most votes wins.
        return int(np.argmax(votes))

    def predict(self, X_test):
        """Predict one sample at a time, like the predict() method in the notes."""
        return np.array([self._predict_one(x) for x in X_test])

    def score(self, X_test, y_test):
        """Accuracy = fraction of predictions that match their true labels."""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))
