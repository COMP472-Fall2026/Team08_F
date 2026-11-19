"""k-nearest neighbours classifier, written from scratch with NumPy (based on the COMP 472 lab)."""

import numpy as np


class KNNClassifier:
    """Predicts a sample's class from the classes of its k closest training samples.

    distance: "euclidean" (straight-line distance) or "cosine" (1 - cosine similarity,
        which only looks at the direction of the vectors, not their length).
    weighted: if True, closer neighbours get a bigger say in the vote (weight 1 / distance);
        otherwise every neighbour gets one vote.
    """

    def __init__(self, k=3, distance="euclidean", weighted=False):
        self.k = k
        self.distance = distance
        self.weighted = weighted

    def fit(self, X_train, y_train):
        """k-NN has no training step: it just stores the training data to compare against later."""
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)

    def _predict_one(self, x):
        """Predict the label of one sample in the same steps as the notes."""
        x = np.asarray(x, dtype=np.float32)

        # Step 1: distance from x to every training sample.
        if self.distance == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, which computes all distances at once.
            # np.maximum(..., 0) clears tiny negative values caused by rounding.
            squared = np.sum(self.X_train ** 2, axis=1) + np.sum(x ** 2) - 2 * (self.X_train @ x)
            distances = np.sqrt(np.maximum(squared, 0))
        elif self.distance == "cosine":
            cosine_sim = np.dot(self.X_train, x) / (
            np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x)
            )

            distances = 1 - cosine_sim  # similar direction -> distance close to 0

        # Step 2: the k nearest training samples. argpartition finds them without sorting everything.
        nearest = np.argpartition(distances, self.k - 1)[:self.k]
        labels = self.y_train[nearest]

        # Step 3: vote. bincount adds up one vote (or one weight) per neighbour for each class.
        if not self.weighted:
            votes = np.bincount(labels)
        else:
            neighbor_distances = distances[nearest]
            weights = 1 / (neighbor_distances + 1e-10)  # the small constant avoids dividing by zero
            votes = np.bincount(labels, weights=weights)

        # The class with the most votes wins (ties go to the smaller class number).
        return int(np.argmax(votes))

    def predict(self, X_test):
        """Predict one sample at a time, like the predict() method in the notes."""
        return np.array([self._predict_one(x) for x in X_test])

    def score(self, X_test, y_test):
        """Accuracy = fraction of predictions that match their true labels."""
        predictions = self.predict(X_test)
        return float(np.mean(predictions == y_test))
