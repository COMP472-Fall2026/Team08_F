"""k-means clustering, written from scratch with NumPy."""

import numpy as np


class KMeans:
    """Groups samples into k clusters, each represented by the mean of its samples (its centroid).

    Clustering only looks at the features: the true labels are never used here.
    seed fixes the random choice of starting centroids, so runs can be repeated.
    """

    def __init__(self, k=3, seed=42):
        self.k = k
        self.seed = seed

    def fit(self, X):
        """Find the k centroids, then store each sample's cluster in self.labels."""
        # Start from k different samples picked at random as the first centroids.
        np.random.seed(self.seed)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)].copy()

        while True:
            # Assignment step: put every sample in the cluster of its nearest centroid.
            labels = self.predict(X)

            # Update step: move each centroid to the mean of the samples now in its cluster.
            new_centroids = self.centroids.copy()

            for i in range(self.k):
                cluster = X[labels == i]

                if len(cluster) > 0:  # an empty cluster keeps its old centroid
                    new_centroids[i] = np.mean(cluster, axis=0)

            # Stop once the centroids no longer move (within a small tolerance).
            if np.allclose(self.centroids, new_centroids):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        self.labels = self.predict(X)
        return self

    def predict(self, X):
        """Return the index of the nearest centroid for each sample (Euclidean distance)."""
        # One row of distances per centroid; argmin down each column picks the closest one.
        distances = np.array([
            np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            for centroid in self.centroids
        ])

        return np.argmin(distances, axis=0)
