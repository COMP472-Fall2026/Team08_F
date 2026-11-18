
"""k-Means clustering algorithm."""

import numpy as np


class KMeans:
    def __init__(self, k=3, seed=42):
        self.k = k
        self.seed = seed

    def fit(self, X):
        np.random.seed(self.seed)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)].copy()

        while True:
            labels = self.predict(X)

            new_centroids = self.centroids.copy()

            for i in range(self.k):
                cluster = X[labels == i]

                if len(cluster) > 0:
                    new_centroids[i] = np.mean(cluster, axis=0)

            if np.allclose(self.centroids, new_centroids):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        self.labels = self.predict(X)
        return self

    def predict(self, X):
        distances = np.array([
            np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            for centroid in self.centroids
        ])

        return np.argmin(distances, axis=0)
