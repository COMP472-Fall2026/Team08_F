
"""k-Means clustering algorithm."""

import numpy as np


class KMeans:
    def __init__(self, k=3, max_iterations=100, seed=42):
        self.k = k
        self.max_iterations = max_iterations
        self.seed = seed

    def fit(self, X):
        # Randomly select k samples as initial centroids.
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iterations):

            # Step 1: Assign each sample to its closest centroid.
            labels = self.predict(X)

            # Step 2: Calculate the new centroids.
            new_centroids = self.centroids.copy()

            for i in range(self.k):
                # Get all samples belonging to cluster i.
                cluster = X[labels == i]

                # Calculate the mean of the cluster.
                if len(cluster) > 0:
                    new_centroids[i] = np.mean(cluster, axis=0)

            # Step 3: Stop if the centroids have converged.
            if np.allclose(self.centroids, new_centroids):
                self.centroids = new_centroids
                break

            # Update the centroids for the next iteration.
            self.centroids = new_centroids

        self.labels = self.predict(X)
        return self

    def predict(self, X):
        # Calculate the Euclidean distance to each centroid.
        distances = np.array([
            np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            for centroid in self.centroids
        ])

        # Assign each sample to its closest centroid.
        return np.argmin(distances, axis=0)
