
from models.kmeans import KMeans
from utils.data_loader import load_dataset
import numpy as np


# Experiment settings.
K_VALUES = [3, 6, 9]
SEED = 42
MAX_ITERATIONS = 100

# Load the sentiment dataset.
data = load_dataset("sentiment", "train")

# Run an experiment for each value of k.
for k in K_VALUES:

    print("\n========================================")
    print(f"K-MEANS: k = {k}")
    print("========================================")

    # Create and fit the model without using actual labels.
    model = KMeans(k=k, max_iterations=MAX_ITERATIONS, seed=SEED)
    model.fit(data.X)

    # Get the assigned clusters.
    labels = model.labels

    correct = 0

    # Analyze each cluster.
    for i in range(k):

        # Get the actual sentiment labels of samples in this cluster.
        actual_labels = data.y[labels == i]

        negative = np.sum(actual_labels == 0)
        neutral = np.sum(actual_labels == 1)
        positive = np.sum(actual_labels == 2)

        print(f"\nCluster {i}: {len(actual_labels)} samples")
        print("Negative:", negative)
        print("Neutral: ", neutral)
        print("Positive:", positive)

        # Count samples belonging to the dominant class.
        correct += max(negative, neutral, positive)

    # Calculate overall cluster purity.
    purity = correct / len(data.X)

    print(f"\nCluster purity: {purity:.2%}")
