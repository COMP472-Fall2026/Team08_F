
from models.kmeans import KMeans
from utils.data_loader import load_dataset
import numpy as np


K_VALUES = [3, 6, 9]
SEED = 42

data = load_dataset("sentiment", "train")

for k in K_VALUES:

    print(f"K-MEANS CLUSTERING: k = {k}")

    model = KMeans(k=k, seed=SEED)
    model.fit(data.X)

    labels = model.labels

    correct = 0

    for i in range(k):

        actual_labels = data.y[labels == i]

        negative = np.sum(actual_labels == 0)
        neutral = np.sum(actual_labels == 1)
        positive = np.sum(actual_labels == 2)

        print(f"\nCluster {i}: {len(actual_labels)} samples")
        print("Negative:", negative)
        print("Neutral: ", neutral)
        print("Positive:", positive)

        correct += max(negative, neutral, positive)

    purity = correct / len(data.X)

    print(f"\nCluster purity: {purity:.2%}")
