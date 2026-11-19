"""k-Means clustering on the sentiment training data, compared with the true classes.

python experiments/kmeans_experiment.py

Clustering uses the features only; the true labels are used afterwards to measure how
well the clusters match the sentiment classes. Results are saved to results/kmeans/:
experiments.csv (purity and WCSS for each k) and cluster_composition.csv (class counts
in each cluster).
"""

import sys
from pathlib import Path
from time import perf_counter

# Let the script run directly (python experiments/...) as well as with python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.kmeans import KMeans
from utils.data_loader import load_dataset
from utils.results import append_csv, clear_results, save_run

ALGORITHM = "kmeans"
DATASET = "sentiment"
K_VALUES = list(range(1, 26, 2))  # k = 1, 3, 5, ..., 25
SEED = 42
CLASS_NAMES = ["negative", "neutral", "positive"]


def main():
    """Cluster the sentiment data for each k and save purity, WCSS and cluster composition."""
    folder = clear_results(ALGORITHM)
    # Clustering is unsupervised, so it uses all labeled rows (no train/validation split).
    data = load_dataset(DATASET)
    X, y = data.X, data.y

    for k in K_VALUES:
        start = perf_counter()
        model = KMeans(k=k, seed=SEED).fit(X)  # features only; labels are not used here
        labels = model.labels

        dominant, wcss, rows = 0, 0.0, []
        for cluster in range(k):
            members = labels == cluster
            # True labels are used only after clustering, to describe each cluster.
            counts = np.bincount(y[members], minlength=len(CLASS_NAMES))
            dominant += int(counts.max())
            # WCSS: sum of squared distances from each point to its cluster's centroid.
            wcss += float(np.sum((X[members] - model.centroids[cluster]) ** 2, dtype=np.float64))
            rows.append({"k": k, "cluster": cluster, "size": int(members.sum()),
                         **dict(zip(CLASS_NAMES, counts.tolist()))})

        # Purity: share of points that belong to their cluster's most common class.
        purity = dominant / len(X)
        save_run(ALGORITHM, DATASET, sweep="k", experiment=f"k={k}", settings={"k": k, "seed": SEED},
                 metrics={"purity": purity, "wcss": wcss, "seconds": round(perf_counter() - start, 2)})
        append_csv(pd.DataFrame(rows), folder / "cluster_composition.csv")
        print(f"k={k}: purity {purity:.2%}, WCSS {wcss:,.0f}", flush=True)


if __name__ == "__main__":
    main()
