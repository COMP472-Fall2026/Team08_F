"""Analyze k-Means clustering on the sentiment training dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save graphs without opening GUI windows.
import matplotlib.pyplot as plt

# Locate the repository root, even when running the file directly in VS Code.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.kmeans import KMeans
from utils.data_loader import load_dataset

# Use the same settings as kmeans_experiment.py.
K_VALUES = [3, 6, 9]
SEED = 42
OUTPUT = ROOT / "results" / "kmeans_analysis"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset("sentiment", "train", data_dir=ROOT / "data")
    X = data.X
    y = data.y
    summary = []
    all_clusters = []

    for k in K_VALUES:
        print(f"Running k-Means with k = {k}...", flush=True)

        # Fit using features only. Actual sentiment labels are not used here.
        model = KMeans(k=k, seed=SEED)
        model.fit(X)
        labels = model.labels

        dominant_count = 0
        wcss = 0.0
        cluster_rows = []

        for i in range(k):
            # Find the samples assigned to cluster i.
            mask = labels == i
            cluster = X[mask]

            # Use the true labels only AFTER clustering, to evaluate composition.
            counts = np.bincount(y[mask], minlength=3)
            size = len(cluster)
            dominant_count += int(counts.max())

            # WCSS: sum of squared distances to each sample's assigned centroid.
            wcss += float(np.sum((cluster - model.centroids[i]) ** 2,
                                 dtype=np.float64))

            # An empty cluster has no dominant class or defined purity.
            row = {
                "k": k,
                "cluster": i,
                "size": size,
                "negative": int(counts[0]),
                "neutral": int(counts[1]),
                "positive": int(counts[2]),
                "dominant_class": ["Negative", "Neutral", "Positive"][counts.argmax()] if size else "",
                "purity": counts.max() / size if size else np.nan,
            }
            cluster_rows.append(row)

        # Overall purity is the fraction of samples in their cluster's majority class.
        purity = dominant_count / len(X)
        summary.append({"k": k, "purity": purity, "wcss": wcss,
                        "seed": SEED})
        all_clusters.extend(cluster_rows)
        print(f"  Purity: {purity:.2%} | WCSS: {wcss:,.2f}")

        # Save after each experiment so completed results are not lost.
        pd.DataFrame(summary).to_csv(OUTPUT / "kmeans_summary.csv", index=False)
        pd.DataFrame(all_clusters).to_csv(OUTPUT / "cluster_composition.csv", index=False)

        # Plot the percentage of each sentiment class inside each cluster.
        table = pd.DataFrame(cluster_rows).set_index("cluster")
        percentages = table[["negative", "neutral", "positive"]].div(
            table["size"].replace(0, np.nan), axis=0
        ).fillna(0) * 100
        ax = percentages.plot(kind="bar", stacked=True, figsize=(10, 5),
                              color=["#d95f59", "#969696", "#38966c"])
        ax.set_title(f"Sentiment composition by cluster (k = {k})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Percentage of cluster (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Actual sentiment")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT / f"cluster_composition_k{k}.png", dpi=180)
        plt.close()

    # Plot purity against the number of clusters.
    results = pd.DataFrame(summary)
    plt.figure(figsize=(8, 5))
    plt.plot(results["k"], results["purity"] * 100, marker="o")
    plt.axhline(100 / 3, linestyle="--", color="gray", label="Balanced-class reference")
    plt.xticks(K_VALUES)
    plt.ylim(0, 100)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Cluster purity (%)")
    plt.title("k-Means purity versus k")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "purity_vs_k.png", dpi=180)
    plt.close()

    # Elbow plot: lower WCSS means samples are closer to their centroids.
    plt.figure(figsize=(8, 5))
    plt.plot(results["k"], results["wcss"], marker="o")
    plt.xticks(K_VALUES)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster sum of squares (WCSS)")
    plt.title("k-Means elbow plot")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / "elbow_plot.png", dpi=180)
    plt.close()

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
