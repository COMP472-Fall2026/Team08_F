"""Generate kNN accuracy plots and confusion matrices."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.knn import KNNClassifier
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "knn_analysis"

TASKS = ["image", "sentiment"]


def plot_accuracy(results, task):
    # Plot accuracy against k for each hyperparameter combination.
    plt.figure(figsize=(10, 6))

    for (distance, scaling, weighted), group in results.groupby(
        ["distance", "scaling", "weighted"]
    ):
        group = group.sort_values("k")

        voting = "weighted" if weighted else "unweighted"

        plt.plot(
            group["k"],
            group["accuracy"] * 100,
            marker="o",
            label=f"{distance}, {scaling}, {voting}"
        )

    plt.title(f"{task.title()} - Accuracy vs k")
    plt.xlabel("Number of neighbors (k)")
    plt.ylabel("Validation accuracy (%)")
    plt.xticks(sorted(results["k"].unique()))
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{task}_accuracy_vs_k.png", dpi=180)
    plt.close()


def plot_confusion_matrix(task, best):
    # Load and split the dataset.
    data = load_dataset(task, "train", data_dir=DATA_DIR)
    split = split_dataset(data, validation_size=0.2, seed=42)

    # Apply the selected preprocessing.
    if best["scaling"] == "standardized":
        split, _ = standardize_split(split)

    # Train the best kNN configuration.
    model = KNNClassifier(
        k=int(best["k"]),
        distance=best["distance"],
        weighted=bool(best["weighted"])
    )

    model.fit(split.train.X, split.train.y)

    # Predict validation samples.
    actual = split.validation.y
    predicted = model.predict(split.validation.X)

    # Calculate confusion matrix.
    labels = np.arange(50 if task == "image" else 3)

    names = (
        [str(i) for i in labels]
        if task == "image"
        else ["Negative", "Neutral", "Positive"]
    )

    matrix = confusion_matrix(actual, predicted, labels=labels)

    # Save confusion matrix as CSV.
    pd.DataFrame(
        matrix,
        index=names,
        columns=names
    ).to_csv(OUTPUT_DIR / f"{task}_confusion_matrix.csv")

    # Convert counts to percentages.
    totals = matrix.sum(axis=1, keepdims=True)

    percentages = np.divide(
        matrix * 100.0,
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals != 0
    )

    # Plot confusion matrix.
    fig, ax = plt.subplots(
        figsize=(11, 9) if task == "image" else (6, 5)
    )

    heatmap = ax.imshow(
        percentages,
        cmap="Blues",
        vmin=0,
        vmax=100
    )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Percentage of actual class (%)"
    )

    ax.set_xticks(labels, names, rotation=90 if task == "image" else 0)
    ax.set_yticks(labels, names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{task.title()} - Confusion Matrix")

    # Display percentages and counts for sentiment.
    if task == "sentiment":
        for i in labels:
            for j in labels:
                ax.text(
                    j, i,
                    f"{percentages[i, j]:.1f}%\n({matrix[i, j]})",
                    ha="center",
                    va="center",
                    color="white" if percentages[i, j] > 50 else "black"
                )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{task}_confusion_matrix.png", dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results from knn_experiment.py.
    results = pd.read_csv(RESULTS_DIR / "knn_results.csv")

    results["weighted"] = (
        results["weighted"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    for task in TASKS:
        task_results = results[results["task"] == task]

        if task_results.empty:
            continue

        # Generate accuracy plot.
        plot_accuracy(task_results, task)

        # Find the best configuration.
        best = task_results.loc[task_results["accuracy"].idxmax()]

        # Generate confusion matrix.
        plot_confusion_matrix(task, best)

    print(f"Plots and confusion matrices saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()