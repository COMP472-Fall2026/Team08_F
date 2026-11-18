"""Run kNN experiments for image and sentiment data; just run this file."""

import csv
import sys
from pathlib import Path
from time import perf_counter

# Find the project folder when you click "Run Python File" in VS Code.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.knn import KNNClassifier
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split

# Edit these lists to choose what you want to compare.
TASKS = ["image", "sentiment"]
K_VALUES = [1, 3, 5, 10, 20]
DISTANCES = ["euclidean", "cosine"]
SCALING = ["unscaled", "standardized"]
WEIGHTING = [False, True]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def save_results(filename, rows):
    """Save a results table after each completed experiment."""
    if not rows:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_results = []
    best_results = []

    for task in TASKS:
        # The CSV must exist locally; hidden Kaggle test labels are not used.
        if not (DATA_DIR / f"{task}_train.csv").exists():
            print(f"Skipping {task}: missing data/{task}_train.csv")
            continue

        # Split labeled data: 80% training, 20% validation (same seed for all runs).
        data = load_dataset(task, "train", data_dir=DATA_DIR)
        split = split_dataset(data, validation_size=0.2, seed=42)

        # Make an optional standardized version using TRAINING statistics only.
        scaled_split, _ = standardize_split(split)
        versions = {"unscaled": split, "standardized": scaled_split}

        rows = []
        total = len(SCALING) * len(DISTANCES) * len(WEIGHTING) * len(K_VALUES)
        print(f"\n{task.upper()}: {total} experiments", flush=True)

        for scaling in SCALING:
            for distance in DISTANCES:
                for weighted in WEIGHTING:
                    for k in K_VALUES:
                        # Use the selected training/validation feature values.
                        current = versions[scaling]

                        # Follow the notes: fit -> predict -> calculate accuracy.
                        start = perf_counter()
                        knn = KNNClassifier(k=k, distance=distance, weighted=weighted)
                        knn.fit(current.train.X, current.train.y)
                        accuracy = knn.score(current.validation.X, current.validation.y)
                        seconds = perf_counter() - start

                        result = {
                            "task": task,
                            "scaling": scaling,
                            "distance": distance,
                            "weighted": weighted,
                            "k": k,
                            "accuracy": round(accuracy, 5),
                            "seconds": round(seconds, 2),
                        }
                        rows.append(result)
                        all_results.append(result)
                        print(f"[{len(rows)}/{total}] {scaling}, {distance}, "
                              f"weighted={weighted}, k={k}: {accuracy:.2%} "
                              f"({seconds:.1f}s)", flush=True)

                        # Write results immediately so Ctrl+C doesn't erase finished work.
                        save_results(f"knn_{task}_results.csv", rows)
                        save_results("knn_results.csv", all_results)

        # Pick the highest validation accuracy for this dataset.
        best = max(rows, key=lambda row: row["accuracy"])
        best_results.append(best)
        save_results("knn_best.csv", best_results)
        print(f"Best {task} result: {best}", flush=True)

    print(f"\nResults saved to: {RESULTS_DIR}")
    if len(best_results) == 2:
        average = (best_results[0]["accuracy"] + best_results[1]["accuracy"]) / 2
        print(f"Average of best validation accuracies: {average:.2%}")


if __name__ == "__main__":
    main()
