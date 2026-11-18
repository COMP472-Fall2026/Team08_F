"""Run kNN experiments for image and sentiment data; just run this file."""

import csv
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.knn import KNNClassifier
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split

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
        if not (DATA_DIR / f"{task}_train.csv").exists():
            print(f"Skipping {task}: missing data/{task}_train.csv")
            continue

        data = load_dataset(task, "train", data_dir=DATA_DIR)
        split = split_dataset(data, validation_size=0.2, seed=42)

        scaled_split, _ = standardize_split(split)
        versions = {"unscaled": split, "standardized": scaled_split}

        rows = []
        total = len(SCALING) * len(DISTANCES) * len(WEIGHTING) * len(K_VALUES)
        print(f"\n{task.upper()}: {total} experiments", flush=True)

        for scaling in SCALING:
            for distance in DISTANCES:
                for weighted in WEIGHTING:
                    for k in K_VALUES:
                        current = versions[scaling]

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

                        save_results(f"knn_{task}_results.csv", rows)
                        save_results("knn_results.csv", all_results)

        best = max(rows, key=lambda row: row["accuracy"])
        best_results.append(best)
        save_results("knn_best.csv", best_results)
        print(f"Best {task} result: {best}", flush=True)

    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
