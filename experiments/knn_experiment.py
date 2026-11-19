"""k-NN experiments: every combination of k, distance, scaling and voting, on both datasets.

python experiments/knn_experiment.py

Each run trains on the shared 80% split and is scored on the 20% validation split.
Results are saved to results/knn/ (see utils/results.py).
"""

import sys
from pathlib import Path
from time import perf_counter

# Let the script run directly (python experiments/...) as well as with python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from models.knn import KNNClassifier
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split
from utils.results import clear_results, save_run

ALGORITHM = "knn"
TASKS = ["image", "sentiment"]
K_VALUES = [1, 3, 5, 10, 20]
DISTANCES = ["euclidean", "cosine"]
SCALING = ["unscaled", "standardized"]
WEIGHTING = [False, True]


def main():
    """Run all 40 k-NN settings on each dataset and save the results."""
    clear_results(ALGORITHM)
    for task in TASKS:
        data = load_dataset(task)
        split = split_dataset(data, validation_size=0.2, seed=42)
        # Standardization is fitted on the training split only, then applied to validation.
        scaled_split, _ = standardize_split(split)
        versions = {"unscaled": split, "standardized": scaled_split}

        total = len(SCALING) * len(DISTANCES) * len(WEIGHTING) * len(K_VALUES)
        print(f"\n{task.upper()}: {total} experiments", flush=True)
        count = 0
        for scaling in SCALING:
            current = versions[scaling]
            for distance in DISTANCES:
                for weighted in WEIGHTING:
                    for k in K_VALUES:
                        start = perf_counter()
                        knn = KNNClassifier(k=k, distance=distance, weighted=weighted)
                        knn.fit(current.train.X, current.train.y)
                        predicted = knn.predict(current.validation.X)
                        accuracy = float(np.mean(predicted == current.validation.y))
                        voting = "weighted" if weighted else "majority"

                        save_run(ALGORITHM, task, sweep="k",
                                 experiment=f"k={k}, {distance}, {scaling}, {voting}",
                                 settings={"k": k, "distance": distance, "scaling": scaling, "voting": voting},
                                 metrics={"val_accuracy": accuracy, "seconds": round(perf_counter() - start, 2)},
                                 ids=current.validation.ids, y_true=current.validation.y, y_pred=predicted)
                        count += 1
                        print(f"[{count}/{total}] {scaling}, {distance}, {voting}, k={k}: {accuracy:.2%}", flush=True)


if __name__ == "__main__":
    main()
