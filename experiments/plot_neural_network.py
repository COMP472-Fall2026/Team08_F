"""Plot saved NN runs without retraining.

python -m experiments.plot_neural_network --run-id RUN_ID
python -m experiments.plot_neural_network --dataset image --compare word_dropout
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_loader import DATASET_SPECS
from utils.evaluation import classification_metrics


def plot_run(run_id: str, results_dir="results/neural_network") -> list[Path]:
    """Save learning curves, a confusion matrix, and per-class metrics for one run."""
    run_dir = Path(results_dir) / "runs" / run_id
    dataset = json.loads((run_dir / "config.json").read_text())["dataset"]
    history = pd.read_csv(run_dir / "history.csv")
    predictions = pd.read_csv(run_dir / "validation_predictions.csv")
    n_classes = DATASET_SPECS[dataset][1]
    report = classification_metrics(predictions.true_label, predictions.predicted_label, n_classes)
    matrix = report["confusion_matrix"]

    paths = [run_dir / "per_class_metrics.csv", run_dir / "confusion_matrix.csv"]
    report["per_class"].to_csv(paths[0], index=False)
    pd.DataFrame(matrix).to_csv(paths[1], index_label="true_label")
    output = run_dir / "figures"
    output.mkdir(exist_ok=True)

    for metric in ("loss", "accuracy"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for subset in ("train", "val"):
            ax.plot(history.epoch, history[f"{subset}_{metric}"], label=subset)
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{dataset}: {metric}")
        ax.legend()
        paths.append(_save(fig, output / f"{metric}.png"))

    fig, ax = plt.subplots(figsize=(10, 9) if n_classes > 10 else (6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xlabel="Predicted class", ylabel="True class", title=f"{dataset}: confusion matrix")
    ax.tick_params(labelsize=6 if n_classes > 10 else 10)
    if n_classes <= 10:
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Examples")
    paths.append(_save(fig, output / "confusion_matrix.png"))
    return paths


def plot_comparison(dataset: str, parameter: str, results_dir="results/neural_network",
                    run_ids: list[str] | None = None) -> Path:
    """Bar chart of train/validation accuracy for runs that differ in one setting."""
    root = Path(results_dir)
    runs = pd.read_csv(root / "experiments.csv")
    runs = runs[runs.dataset == dataset]
    if run_ids:
        runs = runs[runs.run_id.isin(run_ids)]
    if runs.empty:
        raise ValueError("No completed runs match the selection.")
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 5))
    positions = np.arange(len(runs))
    ax.bar(positions - 0.2, runs.train_accuracy, width=0.4, label="train")
    ax.bar(positions + 0.2, runs.val_accuracy, width=0.4, label="validation")
    ax.set(xticks=positions, xticklabels=[str(v) for v in runs[parameter]], xlabel=parameter,
           ylabel="Accuracy", ylim=(0, 1.05), title=f"{dataset}: {parameter} comparison")
    ax.legend()
    (root / "figures").mkdir(exist_ok=True)
    return _save(fig, root / "figures" / f"{dataset}_{parameter}.png")


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/neural_network")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--compare", help="experiments.csv column, e.g. word_dropout or hidden_sizes")
    parser.add_argument("--dataset", choices=DATASET_SPECS)
    parser.add_argument("--run-ids", nargs="+")
    args = parser.parse_args()
    if args.run_id:
        print(*plot_run(args.run_id, args.results_dir), sep="\n")
    elif not args.dataset:
        parser.error("--dataset is required with --compare")
    else:
        print(plot_comparison(args.dataset, args.compare, args.results_dir, args.run_ids))


if __name__ == "__main__":
    main()
