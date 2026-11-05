"""Plot saved runs; no retraining. Run with --help for run/comparison options."""

import argparse
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_loader import DATASET_SPECS
from utils.evaluation import classification_metrics


def plot_run(run_id: str, results_dir="results/neural_network") -> list[Path]:
    """Derive curves, confusion matrix, and class metrics from saved run data.

    No checkpoint or precomputed metrics files are needed. Preserve existing
    analysis files when this command is repeated.
    """
    root = Path(results_dir)
    summary = pd.read_csv(root / "experiments.csv").set_index("run_id").loc[run_id]
    history = pd.read_csv(root / "epoch_history.csv")
    history = history[history.run_id == run_id]
    run_dir = root / "runs" / run_id
    predictions = pd.read_csv(run_dir / "validation_predictions.csv")
    n_classes = DATASET_SPECS[summary.dataset][1]
    report = classification_metrics(predictions.true_label, predictions.predicted_label, n_classes)
    matrix = pd.DataFrame(report["confusion_matrix"])
    paths = []
    # Derive these tables only when analysis is requested, not during training.
    for name, table, index_label in (
        ("confusion_matrix.csv", matrix, "true_label"),
        ("per_class_metrics.csv", report["per_class"], None),
    ):
        path = run_dir / name
        if not path.exists():
            table.to_csv(path, index=index_label is not None, index_label=index_label)
        paths.append(path)
    output = run_dir / "figures"
    output.mkdir(exist_ok=True)
    for metric in ("loss", "accuracy"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for subset in ("train", "val"):
            ax.plot(history.epoch, history[f"{subset}_{metric}"], label=subset)
        ax.axvline(summary.best_epoch, color="gray", linestyle="--", label="selected epoch")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{summary.dataset}: {metric}")
        ax.legend()
        fig.tight_layout()
        path = output / f"{metric}.png"
        if not path.exists():
            fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 9) if n_classes > 10 else (6, 5))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xlabel="Predicted class", ylabel="True class", title=f"{summary.dataset}: confusion matrix")
    ax.tick_params(axis="both", labelsize=6 if n_classes > 10 else 10)
    if n_classes <= 10:
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Examples")
    fig.tight_layout()
    path = output / "confusion_matrix.png"
    if not path.exists():
        fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_comparison(dataset: str, parameter: str, results_dir="results/neural_network",
                    run_ids: list[str] | None = None) -> Path:
    """Compare selected completed runs on one task; inspect other settings for fairness."""
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
    labels = [f"{value}\n{run_id[-8:]}" for value, run_id in zip(runs[parameter], runs.run_id)]
    ax.set(xticks=positions, xticklabels=labels, xlabel=parameter, ylabel="Accuracy",
           ylim=(0, 1.05), title=f"{dataset}: {parameter} comparison")
    ax.legend()
    fig.tight_layout()
    # A unique filename preserves earlier comparison figures.
    output = root / "figures"
    output.mkdir(exist_ok=True)
    path = output / f"{dataset}_{parameter}_{uuid4().hex[:8]}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/neural_network")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--compare", choices=("learning_rate", "hidden_sizes", "dropout",
                                                "weight_decay", "batch_size", "preprocessing"))
    parser.add_argument("--dataset", choices=("image", "sentiment"))
    parser.add_argument("--run-ids", nargs="+")
    args = parser.parse_args()
    if args.run_id:
        print(*plot_run(args.run_id, args.results_dir), sep="\n")
    else:
        if not args.dataset:
            parser.error("--dataset is required with --compare")
        print(plot_comparison(args.dataset, args.compare, args.results_dir, args.run_ids))


if __name__ == "__main__":
    main()
