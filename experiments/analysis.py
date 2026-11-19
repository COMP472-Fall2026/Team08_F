"""Plots and tables from the saved results of every algorithm (no retraining).

python experiments/analysis.py                  # every algorithm that has results
python experiments/analysis.py --algorithm knn  # one algorithm

Reads results/<algorithm>/ (written by the *_experiment.py scripts) and saves to
results/<algorithm>/figures/:
    <dataset>_<sweep>.png        accuracy (or purity) for each value tried in a sweep
    <dataset>_confusion.png/csv  confusion matrix of the final model (or best run) on the validation split
    <dataset>_per_class.csv      precision, recall and F1 of that run
    learning_curves/             per-epoch curves, if the algorithm saved history.csv
    cluster_composition_k*.png   class mix of each cluster, for k-means
When more than one classifier has results, results/comparison/ gets the validation
accuracy of each classifier's final model and how often their mistakes overlap.
"""

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

# Let the script run directly (python experiments/...) as well as with python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from utils.data_loader import DATASET_SPECS
from utils.metrics import confusion_matrix, per_class_report
from utils.results import ID_COLUMNS, METRIC_COLUMNS, RESULTS_DIR, read_results

# Colour-blind-safe categorical palette, always used in this order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_NAMES = {"sentiment": ["negative", "neutral", "positive"]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True,
                     "legend.frameon": False, "figure.constrained_layout.use": True})


def pct(value: float) -> str:
    """Format a fraction as a percentage, rounding halves up (0.5405 -> "54.1%"), as in the report."""
    return f"{np.floor(1000 * round(float(value), 6) + 0.5) / 10:.1f}%"


def sort_key(value):
    """Order sweep values sensibly: numbers ascending, layer lists by size, None last, text after."""
    text = str(value)
    if text == "None":
        return (1, float("inf"), text)
    try:
        parsed = json.loads(text)
    except ValueError:
        return (2, 0.0, text)
    if isinstance(parsed, list):  # hidden layer sizes, e.g. [4096, 4096]
        return (0, len(parsed) * 1e7 + parsed[0], text)
    if isinstance(parsed, (int, float)):
        return (0, float(parsed), text)
    return (2, 0.0, text)


def setting_columns(runs: pd.DataFrame) -> list[str]:
    """The columns of experiments.csv that are settings (everything except IDs and metrics)."""
    return [c for c in runs.columns if c not in ID_COLUMNS + METRIC_COLUMNS]


def plot_sweeps(algorithm: str, runs: pd.DataFrame, out: Path) -> None:
    """One figure per (dataset, sweep): the metric for each value tried.

    The plotted x-axis is the setting that changes within the sweep. If other settings
    change too (e.g. k-NN's distance and scaling), each combination is its own line.
    A "baseline" run, if present, is added to every sweep as the reference point.
    """
    settings = setting_columns(runs)
    if "val_accuracy" in runs:
        panels = [("val_accuracy", "Accuracy")]
    else:
        panels = [(m, m.upper() if m == "wcss" else m.capitalize()) for m in ("purity", "wcss") if m in runs]
    for dataset, by_dataset in runs.groupby("dataset"):
        baseline = by_dataset[by_dataset.sweep == "baseline"]
        for sweep, group in by_dataset.groupby("sweep"):
            if sweep in ("baseline", "custom"):
                continue
            group = pd.concat([group, baseline], ignore_index=True)
            varying = [c for c in settings if group[c].astype(str).nunique() > 1]
            if not varying:
                continue
            # The x-axis is the setting the sweep is named after, or else the first one that changes.
            x = next((c for c in sorted(varying, key=len, reverse=True) if str(sweep).startswith(c)), varying[0])
            series_columns = [c for c in varying if c != x]
            order = sorted(group[x].astype(str).unique(), key=sort_key)
            position = {value: i for i, value in enumerate(order)}

            fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.6), squeeze=False)
            for ax, (metric, label) in zip(axes[0], panels):
                series = group.groupby(series_columns, dropna=False) if series_columns else [((), group)]
                single = len(series_columns) == 0
                for i, (key, part) in enumerate(series):
                    part = part.assign(_x=part[x].astype(str).map(position)).sort_values("_x")
                    color = COLORS[i % len(COLORS)]
                    key = key if isinstance(key, tuple) else (key,)
                    name = ", ".join(f"{c}={v}" for c, v in zip(series_columns, key)) or "validation"
                    ax.plot(part._x, part[metric], marker="o", color=color, lw=2, label=name)
                    # Train accuracy is drawn only for single-line plots, to keep them readable.
                    if single and metric == "val_accuracy" and "train_accuracy" in part and part.train_accuracy.notna().any():
                        ax.plot(part._x, part.train_accuracy, marker="o", color=COLORS[1], lw=2, ls="--", label="train")
                if metric in ("val_accuracy", "purity"):
                    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                ax.set_xticks(range(len(order)), order, rotation=30 if len(order) > 6 else 0)
                ax.set(xlabel=x, ylabel=label, title=f"{dataset}: {label.lower()} vs {x}")
                ax.legend(fontsize=7)
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(sweep)).strip("_")
            fig.savefig(out / f"{dataset}_{safe}.png", dpi=150)
            plt.close(fig)


FINAL_SWEEPS = ("final", "ensemble")  # sweeps that hold an algorithm's chosen final model


def analyse_best_run(runs: pd.DataFrame, predictions: pd.DataFrame, out: Path) -> dict:
    """Confusion matrix and per-class metrics of each dataset's final model; returns those runs.

    The final model is the run in a "final" or "ensemble" sweep if the algorithm has one
    (the model actually chosen), otherwise the run with the best validation accuracy.
    """
    best_runs = {}
    with_predictions = runs[runs.run_id.isin(set(predictions.run_id))]
    for dataset, group in with_predictions.groupby("dataset"):
        final = group[group.sweep.isin(FINAL_SWEEPS)]
        candidates = final if len(final) else group
        best = candidates.loc[candidates.val_accuracy.idxmax()]
        rows = predictions[predictions.run_id == best.run_id]
        n_classes = DATASET_SPECS[dataset][1]
        names = CLASS_NAMES.get(dataset, [str(c) for c in range(n_classes)])
        cm, _ = confusion_matrix(rows.true_label, rows.predicted_label, labels=np.arange(n_classes))
        report = per_class_report(rows.true_label, rows.predicted_label, labels=np.arange(n_classes))
        pd.DataFrame({"class": names, "precision": report["precision"].round(3), "recall": report["recall"].round(3),
                      "f1": report["f1"].round(3), "support": report["support"].astype(int)}
                     ).to_csv(out / f"{dataset}_per_class.csv", index=False)
        pd.DataFrame(cm, index=names, columns=names).to_csv(out / f"{dataset}_confusion.csv")

        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        small = n_classes <= 10
        fig, ax = plt.subplots(figsize=(4.6, 3.8) if small else (8, 7))
        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        if small:
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, f"{pct(share[i, j])}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                            color="white" if share[i, j] > 0.5 else "black")
        ticks = range(0, n_classes, 1 if small else 5)
        ax.set_xticks(list(ticks), [names[t] for t in ticks])
        ax.set_yticks(list(ticks), [names[t] for t in ticks])
        ax.grid(False)
        ax.set(xlabel="Predicted class", ylabel="True class",
               title=f"{dataset}: final model ({pct(best.val_accuracy)})\n{best.experiment}"[:120])
        fig.colorbar(image, ax=ax, format=PercentFormatter(1.0), label="Share of true class")
        fig.savefig(out / f"{dataset}_confusion.png", dpi=150)
        plt.close(fig)
        best_runs[dataset] = (best, rows)
    return best_runs


def plot_learning_curves(runs: pd.DataFrame, history: pd.DataFrame, out: Path) -> None:
    """Train and validation accuracy per epoch for every run (each seed of an ensemble)."""
    out.mkdir(exist_ok=True)
    for run_id, curves in history.groupby("run_id"):
        run = runs[runs.run_id == run_id]
        if run.empty:
            continue
        run = run.iloc[0]
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for seed, part in curves.groupby("seed"):
            suffix = f" (seed {seed})" if curves.seed.nunique() > 1 else ""
            ax.plot(part.epoch, part.train_accuracy, color=COLORS[0], lw=2, label="train" + suffix)
            ax.plot(part.epoch, part.val_accuracy, color=COLORS[1], lw=2, label="validation" + suffix)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02), title=f"{run.dataset}: {run.experiment}")
        ax.legend(fontsize=7)
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(run.experiment)).strip("_")
        fig.savefig(out / f"{run.dataset}_{safe}.png", dpi=150)
        plt.close(fig)


def plot_cluster_composition(composition: pd.DataFrame, out: Path) -> None:
    """Stacked bars: the share of each true class inside each cluster, one figure per k."""
    classes = [c for c in composition.columns if c not in ("k", "cluster", "size")]
    for k, part in composition.groupby("k"):
        shares = part[classes].div(part["size"].replace(0, np.nan), axis=0).fillna(0)
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        bottom = np.zeros(len(part))
        for i, name in enumerate(classes):
            ax.bar(part.cluster.astype(str), shares[name], bottom=bottom, color=COLORS[i], label=name)
            bottom += shares[name].to_numpy()
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(xlabel="Cluster", ylabel="Share of cluster", title=f"Class mix of each cluster (k = {k})")
        ax.legend(fontsize=7)
        fig.savefig(out / f"cluster_composition_k{k}.png", dpi=150)
        plt.close(fig)


def compare(best_runs: dict, out: Path) -> None:
    """Validation accuracy of each classifier's final model, and how often their mistakes overlap."""
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"algorithm": algorithm, "dataset": dataset, "experiment": best.experiment,
                           "val_accuracy": round(best.val_accuracy, 4)}
                          for algorithm, by_dataset in best_runs.items()
                          for dataset, (best, _) in by_dataset.items()])
    table.to_csv(out / "final_validation_accuracy.csv", index=False)

    overlap = []
    for dataset in table.dataset.unique():
        available = [a for a in best_runs if dataset in best_runs[a]]
        for first, second in combinations(available, 2):
            a = best_runs[first][dataset][1][["id", "true_label", "predicted_label"]]
            b = best_runs[second][dataset][1][["id", "predicted_label"]]
            both = a.merge(b, on="id", suffixes=(f"_{first}", f"_{second}"))
            wrong_a = both[f"predicted_label_{first}"] != both.true_label
            wrong_b = both[f"predicted_label_{second}"] != both.true_label
            overlap.append({"dataset": dataset, "model_a": first, "model_b": second,
                            "both_correct": round(float((~wrong_a & ~wrong_b).mean()), 4),
                            "only_a_wrong": round(float((wrong_a & ~wrong_b).mean()), 4),
                            "only_b_wrong": round(float((~wrong_a & wrong_b).mean()), 4),
                            "both_wrong": round(float((wrong_a & wrong_b).mean()), 4)})
    pd.DataFrame(overlap).to_csv(out / "error_overlap.csv", index=False)

    datasets = list(table.dataset.unique())
    algorithms = list(table.algorithm.unique())
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.8 / len(algorithms)
    for i, algorithm in enumerate(algorithms):
        values = [table[(table.algorithm == algorithm) & (table.dataset == d)].val_accuracy.sum() for d in datasets]
        bars = ax.bar(np.arange(len(datasets)) + i * width, values, width * 0.9, color=COLORS[i], label=algorithm)
        ax.bar_label(bars, labels=[pct(v) if v else "" for v in values], fontsize=7)
    ax.set_xticks(np.arange(len(datasets)) + width * (len(algorithms) - 1) / 2, datasets)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(ylabel="Validation accuracy", ylim=(0, 1), title="Validation accuracy of each classifier's final model")
    ax.legend(fontsize=7)
    fig.savefig(out / "final_validation_accuracy.png", dpi=150)
    plt.close(fig)
    print(table.to_string(index=False))


def analyse(algorithm: str, results_dir: Path | None = None) -> dict:
    """Make every figure and table for one algorithm; return its best runs per dataset."""
    folder = Path(results_dir or RESULTS_DIR) / algorithm
    runs = read_results(folder / "experiments.csv")
    out = folder / "figures"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    plot_sweeps(algorithm, runs, out)
    best_runs = {}
    if (folder / "validation_predictions.csv").exists() and "val_accuracy" in runs:
        best_runs = analyse_best_run(runs, read_results(folder / "validation_predictions.csv"), out)
    if (folder / "history.csv").exists():
        plot_learning_curves(runs, read_results(folder / "history.csv"), out / "learning_curves")
    if (folder / "cluster_composition.csv").exists():
        plot_cluster_composition(read_results(folder / "cluster_composition.csv"), out)
    print(f"{algorithm}: figures and tables saved to {out}")
    return best_runs


def main():
    """Analyse every algorithm that has results (or just --algorithm), then compare them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", help="only this results folder (e.g. knn); default: all of them")
    args = parser.parse_args()
    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = sorted(p.parent.name for p in RESULTS_DIR.glob("*/experiments.csv"))
    if not algorithms:
        sys.exit("No results found. Run the experiment scripts first (see README).")
    best_runs = {algorithm: analyse(algorithm) for algorithm in algorithms}
    classifiers = {a: b for a, b in best_runs.items() if b}
    if len(classifiers) > 1:
        compare(classifiers, RESULTS_DIR / "comparison")


if __name__ == "__main__":
    main()
