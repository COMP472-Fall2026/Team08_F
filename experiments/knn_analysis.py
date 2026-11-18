"""Analyze completed kNN validation experiments. Run this file in VS Code."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Find the project root even when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.knn import KNNClassifier
from utils.data_loader import load_dataset
from utils.preprocessing import split_dataset, standardize_split

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "knn_analysis"
TASKS = ["image", "sentiment"]


def plot_accuracy(results, task):
    """Compare k while keeping distance, scaling and voting fixed."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Each line represents one combination of the other hyperparameters.
    for (distance, scaling, weighted), group in results.groupby(
        ["distance", "scaling", "weighted"]
    ):
        group = group.sort_values("k")
        voting = "weighted" if weighted else "unweighted"
        ax.plot(group["k"], group["accuracy"] * 100, marker="o",
                label=f"{distance}, {scaling}, {voting}")

    ax.set_title(f"{task.title()} — validation accuracy versus k")
    ax.set_xlabel("Number of neighbors (k)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(sorted(results["k"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{task}_accuracy_vs_k.png", dpi=180)
    plt.close(fig)


def analyze_errors(task, best):
    """Recreate the validation predictions for the best tested configuration."""
    data = load_dataset(task, "train", data_dir=DATA_DIR)
    # Use exactly the same split as knn_experiment.py.
    split = split_dataset(data, validation_size=0.2, seed=42)

    # Standardize only if that configuration used standardized features.
    if best["scaling"] == "standardized":
        split, _ = standardize_split(split)
    elif best["scaling"] != "unscaled":
        raise ValueError("Unknown scaling setting")

    model = KNNClassifier(k=int(best["k"]), distance=best["distance"],
                          weighted=bool(best["weighted"]))
    model.fit(split.train.X, split.train.y)
    actual = split.validation.y
    predicted = model.predict(split.validation.X)
    accuracy = np.mean(actual == predicted)

    # A mismatch can indicate the model, split or settings changed since the experiment.
    if not np.isclose(accuracy, best["accuracy"], atol=0.00001, rtol=0):
        print(f"WARNING: {task} accuracy differs from saved results. "
              "Check your model and rerun knn_experiment.py if needed.")

    # Rows = actual class; columns = predicted class.
    labels = np.arange(50 if task == "image" else 3)
    names = [str(i) for i in labels] if task == "image" else ["Negative", "Neutral", "Positive"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        ANALYSIS_DIR / f"{task}_confusion_matrix.csv", index_label="actual/predicted")

    # Plot the percentage of each true class assigned to each predicted class.
    # This makes the diagonal the recall (per-class accuracy).
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix * 100.0, totals,
                            out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    fig, ax = plt.subplots(figsize=(11, 9) if task == "image" else (6, 5))
    heatmap = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(heatmap, ax=ax, label="Percentage of actual class (%)")
    ax.set_xticks(labels, names, rotation=90 if task == "image" else 0)
    ax.set_yticks(labels, names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{task.title()} — validation confusion matrix")
    if task == "sentiment":
        for i in labels:
            for j in labels:
                ax.text(j, i, f"{percentages[i, j]:.1f}%\n({matrix[i, j]})",
                        ha="center", va="center",
                        color="white" if percentages[i, j] > 50 else "black")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{task}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Use standard evaluation metrics, as permitted by the competition.
    report = classification_report(actual, predicted, labels=labels,
                                   target_names=names, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.loc[names].copy()
    table.to_csv(ANALYSIS_DIR / f"{task}_class_metrics.csv", index_label="class")
    fig, ax = plt.subplots(figsize=(12, 5) if task == "image" else (6, 4))
    ax.bar(names, table["recall"] * 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Correct predictions within class (%)")
    ax.set_xlabel("Actual class")
    ax.set_title(f"{task.title()} — per-class validation accuracy (recall)")
    ax.tick_params(axis="x", labelrotation=90 if task == "image" else 0)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{task}_class_accuracy.png", dpi=180)
    plt.close(fig)

    # Save validation errors with IDs so teammates can compare model mistakes.
    errors = pd.DataFrame({"id": split.validation.ids, "actual": actual,
                           "predicted": predicted})
    errors = errors[errors["actual"] != errors["predicted"]]
    errors.to_csv(ANALYSIS_DIR / f"{task}_errors.csv", index=False)
    pairs = (errors.groupby(["actual", "predicted"]).size()
             .reset_index(name="count").sort_values("count", ascending=False))
    pairs.to_csv(ANALYSIS_DIR / f"{task}_confused_pairs.csv", index=False)

    print(f"\n{task.upper()} | k={best['k']}, {best['distance']}, "
          f"{best['scaling']}, weighted={best['weighted']}")
    print(f"Saved validation accuracy: {best['accuracy']:.2%}")
    print(f"Recomputed validation accuracy: {accuracy:.2%}")
    print(f"Errors: {len(errors)} / {len(actual)}")
    print("Lowest-recall classes:")
    print(table.sort_values("recall")[["recall", "support"]].head(5).to_string())
    print("Most common misclassifications (actual -> predicted):")
    print(pairs.head(5).to_string(index=False))
    return {"task": task, "k": int(best["k"]), "distance": best["distance"],
            "scaling": best["scaling"], "weighted": bool(best["weighted"]),
            "saved_accuracy": best["accuracy"], "validation_accuracy": accuracy,
            "macro_f1": report["macro avg"]["f1-score"], "errors": len(errors)}


def main():
    # Read the results already produced by knn_experiment.py.
    results = pd.read_csv(RESULTS_DIR / "knn_results.csv")
    results["weighted"] = results["weighted"].astype(str).str.lower().map(
        {"true": True, "false": False})
    if results["weighted"].isna().any():
        raise ValueError("Unexpected weighted setting in results CSV")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for task in TASKS:
        task_results = results[results["task"] == task]
        if task_results.empty:
            print(f"Skipping {task}: no saved results")
            continue
        plot_accuracy(task_results, task)

        # Compare hyperparameters while keeping the remaining settings fixed.
        # Pivot CSV exposes the original measured accuracies, not invented values.
        comparison = task_results.pivot_table(
            index=["k", "distance"], columns=["scaling", "weighted"],
            values="accuracy", aggfunc="max")
        comparison.to_csv(ANALYSIS_DIR / f"{task}_hyperparameter_comparison.csv")
        best = task_results.loc[task_results["accuracy"].idxmax()]
        if not (DATA_DIR / f"{task}_train.csv").exists():
            print(f"Skipping {task} error analysis: training CSV is missing")
            continue
        summary.append(analyze_errors(task, best))

    if summary:
        pd.DataFrame(summary).to_csv(ANALYSIS_DIR / "knn_analysis_summary.csv", index=False)
    print(f"\nAnalysis saved in: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
