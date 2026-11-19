"""Save experiment results in the same format for every algorithm.

Each algorithm writes to results/<algorithm>/:
    experiments.csv             one row per run: dataset, sweep, experiment name, settings, metrics
    validation_predictions.csv  one row per validation example per run: run_id, id, true_label, predicted_label

experiments/analysis.py reads these files to make the plots for every algorithm.
"""

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Columns that identify a run, and the metric columns a run may report. Every other
# column in experiments.csv is a setting (e.g. k, max_depth, learning_rate).
ID_COLUMNS = ["run_id", "dataset", "sweep", "experiment"]
METRIC_COLUMNS = [
    "train_accuracy", "val_accuracy", "train_loss", "val_loss", "oob_accuracy",
    "purity", "wcss", "seconds", "parameter_count", "n_leaves", "depth_reached", "features_per_split",
]


def clear_results(algorithm: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Delete an algorithm's previous results so a full run starts from empty files."""
    folder = Path(results_dir) / algorithm
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def read_results(path: Path) -> pd.DataFrame:
    """Read a results CSV. Only empty cells count as missing, so "None" (e.g. no depth limit) stays text."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def append_csv(rows: pd.DataFrame, path: Path) -> None:
    """Append rows to a CSV file, creating it if needed. Columns are matched by name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = pd.concat([read_results(path), rows], ignore_index=True)
    rows.to_csv(path, index=False)


def save_run(algorithm: str, dataset: str, sweep: str, experiment: str, settings: dict, metrics: dict, *,
             ids=None, y_true=None, y_pred=None, results_dir: Path = RESULTS_DIR) -> str:
    """Record one run and return its run_id.

    sweep groups runs that are compared in one plot (e.g. "k" or "max_depth");
    experiment is a short readable name for this run (e.g. "k=5, cosine").
    If ids, y_true and y_pred are given, the validation predictions are saved too.
    """
    unknown = set(metrics) - set(METRIC_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown metric names {sorted(unknown)}; add them to METRIC_COLUMNS.")
    # Lists (e.g. hidden layer sizes) are stored as JSON text and None as "None",
    # so every value survives the round trip through CSV unambiguously.
    clean = {key: json.dumps(value) if isinstance(value, (list, tuple))
             else "None" if value is None else value for key, value in settings.items()}
    run_id = f"{dataset}_{uuid4().hex[:8]}"
    folder = Path(results_dir) / algorithm
    append_csv(pd.DataFrame([{"run_id": run_id, "dataset": dataset, "sweep": sweep, "experiment": experiment,
                              **clean, **metrics}]), folder / "experiments.csv")
    if ids is not None:
        append_csv(pd.DataFrame({"run_id": run_id, "id": ids, "true_label": y_true, "predicted_label": y_pred}),
                   folder / "validation_predictions.csv")
    return run_id
