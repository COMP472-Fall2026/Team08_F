"""Load competition feature vectors without treating IDs or labels as features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# task: (feature count, class count, labeled row count, hidden test row count)
DATASET_SPECS = {
    "image": (500, 50, 20_000, 5_000),
    "sentiment": (768, 3, 12_000, 5_000),
}


@dataclass
class Dataset:
    """Aligned arrays in CSV row order; y is None for hidden test data."""

    X: np.ndarray  # Rows are samples; columns are numerical features.
    y: np.ndarray | None  # One class label per sample, when available.
    ids: np.ndarray  # Identifiers for matching samples to predictions.


def load_dataset(
    task: str,
    split: str = "train",
    data_dir: str | Path = "data",
    *,
    validate_row_count: bool = True,
) -> Dataset:
    """Load {task}_{split}.csv and return features, labels, and IDs.

    Use task='image' or 'sentiment' and split='train' or 'test'. Set
    validate_row_count=False for small fixtures. Features follow numeric column
    order; IDs stay strings; test data has no labels (y=None).
    """
    if task not in DATASET_SPECS:
        raise ValueError("task must be 'image' or 'sentiment'.")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    n_features, _, n_train, n_test = DATASET_SPECS[task]
    path = Path(data_dir) / f"{task}_{split}.csv"
    # Read IDs as text to preserve identifiers such as "000123".
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    # Build numeric order explicitly: feature_2 must come before feature_10.
    features = [f"feature_{i}" for i in range(n_features)]
    expected = {"id", *features}
    if split == "train":
        expected.add("label")
    if set(frame.columns) != expected:
        raise ValueError(f"{path}: invalid columns for {task} {split} data.")
    if frame.empty:
        raise ValueError(f"{path}: empty datasets are not supported.")
    expected_rows = n_train if split == "train" else n_test
    # Small synthetic tests can skip the official competition row counts.
    if validate_row_count and len(frame) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(frame)}.")
    # Pandas/NumPy report file-reading and numeric-conversion errors directly.
    # Select only features for X so neither IDs nor target labels enter a model.
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64) if split == "train" else None
    ids = frame["id"].to_numpy(dtype=str)
    return Dataset(X=X, y=y, ids=ids)
