"""Load the Kaggle CSV files into feature, label and ID arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# task: (number of features, number of classes, training rows, test rows)
DATASET_SPECS = {
    "image": (500, 50, 20_000, 5_000),
    "sentiment": (768, 3, 12_000, 5_000),
}


@dataclass
class Dataset:
    """Features, labels and IDs for the same rows, in file order. y is None for test data."""

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
    """Load data/{task}_{split}.csv, e.g. load_dataset("image") or load_dataset("sentiment", "test").

    Checks the columns and number of rows (tests with small files can turn the
    row check off with validate_row_count=False). Test data has no labels, so y is None.
    """
    if task not in DATASET_SPECS:
        raise ValueError("task must be 'image' or 'sentiment'.")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    n_features, _, n_train, n_test = DATASET_SPECS[task]
    path = Path(data_dir) / f"{task}_{split}.csv"
    # Read IDs as text to preserve identifiers such as "000123".
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    # List the columns in numeric order (feature_2 before feature_10, unlike alphabetical order).
    features = [f"feature_{i}" for i in range(n_features)]
    expected = {"id", *features}
    if split == "train":
        expected.add("label")
    if set(frame.columns) != expected:
        raise ValueError(f"{path}: invalid columns for {task} {split} data.")
    if frame.empty:
        raise ValueError(f"{path}: empty datasets are not supported.")
    expected_rows = n_train if split == "train" else n_test
    if validate_row_count and len(frame) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(frame)}.")
    # X contains only the feature columns, so IDs and labels can never be used as features.
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64) if split == "train" else None
    ids = frame["id"].to_numpy(dtype=str)
    return Dataset(X=X, y=y, ids=ids)
