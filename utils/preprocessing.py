"""Shared stratified splits and optional training-only standardization."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from utils.data_loader import Dataset


@dataclass
class DatasetSplit:
    """Subsets plus positional indices into the original loaded Dataset."""

    train: Dataset
    validation: Dataset
    train_indices: np.ndarray
    validation_indices: np.ndarray


def split_dataset(
    data: Dataset, *, validation_size: float = 0.2, seed: int = 42
) -> DatasetSplit:
    """Stratify labeled data using the team's default 80/20 split and seed 42.

    Sort by string ID before sampling so CSV row reordering does not change
    membership. All classifiers should use these defaults on the same dataset.
    Returned IDs identify the examples; indices refer to the input row order.
    No learned preprocessing is performed here.
    """
    if data.y is None:
        raise ValueError("Cannot split unlabeled test data.")
    if (
        data.X.ndim != 2 or data.y.ndim != 1 or data.ids.ndim != 1
        or len(data.X) != len(data.y) or len(data.X) != len(data.ids)
    ):
        raise ValueError("X, y, and ids must have aligned rows and valid dimensions.")
    ids = data.ids.astype(str)
    if len(np.unique(ids)) != len(ids) or np.any(np.char.strip(ids) == ""):
        raise ValueError("IDs must be nonempty and unique.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be a fraction strictly between 0 and 1.")
    # These are original row positions, ordered by ID before random sampling.
    canonical_indices = np.argsort(ids)
    train_indices, validation_indices = train_test_split(
        canonical_indices,
        test_size=validation_size,
        random_state=seed,  # Repeat the same random choices across classifiers.
        stratify=data.y[canonical_indices],  # Preserve class proportions.
    )

    def subset(indices: np.ndarray) -> Dataset:
        # Apply identical row positions to all three arrays to keep them aligned.
        return Dataset(data.X[indices], data.y[indices], data.ids[indices])

    return DatasetSplit(
        subset(train_indices), subset(validation_indices),
        train_indices, validation_indices,
    )


class Standardizer:
    """Per-feature (X - training mean) / training standard deviation.

    Call fit only on the training subset during model selection. Constant
    training features use scale 1 to avoid division by zero. Statistics use
    float64 accumulation; transformed model inputs are float32.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        """Learn statistics from training features only; return this scaler."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a nonempty, two-dimensional feature array.")
        # axis=0 computes one statistic per feature, across training samples.
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # A constant training feature becomes zero after centering; avoid 0/0.
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply already fitted statistics without modifying X or the scaler."""
        if self.mean_ is None:
            raise ValueError("Fit the Standardizer on training data before transform.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean_):
            raise ValueError("Feature count does not match fitted training data.")
        # Reuse training statistics, even when X is validation or test data.
        return ((X - self.mean_) / self.scale_).astype(np.float32)


def recover_counts(X: np.ndarray) -> np.ndarray:
    """Recover integer visual-word counts from L2-normalized image feature rows.

    Each image row equals counts / ||counts||, so dividing by the smallest
    positive value gives the counts, assuming the smallest count is 1. This is
    a fixed per-row transform, so it learns nothing and cannot leak data. Raise
    an error rather than guess if a row is not a scaled count vector.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.size == 0 or (X < 0).any():
        raise ValueError("X must be a nonempty 2-D array of nonnegative values.")
    smallest = np.where(X > 0, X, np.inf).min(axis=1, keepdims=True)
    if not np.isfinite(smallest).all():
        raise ValueError("Every row needs at least one positive value.")
    ratios = X / smallest
    counts = np.rint(ratios)
    if np.abs(ratios - counts).max() > 1e-3:
        raise ValueError("Rows are not scaled integer counts.")
    return counts.astype(np.float32)


def standardize_split(split: DatasetSplit) -> tuple[DatasetSplit, Standardizer]:
    """Return a scaled copy and scaler, fitting exclusively on split.train.X.

    Reuse the returned scaler to transform hidden test features. The original
    split remains unscaled so models can compare preprocessing choices fairly.
    """
    # Fit once on training data so validation cannot influence preprocessing.
    scaler = Standardizer().fit(split.train.X)

    def scaled(data: Dataset) -> Dataset:
        # Copy labels and IDs so the returned dataset remains independent.
        return Dataset(scaler.transform(data.X), data.y.copy(), data.ids.copy())

    return DatasetSplit(
        scaled(split.train), scaled(split.validation),
        split.train_indices.copy(), split.validation_indices.copy(),
    ), scaler
