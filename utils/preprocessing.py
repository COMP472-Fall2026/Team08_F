"""Shared preprocessing: the train/validation split, standardization and image word counts."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from utils.data_loader import Dataset


@dataclass
class DatasetSplit:
    """The training and validation subsets, plus their row positions in the original Dataset."""

    train: Dataset
    validation: Dataset
    train_indices: np.ndarray
    validation_indices: np.ndarray


def split_dataset(
    data: Dataset, *, validation_size: float = 0.2, seed: int = 42
) -> DatasetSplit:
    """Split labeled data into 80% training and 20% validation, keeping class proportions.

    Every classifier uses these defaults, so they are all evaluated on the same
    examples. Rows are sorted by ID first, so the split is the same even if the
    CSV rows are reordered.
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
    # Row positions sorted by ID; the random split is taken from this order.
    canonical_indices = np.argsort(ids)
    train_indices, validation_indices = train_test_split(
        canonical_indices,
        test_size=validation_size,
        random_state=seed,  # same split every time
        stratify=data.y[canonical_indices],  # same class proportions in both subsets
    )

    def subset(indices: np.ndarray) -> Dataset:
        """The rows at these positions, as a Dataset."""
        # Take the same rows from X, y and ids so they stay aligned.
        return Dataset(data.X[indices], data.y[indices], data.ids[indices])

    return DatasetSplit(
        subset(train_indices), subset(validation_indices),
        train_indices, validation_indices,
    )


class Standardizer:
    """Rescale each feature to mean 0 and standard deviation 1, using training statistics.

    Fit it on training data only, then use transform on validation and test data.
    Features that are constant in the training data are left unscaled (scale 1).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        """Compute each feature's mean and standard deviation from training data."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a nonempty, two-dimensional feature array.")
        # axis=0: one mean and standard deviation per feature (column).
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # A constant feature has standard deviation 0; use 1 to avoid dividing by zero.
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X with the statistics learned in fit."""
        if self.mean_ is None:
            raise ValueError("Fit the Standardizer on training data before transform.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean_):
            raise ValueError("Feature count does not match fitted training data.")
        return ((X - self.mean_) / self.scale_).astype(np.float32)


def recover_counts(X: np.ndarray) -> np.ndarray:
    """Turn image feature rows back into integer visual-word counts.

    Each row is a vector of word counts divided by its length. The smallest
    non-zero value comes from a word counted once, so dividing the row by it
    gives the original counts. This uses only the row itself, so nothing is
    learned from other data. Raises an error if a row is not a scaled count vector.
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
    """Standardize a split, fitting the scaler on the training subset only.

    Returns the scaled split and the scaler; use scaler.transform on the test
    features too. The original split is not changed.
    """
    scaler = Standardizer().fit(split.train.X)

    def scaled(data: Dataset) -> Dataset:
        """A standardized copy of a Dataset."""
        return Dataset(scaler.transform(data.X), data.y.copy(), data.ids.copy())

    return DatasetSplit(
        scaled(split.train), scaled(split.validation),
        split.train_indices.copy(), split.validation_indices.copy(),
    ), scaler
