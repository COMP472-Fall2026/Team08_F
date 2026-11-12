"""Synthetic correctness tests; no competition files or hidden labels required."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils.data_loader import DATASET_SPECS, Dataset, load_dataset
from utils.preprocessing import Standardizer, recover_counts, split_dataset, standardize_split


class LoaderTests(unittest.TestCase):
    def setUp(self):
        # Temporary CSVs exercise file loading without touching Kaggle data.
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def frame(self, task="image", split="train", rows=30):
        features, classes, _, _ = DATASET_SPECS[task]
        # Reverse columns to verify the loader restores numeric feature order.
        frame = pd.DataFrame({
            f"feature_{i}": np.arange(rows) + i / 1000
            for i in reversed(range(features))
        })
        frame["id"] = [f"{i:06d}" for i in range(rows)]
        if split == "train":
            frame["label"] = np.arange(rows) % min(classes, 3)
        return frame

    def load(self, frame, task="image", split="train", **kwargs):
        frame.to_csv(self.root / f"{task}_{split}.csv", index=False)
        return load_dataset(task, split, self.root, **kwargs)

    def test_both_tasks_and_splits_preserve_order_types_and_ids(self):
        for task, (features, _, _, _) in DATASET_SPECS.items():
            for split in ("train", "test"):
                with self.subTest(task=task, split=split):
                    frame = self.frame(task, split)
                    data = self.load(frame, task, split, validate_row_count=False)
                    self.assertEqual(data.X.shape, (30, features))
                    self.assertEqual(data.X.dtype, np.float32)
                    np.testing.assert_array_equal(data.ids, frame["id"])
                    np.testing.assert_allclose(data.X[0], np.arange(features) / 1000)
                    if split == "train":
                        self.assertEqual(data.y.dtype, np.int64)
                        np.testing.assert_array_equal(data.y, frame["label"])
                    else:
                        self.assertIsNone(data.y)

    def test_row_count_is_strict_by_default(self):
        with self.assertRaisesRegex(ValueError, "expected 20000 rows"):
            self.load(self.frame())

    def test_invalid_columns(self):
        for frame in (
            self.frame().drop(columns="feature_499"),
            self.frame().drop(columns="label"),
            self.frame().assign(extra=1),
        ):
            with self.subTest(columns=frame.shape):
                with self.assertRaisesRegex(ValueError, "invalid columns"):
                    self.load(frame, validate_row_count=False)

    def test_hidden_test_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid columns"):
            self.load(self.frame(), split="test", validate_row_count=False)

    def test_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "empty datasets"):
            self.load(self.frame(rows=0), validate_row_count=False)

    def test_invalid_arguments_and_missing_file(self):
        with self.assertRaisesRegex(ValueError, "task must"):
            load_dataset("unknown", data_dir=self.root)
        with self.assertRaisesRegex(ValueError, "split must"):
            load_dataset("image", "validation", self.root)
        with self.assertRaises(FileNotFoundError):
            load_dataset("image", data_dir=self.root)

    def test_load_split_scale_integration_for_both_tasks(self):
        for task in DATASET_SPECS:
            data = self.load(self.frame(task), task, validate_row_count=False)
            raw = split_dataset(data)
            scaled, scaler = standardize_split(raw)
            self.assertEqual(scaled.train.X.shape[0], 24)
            self.assertEqual(scaled.validation.X.shape[0], 6)
            np.testing.assert_array_equal(raw.train.ids, scaled.train.ids)
            np.testing.assert_allclose(scaled.train.X.mean(axis=0), 0, atol=1e-6)
            hidden = self.load(self.frame(task, "test"), task, "test", validate_row_count=False)
            self.assertEqual(scaler.transform(hidden.X).shape, hidden.X.shape)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        # Unequal class counts make it clear whether stratification works.
        self.data = Dataset(
            np.arange(200, dtype=np.float32).reshape(100, 2),
            np.repeat([0, 1, 2], [50, 30, 20]),
            np.array([f"sample_{i:03d}" for i in range(100)]),
        )

    def test_split_stratification_coverage_and_alignment(self):
        split = split_dataset(self.data)
        self.assertEqual(len(split.train.X), 80)
        self.assertEqual(len(split.validation.X), 20)
        np.testing.assert_array_equal(np.bincount(split.validation.y), [10, 6, 4])
        np.testing.assert_array_equal(np.bincount(split.train.y), [40, 24, 16])
        self.assertEqual(len(np.intersect1d(split.train_indices, split.validation_indices)), 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([split.train_indices, split.validation_indices])),
            np.arange(100),
        )
        for subset, indices in ((split.train, split.train_indices),
                                (split.validation, split.validation_indices)):
            np.testing.assert_array_equal(subset.X, self.data.X[indices])
            np.testing.assert_array_equal(subset.y, self.data.y[indices])
            np.testing.assert_array_equal(subset.ids, self.data.ids[indices])

    def test_reproducible_even_after_row_reordering(self):
        first = split_dataset(self.data)
        second = split_dataset(self.data)
        np.testing.assert_array_equal(first.train_indices, second.train_indices)
        np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
        order = np.random.default_rng(123).permutation(100)
        shuffled = Dataset(self.data.X[order], self.data.y[order], self.data.ids[order])
        reordered = split_dataset(shuffled)
        np.testing.assert_array_equal(first.train.ids, reordered.train.ids)
        np.testing.assert_array_equal(first.validation.ids, reordered.validation.ids)
        self.assertFalse(np.array_equal(first.train.ids, split_dataset(self.data, seed=7).train.ids))

    def test_invalid_splits(self):
        for size in (0.0, 1.0, -0.2, 20):
            with self.assertRaisesRegex(ValueError, "validation_size"):
                split_dataset(self.data, validation_size=size)
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            split_dataset(Dataset(self.data.X, None, self.data.ids))
        with self.assertRaisesRegex(ValueError, "aligned"):
            split_dataset(Dataset(self.data.X[:-1], self.data.y, self.data.ids))
        with self.assertRaisesRegex(ValueError, "IDs must"):
            split_dataset(Dataset(self.data.X, self.data.y, np.repeat("same", 100)))
        with self.assertRaises(ValueError):
            split_dataset(Dataset(self.data.X[:3], np.array([0, 0, 1]), self.data.ids[:3]))

    def test_scaling_uses_training_only_and_does_not_mutate(self):
        split = split_dataset(self.data)
        split.train.X[:, 1] = 5  # Constant training feature.
        split.validation.X[:] = 10000  # Deliberately different validation distribution.
        original_train = split.train.X.copy()
        original_validation = split.validation.X.copy()
        scaled, scaler = standardize_split(split)
        # Large validation values must not affect the fitted mean or scale.
        np.testing.assert_allclose(scaler.mean_, original_train.mean(axis=0, dtype=np.float64))
        expected_scale = original_train.std(axis=0, dtype=np.float64)
        expected_scale[1] = 1
        np.testing.assert_allclose(scaler.scale_, expected_scale)
        np.testing.assert_allclose(scaled.validation.X, (original_validation - scaler.mean_) / expected_scale)
        np.testing.assert_allclose(scaled.train.X.mean(axis=0), 0, atol=1e-6)
        self.assertEqual(scaled.train.X.dtype, np.float32)
        np.testing.assert_array_equal(split.train.X, original_train)
        np.testing.assert_array_equal(split.validation.X, original_validation)
        old_mean, old_scale = scaler.mean_.copy(), scaler.scale_.copy()
        scaler.transform(np.full((2, 2), -5000))
        np.testing.assert_array_equal(scaler.mean_, old_mean)
        np.testing.assert_array_equal(scaler.scale_, old_scale)

    def test_standardizer_errors(self):
        with self.assertRaisesRegex(ValueError, "Fit the Standardizer"):
            Standardizer().transform(self.data.X)
        for X in (np.array([1, 2]), np.empty((0, 2))):
            with self.assertRaisesRegex(ValueError, "feature array"):
                Standardizer().fit(X)
        scaler = Standardizer().fit(self.data.X)
        with self.assertRaisesRegex(ValueError, "Feature count"):
            scaler.transform(np.ones((3, 4)))


class RecoverCountsTests(unittest.TestCase):
    def test_recovers_exact_integer_counts(self):
        counts = np.array([[1, 0, 3, 2], [0, 5, 1, 0]], dtype=float)
        rows = counts / np.linalg.norm(counts, axis=1, keepdims=True)
        np.testing.assert_array_equal(recover_counts(rows.astype(np.float32)), counts)

    def test_rejects_rows_that_are_not_counts(self):
        for X in (np.array([[0.5, 0.75]]), np.array([[0.0, 0.0]]), np.array([[-1.0, 1.0]]), np.zeros(3)):
            with self.subTest(X=X), self.assertRaises(ValueError):
                recover_counts(X)


if __name__ == "__main__":
    unittest.main()
