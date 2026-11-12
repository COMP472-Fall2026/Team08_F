"""Synthetic integration tests: experiment records, plots, metrics, and submission merging."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.plot_neural_network import plot_comparison, plot_run
from experiments.train_neural_network import experiment_config, run_experiment
from submission import make_submission
from utils.data_loader import Dataset
from utils.evaluation import classification_metrics
from utils.preprocessing import split_dataset


def count_rows(rows, features=500, seed=0):
    """L2-normalized sparse count rows, each containing a count of 1."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.3, size=(rows, features)).astype(np.float64)
    counts[np.arange(rows), rng.integers(0, features, rows)] = 1
    return (counts / np.linalg.norm(counts, axis=1, keepdims=True)).astype(np.float32), counts


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        X, _ = count_rows(rows=30)
        self.data = Dataset(X, np.tile([0, 1, 2], 10), np.array([f"img_train_{i:03d}" for i in range(30)]))

    def run_small(self, **overrides):
        return run_experiment("image", {"hidden_sizes": [8], "epochs": 2, **overrides},
                              results_dir=self.root, data=self.data)

    def test_run_saves_final_epoch_records_and_plots(self):
        summary = self.run_small()
        folder = self.root / "runs" / summary["run_id"]
        history = pd.read_csv(folder / "history.csv")
        self.assertEqual(list(history.epoch), [1, 2])
        self.assertAlmostEqual(summary["val_accuracy"], history.val_accuracy.iloc[-1])
        saved = pd.read_csv(folder / "validation_predictions.csv")
        split = split_dataset(self.data)
        np.testing.assert_array_equal(saved.id, split.validation.ids)
        np.testing.assert_array_equal(saved.true_label, split.validation.y)
        self.assertAlmostEqual(np.mean(saved.true_label == saved.predicted_label), summary["val_accuracy"])
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual((config["split_seed"], config["word_dropout"]), (42, 0.4))

        self.run_small(word_dropout=0.0)
        self.assertEqual(len(pd.read_csv(self.root / "experiments.csv")), 2)
        for path in plot_run(summary["run_id"], self.root):
            self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(len(pd.read_csv(folder / "per_class_metrics.csv")), 50)
        self.assertGreater(plot_comparison("image", "word_dropout", self.root).stat().st_size, 0)

    def test_sentiment_defaults_run(self):
        rng = np.random.default_rng(8)
        data = Dataset(rng.normal(size=(30, 768)).astype(np.float32), self.data.y, self.data.ids)
        summary = run_experiment("sentiment", {"hidden_sizes": [8], "epochs": 1},
                                 results_dir=self.root, data=data)
        self.assertEqual(summary["preprocessing"], "standardized")

    def test_invalid_options(self):
        for changes in ({"epochs": 0}, {"learning_rate": -1}, {"batch_size": 0}, {"schedule": "step"},
                        {"optimizer": "sgd"}, {"preprocessing": "pca"}, {"word_dropout": 1.0},
                        {"preprocessing": "standardized"}, {"patience": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                experiment_config("image", **changes)

    def test_classification_metrics(self):
        metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1], 3)
        self.assertEqual(metrics["accuracy"], 0.5)
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics["per_class"].loc[1, "precision"], 1 / 3)


class SubmissionTests(unittest.TestCase):
    sample = pd.DataFrame({"id": ["text_1", "img_0", "text_0", "img_1"]})

    def predictions(self, image_labels=(49, 3), sentiment_labels=(2, 0)):
        return {"image": pd.DataFrame({"id": ["img_0", "img_1"], "label": list(image_labels)}),
                "sentiment": pd.DataFrame({"id": ["text_0", "text_1"], "label": list(sentiment_labels)})}

    def test_merges_by_id_in_template_order(self):
        submission = make_submission(self.predictions(), self.sample)
        self.assertEqual(list(submission.columns), ["id", "label"])
        self.assertEqual(list(submission.id), list(self.sample.id))
        self.assertEqual(list(submission.label), [0, 49, 2, 3])

    def test_rejects_missing_extra_duplicate_and_out_of_range(self):
        bad = [
            {**self.predictions(), "sentiment": pd.DataFrame({"id": ["text_0"], "label": [1]})},
            {**self.predictions(), "extra": pd.DataFrame({"id": ["img_9"], "label": [1]})},
            {**self.predictions(), "image": pd.DataFrame({"id": ["img_0", "img_0", "img_1"], "label": [1, 1, 1]})},
            self.predictions(image_labels=(50, 0)),
            self.predictions(sentiment_labels=(3, 0)),
        ]
        for predictions in bad:
            with self.subTest(predictions=predictions), self.assertRaises(Exception):
                make_submission(predictions, self.sample)


if __name__ == "__main__":
    unittest.main()
