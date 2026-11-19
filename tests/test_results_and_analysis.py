"""Tests for the shared results format (utils/results.py) and the analysis script, on made-up results."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments import analysis
from utils.results import clear_results, read_results, save_run


class SaveRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_runs_and_predictions_are_appended_and_linked(self):
        ids, y = np.array(["a", "b", "c"]), np.array([0, 1, 2])
        first = save_run("knn", "sentiment", "k", "k=1", {"k": 1, "distance": "cosine"}, {"val_accuracy": 1.0},
                         ids=ids, y_true=y, y_pred=y, results_dir=self.root)
        second = save_run("knn", "sentiment", "k", "k=3", {"k": 3, "distance": "cosine"}, {"val_accuracy": 0.5},
                          ids=ids, y_true=y, y_pred=np.array([0, 0, 0]), results_dir=self.root)
        runs = pd.read_csv(self.root / "knn" / "experiments.csv")
        self.assertEqual(list(runs.run_id), [first, second])
        self.assertEqual(list(runs.columns[:4]), ["run_id", "dataset", "sweep", "experiment"])
        predictions = pd.read_csv(self.root / "knn" / "validation_predictions.csv")
        self.assertEqual(len(predictions), 6)
        self.assertEqual(set(predictions.run_id), {first, second})

    def test_lists_and_none_are_stored_as_text(self):
        save_run("neural_network", "image", "depth", "d", {"hidden_sizes": [8, 8], "max_depth": None},
                 {"val_accuracy": 0.1}, results_dir=self.root)
        row = read_results(self.root / "neural_network" / "experiments.csv").iloc[0]
        self.assertEqual((row.hidden_sizes, row.max_depth), ("[8, 8]", "None"))

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            save_run("knn", "image", "k", "x", {"k": 1}, {"accuracy": 0.5}, results_dir=self.root)

    def test_clear_results_starts_empty(self):
        save_run("knn", "image", "k", "x", {"k": 1}, {"val_accuracy": 0.5}, results_dir=self.root)
        folder = clear_results("knn", self.root)
        self.assertEqual(list(folder.iterdir()), [])


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.default_rng(0)
        ids = np.array([f"t{i}" for i in range(30)])
        y = np.tile([0, 1, 2], 10)
        # Two classifiers with a sweep each, plus a clustering result.
        for k in (1, 3, 5):
            for distance in ("euclidean", "cosine"):
                predicted = np.where(rng.random(30) < 0.6, y, (y + 1) % 3)
                save_run("knn", "sentiment", "k", f"k={k} {distance}", {"k": k, "distance": distance},
                         {"val_accuracy": float(np.mean(predicted == y))},
                         ids=ids, y_true=y, y_pred=predicted, results_dir=self.root)
        for name, sweep, size in (("baseline", "baseline", [8]), ("wide", "width", [16]), ("deep", "depth", [8, 8])):
            predicted = np.where(rng.random(30) < 0.8, y, (y + 2) % 3)
            save_run("neural_network", "sentiment", sweep, name, {"hidden_sizes": size},
                     {"train_accuracy": 0.9, "val_accuracy": float(np.mean(predicted == y))},
                     ids=ids, y_true=y, y_pred=predicted, results_dir=self.root)
        for k in (3, 6):
            save_run("kmeans", "sentiment", "k", f"k={k}", {"k": k}, {"purity": 0.4, "wcss": 100.0 / k},
                     results_dir=self.root)
        pd.DataFrame({"k": [3, 3, 3], "cluster": [0, 1, 2], "size": [10, 10, 10], "negative": [5, 3, 2],
                      "neutral": [3, 4, 3], "positive": [2, 3, 5]}).to_csv(
            self.root / "kmeans" / "cluster_composition.csv", index=False)

    def test_every_algorithm_gets_figures_and_the_classifiers_are_compared(self):
        with mock.patch.object(analysis, "RESULTS_DIR", self.root), mock.patch("sys.argv", ["analysis.py"]):
            analysis.main()
        knn = self.root / "knn" / "figures"
        self.assertTrue((knn / "sentiment_k.png").exists())
        self.assertTrue((knn / "sentiment_confusion.png").exists())
        self.assertEqual(len(pd.read_csv(knn / "sentiment_per_class.csv")), 3)
        network = self.root / "neural_network" / "figures"
        self.assertTrue((network / "sentiment_width.png").exists())   # the baseline is added to each sweep
        self.assertTrue((network / "sentiment_depth.png").exists())
        kmeans = self.root / "kmeans" / "figures"
        self.assertTrue((kmeans / "sentiment_k.png").exists())
        self.assertTrue((kmeans / "cluster_composition_k3.png").exists())

        best = pd.read_csv(self.root / "comparison" / "final_validation_accuracy.csv")
        self.assertEqual(set(best.algorithm), {"knn", "neural_network"})   # clustering has no accuracy
        overlap = pd.read_csv(self.root / "comparison" / "error_overlap.csv").iloc[0]
        parts = overlap[["both_correct", "only_a_wrong", "only_b_wrong", "both_wrong"]].sum()
        self.assertAlmostEqual(parts, 1.0, places=3)

    def test_sweep_values_are_ordered_numerically_with_none_last(self):
        values = ["None", "12", "4", "[4096, 4096]", "[512]", "sqrt"]
        self.assertEqual(sorted(values, key=analysis.sort_key), ["4", "12", "[512]", "[4096, 4096]", "None", "sqrt"])


if __name__ == "__main__":
    unittest.main()
