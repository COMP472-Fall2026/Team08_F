"""Synthetic integration tests for training, saved results, and evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.train_neural_network import (
    evaluate, experiment_config, run_experiment,
)
from experiments.plot_neural_network import plot_comparison, plot_run
from models.neural_network import NeuralNetwork
from utils.data_loader import Dataset
from utils.evaluation import classification_metrics
from utils.preprocessing import split_dataset


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Full expected feature dimension, but only 30 artificial labeled samples.
        rng = np.random.default_rng(7)
        self.data = Dataset(rng.normal(size=(30, 500)).astype(np.float32),
                            np.tile([0, 1, 2], 10), np.array([f"sample_{i}" for i in range(30)]))

    def run_small(self, **overrides):
        return run_experiment("image", experiment_config(
            hidden_sizes=[8], batch_size=7, max_epochs=3, patience=2, **overrides,
        ), results_dir=self.root, data=self.data)

    def test_preset_and_explicit_overrides(self):
        config = experiment_config("dropout_0.2", learning_rate=0.01)
        self.assertEqual(config["dropout"], 0.2)
        self.assertEqual(config["learning_rate"], 0.01)
        self.assertEqual(config["hidden_sizes"], [256])

    def test_evaluation_weights_batches_and_does_not_update_parameters(self):
        model = nn.Linear(2, 3)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        X = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        y = torch.tensor([0, 1, 2, 0, 1])
        state = {key: value.clone() for key, value in model.state_dict().items()}
        result = evaluate(model, DataLoader(TensorDataset(X, y), batch_size=3))
        self.assertAlmostEqual(result["loss"], np.log(3), places=6)
        self.assertEqual(result["accuracy"], 0.4)
        np.testing.assert_array_equal(result["y_true"], y.numpy())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_early_stopping_and_checkpoint_summary(self):
        # Freeze updates: accuracy ties after epoch 1 must trigger patience=2.
        with patch("torch.optim.Adam.step"):
            summary = self.run_small()
        self.assertEqual(summary["best_epoch"], 1)
        self.assertEqual(summary["epochs_trained"], 3)

    def verify_saved_run(self, summary):
        run_id = summary["run_id"]
        history = pd.read_csv(self.root / "epoch_history.csv")
        selected = history[(history.run_id == run_id) & (history.epoch == summary["best_epoch"])].iloc[0]
        for metric in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            self.assertAlmostEqual(summary[metric], selected[metric], places=6)
        checkpoint = torch.load(self.root / "checkpoints" / f"{run_id}.pt", weights_only=True)
        self.assertEqual(checkpoint["best_epoch"], summary["best_epoch"])
        split = split_dataset(self.data)
        np.testing.assert_allclose(checkpoint["scaler_mean"].numpy(), split.train.X.mean(axis=0, dtype=np.float64))
        X = ((split.validation.X - checkpoint["scaler_mean"].numpy()) /
             checkpoint["scaler_scale"].numpy()).astype(np.float32)
        model = NeuralNetwork(500, 50, hidden_sizes=[8])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(X)).argmax(dim=1).numpy()
        folder = self.root / "runs" / run_id
        saved = pd.read_csv(folder / "validation_predictions.csv")
        np.testing.assert_array_equal(saved.id, split.validation.ids)
        np.testing.assert_array_equal(saved.true_label, split.validation.y)
        np.testing.assert_array_equal(saved.predicted_label, predictions)
        self.assertAlmostEqual(np.mean(predictions == split.validation.y), summary["val_accuracy"])
        self.assertFalse((folder / "per_class_metrics.csv").exists())
        self.assertFalse((folder / "confusion_matrix.csv").exists())
        self.assertFalse((folder / "figures").exists())
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(config["split_seed"], 42)
        self.assertEqual(len(pd.read_csv(folder / "split_ids.csv")), len(self.data.ids))

    def test_completed_runs_append_without_overwriting_and_plots_work(self):
        first = self.run_small()
        self.verify_saved_run(first)
        original = (self.root / "checkpoints" / f"{first['run_id']}.pt").read_bytes()
        second = self.run_small(seed=7)
        self.verify_saved_run(second)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual((self.root / "checkpoints" / f"{first['run_id']}.pt").read_bytes(), original)
        table = pd.read_csv(self.root / "experiments.csv")
        self.assertEqual(len(table), 2)
        folder = self.root / "runs" / first["run_id"]
        # Analysis must work with only history and predictions, preserving core records.
        core_files = [folder / "validation_predictions.csv", self.root / "experiments.csv",
                      self.root / "epoch_history.csv"]
        originals = {path: path.read_bytes() for path in core_files}
        paths = plot_run(first["run_id"], self.root)
        for path in paths:
            self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(len(pd.read_csv(folder / "per_class_metrics.csv")), 50)
        matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0)
        self.assertEqual(matrix.to_numpy().sum(), len(split_dataset(self.data).validation.ids))
        per_class = pd.read_csv(folder / "per_class_metrics.csv")
        self.assertEqual(per_class.support.sum(), matrix.to_numpy().sum())
        # Repeating analysis must leave both earlier analysis and core records intact.
        originals.update({path: path.read_bytes() for path in paths})
        plot_run(first["run_id"], self.root)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        path = plot_comparison("image", "learning_rate", self.root)
        self.assertGreater(path.stat().st_size, 0)

    def test_classification_metrics(self):
        metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1], 3)
        self.assertEqual(metrics["accuracy"], 0.5)
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics["per_class"].loc[1, "precision"], 1 / 3)
        self.assertEqual(metrics["per_class"].loc[2, "recall"], 0)

    def test_sentiment_without_scaling(self):
        rng = np.random.default_rng(8)
        data = Dataset(rng.normal(size=(30, 768)).astype(np.float32),
                       self.data.y, self.data.ids)
        config = experiment_config(hidden_sizes=[8], max_epochs=1, preprocessing="none", dropout=0.2)
        summary = run_experiment("sentiment", config, data=data, results_dir=self.root)
        checkpoint = torch.load(self.root / "checkpoints" / f"{summary['run_id']}.pt", weights_only=True)
        self.assertIsNone(checkpoint["scaler_mean"])
        self.assertIsNone(checkpoint["scaler_scale"])
        self.assertEqual(summary["epochs_trained"], 1)
        # Saved settings should match the run, regardless of the selected configuration.
        for key, value in config.items():
            self.assertEqual(checkpoint["config"][key], value)

    def test_basic_invalid_training_options(self):
        for changes in ({"max_epochs": 0}, {"patience": 0}, {"learning_rate": -1},
                        {"batch_size": 0}, {"preprocessing": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_experiment("image", changes, data=self.data, results_dir=self.root)


if __name__ == "__main__":
    unittest.main()
