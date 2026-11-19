"""Synthetic tests for the neural network, experiment CSVs, and Kaggle submission merging.

No competition files are needed; data is generated to look like the real features.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.neural_network_experiment import EXPERIMENTS, experiment_config, run_experiment
from models.neural_network import InputPipeline, NeuralNetwork, predict_proba, train_model, word_dropout
from submission import make_submission
from utils.data_loader import Dataset
from utils.preprocessing import split_dataset


def count_rows(rows=60, features=500, seed=0):
    """L2-normalized sparse count rows (each with at least one count of 1), like the image data."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.3, size=(rows, features)).astype(np.float64)
    counts[np.arange(rows), rng.integers(0, features, rows)] = 1
    return (counts / np.linalg.norm(counts, axis=1, keepdims=True)).astype(np.float32), counts


CONFIG = {"hidden_sizes": [16], "activation": "GELU", "dropout": 0.0, "word_dropout": 0.0,
          "preprocessing": "standardized", "optimizer": "AdamW", "learning_rate": 0.01,
          "weight_decay": 0.0, "schedule": "constant", "batch_size": 8, "epochs": 30, "seed": 0}


class NetworkTests(unittest.TestCase):
    def test_output_shapes_for_both_tasks_and_layers(self):
        for size, classes, hidden in ((500, 50, [4096]), (768, 3, [256, 128])):
            model = NeuralNetwork(size, classes, hidden, dropout=0.5).eval()
            self.assertEqual(model(torch.zeros(4, size)).shape, (4, classes))

    def test_invalid_configuration(self):
        for options in ({"input_size": 0}, {"num_classes": 1}, {"hidden_sizes": []},
                        {"hidden_sizes": [0]}, {"dropout": 1.0}, {"activation": "tanh"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NeuralNetwork(**{"input_size": 2, "num_classes": 3, **options})


class WordDropoutTests(unittest.TestCase):
    def test_zero_probability_is_identity(self):
        counts = torch.tensor([[2.0, 0, 1], [0, 3, 0]])
        self.assertIs(word_dropout(counts, 0.0), counts)

    def test_only_removes_words_keeps_rows_nonempty_and_is_reproducible(self):
        counts = torch.from_numpy(count_rows()[1]).float()
        first = word_dropout(counts, 0.4, torch.Generator().manual_seed(1))
        second = word_dropout(counts, 0.4, torch.Generator().manual_seed(1))
        torch.testing.assert_close(first, second)
        self.assertTrue((first <= counts).all() and (first == first.round()).all())
        self.assertTrue((first.sum(dim=1) > 0).all())
        kept = first.sum() / counts.sum()
        self.assertGreater(kept.item(), 0.5)
        self.assertLess(kept.item(), 0.7)


class InputPipelineTests(unittest.TestCase):
    def test_counts_pipeline_reproduces_unit_rows_with_training_scale(self):
        X, _ = count_rows()
        train, validation = X[:40], X[40:]
        pipeline = InputPipeline("counts").fit(train)
        self.assertAlmostEqual(pipeline.count_scale, float(train.std()), places=5)
        inputs = pipeline.to_input(pipeline.encode(validation))
        np.testing.assert_allclose(inputs.numpy() * pipeline.count_scale, validation, atol=1e-6)

    def test_standardized_pipeline_uses_training_statistics_only(self):
        X = np.random.default_rng(3).normal(size=(50, 4)).astype(np.float32)
        pipeline = InputPipeline("standardized").fit(X[:30])
        np.testing.assert_allclose(pipeline.scaler.mean_, X[:30].mean(axis=0, dtype=np.float64))
        encoded = pipeline.encode(X[30:]).numpy()
        np.testing.assert_allclose(encoded, (X[30:] - pipeline.scaler.mean_) / pipeline.scaler.scale_, atol=1e-5)

    def test_invalid_kind_and_unfitted_use(self):
        with self.assertRaises(ValueError):
            InputPipeline("pca")
        with self.assertRaises(ValueError):
            InputPipeline("standardized").encode(np.zeros((2, 2)))


class TrainingTests(unittest.TestCase):
    def separable_data(self):
        rng = np.random.default_rng(0)
        y = np.tile([0, 1, 2], 20)
        X = rng.normal(scale=0.3, size=(60, 4)).astype(np.float32)
        X[np.arange(60), y] += 3
        return X, y

    def test_learns_separable_data_and_is_deterministic(self):
        X, y = self.separable_data()
        model, pipeline, history = train_model(CONFIG, X, y, 3, X, y, verbose=False)
        self.assertEqual(len(history), CONFIG["epochs"])
        self.assertEqual(history[-1]["val_accuracy"], 1.0)
        probabilities = predict_proba(model, pipeline, X)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)
        again = predict_proba(*train_model(CONFIG, X, y, 3, verbose=False)[:2], X)
        np.testing.assert_allclose(probabilities, again, rtol=1e-5)

    def test_onecycle_schedule_and_word_dropout_train_on_count_rows(self):
        X, _ = count_rows()
        y = np.arange(len(X)) % 3
        config = {**CONFIG, "preprocessing": "counts", "word_dropout": 0.4, "schedule": "onecycle",
                  "epochs": 3, "optimizer": "Adam"}
        _, _, history = train_model(config, X, y, 3, verbose=False)
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3])
        self.assertNotIn("val_accuracy", history[-1])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "neural_network"  # where save_run writes this algorithm's files
        X, _ = count_rows(rows=30)
        self.data = Dataset(X, np.tile([0, 1, 2], 10), np.array([f"img_train_{i:03d}" for i in range(30)]))

    def run_small(self, **overrides):
        return run_experiment("image", {"hidden_sizes": [8], "epochs": 2, **overrides},
                              results_dir=self.root, data=self.data, verbose=False)

    def test_run_appends_final_epoch_records_to_csvs(self):
        summary = self.run_small()
        history = pd.read_csv(self.folder / "history.csv")
        self.assertEqual(list(history.epoch), [1, 2])
        self.assertAlmostEqual(summary["val_accuracy"], history.val_accuracy.iloc[-1])
        saved = pd.read_csv(self.folder / "validation_predictions.csv")
        split = split_dataset(self.data)
        np.testing.assert_array_equal(saved.id, split.validation.ids)
        np.testing.assert_array_equal(saved.true_label, split.validation.y)
        self.assertAlmostEqual(np.mean(saved.true_label == saved.predicted_label), summary["val_accuracy"])

        # A second run adds rows to the same files; run_id links rows across them.
        second = self.run_small(word_dropout=0.0)
        experiments = pd.read_csv(self.folder / "experiments.csv")
        self.assertEqual(list(experiments.run_id), [summary["run_id"], second["run_id"]])
        self.assertEqual(list(experiments.word_dropout), [0.4, 0.0])
        self.assertEqual(set(pd.read_csv(self.folder / "history.csv").run_id), set(experiments.run_id))

    def test_ensemble_averages_one_model_per_seed_into_one_row(self):
        single = [self.run_small(seed=seed) for seed in (1, 2)]
        ensemble = run_experiment("image", {"hidden_sizes": [8], "epochs": 2}, seeds=[1, 2],
                                  results_dir=self.root, data=self.data, verbose=False)
        self.assertEqual(ensemble["seed"], "1 2")
        history = pd.read_csv(self.folder / "history.csv")
        self.assertEqual(list(history[history.run_id == ensemble["run_id"]].seed), [1, 1, 2, 2])
        # Each member trains exactly like the matching single-seed run.
        members = history[history.run_id == ensemble["run_id"]].groupby("seed").val_loss.last()
        self.assertAlmostEqual(members[1], single[0]["val_loss"], places=4)
        self.assertAlmostEqual(members[2], single[1]["val_loss"], places=4)
        self.assertEqual(len(pd.read_csv(self.folder / "experiments.csv")), 3)

    def test_planned_experiments_are_valid_and_uniquely_named(self):
        for dataset, experiments in EXPERIMENTS.items():
            self.assertEqual(len({name for name, _, _ in experiments}), len(experiments))
            for name, _, overrides in experiments:
                with self.subTest(dataset=dataset, experiment=name):
                    experiment_config(dataset, **overrides)

    def test_sentiment_defaults_run(self):
        rng = np.random.default_rng(8)
        data = Dataset(rng.normal(size=(30, 768)).astype(np.float32), self.data.y, self.data.ids)
        summary = run_experiment("sentiment", {"hidden_sizes": [8], "epochs": 1},
                                 results_dir=self.root, data=data, verbose=False)
        self.assertEqual(summary["preprocessing"], "standardized")

    def test_invalid_options(self):
        for changes in ({"epochs": 0}, {"learning_rate": -1}, {"batch_size": 0}, {"schedule": "step"},
                        {"optimizer": "sgd"}, {"preprocessing": "pca"}, {"word_dropout": 1.0},
                        {"preprocessing": "standardized"}, {"patience": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                experiment_config("image", **changes)


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
