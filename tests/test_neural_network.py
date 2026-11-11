"""Synthetic tests for the network, input pipeline, word dropout, and training loop."""

import unittest

import numpy as np
import torch

from models.neural_network import InputPipeline, NeuralNetwork, predict_proba, train_model, word_dropout


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


if __name__ == "__main__":
    unittest.main()
