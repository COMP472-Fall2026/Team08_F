"""Check network dimensions and learning on a tiny synthetic problem."""

import unittest

import torch
from torch import nn

from models.neural_network import NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
    def test_multiple_hidden_layers_and_dropout(self):
        model = NeuralNetwork(500, 50, hidden_sizes=[256, 128, 64], dropout=0.5)
        X = torch.ones(4, 500)
        model.eval()
        torch.testing.assert_close(model(X), model(X), rtol=0, atol=0)
        self.assertEqual(model(X).shape, (4, 50))

    def test_both_task_output_shapes(self):
        for input_size, num_classes in ((500, 50), (768, 3)):
            with self.subTest(input_size=input_size):
                model = NeuralNetwork(input_size, num_classes)
                model.eval()
                with torch.no_grad():
                    scores = model(torch.zeros(4, input_size))
                self.assertEqual(scores.shape, (4, num_classes))
                self.assertTrue(torch.isfinite(scores).all().item())

    def test_invalid_configuration(self):
        for options in ({"input_size": 0}, {"num_classes": 1},
                        {"hidden_sizes": []}, {"hidden_sizes": [0]}, {"dropout": 1.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NeuralNetwork(**{"input_size": 2, "num_classes": 3, **options})

    def test_can_learn_small_synthetic_dataset(self):
        torch.manual_seed(42)
        model = NeuralNetwork(2, 2, hidden_size=8)
        # Two easily separated groups; this verifies that gradients train the model.
        X = torch.tensor([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
        y = torch.tensor([0, 0, 1, 1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        initial_loss = criterion(model(X), y).item()

        model.train()
        for _ in range(40):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(X)
            final_loss = criterion(scores, y).item()
        self.assertLess(final_loss, initial_loss * 0.1)
        torch.testing.assert_close(scores.argmax(dim=1), y)


if __name__ == "__main__":
    unittest.main()
