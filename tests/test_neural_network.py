"""Check network dimensions and learning on a tiny synthetic problem."""

import unittest

import torch
from torch import nn

from models.neural_network import NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
    def test_both_task_shapes_and_class_ranges(self):
        for input_size, num_classes in ((500, 50), (768, 3)):
            with self.subTest(input_size=input_size):
                model = NeuralNetwork(input_size, num_classes)
                model.eval()
                with torch.no_grad():
                    scores = model(torch.zeros(4, input_size))
                self.assertEqual(scores.shape, (4, num_classes))
                self.assertTrue(torch.isfinite(scores).all().item())
                predictions = scores.argmax(dim=1)
                self.assertTrue(((predictions >= 0) & (predictions < num_classes)).all().item())

    def test_seed_reproduces_initialization(self):
        torch.manual_seed(42)
        first = NeuralNetwork(2, 3, hidden_size=8)
        torch.manual_seed(42)
        second = NeuralNetwork(2, 3, hidden_size=8)
        X = torch.tensor([[1.0, -1.0]])
        torch.testing.assert_close(first(X), second(X), rtol=0, atol=0)

    def test_invalid_layer_sizes(self):
        for sizes in ((0, 3, 8), (2, 1, 8), (2, 3, 0)):
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    NeuralNetwork(*sizes)

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
