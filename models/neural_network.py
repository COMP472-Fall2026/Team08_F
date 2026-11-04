"""A simple feed-forward classifier for the provided numerical features."""

from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """One hidden layer with ReLU, followed by one score per class.

    Create separate models for the two tasks:
        image_model = NeuralNetwork(input_size=500, num_classes=50)
        sentiment_model = NeuralNetwork(input_size=768, num_classes=3)

    hidden_size=256 is a baseline to investigate, not a required architecture.
    Set torch.manual_seed(42) before construction for repeatable initialization.
    Training and preprocessing belong in the experiment script.
    """

    def __init__(
        self, input_size: int, num_classes: int, hidden_size: int = 256
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_classes < 2:
            raise ValueError(
                "input_size and hidden_size must be positive; num_classes must be at least 2."
            )

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),  # Nonlinearity lets the network learn more than a linear mapping.
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, X: Tensor) -> Tensor:
        """Map float32 features (batch, input_size) to scores (batch, num_classes).

        Pass these raw scores (logits) directly to nn.CrossEntropyLoss, which
        handles the probability calculation internally. For predicted labels,
        use model.eval() and torch.no_grad(), then model(X).argmax(dim=1).
        """
        return self.layers(X)
