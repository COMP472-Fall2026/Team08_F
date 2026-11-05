"""A simple feed-forward classifier for the provided numerical features."""

from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """Configurable hidden layers with ReLU, followed by one score per class.

    Create separate models for the two tasks:
        image_model = NeuralNetwork(input_size=500, num_classes=50)
        sentiment_model = NeuralNetwork(input_size=768, num_classes=3)

    hidden_sizes=[256] is the baseline; try [256, 128] for two hidden layers.
    hidden_size is retained for compatibility with the original single layer.
    Set torch.manual_seed(42) before construction for repeatable initialization.
    Training and preprocessing belong in the experiment script.
    """

    def __init__(
        self, input_size: int, num_classes: int, hidden_size: int = 256,
        *, hidden_sizes: list[int] | None = None, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        sizes = [hidden_size] if hidden_sizes is None else list(hidden_sizes)
        if input_size < 1 or num_classes < 2 or not sizes or any(size < 1 for size in sizes):
            raise ValueError(
                "Input and hidden sizes must be positive; num_classes must be at least 2."
            )

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")

        layers = []
        previous_size = input_size
        for size in sizes:
            layers.extend([nn.Linear(previous_size, size), nn.ReLU()])
            # Dropout randomly disables hidden activations during training only.
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_size = size
        layers.append(nn.Linear(previous_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """Map float32 features (batch, input_size) to scores (batch, num_classes).

        Pass these raw scores (logits) directly to nn.CrossEntropyLoss, which
        handles the probability calculation internally. For predicted labels,
        use model.eval() and torch.no_grad(), then model(X).argmax(dim=1).
        """
        return self.layers(X)
