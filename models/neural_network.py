"""Feed-forward neural network: the model, its input preprocessing, training and prediction.

Used by experiments/neural_network_experiment.py (validation experiments) and
submission.py (final Kaggle predictions).
"""

import math
import random

import numpy as np
import torch
from torch import Tensor, nn

from utils.preprocessing import Standardizer, recover_counts


class NeuralNetwork(nn.Module):
    """Fully connected network: input -> hidden layers -> one score per class.

    Each hidden layer is Linear -> activation -> dropout. For example,
    NeuralNetwork(500, 50, [4096]) is the image model and
    NeuralNetwork(768, 3, [256]) is the sentiment model.
    """

    def __init__(
        self, input_size: int, num_classes: int, hidden_sizes=(256,), *,
        dropout: float = 0.0, activation: str = "GELU",
    ) -> None:
        super().__init__()
        sizes = list(hidden_sizes)
        if input_size < 1 or num_classes < 2 or not sizes or any(size < 1 for size in sizes):
            raise ValueError("Input and hidden sizes must be positive; num_classes must be at least 2.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")
        activation_class = {"relu": nn.ReLU, "gelu": nn.GELU}.get(activation.lower())
        if activation_class is None:
            raise ValueError("activation must be ReLU or GELU.")

        layers, previous = [], input_size
        for size in sizes:
            layers += [nn.Linear(previous, size), activation_class()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # only active in training mode
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        # Raw scores (logits). The loss function applies softmax itself.
        return self.layers(X)


class InputPipeline:
    """Prepares features for the network. Fitted on training data only.

    "standardized" (sentiment): each feature becomes (x - mean) / std.
    "counts" (image): each row is a bag of visual words scaled to length 1. We
        recover the integer word counts so that word dropout can remove words
        during training, then scale each row back to length 1 before it enters
        the network.
    """

    KINDS = ("standardized", "counts")

    def __init__(self, kind: str) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"preprocessing must be one of {self.KINDS}.")
        self.kind = kind
        self.scaler: Standardizer | None = None
        self.count_scale: float | None = None

    def fit(self, X: np.ndarray) -> "InputPipeline":
        if self.kind == "standardized":
            self.scaler = Standardizer().fit(X)
        else:
            # Length-1 rows have very small values, so we also divide by their
            # overall standard deviation to bring the inputs to a typical scale.
            counts = torch.from_numpy(recover_counts(X))
            self.count_scale = float(_unit_rows(counts).std())
        return self

    def encode(self, X: np.ndarray) -> Tensor:
        """Convert features once: word counts for images, standardized values for sentiment."""
        if self.kind == "counts":
            return torch.from_numpy(recover_counts(X))
        if self.scaler is None:
            raise ValueError("Fit the InputPipeline before encoding data.")
        return torch.from_numpy(self.scaler.transform(X))

    def to_input(self, batch: Tensor) -> Tensor:
        """Turn an encoded batch into network input (images: back to scaled length-1 rows)."""
        if self.kind == "standardized":
            return batch
        if self.count_scale is None:
            raise ValueError("Fit the InputPipeline before converting batches.")
        return _unit_rows(batch) / self.count_scale


def _unit_rows(X: Tensor) -> Tensor:
    """Scale every row to length 1 (the clamp avoids dividing by zero)."""
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)


def word_dropout(counts: Tensor, p: float, generator: torch.Generator | None = None) -> Tensor:
    """Randomly remove each counted visual word with probability p (training batches only).

    For example, a word counted 5 times keeps between 0 and 5 copies. If a row
    would lose every word, it keeps its original counts instead.
    """
    if p == 0:
        return counts
    kept = torch.binomial(counts, torch.full_like(counts, 1 - p), generator=generator)
    empty = kept.sum(dim=1) == 0
    kept[empty] = counts[empty]
    return kept


def set_seed(seed: int) -> None:
    """Seed every random number generator so runs can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    config: dict, X: np.ndarray, y: np.ndarray, num_classes: int,
    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, *, verbose: bool = True,
) -> tuple[NeuralNetwork, InputPipeline, list[dict]]:
    """Train for config["epochs"] epochs and return (model, pipeline, history).

    history has one entry per epoch with the training loss and accuracy, plus the
    validation loss and accuracy if validation data is given. Validation data is
    only measured: it never affects training, and the model from the last epoch
    is the one returned.
    """
    set_seed(config["seed"])
    pipeline = InputPipeline(config["preprocessing"]).fit(X)  # training data only
    X_train, y_train = pipeline.encode(X), torch.tensor(y, dtype=torch.long)
    model = NeuralNetwork(
        X_train.shape[1], num_classes, config["hidden_sizes"],
        dropout=config["dropout"], activation=config["activation"],
    )
    optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[config["optimizer"].lower()]
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
    batch_size, epochs = config["batch_size"], config["epochs"]

    # One-cycle schedule: the learning rate rises to its maximum over the first
    # 10% of batches, then falls along a cosine curve to almost zero.
    scheduler = None
    if config["schedule"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config["learning_rate"],
            total_steps=epochs * math.ceil(len(X_train) / batch_size), pct_start=0.1,
        )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config["seed"])  # shuffling and word dropout
    validation = None if X_val is None else (pipeline.encode(X_val), torch.tensor(y_val, dtype=torch.long))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train), generator=generator)  # new shuffle each epoch
        for start in range(0, len(X_train), batch_size):
            index = order[start:start + batch_size]
            batch = word_dropout(X_train[index], config["word_dropout"], generator)
            loss = criterion(model(pipeline.to_input(batch)), y_train[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Measure the whole training set (without word dropout) and the validation set.
        row = {"epoch": epoch, **_metrics(model, pipeline, X_train, y_train, "train")}
        if validation is not None:
            row.update(_metrics(model, pipeline, *validation, "val"))
        history.append(row)
        if verbose:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    return model, pipeline, history


def _scores(model: NeuralNetwork, pipeline: InputPipeline, encoded: Tensor, batch_size=1024) -> Tensor:
    """Network scores for encoded data, in evaluation mode (dropout off)."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(pipeline.to_input(part)) for part in encoded.split(batch_size)])


def _metrics(model, pipeline, encoded: Tensor, y: Tensor, prefix: str) -> dict:
    """Loss and accuracy, e.g. {"val_loss": ..., "val_accuracy": ...}."""
    scores = _scores(model, pipeline, encoded)
    return {f"{prefix}_loss": nn.functional.cross_entropy(scores, y).item(),
            f"{prefix}_accuracy": (scores.argmax(dim=1) == y).float().mean().item()}


def predict_proba(model: NeuralNetwork, pipeline: InputPipeline, X: np.ndarray) -> np.ndarray:
    """Class probabilities for loaded features: one row per example, one column per class."""
    return torch.softmax(_scores(model, pipeline, pipeline.encode(X)), dim=1).numpy()
