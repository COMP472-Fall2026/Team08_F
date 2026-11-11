"""Feed-forward classifier plus the training and prediction code shared by
experiments and the final Kaggle submission."""

import random

import numpy as np
import torch
from torch import Tensor, nn

from utils.preprocessing import Standardizer, recover_counts


class NeuralNetwork(nn.Module):
    """Fully connected layers: input -> hidden layers -> one score per class.

    Create separate models for the two tasks, e.g. NeuralNetwork(500, 50) for
    images and NeuralNetwork(768, 3) for sentiment. Dropout is applied after
    each hidden activation during training only.
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
        self.activation_name = activation_class.__name__

        layers, previous = [], input_size
        for size in sizes:
            layers += [nn.Linear(previous, size), activation_class()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """Return raw class scores (logits); nn.CrossEntropyLoss applies softmax."""
        return self.layers(X)


class InputPipeline:
    """Turn loaded features into network inputs, using training statistics only.

    "standardized": per-feature (x - mean) / std, used for sentiment.
    "counts": image rows are L2-normalized visual-word counts. Rows are stored as
        integer counts so word dropout can remove words; each (possibly thinned)
        row is then re-normalized to length 1 and divided by one global scale.
    "none": features are used unchanged.
    """

    KINDS = ("standardized", "counts", "none")

    def __init__(self, kind: str) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"preprocessing must be one of {self.KINDS}.")
        self.kind = kind
        self.scaler: Standardizer | None = None
        self.count_scale: float | None = None

    def fit(self, X: np.ndarray) -> "InputPipeline":
        if self.kind == "standardized":
            self.scaler = Standardizer().fit(X)
        elif self.kind == "counts":
            counts = torch.from_numpy(recover_counts(X))
            self.count_scale = float(_unit_rows(counts).std())
        return self

    def encode(self, X: np.ndarray) -> Tensor:
        """Stored representation: counts for "counts", final inputs otherwise."""
        if self.kind == "counts":
            return torch.from_numpy(recover_counts(X))
        if self.kind == "standardized":
            if self.scaler is None:
                raise ValueError("Fit the InputPipeline before encoding data.")
            X = self.scaler.transform(X)
        return torch.tensor(X, dtype=torch.float32)

    def to_input(self, batch: Tensor) -> Tensor:
        """Convert an encoded batch (after any augmentation) to network input."""
        if self.kind != "counts":
            return batch
        if self.count_scale is None:
            raise ValueError("Fit the InputPipeline before converting batches.")
        return _unit_rows(batch) / self.count_scale


def _unit_rows(X: Tensor) -> Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)


def word_dropout(counts: Tensor, p: float, generator: torch.Generator | None = None) -> Tensor:
    """Remove each counted visual word independently with probability p.

    A row with count 5 keeps Binomial(5, 1 - p) of that word. Rows that would
    become empty keep their original counts. Used on training batches only.
    """
    if p == 0:
        return counts
    kept = torch.binomial(counts, torch.full_like(counts, 1 - p), generator=generator)
    empty = kept.sum(dim=1) == 0
    kept[empty] = counts[empty]
    return kept


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    config: dict, X: np.ndarray, y: np.ndarray, num_classes: int,
    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, *, verbose: bool = True,
) -> tuple[NeuralNetwork, InputPipeline, list[dict]]:
    """Train for a fixed number of epochs and return (model, pipeline, history).

    Preprocessing is fitted on X only. When validation data is given, both
    subsets are evaluated after every epoch; validation never updates weights
    or chooses an epoch, so the reported model is always the final epoch.
    """
    set_seed(config["seed"])
    pipeline = InputPipeline(config["preprocessing"]).fit(X)
    X_train, y_train = pipeline.encode(X), torch.tensor(y, dtype=torch.long)
    model = NeuralNetwork(
        X_train.shape[1], num_classes, config["hidden_sizes"],
        dropout=config["dropout"], activation=config["activation"],
    )
    optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[config["optimizer"].lower()]
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
    batch_size, epochs = config["batch_size"], config["epochs"]
    steps_per_epoch = -(-len(X_train) // batch_size)
    # One-cycle: warm up to the learning rate over the first 10% of steps, then
    # decrease it along a cosine curve to almost zero by the final step.
    scheduler = None
    if config["schedule"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config["learning_rate"], total_steps=epochs * steps_per_epoch, pct_start=0.1,
        )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config["seed"])
    validation = None if X_val is None else (pipeline.encode(X_val), torch.tensor(y_val, dtype=torch.long))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), batch_size):
            index = order[start:start + batch_size]
            batch = word_dropout(X_train[index], config["word_dropout"], generator)
            loss = criterion(model(pipeline.to_input(batch)), y_train[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        row = {"epoch": epoch, **_metrics(model, pipeline, X_train, y_train, "train")}
        if validation is not None:
            row.update(_metrics(model, pipeline, *validation, "val"))
        history.append(row)
        if verbose:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    return model, pipeline, history


def _scores(model: NeuralNetwork, pipeline: InputPipeline, encoded: Tensor, batch_size=1024) -> Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(pipeline.to_input(part)) for part in encoded.split(batch_size)])


def _metrics(model, pipeline, encoded: Tensor, y: Tensor, prefix: str) -> dict:
    scores = _scores(model, pipeline, encoded)
    return {f"{prefix}_loss": nn.functional.cross_entropy(scores, y).item(),
            f"{prefix}_accuracy": (scores.argmax(dim=1) == y).float().mean().item()}


def predict_proba(model: NeuralNetwork, pipeline: InputPipeline, X: np.ndarray) -> np.ndarray:
    """Class probabilities for raw loaded features, one row per sample."""
    return torch.softmax(_scores(model, pipeline, pipeline.encode(X)), dim=1).numpy()
