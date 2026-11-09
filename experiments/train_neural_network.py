"""Run one NN experiment: python -m experiments.train_neural_network --dataset image."""

import argparse
import json
from pathlib import Path
import platform
import random
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.neural_network import NeuralNetwork
from utils.data_loader import DATASET_SPECS, Dataset, load_dataset
from utils.preprocessing import split_dataset, standardize_split


BASELINE = {
    "hidden_sizes": [256], "learning_rate": 0.001, "batch_size": 64,
    "dropout": 0.0, "weight_decay": 0.0, "max_epochs": 50,
    "patience": 10, "seed": 42, "preprocessing": "standardized",
    "activation": "ReLU",
}
# Each preset changes only the stated baseline setting; no automatic grid runs.
PRESETS = {
    "baseline": {},
    "lr_0.01": {"learning_rate": 0.01},
    "lr_0.0001": {"learning_rate": 0.0001},
    "hidden_64": {"hidden_sizes": [64]},
    "hidden_128": {"hidden_sizes": [128]},
    "hidden_256_128": {"hidden_sizes": [256, 128]},
    "hidden_256_128_64": {"hidden_sizes": [256, 128, 64]},
    "dropout_0.2": {"dropout": 0.2},
    "dropout_0.5": {"dropout": 0.5},
    "weight_decay_0.0001": {"weight_decay": 0.0001},
}
SUMMARY_FIELDS = [
    "run_id", "dataset", "hidden_sizes", "activation", "learning_rate",
    "batch_size", "optimizer", "dropout", "weight_decay", "preprocessing",
    "seed", "max_epochs", "epochs_trained", "best_epoch", "train_accuracy",
    "val_accuracy", "train_loss", "val_loss", "training_time_seconds", "parameter_count",
]
HISTORY_FIELDS = ["run_id", "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]


def experiment_config(preset="baseline", **overrides) -> dict:
    """Get baseline + named changes + optional explicit overrides."""
    config = {**BASELINE, **PRESETS[preset], **overrides}
    config["hidden_sizes"] = list(config["hidden_sizes"])
    return config


def append_row(path: Path, fields: list[str], row: dict) -> None:
    """Append one record; run experiments sequentially in a results directory."""
    new_file = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row], columns=fields).to_csv(
        path, mode="a", header=new_file, index=False,
    )


def evaluate(model, loader) -> dict:
    """Measure a fixed model on all examples, without dropout or gradients.

    Use summed losses divided by sample count so a smaller last batch has the
    correct weight. Evaluation loaders must not shuffle, preserving ID order.
    """
    model.eval()
    total_loss = 0.0
    labels, predictions = [], []
    with torch.no_grad():
        for X, y in loader:
            scores = model(X)
            total_loss += nn.functional.cross_entropy(scores, y, reduction="sum").item()
            labels.append(y)
            predictions.append(scores.argmax(dim=1))
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(predictions).numpy()
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
        "y_true": y_true, "y_pred": y_pred,
    }


def run_experiment(
    dataset: str, config: dict | None = None, *, data_dir="data",
    results_dir="results/neural_network", data: Dataset | None = None,
) -> dict:
    """Train one run and return its best-checkpoint summary.

    Normally load the genuine training CSV. The optional data argument lets
    synthetic tests exercise the entire pipeline without competition files.
    Training runs on CPU. Only the labeled training CSV is read.
    """
    config = experiment_config(**(config or {}))
    for key in ("batch_size", "max_epochs", "patience"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if config["preprocessing"] not in ("standardized", "none"):
        raise ValueError("preprocessing must be 'standardized' or 'none'.")
    input_size, num_classes, _, _ = DATASET_SPECS[dataset]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    data = load_dataset(dataset, data_dir=data_dir) if data is None else data
    # Split seed is always 42, independent of the training/initialization seed.
    split = split_dataset(data, validation_size=0.2, seed=42)
    scaler = None
    if config["preprocessing"] == "standardized":
        split, scaler = standardize_split(split)

    def tensor_data(subset):
        return TensorDataset(
            torch.as_tensor(subset.X, dtype=torch.float32),
            torch.as_tensor(subset.y, dtype=torch.long),
        )

    train_data, val_data = tensor_data(split.train), tensor_data(split.validation)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, generator=generator)
    train_eval_loader = DataLoader(train_data, batch_size=config["batch_size"])
    val_loader = DataLoader(val_data, batch_size=config["batch_size"])
    model = NeuralNetwork(
        input_size, num_classes, hidden_sizes=config["hidden_sizes"], dropout=config["dropout"],
        activation=config["activation"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    root = Path(results_dir)
    run_id = f"{dataset}_{uuid4().hex}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    metadata = {
        **config, "dataset": dataset, "run_id": run_id, "device": "cpu",
        "split_seed": 42, "validation_size": 0.2, "activation": model.activation_name, "optimizer": "Adam",
        "selection_rule": "highest validation accuracy; earliest epoch wins ties",
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame({"id": np.concatenate([split.train.ids, split.validation.ids]),
                  "subset": ["train"] * len(split.train.ids) + ["validation"] * len(split.validation.ids)
                  }).to_csv(run_dir / "split_ids.csv", index=False)

    best_accuracy, stale_epochs = -1.0, 0
    start = perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()  # Re-enable dropout for gradient updates.
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        # Compare train and validation using the same post-epoch evaluation.
        train_metrics = evaluate(model, train_eval_loader)
        val_metrics = evaluate(model, val_loader)
        row = {"run_id": run_id, "epoch": epoch,
               "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"]}
        append_row(root / "epoch_history.csv", HISTORY_FIELDS, row)
        print(f"{run_id} epoch {epoch}: train loss={row['train_loss']:.4f} "
              f"acc={row['train_accuracy']:.4f}; val loss={row['val_loss']:.4f} "
              f"acc={row['val_accuracy']:.4f}", flush=True)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch, stale_epochs = val_metrics["accuracy"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": metadata, "best_epoch": best_epoch,
                "scaler_mean": None if scaler is None else torch.from_numpy(scaler.mean_),
                "scaler_scale": None if scaler is None else torch.from_numpy(scaler.scale_),
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break
    training_time = perf_counter() - start

    # Restore selected weights before reporting metrics or saving predictions.
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = evaluate(model, train_eval_loader)
    val_metrics = evaluate(model, val_loader)
    pd.DataFrame({"id": split.validation.ids, "true_label": val_metrics["y_true"],
                  "predicted_label": val_metrics["y_pred"]}).to_csv(
        run_dir / "validation_predictions.csv", index=False,
    )
    summary = {key: config[key] for key in (
        "learning_rate", "batch_size", "dropout", "weight_decay", "preprocessing", "seed", "max_epochs",
    )}
    summary.update(
        run_id=run_id, dataset=dataset, hidden_sizes=json.dumps(config["hidden_sizes"]),
        activation=model.activation_name, optimizer="Adam", epochs_trained=epoch, best_epoch=best_epoch,
        train_accuracy=train_metrics["accuracy"], val_accuracy=val_metrics["accuracy"],
        train_loss=train_metrics["loss"], val_loss=val_metrics["loss"],
        training_time_seconds=training_time,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    append_row(root / "experiments.csv", SUMMARY_FIELDS, summary)
    print(f"Completed {run_id}; selected epoch {best_epoch}, validation accuracy {best_accuracy:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_SPECS, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="baseline")
    parser.add_argument("--hidden-sizes", type=int, nargs="+")
    parser.add_argument("--activation", type=str.lower, choices=("relu", "gelu"))
    for name, value_type in (("learning_rate", float), ("weight_decay", float), ("dropout", float),
                             ("batch_size", int), ("max_epochs", int), ("patience", int), ("seed", int)):
        parser.add_argument("--" + name.replace("_", "-"), type=value_type)
    parser.add_argument("--preprocessing", choices=("standardized", "none"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/neural_network")
    args = vars(parser.parse_args())
    options = {key: args.pop(key) for key in ("dataset", "data_dir", "results_dir")}
    preset = args.pop("preset")
    config = experiment_config(preset, **{key: value for key, value in args.items() if value is not None})
    run_experiment(config=config, **options)


if __name__ == "__main__":
    main()
