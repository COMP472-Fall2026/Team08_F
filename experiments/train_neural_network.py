"""Run one NN experiment on the shared split: python -m experiments.train_neural_network --dataset image.

Override any default with a flag, e.g. --word-dropout 0 or --hidden-sizes 512.
"""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from models.neural_network import InputPipeline, predict_proba, train_model
from utils.data_loader import DATASET_SPECS, Dataset, load_dataset
from utils.preprocessing import split_dataset


# Selected settings for each task; experiments change one setting at a time.
DEFAULTS = {
    "image": {
        "hidden_sizes": [4096], "activation": "GELU", "dropout": 0.6, "word_dropout": 0.4,
        "preprocessing": "counts", "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.05,
        "schedule": "onecycle", "batch_size": 128, "epochs": 40, "seed": 42,
    },
    "sentiment": {
        "hidden_sizes": [256], "activation": "GELU", "dropout": 0.2, "word_dropout": 0.0,
        "preprocessing": "standardized", "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.0,
        "schedule": "constant", "batch_size": 64, "epochs": 15, "seed": 42,
    },
}
METRICS = ["train_accuracy", "val_accuracy", "train_loss", "val_loss"]


def experiment_config(dataset: str, **overrides) -> dict:
    """Task defaults plus explicit overrides, validated."""
    unknown = set(overrides) - set(DEFAULTS[dataset])
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    config = {**DEFAULTS[dataset], **overrides}
    config["hidden_sizes"] = list(config["hidden_sizes"])
    for key in ("batch_size", "epochs"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if not 0 <= config["word_dropout"] < 1:
        raise ValueError("word_dropout must be in [0, 1).")
    if config["word_dropout"] > 0 and config["preprocessing"] != "counts":
        raise ValueError("word_dropout requires preprocessing='counts'.")
    if config["preprocessing"] not in InputPipeline.KINDS:
        raise ValueError(f"preprocessing must be one of {InputPipeline.KINDS}.")
    if config["optimizer"].lower() not in ("adam", "adamw"):
        raise ValueError("optimizer must be Adam or AdamW.")
    if config["schedule"] not in ("constant", "onecycle"):
        raise ValueError("schedule must be 'constant' or 'onecycle'.")
    return config


def run_experiment(dataset: str, overrides: dict | None = None, *, data_dir="data",
                   results_dir="results/neural_network", data: Dataset | None = None) -> dict:
    """Train on the 80% split, evaluate the final epoch on the 20% split, save records.

    Saves runs/<run_id>/{config.json, history.csv, validation_predictions.csv}
    and appends one summary row to experiments.csv. The optional data argument
    lets synthetic tests run the pipeline without competition files.
    """
    config = experiment_config(dataset, **(overrides or {}))
    data = load_dataset(dataset, data_dir=data_dir) if data is None else data
    split = split_dataset(data, validation_size=0.2, seed=42)
    num_classes = DATASET_SPECS[dataset][1]

    start = perf_counter()
    model, pipeline, history = train_model(
        config, split.train.X, split.train.y, num_classes, split.validation.X, split.validation.y,
    )
    seconds = perf_counter() - start
    predictions = predict_proba(model, pipeline, split.validation.X).argmax(axis=1)

    run_id = f"{dataset}_{uuid4().hex[:12]}"
    run_dir = Path(results_dir) / "runs" / run_id
    run_dir.mkdir(parents=True)
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}
    (run_dir / "config.json").write_text(json.dumps(
        {"run_id": run_id, "dataset": dataset, **config, "split_seed": 42, "validation_size": 0.2,
         "versions": versions}, indent=2) + "\n")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({"id": split.validation.ids, "true_label": split.validation.y,
                  "predicted_label": predictions}).to_csv(run_dir / "validation_predictions.csv", index=False)

    summary = {"run_id": run_id, "dataset": dataset, **config,
               "hidden_sizes": json.dumps(config["hidden_sizes"]),
               **{key: history[-1][key] for key in METRICS},
               "training_time_seconds": round(seconds, 1),
               "parameter_count": sum(p.numel() for p in model.parameters())}
    table = Path(results_dir) / "experiments.csv"
    pd.DataFrame([summary]).to_csv(table, mode="a", header=not table.exists(), index=False)
    print(f"{run_id}: final-epoch validation accuracy {summary['val_accuracy']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASET_SPECS, required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+")
    for name, kind in (("activation", str), ("dropout", float), ("word_dropout", float),
                       ("preprocessing", str), ("optimizer", str), ("learning_rate", float),
                       ("weight_decay", float), ("schedule", str), ("batch_size", int),
                       ("epochs", int), ("seed", int)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/neural_network")
    args = vars(parser.parse_args())
    options = {key: args.pop(key) for key in ("dataset", "data_dir", "results_dir")}
    run_experiment(options.pop("dataset"), {k: v for k, v in args.items() if v is not None}, **options)


if __name__ == "__main__":
    main()
