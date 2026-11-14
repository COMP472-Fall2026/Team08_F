"""Neural network experiments: train on the 80% split, evaluate on the 20% validation split.

python -m experiments.train_neural_network --dataset image --all          # all experiments
python -m experiments.train_neural_network --dataset image                # final settings
python -m experiments.train_neural_network --dataset image --dropout 0.2  # change a setting
python -m experiments.train_neural_network --dataset image --ensemble     # 3-seed final model

Each run adds rows to three CSV files in results/neural_network/, linked by run_id:
experiments.csv (one row per run), history.csv (one per epoch) and
validation_predictions.csv (one per validation example).
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from models.neural_network import InputPipeline, predict_proba, train_model
from utils.data_loader import DATASET_SPECS, Dataset, load_dataset
from utils.preprocessing import split_dataset


# Final settings for each task. Every experiment below changes one of them.
DEFAULTS = {
    "image": {
        "hidden_sizes": [4096], "activation": "GELU", "dropout": 0.4, "word_dropout": 0.4,
        "preprocessing": "counts", "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.05,
        "schedule": "onecycle", "batch_size": 128, "epochs": 40, "seed": 42,
    },
    "sentiment": {
        "hidden_sizes": [256], "activation": "GELU", "dropout": 0.2, "word_dropout": 0.0,
        "preprocessing": "standardized", "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.0,
        "schedule": "constant", "batch_size": 64, "epochs": 15, "seed": 42,
    },
}

# Experiments as (name, settings changed from DEFAULTS). Changing one setting at a
# time means any change in accuracy can be attributed to that setting. Both tasks
# try learning rates 0.01, 0.001 and 0.0001 (one of them is the default).
EXPERIMENTS = {
    "image": [
        ("baseline", {}),
        ("width 512", {"hidden_sizes": [512]}),                   # layer size
        ("width 1024", {"hidden_sizes": [1024]}),
        ("depth 2", {"hidden_sizes": [4096, 4096]}),              # number of hidden layers
        ("depth 3", {"hidden_sizes": [4096, 4096, 4096]}),
        ("ReLU", {"activation": "ReLU"}),                         # activation function
        ("no dropout", {"dropout": 0.0}),                         # regularization
        ("no word dropout", {"word_dropout": 0.0}),
        ("learning rate 0.01", {"learning_rate": 1e-2}),          # learning rate
        ("learning rate 0.0001", {"learning_rate": 1e-4}),
        # Word dropout only works on counts, so compare this run with "no word dropout":
        # the only difference between the two is the input preprocessing.
        ("standardized input", {"preprocessing": "standardized", "word_dropout": 0.0}),
    ],
    "sentiment": [
        ("baseline", {}),
        ("width 64", {"hidden_sizes": [64]}),                     # layer size
        ("width 1024", {"hidden_sizes": [1024]}),
        ("depth 2", {"hidden_sizes": [256, 256]}),                # number of hidden layers
        ("depth 3", {"hidden_sizes": [256, 256, 256]}),
        ("ReLU", {"activation": "ReLU"}),                         # activation function
        ("dropout 0.5", {"dropout": 0.5}),                        # regularization
        ("50 epochs", {"epochs": 50}),                            # overfitting over time
        ("learning rate 0.001", {"learning_rate": 1e-3}),         # learning rate
        ("learning rate 0.01", {"learning_rate": 1e-2}),
    ],
}
METRICS = ["train_accuracy", "val_accuracy", "train_loss", "val_loss"]
ENSEMBLE_SEEDS = [42, 7, 123]  # the final model averages one network per seed


def experiment_config(dataset: str, **overrides) -> dict:
    """Return the task's default settings with the given changes, after checking them."""
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


def run_experiment(dataset: str, overrides: dict | None = None, *, name="custom", seeds: list[int] | None = None,
                   data_dir="data", results_dir="results/neural_network", data: Dataset | None = None) -> dict:
    """Train on the 80% split, evaluate on the 20% validation split and save the results.

    If several seeds are given, one model is trained per seed and their predicted
    probabilities are averaged (an ensemble, as in submission.py). The data
    argument lets tests pass in small made-up datasets.
    """
    config = experiment_config(dataset, **(overrides or {}))
    seeds = seeds or [config["seed"]]
    data = load_dataset(dataset, data_dir=data_dir) if data is None else data
    split = split_dataset(data, validation_size=0.2, seed=42)
    subsets = {"train": split.train, "val": split.validation}

    start = perf_counter()
    probabilities, histories = {subset: [] for subset in subsets}, []
    for seed in seeds:
        model, pipeline, history = train_model(
            {**config, "seed": seed}, split.train.X, split.train.y, DATASET_SPECS[dataset][1],
            split.validation.X, split.validation.y,
        )
        histories.append(pd.DataFrame(history).assign(seed=seed))
        for subset, part in subsets.items():
            probabilities[subset].append(predict_proba(model, pipeline, part.X))
    seconds = perf_counter() - start

    # Average the probabilities over the models (with one seed this is just that model).
    metrics = {}
    for subset, part in subsets.items():
        mean = np.mean(probabilities[subset], axis=0)
        metrics[f"{subset}_accuracy"] = float(np.mean(mean.argmax(axis=1) == part.y))
        metrics[f"{subset}_loss"] = float(-np.mean(np.log(mean[np.arange(len(part.y)), part.y] + 1e-12)))
    predictions = np.mean(probabilities["val"], axis=0).argmax(axis=1)

    run_id = f"{dataset}_{uuid4().hex[:8]}"
    summary = {"run_id": run_id, "dataset": dataset, "experiment": name, **config,
               "hidden_sizes": json.dumps(config["hidden_sizes"]), "seed": " ".join(map(str, seeds)),
               **{key: metrics[key] for key in METRICS},
               "training_time_seconds": round(seconds, 1),
               "parameter_count": sum(p.numel() for p in model.parameters())}
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    append_csv(pd.DataFrame([summary]), results / "experiments.csv")
    append_csv(pd.concat(histories).assign(run_id=run_id), results / "history.csv")
    append_csv(pd.DataFrame({"run_id": run_id, "id": split.validation.ids, "true_label": split.validation.y,
                             "predicted_label": predictions}), results / "validation_predictions.csv")
    print(f"{run_id} ({name}): final-epoch validation accuracy {summary['val_accuracy']:.4f}", flush=True)
    return summary


def append_csv(rows: pd.DataFrame, path: Path) -> None:
    """Append rows to a CSV file, creating it if needed. Columns are matched by name."""
    if path.exists():
        rows = pd.concat([pd.read_csv(path), rows], ignore_index=True)
    rows.to_csv(path, index=False)


def run_all(dataset: str, data_dir="data", results_dir="results/neural_network") -> None:
    """Run every experiment for one task:

    1. each experiment in EXPERIMENTS (seed 42);
    2. the baseline with the other ensemble seeds, to see how much accuracy
       changes from the random seed alone;
    3. the final model (3-seed ensemble).
    """
    data = load_dataset(dataset, data_dir=data_dir)
    for name, overrides in EXPERIMENTS[dataset]:
        run_experiment(dataset, overrides, name=name, data=data, results_dir=results_dir)
    for seed in ENSEMBLE_SEEDS[1:]:
        run_experiment(dataset, {"seed": seed}, name=f"baseline seed {seed}", data=data, results_dir=results_dir)
    run_experiment(dataset, name="final ensemble", seeds=ENSEMBLE_SEEDS, data=data, results_dir=results_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASET_SPECS, required=True)
    parser.add_argument("--all", action="store_true",
                        help="run all experiments, the seed repeats and the final ensemble")
    parser.add_argument("--ensemble", action="store_true",
                        help=f"average one model per seed {ENSEMBLE_SEEDS} (the final model)")
    parser.add_argument("--hidden-sizes", type=int, nargs="+")
    for name, kind in (("activation", str), ("dropout", float), ("word_dropout", float),
                       ("preprocessing", str), ("optimizer", str), ("learning_rate", float),
                       ("weight_decay", float), ("schedule", str), ("batch_size", int),
                       ("epochs", int), ("seed", int)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/neural_network")
    args = vars(parser.parse_args())
    dataset, everything, ensemble = args.pop("dataset"), args.pop("all"), args.pop("ensemble")
    if ensemble and args["seed"] is not None:
        parser.error("--ensemble uses fixed seeds; do not combine it with --seed")
    paths = {key: args.pop(key) for key in ("data_dir", "results_dir")}
    if everything:
        run_all(dataset, **paths)
    else:
        run_experiment(dataset, {k: v for k, v in args.items() if v is not None},
                       name="ensemble" if ensemble else "custom",
                       seeds=ENSEMBLE_SEEDS if ensemble else None, **paths)


if __name__ == "__main__":
    main()
