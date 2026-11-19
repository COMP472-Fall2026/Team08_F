"""Neural network experiments: train on the 80% split, evaluate on the 20% validation split.

python experiments/neural_network_experiment.py                                    # all experiments, both datasets
python experiments/neural_network_experiment.py --dataset image --dropout 0.2      # one run with a setting changed
python experiments/neural_network_experiment.py --dataset image --ensemble         # the 3-seed final model

Results are saved to results/neural_network/ (see utils/results.py): experiments.csv,
validation_predictions.csv, and history.csv (one row per epoch, for learning curves).
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Let the script run directly (python experiments/...) as well as with python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.neural_network import InputPipeline, predict_proba, train_model
from utils.data_loader import DATASET_SPECS, Dataset, load_dataset
from utils.preprocessing import split_dataset
from utils.results import RESULTS_DIR, append_csv, clear_results, save_run

ALGORITHM = "neural_network"

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

# Experiments as (name, sweep, settings changed from DEFAULTS). Changing one setting at
# a time means any change in accuracy can be attributed to that setting. Runs with the
# same sweep are plotted together, next to the baseline. Both tasks try learning rates
# 0.01, 0.001 and 0.0001 (one of them is the default).
EXPERIMENTS = {
    "image": [
        ("baseline", "baseline", {}),
        ("width 512", "width", {"hidden_sizes": [512]}),
        ("width 1024", "width", {"hidden_sizes": [1024]}),
        ("depth 2", "depth", {"hidden_sizes": [4096, 4096]}),
        ("depth 3", "depth", {"hidden_sizes": [4096, 4096, 4096]}),
        ("ReLU", "activation", {"activation": "ReLU"}),
        ("no dropout", "dropout", {"dropout": 0.0}),
        ("no word dropout", "word_dropout", {"word_dropout": 0.0}),
        ("learning rate 0.01", "learning_rate", {"learning_rate": 1e-2}),
        ("learning rate 0.0001", "learning_rate", {"learning_rate": 1e-4}),
        # Word dropout only works on counts, so compare this run with "no word dropout":
        # the only difference between the two is the input preprocessing.
        ("standardized input", "preprocessing", {"preprocessing": "standardized", "word_dropout": 0.0}),
    ],
    "sentiment": [
        ("baseline", "baseline", {}),
        ("width 64", "width", {"hidden_sizes": [64]}),
        ("width 1024", "width", {"hidden_sizes": [1024]}),
        ("depth 2", "depth", {"hidden_sizes": [256, 256]}),
        ("depth 3", "depth", {"hidden_sizes": [256, 256, 256]}),
        ("ReLU", "activation", {"activation": "ReLU"}),
        ("dropout 0.5", "dropout", {"dropout": 0.5}),
        ("50 epochs", "epochs", {"epochs": 50}),
        ("learning rate 0.001", "learning_rate", {"learning_rate": 1e-3}),
        ("learning rate 0.01", "learning_rate", {"learning_rate": 1e-2}),
    ],
}
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


def run_experiment(dataset: str, overrides: dict | None = None, *, name="custom", sweep="custom",
                   seeds: list[int] | None = None, data_dir="data", results_dir=RESULTS_DIR,
                   data: Dataset | None = None, verbose: bool = True) -> dict:
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
            split.validation.X, split.validation.y, verbose=verbose,
        )
        histories.append(pd.DataFrame(history).assign(seed=seed))
        for subset, part in subsets.items():
            probabilities[subset].append(predict_proba(model, pipeline, part.X))

    # Average the probabilities over the models (with one seed this is just that model).
    metrics = {}
    for subset, part in subsets.items():
        mean = np.mean(probabilities[subset], axis=0)
        metrics[f"{subset}_accuracy"] = float(np.mean(mean.argmax(axis=1) == part.y))
        metrics[f"{subset}_loss"] = float(-np.mean(np.log(mean[np.arange(len(part.y)), part.y] + 1e-12)))
    metrics["seconds"] = round(perf_counter() - start, 1)
    metrics["parameter_count"] = sum(p.numel() for p in model.parameters())

    settings = {**config, "seed": " ".join(map(str, seeds))}
    predictions = np.mean(probabilities["val"], axis=0).argmax(axis=1)
    run_id = save_run(ALGORITHM, dataset, sweep, name, settings, metrics, ids=split.validation.ids,
                      y_true=split.validation.y, y_pred=predictions, results_dir=results_dir)
    append_csv(pd.concat(histories).assign(run_id=run_id), Path(results_dir) / ALGORITHM / "history.csv")
    print(f"{run_id} ({name}): validation accuracy {metrics['val_accuracy']:.4f}", flush=True)
    return {"run_id": run_id, **settings, **metrics}


def run_all(dataset: str, data_dir="data", results_dir=RESULTS_DIR) -> None:
    """Run every experiment for one task:

    1. each experiment in EXPERIMENTS (seed 42);
    2. the baseline with the other ensemble seeds, to see how much accuracy
       changes from the random seed alone;
    3. the final model (3-seed ensemble).
    """
    data = load_dataset(dataset, data_dir=data_dir)
    options = {"data": data, "results_dir": results_dir, "verbose": False}
    for name, sweep, overrides in EXPERIMENTS[dataset]:
        run_experiment(dataset, overrides, name=name, sweep=sweep, **options)
    for seed in ENSEMBLE_SEEDS[1:]:
        run_experiment(dataset, {"seed": seed}, name=f"baseline seed {seed}", sweep="seed", **options)
    run_experiment(dataset, name="final ensemble", sweep="ensemble", seeds=ENSEMBLE_SEEDS, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*DATASET_SPECS, "both"], default="both")
    parser.add_argument("--ensemble", action="store_true",
                        help=f"one run of the final model: average one network per seed {ENSEMBLE_SEEDS}")
    parser.add_argument("--hidden-sizes", type=int, nargs="+")
    for name, kind in (("activation", str), ("dropout", float), ("word_dropout", float),
                       ("preprocessing", str), ("optimizer", str), ("learning_rate", float),
                       ("weight_decay", float), ("schedule", str), ("batch_size", int),
                       ("epochs", int), ("seed", int)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--data-dir", default="data")
    args = vars(parser.parse_args())
    dataset, ensemble, data_dir = args.pop("dataset"), args.pop("ensemble"), args.pop("data_dir")
    if ensemble and args["seed"] is not None:
        parser.error("--ensemble uses fixed seeds; do not combine it with --seed")
    datasets = list(DATASET_SPECS) if dataset == "both" else [dataset]
    changes = {key: value for key, value in args.items() if value is not None}

    if not changes and not ensemble:
        # No settings given: run every experiment. A full run of both datasets starts from empty files.
        if dataset == "both":
            clear_results(ALGORITHM)
        for name in datasets:
            run_all(name, data_dir=data_dir)
    else:
        for name in datasets:
            run_experiment(name, changes, name="ensemble" if ensemble else "custom",
                           sweep="ensemble" if ensemble else "custom",
                           seeds=ENSEMBLE_SEEDS if ensemble else None, data_dir=data_dir)


if __name__ == "__main__":
    main()
