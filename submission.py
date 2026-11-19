"""Train the final neural networks on all labeled data and write submissions/submission.csv.

python submission.py              # average of 3 models per task (seeds 42, 7, 123)
python submission.py --seeds 42   # one model per task

The test files have no labels; they are only used to make predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.neural_network_experiment import ENSEMBLE_SEEDS, experiment_config
from models.neural_network import predict_proba, train_model
from utils.data_loader import DATASET_SPECS, load_dataset

PREFIXES = {"image": "img_", "sentiment": "text_"}


def make_submission(predictions: dict[str, pd.DataFrame], sample: pd.DataFrame) -> pd.DataFrame:
    """Combine each task's (id, label) predictions into one table in the template's order.

    Rows are matched by ID, so the order of the test files does not matter.
    Raises an error if an ID is missing, repeated or unknown, or a label is out of range.
    """
    combined = pd.concat(predictions.values(), ignore_index=True)
    submission = sample[["id"]].merge(combined, on="id", how="left", validate="one_to_one")
    if len(submission) != len(sample) or submission["label"].isna().any():
        raise ValueError("Every template ID needs exactly one prediction.")
    if len(combined) != len(sample):
        raise ValueError("Predictions contain IDs that are not in the template.")
    submission["label"] = submission["label"].astype(int)
    for task, prefix in PREFIXES.items():
        labels = submission.loc[submission["id"].str.startswith(prefix), "label"]
        if not labels.between(0, DATASET_SPECS[task][1] - 1).all():
            raise ValueError(f"{task} labels are out of range.")
    return submission


def main(data_dir="data", output_dir="submissions", seeds=ENSEMBLE_SEEDS):
    """For each task, train one model per seed on all labeled data and average their predictions."""
    predictions, configs = {}, {}
    for task in PREFIXES:
        config = experiment_config(task)
        train, test = load_dataset(task, data_dir=data_dir), load_dataset(task, "test", data_dir=data_dir)
        probabilities = []
        for seed in seeds:
            print(f"Training final {task} model (seed {seed}) on {len(train.y)} labeled examples")
            # Same settings as in the experiments; preprocessing is refitted on all labeled data.
            model, pipeline, _ = train_model({**config, "seed": seed}, train.X, train.y, DATASET_SPECS[task][1])
            probabilities.append(predict_proba(model, pipeline, test.X))
        labels = np.mean(probabilities, axis=0).argmax(axis=1)
        predictions[task] = pd.DataFrame({"id": test.ids, "label": labels})
        configs[task] = {**config, "seed": None, "ensemble_seeds": list(seeds)}
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv", dtype={"id": str}, keep_default_na=False)
    submission = make_submission(predictions, sample)
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)
    (output / "submission_config.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"Wrote {output / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=ENSEMBLE_SEEDS,
                        help="one model is trained per seed and their probabilities are averaged")
    main(seeds=parser.parse_args().seeds)
