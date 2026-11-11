"""Train the final NN models on all labeled data and write submissions/submission.csv.

Run from the repository root: python submission.py
Hidden test features are only used for prediction; no test labels exist locally.
"""

import json
from pathlib import Path

import pandas as pd

from experiments.train_neural_network import experiment_config
from models.neural_network import predict_proba, train_model
from utils.data_loader import DATASET_SPECS, load_dataset

PREFIXES = {"image": "img_", "sentiment": "text_"}


def make_submission(predictions: dict[str, pd.DataFrame], sample: pd.DataFrame) -> pd.DataFrame:
    """Merge per-task (id, label) predictions onto the template's IDs and validate.

    Merging by ID means row order in the test files and template never matters.
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


def main(data_dir="data", output_dir="submissions"):
    predictions, configs = {}, {}
    for task in PREFIXES:
        config = experiment_config(task)
        train, test = load_dataset(task, data_dir=data_dir), load_dataset(task, "test", data_dir=data_dir)
        print(f"Training final {task} model on {len(train.y)} labeled examples")
        # Same settings and epoch count as validation, but preprocessing is refit on all labeled data.
        model, pipeline, _ = train_model(config, train.X, train.y, DATASET_SPECS[task][1])
        labels = predict_proba(model, pipeline, test.X).argmax(axis=1)
        predictions[task] = pd.DataFrame({"id": test.ids, "label": labels})
        configs[task] = config
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv", dtype={"id": str}, keep_default_na=False)
    submission = make_submission(predictions, sample)
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)
    (output / "submission_config.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"Wrote {output / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
