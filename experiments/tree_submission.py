"""Train the final random forests on all labeled data and write a tree-model submission.

Run from the repository root:

    python experiments/tree_submission.py                # holdout check, then refit on 100%
    python experiments/tree_submission.py --no-holdout   # skip the holdout check

The settings below are the ones chosen in experiments/tree_experiment.py. The two
tasks disagree, so each gets its own configuration. The submission table is
assembled and validated by submission.make_submission, the same helper the neural
network submission uses.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.random_forest import RandomForestClassifier
from submission import make_submission
from utils.data_loader import load_dataset
from utils.metrics import accuracy, overall_competition_score
from utils.preprocessing import split_dataset

# Chosen in experiments/tree_experiment.py; see the report for the sweeps behind them.
FINAL_SETTINGS = {
    "image": dict(n_estimators=300, criterion="gini", max_depth=None,
                  min_samples_leaf=1, max_features=0.05, n_bins=64),
    "sentiment": dict(n_estimators=300, criterion="gini", max_depth=12,
                      min_samples_leaf=1, max_features="sqrt", n_bins=64),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-holdout", action="store_true",
                        help="skip the held-out validation check before the final refit")
    parser.add_argument("--n-jobs", type=int, default=0, help="0 uses every core")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "submissions"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    predictions, scores = {}, {}

    for task, settings in FINAL_SETTINGS.items():
        train = load_dataset(task, "train", data_dir=data_dir)
        test = load_dataset(task, "test", data_dir=data_dir)

        if not args.no_holdout:
            split = split_dataset(train, seed=args.seed)
            forest = RandomForestClassifier(random_state=0, n_jobs=args.n_jobs, **settings)
            forest.fit(split.train.X, split.train.y)
            scores[task] = accuracy(split.validation.y, forest.predict(split.validation.X))
            print("[%s] held-out validation accuracy: %.4f" % (task, scores[task]))

        # Refit on every labeled row now that the settings are fixed.
        print("[%s] refitting on all %d labeled rows" % (task, len(train.y)))
        final = RandomForestClassifier(random_state=0, n_jobs=args.n_jobs, **settings)
        final.fit(train.X, train.y)
        predictions[task] = pd.DataFrame({"id": test.ids, "label": final.predict(test.X)})

    if scores:
        print("mean of the two task accuracies: %.4f"
              % overall_competition_score(scores["image"], scores["sentiment"]))

    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={"id": str},
                         keep_default_na=False)
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True)
    path = output / "tree_submission.csv"
    make_submission(predictions, sample).to_csv(path, index=False)
    print("Wrote %s" % path)


if __name__ == "__main__":
    main()
