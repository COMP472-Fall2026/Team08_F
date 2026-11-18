"""Decision tree and random forest experiments for the image and sentiment tasks.

Run from the repository root:

    python experiments/tree_experiment.py                      # every sweep, both tasks
    python experiments/tree_experiment.py --task image --sweep depth
    python experiments/tree_experiment.py --subsample 2000     # quick smoke run

Each sweep writes results/<task>_<sweep>.csv.

The models are the from-scratch implementations in models/decision_tree.py and
models/random_forest.py. The train/validation split is the shared one from
utils.preprocessing, so these numbers are directly comparable with the k-NN,
neural network and clustering results.

No scaling is applied. A tree splits on tests of the form feature <= threshold,
so any strictly monotone rescaling maps every candidate threshold onto a
corresponding one and leaves the fitted tree unchanged; standardize_split is
deliberately not used here. The one transform the trees do apply is histogram
binning, computed on training rows only (see models/decision_tree.py).
"""

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier
from utils.data_loader import load_dataset
from utils.metrics import accuracy
from utils.preprocessing import split_dataset

TASKS = ["image", "sentiment"]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DEPTHS = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, None]
LEAF_SIZES = [1, 2, 5, 10, 20, 50, 100]
CRITERIA = ["gini", "entropy", "misclassification"]
CRITERION_DEPTHS = [8, None]
BIN_COUNTS = [8, 16, 32, 64, 128, 255]
TREE_COUNTS = [1, 2, 5, 10, 25, 50, 100]
FEATURE_CHOICES = ["sqrt", "log2", 0.05, 0.1, 0.25, 0.5, 1.0]
MATCHED_DEPTHS = [4, 8, 12, 16, 20, None]

SWEEPS = ["depth", "leaf", "criterion", "bins", "trees", "features", "overfit"]


def save_results(filename, rows):
    """Save a results table; called after each sweep so partial runs keep their output."""
    if not rows:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> results/{filename}")


def fit_tree(train, validation, **kwargs):
    """Fit one tree and report train and validation accuracy plus its size."""
    started = perf_counter()
    tree = DecisionTreeClassifier(random_state=0, **kwargs).fit(train.X, train.y)
    train_accuracy = accuracy(train.y, tree.predict(train.X))
    validation_accuracy = accuracy(validation.y, tree.predict(validation.X))
    return {
        "train_acc": round(train_accuracy, 4),
        "val_acc": round(validation_accuracy, 4),
        "gap": round(train_accuracy - validation_accuracy, 4),
        "n_leaves": tree.n_leaves_,
        "depth_reached": tree.depth_,
        "fit_s": round(perf_counter() - started, 1),
    }


def fit_forest(train, validation, n_jobs=0, **kwargs):
    """Fit one forest and report train and validation accuracy."""
    started = perf_counter()
    forest = RandomForestClassifier(random_state=0, n_jobs=n_jobs, **kwargs)
    forest.fit(train.X, train.y)
    train_accuracy = accuracy(train.y, forest.predict(train.X))
    validation_accuracy = accuracy(validation.y, forest.predict(validation.X))
    row = {
        "train_acc": round(train_accuracy, 4),
        "val_acc": round(validation_accuracy, 4),
        "gap": round(train_accuracy - validation_accuracy, 4),
        "fit_s": round(perf_counter() - started, 1),
    }
    if getattr(forest, "oob_score_", None) is not None:
        row["oob_acc"] = round(float(forest.oob_score_), 4)
    row["k_used"] = forest.max_features_used_
    return row


def sweep_depth(train, validation, args):
    """Single tree, increasing maximum depth: underfitting through to memorization."""
    rows = []
    for depth in DEPTHS:
        row = {"max_depth": depth}
        row.update(fit_tree(train, validation, max_depth=depth,
                            criterion=args.criterion, n_bins=args.n_bins))
        print(f"  depth={depth}: val {row['val_acc']:.4f}")
        rows.append(row)
    return rows


def sweep_leaf(train, validation, args):
    """Single tree, increasing minimum leaf size. This is our only pruning control."""
    rows = []
    for leaf in LEAF_SIZES:
        row = {"min_samples_leaf": leaf}
        row.update(fit_tree(train, validation, min_samples_leaf=leaf,
                            criterion=args.criterion, n_bins=args.n_bins))
        print(f"  min_samples_leaf={leaf}: val {row['val_acc']:.4f}")
        rows.append(row)
    return rows


def sweep_criterion(train, validation, args):
    """Impurity measure at a shallow depth and at unrestricted depth."""
    rows = []
    for criterion in CRITERIA:
        for depth in CRITERION_DEPTHS:
            row = {"criterion": criterion, "max_depth": depth}
            row.update(fit_tree(train, validation, criterion=criterion,
                                max_depth=depth, n_bins=args.n_bins))
            print(f"  {criterion} depth={depth}: val {row['val_acc']:.4f}")
            rows.append(row)
    return rows


def sweep_bins(train, validation, args):
    """Histogram resolution, with and without the sparse-aware binner."""
    rows = []
    for n_bins in BIN_COUNTS:
        for sparse_aware in (False, True):
            row = {"n_bins": n_bins, "sparse_aware": sparse_aware}
            row.update(fit_forest(train, validation, n_jobs=args.n_jobs,
                                  n_estimators=50, n_bins=n_bins,
                                  sparse_aware=sparse_aware))
            print(f"  bins={n_bins} sparse_aware={sparse_aware}: val {row['val_acc']:.4f}")
            rows.append(row)
    return rows


def sweep_trees(train, validation, args):
    """Forest size. More trees is variance reduction, so this should not overfit."""
    rows = []
    for n_trees in TREE_COUNTS:
        row = {"n_trees": n_trees}
        row.update(fit_forest(train, validation, n_jobs=args.n_jobs,
                              n_estimators=n_trees, max_features=args.max_features,
                              n_bins=args.n_bins, oob_score=True))
        print(f"  trees={n_trees}: val {row['val_acc']:.4f}")
        rows.append(row)
    return rows


def sweep_features(train, validation, args):
    """Features considered per split: the decorrelation knob."""
    rows = []
    for choice in FEATURE_CHOICES:
        row = {"max_features": choice}
        row.update(fit_forest(train, validation, n_jobs=args.n_jobs,
                              n_estimators=args.n_estimators, max_features=choice,
                              n_bins=args.n_bins, oob_score=True))
        print(f"  max_features={choice} (k={row['k_used']}): val {row['val_acc']:.4f}")
        rows.append(row)
    return rows


def sweep_overfit(train, validation, args):
    """Tree against forest at matched depth, so the two gaps can be compared fairly."""
    rows = []
    for depth in MATCHED_DEPTHS:
        tree = fit_tree(train, validation, max_depth=depth,
                        criterion=args.criterion, n_bins=args.n_bins)
        forest = fit_forest(train, validation, n_jobs=args.n_jobs, n_estimators=50,
                            max_depth=depth, max_features=args.max_features,
                            n_bins=args.n_bins)
        rows.append({
            "max_depth": depth,
            "dt_train": tree["train_acc"], "dt_val": tree["val_acc"],
            "dt_gap": tree["gap"],
            "rf_train": forest["train_acc"], "rf_val": forest["val_acc"],
            "rf_gap": forest["gap"],
            "val_gain": round(forest["val_acc"] - tree["val_acc"], 4),
        })
        print(f"  depth={depth}: tree {tree['val_acc']:.4f} forest {forest['val_acc']:.4f}")
    return rows


RUNNERS = {
    "depth": sweep_depth, "leaf": sweep_leaf, "criterion": sweep_criterion,
    "bins": sweep_bins, "trees": sweep_trees, "features": sweep_features,
    "overfit": sweep_overfit,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="both", choices=TASKS + ["both"])
    parser.add_argument("--sweep", default="all", choices=SWEEPS + ["all"])
    parser.add_argument("--criterion", default="gini")
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=0, help="0 uses every core")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42,
                        help="the shared split seed; leave at 42 to match the other models")
    parser.add_argument("--subsample", type=int, default=0,
                        help="use only this many training rows (for a quick smoke run)")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    tasks = TASKS if args.task == "both" else [args.task]
    sweeps = SWEEPS if args.sweep == "all" else [args.sweep]
    data_dir = Path(args.data_dir)

    for task in tasks:
        if not (data_dir / f"{task}_train.csv").exists():
            print(f"Skipping {task}: missing {data_dir}/{task}_train.csv")
            continue
        data = load_dataset(task, "train", data_dir=data_dir)
        split = split_dataset(data, validation_size=args.validation_size, seed=args.seed)
        train, validation = split.train, split.validation
        if args.subsample and args.subsample < len(train.y):
            train = type(train)(train.X[:args.subsample], train.y[:args.subsample],
                                train.ids[:args.subsample])
        print(f"\n{task}: {len(train.y)} train / {len(validation.y)} validation, "
              f"{train.X.shape[1]} features, seed {args.seed}")
        for sweep in sweeps:
            print(f"[{task}] {sweep}")
            save_results(f"{task}_{sweep}.csv", RUNNERS[sweep](train, validation, args))


if __name__ == "__main__":
    main()
