"""Decision tree and random forest experiments for the image and sentiment tasks.

python experiments/tree_experiment.py                               # every sweep, both datasets
python experiments/tree_experiment.py --dataset image --sweep depth # one sweep
python experiments/tree_experiment.py --subsample 2000              # quick check on fewer rows

The models are the from-scratch implementations in models/decision_tree.py and
models/random_forest.py. Every run uses the shared split (utils.preprocessing,
seed 42), so the results compare directly with the other classifiers. Results are
saved to results/trees/ (see utils/results.py); the "model" column says whether a
run is a single tree or a forest.

No scaling is applied. A tree splits on tests of the form feature <= threshold, so
any strictly increasing rescaling of a feature leaves the fitted tree unchanged.
The one transform the trees apply is histogram binning, computed on training rows
only (see models/decision_tree.py).
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Let the script run directly (python experiments/...) as well as with python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier
from utils.data_loader import Dataset, load_dataset
from utils.metrics import accuracy
from utils.preprocessing import split_dataset
from utils.results import clear_results, save_run

ALGORITHM = "trees"
TASKS = ["image", "sentiment"]

DEPTHS = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, None]
LEAF_SIZES = [1, 2, 5, 10, 20, 50, 100]
CRITERIA = ["gini", "entropy", "misclassification"]
CRITERION_DEPTHS = [8, None]
BIN_COUNTS = [8, 16, 32, 64, 128, 255]
TREE_COUNTS = [1, 2, 5, 10, 25, 50, 100]
FEATURE_CHOICES = ["sqrt", "log2", 0.05, 0.1, 0.25, 0.5, 1.0]
MATCHED_DEPTHS = [4, 8, 12, 16, 20, None]

# Final forest for each task, chosen from the sweeps below. The two tasks disagree, so
# each has its own settings. Evaluated on the validation split by the "final" sweep.
FINAL_SETTINGS = {
    "image": dict(n_estimators=300, criterion="gini", max_depth=None,
                  min_samples_leaf=1, max_features=0.05, n_bins=64),
    "sentiment": dict(n_estimators=300, criterion="gini", max_depth=12,
                      min_samples_leaf=1, max_features="sqrt", n_bins=64),
}

SWEEPS = ["depth", "leaf", "criterion", "bins", "trees", "features", "overfit", "final"]


def fit_tree(task, train, validation, sweep, **settings):
    """Fit one decision tree, score it on train and validation, and save the run."""
    started = perf_counter()
    tree = DecisionTreeClassifier(random_state=0, **settings).fit(train.X, train.y)
    predicted = tree.predict(validation.X)
    metrics = {
        "train_accuracy": accuracy(train.y, tree.predict(train.X)),
        "val_accuracy": accuracy(validation.y, predicted),
        "n_leaves": tree.n_leaves_,
        "depth_reached": tree.depth_,
        "seconds": round(perf_counter() - started, 1),
    }
    return _save(task, sweep, "tree", settings, metrics, validation, predicted)


def fit_forest(task, train, validation, sweep, n_jobs=0, oob=False, **settings):
    """Fit one random forest, score it on train and validation, and save the run.

    oob=True also records out-of-bag accuracy (slower; used by the trees and features sweeps).
    """
    started = perf_counter()
    forest = RandomForestClassifier(random_state=0, n_jobs=n_jobs, oob_score=oob, **settings)
    forest.fit(train.X, train.y)
    predicted = forest.predict(validation.X)
    metrics = {
        "train_accuracy": accuracy(train.y, forest.predict(train.X)),
        "val_accuracy": accuracy(validation.y, predicted),
        "features_per_split": forest.max_features_used_,
        "seconds": round(perf_counter() - started, 1),
    }
    if getattr(forest, "oob_score_", None) is not None:
        metrics["oob_accuracy"] = float(forest.oob_score_)
    return _save(task, sweep, "forest", settings, metrics, validation, predicted)


def _save(task, sweep, model, settings, metrics, validation, predicted):
    name = f"{model}: " + ", ".join(f"{key}={value}" for key, value in settings.items())
    save_run(ALGORITHM, task, sweep, name, {"model": model, **settings}, metrics,
             ids=validation.ids, y_true=validation.y, y_pred=predicted)
    print(f"  {name}: val {metrics['val_accuracy']:.4f}", flush=True)
    return metrics


def run_sweep(sweep, task, train, validation, args):
    """Run one sweep. Each sweep changes one setting (two for "criterion" and "bins")."""
    common = {"criterion": args.criterion, "n_bins": args.n_bins}
    if sweep == "depth":          # single tree: underfitting through to memorization
        for depth in DEPTHS:
            fit_tree(task, train, validation, "max_depth", max_depth=depth, **common)
    elif sweep == "leaf":         # single tree: minimum leaf size, the pruning control
        for leaf in LEAF_SIZES:
            fit_tree(task, train, validation, "min_samples_leaf", min_samples_leaf=leaf, **common)
    elif sweep == "criterion":    # impurity measure, at a shallow and an unrestricted depth
        for criterion in CRITERIA:
            for depth in CRITERION_DEPTHS:
                fit_tree(task, train, validation, "criterion", criterion=criterion, max_depth=depth,
                         n_bins=args.n_bins)
    elif sweep == "bins":         # histogram resolution, with and without the sparse-aware binner
        for n_bins in BIN_COUNTS:
            for sparse_aware in (False, True):
                fit_forest(task, train, validation, "n_bins", args.n_jobs, n_estimators=50,
                           n_bins=n_bins, sparse_aware=sparse_aware)
    elif sweep == "trees":        # forest size: more trees reduce variance
        for n_trees in TREE_COUNTS:
            fit_forest(task, train, validation, "n_estimators", args.n_jobs, oob=True, n_estimators=n_trees,
                       max_features=args.max_features, n_bins=args.n_bins)
    elif sweep == "features":     # features considered per split: decorrelates the trees
        for choice in FEATURE_CHOICES:
            fit_forest(task, train, validation, "max_features", args.n_jobs, oob=True, n_estimators=args.n_estimators,
                       max_features=choice, n_bins=args.n_bins)
    elif sweep == "overfit":      # tree against forest at matched depth
        for depth in MATCHED_DEPTHS:
            fit_tree(task, train, validation, "max_depth (tree vs forest)", max_depth=depth, **common)
            fit_forest(task, train, validation, "max_depth (tree vs forest)", args.n_jobs, n_estimators=50,
                       max_depth=depth, max_features=args.max_features, n_bins=args.n_bins)
    elif sweep == "final":        # the final per-task forest
        fit_forest(task, train, validation, "final", args.n_jobs, **FINAL_SETTINGS[task])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="both", choices=TASKS + ["both"])
    parser.add_argument("--sweep", default="all", choices=SWEEPS + ["all"])
    parser.add_argument("--criterion", default="gini")
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=0, help="0 uses every core")
    parser.add_argument("--subsample", type=int, default=0,
                        help="use only this many training rows (for a quick check)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    tasks = TASKS if args.dataset == "both" else [args.dataset]
    sweeps = SWEEPS if args.sweep == "all" else [args.sweep]
    if args.dataset == "both" and args.sweep == "all":
        clear_results(ALGORITHM)  # a full run starts from empty files

    for task in tasks:
        split = split_dataset(load_dataset(task, data_dir=args.data_dir), validation_size=0.2, seed=42)
        train, validation = split.train, split.validation
        if args.subsample and args.subsample < len(train.y):
            n = args.subsample
            train = Dataset(train.X[:n], train.y[:n], train.ids[:n])
        print(f"\n{task}: {len(train.y)} train / {len(validation.y)} validation", flush=True)
        for sweep in sweeps:
            print(f"[{task}] {sweep}", flush=True)
            run_sweep(sweep, task, train, validation, args)


if __name__ == "__main__":
    main()
