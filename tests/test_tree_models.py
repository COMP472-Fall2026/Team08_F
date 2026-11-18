"""Correctness checks for the from-scratch tree and forest.

Run:  python -m unittest discover -s tests -v

The sklearn comparisons are skipped when sklearn is absent.  sklearn is used
ONLY as a test oracle here - never by the implementation.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from models.decision_tree import (DecisionTreeClassifier, apply_bins,
                                  compute_bin_edges, impurity_from_counts)
from models.random_forest import RandomForestClassifier
from utils.data_loader import Dataset
from utils.metrics import accuracy, per_class_report
from utils.preprocessing import split_dataset


def stratified_split(y, val_frac=0.2, seed=0):
    """Row positions for a stratified split, using the shared splitter.

    utils.preprocessing.split_dataset works on a Dataset and sorts by ID, so this
    wraps it with throwaway sorted IDs and returns the training and validation row
    positions. The tests then index X and y with them, exactly as before.
    """
    y = np.asarray(y)
    ids = np.array(["r%06d" % i for i in range(len(y))])
    placeholder = Dataset(np.zeros((len(y), 1), dtype=np.float32), y, ids)
    split = split_dataset(placeholder, validation_size=val_frac, seed=seed)
    return split.train_indices, split.validation_indices

FAILURES = []


def check(name, cond, detail=""):
    """Assert a condition, keeping the original name and detail in the message."""
    if not cond:
        FAILURES.append(name)
        raise AssertionError("%s %s" % (name, detail))


def toy(n=600, d=8, c=3, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    cent = rng.normal(0, 2.0, size=(c, d))
    y = rng.integers(0, c, size=n)
    X = cent[y] + rng.normal(0, noise, size=(n, d))
    return X.astype(np.float32), y


def test_impurity():
    pure = np.array([10.0, 0.0, 0.0])
    even = np.array([5.0, 5.0, 5.0])
    check("impurity/gini_pure", abs(impurity_from_counts(pure, "gini")) < 1e-12)
    check("impurity/gini_even",
          abs(float(impurity_from_counts(even, "gini")) - (1 - 3 * (1 / 3) ** 2)) < 1e-12)
    check("impurity/entropy_even",
          abs(float(impurity_from_counts(even, "entropy")) - np.log2(3)) < 1e-12)
    check("impurity/misclass_even",
          abs(float(impurity_from_counts(even, "misclassification")) - 2 / 3) < 1e-12)
    check("impurity/empty_is_zero",
          float(impurity_from_counts(np.zeros(3), "entropy")) == 0.0)


def test_binning_matches_threshold_semantics():
    """The split rule used while fitting (bin <= b) must be identical to the
    rule used while predicting (x < edges[b]).  If these ever disagree the tree
    silently predicts garbage, so this is the single most important test here.
    """
    rng = np.random.default_rng(1)
    X = rng.random((500, 6)).astype(np.float32)
    X[:, 0] = 0.0                      # constant column
    X[rng.random(X.shape) < 0.5] = 0.0  # heavy ties at zero, like the image data
    edges = compute_bin_edges(X, n_bins=16, rng=rng)
    Xb, B = apply_bins(X, edges)
    ok = True
    for j in range(X.shape[1]):
        for b in range(len(edges[j])):
            ok &= bool(np.array_equal(Xb[:, j] <= b, X[:, j] < edges[j][b]))
    check("binning/split_rule_equivalence", ok)
    check("binning/constant_feature_no_edges", len(edges[0]) == 0)
    check("binning/bin_count", B >= 2 and Xb.max() < B)


def test_tree_separable():
    X, y = toy(noise=0.15)
    clf = DecisionTreeClassifier().fit(X, y)
    check("tree/fits_separable_data", accuracy(y, clf.predict(X)) > 0.99,
          "train acc %.4f" % accuracy(y, clf.predict(X)))


def test_tree_depth_limit():
    X, y = toy()
    for d in (1, 3, 5):
        clf = DecisionTreeClassifier(max_depth=d).fit(X, y)
        check("tree/respects_max_depth_%d" % d, clf.depth_ <= d,
              "reached %d" % clf.depth_)
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    check("tree/stump_has_3_nodes", stump.n_nodes_ == 3, str(stump.n_nodes_))


def test_tree_min_samples_leaf():
    X, y = toy()
    for msl in (5, 25, 100):
        clf = DecisionTreeClassifier(min_samples_leaf=msl).fit(X, y)
        leaf_sizes = clf.n_node_samples_[clf.feature_ < 0]
        check("tree/min_samples_leaf_%d" % msl, int(leaf_sizes.min()) >= msl,
              "smallest leaf %d" % leaf_sizes.min())


def test_criteria_all_work():
    X, y = toy(noise=0.5)
    for crit in ("gini", "entropy", "misclassification"):
        clf = DecisionTreeClassifier(criterion=crit, max_depth=6).fit(X, y)
        a = accuracy(y, clf.predict(X))
        check("tree/criterion_%s" % crit, a > 0.80, "train acc %.4f" % a)


def test_predict_consistency():
    """Predicting the training rows must agree with the leaf each row fell into
    during fitting - catches any fit/predict traversal mismatch."""
    X, y = toy(noise=0.8, seed=3)
    clf = DecisionTreeClassifier(max_depth=7).fit(X, y)
    leaves = clf.apply(X)
    check("tree/apply_lands_on_leaves", bool((clf.feature_[leaves] < 0).all()))
    counts = np.bincount(leaves, minlength=clf.n_nodes_)
    stored = clf.n_node_samples_.copy()
    leaf_ids = np.flatnonzero(clf.feature_ < 0)
    check("tree/leaf_sample_counts_match",
          bool(np.array_equal(counts[leaf_ids], stored[leaf_ids])))


def test_forest_beats_tree_on_noise():
    X, y = toy(n=1500, d=20, c=4, seed=7, noise=3.0)
    tr, va = stratified_split(y, 0.3, seed=0)
    dt = DecisionTreeClassifier(random_state=0).fit(X[tr], y[tr])
    rf = RandomForestClassifier(n_estimators=60, n_jobs=1,
                                random_state=0).fit(X[tr], y[tr])
    a_dt = accuracy(y[va], dt.predict(X[va]))
    a_rf = accuracy(y[va], rf.predict(X[va]))
    check("forest/beats_single_tree", a_rf > a_dt,
          "dt %.4f vs rf %.4f" % (a_dt, a_rf))
    g_dt = accuracy(y[tr], dt.predict(X[tr])) - a_dt
    g_rf = accuracy(y[tr], rf.predict(X[tr])) - a_rf
    check("forest/smaller_overfitting_gap", g_rf < g_dt,
          "dt gap %.4f vs rf gap %.4f" % (g_dt, g_rf))


def test_forest_parallel_matches_serial():
    X, y = toy(n=800, d=12, c=3, seed=11, noise=2.0)
    a = RandomForestClassifier(n_estimators=16, n_jobs=1, random_state=42).fit(X, y)
    b = RandomForestClassifier(n_estimators=16, n_jobs=4, random_state=42).fit(X, y)
    check("forest/parallel_equals_serial",
          bool(np.array_equal(a.predict(X), b.predict(X))))


def test_forest_oob():
    X, y = toy(n=900, d=10, c=3, seed=5, noise=2.0)
    rf = RandomForestClassifier(n_estimators=40, oob_score=True, n_jobs=1,
                                random_state=0).fit(X, y)
    tr, va = stratified_split(y, 0.3, seed=1)
    rf2 = RandomForestClassifier(n_estimators=40, n_jobs=1,
                                 random_state=0).fit(X[tr], y[tr])
    held = accuracy(y[va], rf2.predict(X[va]))
    check("forest/oob_is_sane", rf.oob_score_ is not None
          and abs(rf.oob_score_ - held) < 0.12,
          "oob %.4f vs holdout %.4f" % (rf.oob_score_, held))


def test_max_features_resolution():
    X, y = toy(n=400, d=100, c=3)
    for mf, want in (("sqrt", 10), ("log2", 6), (0.25, 25), (None, 100), (7, 7)):
        rf = RandomForestClassifier(n_estimators=2, max_features=mf, n_jobs=1,
                                    random_state=0).fit(X, y)
        check("forest/max_features_%s" % mf, rf.max_features_used_ == want,
              "got %d want %d" % (rf.max_features_used_, want))


def test_sparse_highdim_like_image_task():
    """Shape-alike of the real image data: sparse, non-negative, L2-normalised."""
    rng = np.random.default_rng(13)
    n, d, c = 1200, 200, 10
    cent = rng.random((c, d)) ** 3
    y = rng.integers(0, c, size=n)
    counts = rng.poisson(cent[y] * 8.0).astype(np.float32)
    counts[rng.random(counts.shape) < 0.8] = 0.0
    X = counts / np.maximum(np.linalg.norm(counts, axis=1, keepdims=True), 1e-9)
    tr, va = stratified_split(y, 0.25, seed=0)
    rf = RandomForestClassifier(n_estimators=50, n_jobs=0,
                                random_state=0).fit(X[tr], y[tr])
    a = accuracy(y[va], rf.predict(X[va]))
    check("integration/sparse_highdim", a > 3.0 / c,
          "val acc %.4f vs chance %.4f" % (a, 1.0 / c))


def test_binned_descent_matches_raw():
    """The forest's OOB scorer descends on bin indices while predict() descends
    on raw values.  Both must land on the same leaf for every row, or OOB
    accuracy silently measures a different tree than the one being used.
    """
    from models.random_forest import _predict_binned
    from models.decision_tree import apply_bins
    rng = np.random.default_rng(17)
    X, y = toy(n=900, d=14, c=4, seed=17, noise=2.0)
    X[rng.random(X.shape) < 0.3] = 0.0     # ties, as in the real image data
    clf = DecisionTreeClassifier(max_depth=9, random_state=0).fit(X, y)
    Xb, _ = apply_bins(X, clf.bin_edges_)
    rows = np.arange(X.shape[0])
    check("tree/binned_descent_equals_raw",
          bool(np.array_equal(_predict_binned(clf, Xb, rows),
                              clf.predict_encoded(X))))


def test_sparse_binning_resolution():
    """Plain quantiles collapse on sparse columns: with 85% zeros, every
    quantile below 0.85 lands on zero and folds into one edge, so asking for
    64 bins yields about ten.  Measured on the real image_test.csv: median 10
    of 64.  sparse_aware must recover the budget.
    """
    rng = np.random.default_rng(23)
    X = (rng.random((4000, 40)) ** 2).astype(np.float32)
    X[rng.random(X.shape) < 0.85] = 0.0
    plain = compute_bin_edges(X, n_bins=64, rng=rng, sparse_aware=False)
    smart = compute_bin_edges(X, n_bins=64, rng=rng, sparse_aware=True)
    mp = float(np.median([len(e) + 1 for e in plain]))
    ms = float(np.median([len(e) + 1 for e in smart]))
    check("binning/sparse_aware_recovers_resolution", ms > 3 * mp,
          "plain %.0f bins vs sparse-aware %.0f of 64" % (mp, ms))
    # zero must still be separable from the smallest non-zero
    ok = True
    for j in range(X.shape[1]):
        if len(smart[j]):
            ok &= bool(np.array_equal(X[:, j] < smart[j][0], X[:, j] == 0))
    check("binning/sparse_aware_isolates_zero", ok)
    # and the fit/predict split rule must still agree, which is the invariant
    Xb, _ = apply_bins(X, smart)
    agree = True
    for j in range(X.shape[1]):
        for b in range(len(smart[j])):
            agree &= bool(np.array_equal(Xb[:, j] <= b, X[:, j] < smart[j][b]))
    check("binning/sparse_aware_split_rule_equivalence", agree)


def test_dense_binning_unchanged():
    """The BERT features are dense; sparse_aware must leave them alone."""
    rng = np.random.default_rng(29)
    X = rng.normal(0, 1, size=(2000, 20)).astype(np.float32)
    a = compute_bin_edges(X, n_bins=32, rng=np.random.default_rng(1), sparse_aware=False)
    b = compute_bin_edges(X, n_bins=32, rng=np.random.default_rng(1), sparse_aware=True)
    check("binning/dense_path_untouched",
          all(np.array_equal(x, y) for x, y in zip(a, b)))


def test_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    rep = per_class_report(y, p)
    check("metrics/accuracy", abs(rep["accuracy"] - 4 / 6) < 1e-12)
    check("metrics/support", list(rep["support"]) == [2.0, 2.0, 2.0])


def test_against_sklearn_oracle():
    try:
        from sklearn.tree import DecisionTreeClassifier as SkTree
        from sklearn.ensemble import RandomForestClassifier as SkForest
    except ImportError:
        print("SKIP sklearn oracle (not installed)")
        return
    X, y = toy(n=2000, d=15, c=4, seed=21, noise=2.5)
    tr, va = stratified_split(y, 0.3, seed=0)
    mine = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X[tr], y[tr])
    theirs = SkTree(max_depth=8, random_state=0).fit(X[tr], y[tr])
    a, b = (accuracy(y[va], mine.predict(X[va])),
            accuracy(y[va], theirs.predict(X[va])))
    check("oracle/tree_within_5pts", abs(a - b) < 0.05,
          "mine %.4f sklearn %.4f" % (a, b))
    mine_f = RandomForestClassifier(n_estimators=100, n_jobs=0,
                                    random_state=0).fit(X[tr], y[tr])
    theirs_f = SkForest(n_estimators=100, random_state=0,
                        n_jobs=-1).fit(X[tr], y[tr])
    a, b = (accuracy(y[va], mine_f.predict(X[va])),
            accuracy(y[va], theirs_f.predict(X[va])))
    check("oracle/forest_within_5pts", abs(a - b) < 0.05,
          "mine %.4f sklearn %.4f" % (a, b))


class TreeModelTests(unittest.TestCase):
    """Wraps the checks above so unittest discovery picks them up."""


def _attach_tests():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            setattr(TreeModelTests, name, (lambda f: lambda self: f())(function))


_attach_tests()


if __name__ == "__main__":
    unittest.main()
