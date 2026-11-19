"""Random forest built on top of the from-scratch DecisionTreeClassifier.

Each tree is trained on a bootstrap sample of the training rows (drawn with
replacement) and considers only a random subset of features at each split. The
forest predicts by averaging the trees' class probabilities.

Two things keep it fast enough for 20,000 x 500 data:
  * the features are binned once and the bins are shared by every tree, instead of
    re-binning the data for each of the 300 trees;
  * trees are trained in parallel, one process per CPU core. The binned data is put
    in a module-level variable before the processes start, so each process can read
    it without it being copied to them.
"""

import multiprocessing as mp
import numpy as np

from models.decision_tree import (DecisionTreeClassifier, apply_bins, compute_bin_edges,
                   resolve_max_features)

_SHARED = {}  # binned training data, read by the worker processes


def _fit_one_tree(args):
    """Train one tree of the forest (runs in a worker process).

    Returns the tree and, if requested, its predictions for the rows it did not
    train on (its "out-of-bag" rows), used to compute out-of-bag accuracy.
    """
    seed, params, bootstrap, max_samples, want_oob = args
    Xb = _SHARED["Xb"]
    y = _SHARED["y"]
    edges = _SHARED["edges"]
    B = _SHARED["B"]
    C = _SHARED["C"]
    n = Xb.shape[0]

    # Bootstrap sample: n rows drawn with replacement, so some rows repeat and
    # about a third are left out.
    rng = np.random.default_rng(seed)
    if bootstrap:
        n_draw = n if max_samples is None else max(1, int(round(max_samples * n)))
        idx = rng.integers(0, n, size=n_draw)
    else:
        idx = np.arange(n)

    tree = DecisionTreeClassifier(random_state=seed, **params)
    tree.fit_binned(Xb, y, edges, B, idx, C, seed=seed)
    tree.bin_edges_ = None  # never ship 500 edge arrays back per tree

    oob_idx = oob_pred = None
    if want_oob and bootstrap:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        oob_idx = np.flatnonzero(mask)
        if oob_idx.size:
            oob_pred = _predict_binned(tree, Xb, oob_idx)
    return tree, oob_idx, oob_pred


def _predict_binned(tree, Xb, rows):
    """Descend the tree on bin indices rather than raw values.

    Each node records the bin it split on, so the test is just bin <= split_bin
    and the whole descent stays vectorised.  Used for OOB scoring, where the
    raw matrix is not available in the worker.
    """
    cur = np.zeros(rows.size, dtype=np.int64)
    feat, sb, lf, rt = (tree.feature_, tree.split_bin_, tree.left_, tree.right_)
    while True:
        act = np.flatnonzero(feat[cur] >= 0)
        if act.size == 0:
            break
        nodes = cur[act]
        go_left = Xb[rows[act], feat[nodes]] <= sb[nodes]
        cur[act] = np.where(go_left, lf[nodes], rt[nodes])
    return tree.value_[cur]


class RandomForestClassifier:
    """A forest of decision trees, each trained on a bootstrap sample with random feature subsets.

    max_features is the number of features each split may consider ("sqrt", "log2",
    a fraction or a count); fewer features make the trees more different from each
    other, which helps the average. voting="soft" averages the trees' class
    probabilities; "hard" counts one vote per tree. n_jobs=0 uses every CPU core.
    """

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 min_impurity_decrease=0.0, bootstrap=True, max_samples=None,
                 n_bins=64, n_jobs=1, random_state=None, oob_score=False,
                 voting="soft", verbose=0, sparse_aware=True):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.oob_score = oob_score
        self.voting = voting
        self.verbose = verbose
        self.sparse_aware = sparse_aware

    def fit(self, X, y):
        """Train n_estimators trees on bootstrap samples of (X, y)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        C = len(self.classes_)
        rng = np.random.default_rng(self.random_state)

        # Bin the features once; every tree reuses the same bins.
        edges = compute_bin_edges(X, self.n_bins, rng=rng,
                                  sparse_aware=self.sparse_aware)
        Xb, B = apply_bins(X, edges)
        self.bin_edges_ = edges
        self.n_features_ = X.shape[1]
        self.max_features_used_ = resolve_max_features(self.max_features, X.shape[1])

        params = dict(criterion=self.criterion, max_depth=self.max_depth,
                      min_samples_split=self.min_samples_split,
                      min_samples_leaf=self.min_samples_leaf,
                      max_features=self.max_features,
                      min_impurity_decrease=self.min_impurity_decrease,
                      n_bins=self.n_bins,
                      store_proba=(self.voting == "soft"))

        _SHARED.clear()
        _SHARED.update(Xb=Xb, y=y_enc.astype(np.int64), edges=edges, B=B, C=C)

        # Every tree's seed is drawn now, so the forest is the same whether the trees
        # are trained in parallel or one after another.
        seeds = rng.integers(0, 2**31 - 1, size=self.n_estimators)
        jobs = [(int(s), params, self.bootstrap, self.max_samples,
                 self.oob_score) for s in seeds]

        # Train the trees in parallel; if that is not possible (e.g. on Windows, where
        # "fork" does not exist), train them one after another instead.
        n_jobs = self.n_jobs if self.n_jobs and self.n_jobs > 0 else mp.cpu_count()
        results = None
        if n_jobs > 1 and self.n_estimators > 1:
            try:
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=min(n_jobs, self.n_estimators)) as pool:
                    results = pool.map(_fit_one_tree, jobs,
                                       chunksize=max(1, len(jobs) // (4 * n_jobs)))
            except Exception as exc:  # fall back rather than fail the run
                if self.verbose:
                    print("parallel fit failed (%s); falling back to serial" % exc)
                results = None
        if results is None:
            results = [_fit_one_tree(j) for j in jobs]

        self.estimators_ = [r[0] for r in results]
        for t in self.estimators_:
            t.classes_ = self.classes_

        # Out-of-bag accuracy: each row is predicted only by the trees that did not
        # train on it, and those trees vote.
        self.oob_score_ = None
        if self.oob_score and self.bootstrap:
            votes = np.zeros((X.shape[0], C), dtype=np.int32)
            for _, oob_idx, oob_pred in results:
                if oob_idx is not None and oob_idx.size:
                    np.add.at(votes, (oob_idx, oob_pred), 1)
            seen = votes.sum(axis=1) > 0
            if seen.any():
                self.oob_score_ = float(
                    (votes[seen].argmax(axis=1) == y_enc[seen]).mean())
                self.oob_coverage_ = float(seen.mean())

        _SHARED.clear()
        return self

    # ------------------------------------------------------------- prediction

    def predict_proba(self, X):
        """Average of the trees' class probabilities (or of their votes, with hard voting)."""
        X = np.asarray(X, dtype=np.float32)
        C = len(self.classes_)
        acc = np.zeros((X.shape[0], C), dtype=np.float64)
        for t in self.estimators_:
            if self.voting == "soft" and t.proba_ is not None:
                acc += t.proba_[t.apply(X)]
            else:
                acc[np.arange(X.shape[0]), t.predict_encoded(X)] += 1.0
        return acc / len(self.estimators_)

    def predict(self, X):
        """Predicted class label: the class with the highest averaged probability."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def staged_predict(self, X, checkpoints):
        """Accuracy as a function of forest size, without refitting.

        Yields (n_trees, predicted_labels) at each checkpoint - this is what the
        'how many trees do you need' plot is built from.
        """
        X = np.asarray(X, dtype=np.float32)
        C = len(self.classes_)
        acc = np.zeros((X.shape[0], C), dtype=np.float64)
        want = sorted(set(int(c) for c in checkpoints))
        for i, t in enumerate(self.estimators_, start=1):
            if self.voting == "soft" and t.proba_ is not None:
                acc += t.proba_[t.apply(X)]
            else:
                acc[np.arange(X.shape[0]), t.predict_encoded(X)] += 1.0
            if i in want:
                yield i, self.classes_[acc.argmax(axis=1)]

    @property
    def mean_depth_(self):
        """Average depth of the trees."""
        return float(np.mean([t.depth_ for t in self.estimators_]))

    @property
    def mean_leaves_(self):
        """Average number of leaves per tree."""
        return float(np.mean([t.n_leaves_ for t in self.estimators_]))
