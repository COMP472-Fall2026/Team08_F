"""Random forest built on top of the from-scratch DecisionTreeClassifier.

Two things make this usable at 20k x 500 scale:

  * the quantile binning is computed ONCE and shared by every tree, so growing
    300 trees does not re-bin the data 300 times;
  * trees are grown in parallel with fork-based processes.  The binned matrix
    lives in a module global set before the pool is created, so fork shares it
    copy-on-write and nothing large is ever pickled on the way in.
"""

import multiprocessing as mp
import numpy as np

from models.decision_tree import (DecisionTreeClassifier, apply_bins, compute_bin_edges,
                   resolve_max_features)

_SHARED = {}


def _fit_one_tree(args):
    seed, params, bootstrap, max_samples, want_oob = args
    Xb = _SHARED["Xb"]
    y = _SHARED["y"]
    edges = _SHARED["edges"]
    B = _SHARED["B"]
    C = _SHARED["C"]
    n = Xb.shape[0]

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
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        C = len(self.classes_)
        rng = np.random.default_rng(self.random_state)

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

        seeds = rng.integers(0, 2**31 - 1, size=self.n_estimators)
        jobs = [(int(s), params, self.bootstrap, self.max_samples,
                 self.oob_score) for s in seeds]

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
        return float(np.mean([t.depth_ for t in self.estimators_]))

    @property
    def mean_leaves_(self):
        return float(np.mean([t.n_leaves_ for t in self.estimators_]))
