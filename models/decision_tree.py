"""Decision tree classifier, implemented from scratch with NumPy only.

Split finding is histogram based (the LightGBM trick): every feature is bucketed
once into at most n_bins quantile bins, and each node's best split is then found
with a single vectorised bincount + cumsum over the (feature, bin, class) cube.
A pure-Python threshold scan over 20,000 x 500 data would take hours; this takes
seconds, and it still finds an exact-on-the-grid optimum.

No scikit-learn, no external ML library. NumPy and the standard library only.
"""

import numpy as np

CRITERIA = ("gini", "entropy", "misclassification")


def impurity_from_counts(counts, criterion):
    """Impurity of class-count vectors stored in the last axis.

    counts : array (..., n_classes)
    returns: array (...)
    """
    n = counts.sum(axis=-1)
    p = counts / np.maximum(n, 1.0)[..., None]
    if criterion == "gini":
        imp = 1.0 - np.einsum("...c,...c->...", p, p)
    elif criterion == "entropy":
        safe = np.where(p > 0.0, p, 1.0)
        imp = -(p * np.log2(safe)).sum(axis=-1)
    elif criterion == "misclassification":
        imp = 1.0 - p.max(axis=-1)
    else:
        raise ValueError("criterion must be one of %s" % (CRITERIA,))
    return np.where(n > 0, imp, 0.0)


def compute_bin_edges(X, n_bins=64, max_rows=50000, rng=None,
                      sparse_aware=True, sparse_threshold=0.5):
    """Quantile bin edges, one array of cut points per feature.

    Edges at or below the column minimum are dropped (they would only create an
    empty left bin).  A constant feature gets an empty edge array and is simply
    never splittable.

    sparse_aware exists because plain quantiles collapse on sparse data, and
    the real image features are 85% zeros.  Every quantile below the 0.85 mark
    lands exactly on zero, np.unique folds them into one edge, and asking for
    64 bins yields about TEN usable cut points per column - measured on the
    real image_test.csv: median 10, max 21.  All the resolution is spent
    describing the zeros.

    So when a column is mostly one value, the quantiles are taken over its
    NON-ZERO entries instead, plus a single edge at the smallest non-zero value
    to keep zero separable.  Same budget, roughly 6x the usable thresholds.
    Dense columns (the BERT features) are untouched and take the plain path.
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    if rng is None:
        rng = np.random.default_rng(0)
    Xs = X[rng.choice(n, size=max_rows, replace=False)] if n > max_rows else X
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    qv = np.quantile(Xs, qs, axis=0)
    lo = Xs.min(axis=0)
    frac_nz = (Xs != 0).mean(axis=0)

    edges = []
    for j in range(d):
        if sparse_aware and 0.0 < frac_nz[j] < sparse_threshold:
            nzv = Xs[:, j][Xs[:, j] != 0]
            e = np.unique(np.quantile(nzv, np.linspace(0.0, 1.0, n_bins)[1:-1]))
            e = np.unique(np.concatenate([[nzv.min()], e]))
        else:
            e = np.unique(qv[:, j])
        e = e[e > lo[j]]
        edges.append(np.ascontiguousarray(e, dtype=np.float32))
    return edges


def apply_bins(X, edges):
    """Map raw features to bin indices.  bin(x) == number of edges <= x."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    B = max(1, max(len(e) for e in edges) + 1)
    dt = np.uint8 if B <= 255 else np.uint16
    Xb = np.empty((n, d), dtype=dt)
    for j in range(d):
        if len(edges[j]) == 0:
            Xb[:, j] = 0
        else:
            Xb[:, j] = np.searchsorted(edges[j], X[:, j], side="right")
    return Xb, B


def resolve_max_features(max_features, n_features):
    if max_features is None or max_features == "all":
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("unknown max_features %r" % (max_features,))
    if isinstance(max_features, float):
        return max(1, min(n_features, int(round(max_features * n_features))))
    return max(1, min(n_features, int(max_features)))


class DecisionTreeClassifier:
    """CART-style classification tree with binary splits of the form x < t."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0,
                 n_bins=64, random_state=None, store_proba=True,
                 sparse_aware=True):
        if criterion not in CRITERIA:
            raise ValueError("criterion must be one of %s" % (CRITERIA,))
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.n_bins = n_bins
        self.random_state = random_state
        self.store_proba = store_proba
        self.sparse_aware = sparse_aware

    # ---------------------------------------------------------------- fitting

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        edges = compute_bin_edges(X, self.n_bins, rng=rng,
                                  sparse_aware=self.sparse_aware)
        Xb, B = apply_bins(X, edges)
        self.bin_edges_ = edges
        return self.fit_binned(Xb, y_enc.astype(np.int64), edges, B,
                               np.arange(X.shape[0]), len(self.classes_),
                               classes=self.classes_)

    def fit_binned(self, Xb, y_enc, edges, B, idx, n_classes, classes=None,
                   seed=None):
        """Grow from a pre-binned matrix.  The forest calls this directly so the
        binning is shared by every tree instead of recomputed n_estimators times.
        """
        if classes is not None:
            self.classes_ = classes
        C = int(n_classes)
        n_features = Xb.shape[1]
        k = resolve_max_features(self.max_features, n_features)
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        max_depth = np.inf if self.max_depth is None else self.max_depth
        msl = int(self.min_samples_leaf)

        feature, threshold, left, right, value = [], [], [], [], []
        split_bin = []           # bin index of the split; lets the forest descend
        n_samp, imp_, depth_ = [], [], []   # on binned data without re-deriving it
        proba = [] if self.store_proba else None

        idx = np.asarray(idx)
        root_counts = np.bincount(y_enc[idx], minlength=C).astype(np.float64)
        stack = [(idx, 0, -1, False, root_counts)]

        while stack:
            node_idx, depth, parent, is_left, counts = stack.pop()
            nid = len(feature)
            if parent >= 0:
                if is_left:
                    left[parent] = nid
                else:
                    right[parent] = nid
            m = int(node_idx.size)
            imp = float(impurity_from_counts(counts, self.criterion))
            feature.append(-1)
            threshold.append(np.float32(np.nan))
            split_bin.append(-1)
            left.append(-1)
            right.append(-1)
            value.append(int(counts.argmax()))
            n_samp.append(m)
            imp_.append(imp)
            depth_.append(depth)
            if proba is not None:
                proba.append(counts / max(m, 1))

            if (depth >= max_depth or m < self.min_samples_split
                    or m < 2 * msl or imp <= 0.0):
                continue

            feats = (np.arange(n_features) if k >= n_features
                     else rng.choice(n_features, size=k, replace=False))
            best = self._best_split(Xb, node_idx, y_enc, feats, C, imp, m)
            if best is None:
                continue
            f, b, gain = best
            if gain <= self.min_impurity_decrease:
                continue

            go_left = Xb[node_idx, f] <= b
            li = node_idx[go_left]
            ri = node_idx[~go_left]
            if li.size < msl or ri.size < msl:
                continue
            feature[nid] = int(f)
            threshold[nid] = np.float32(edges[f][b])
            split_bin[nid] = int(b)
            lc = np.bincount(y_enc[li], minlength=C).astype(np.float64)
            stack.append((ri, depth + 1, nid, False, counts - lc))
            stack.append((li, depth + 1, nid, True, lc))

        self.feature_ = np.asarray(feature, dtype=np.int32)
        self.threshold_ = np.asarray(threshold, dtype=np.float32)
        self.split_bin_ = np.asarray(split_bin, dtype=np.int32)
        self.left_ = np.asarray(left, dtype=np.int32)
        self.right_ = np.asarray(right, dtype=np.int32)
        self.value_ = np.asarray(value, dtype=np.int32)
        self.n_node_samples_ = np.asarray(n_samp, dtype=np.int64)
        self.impurity_ = np.asarray(imp_, dtype=np.float64)
        self.node_depth_ = np.asarray(depth_, dtype=np.int32)
        self.proba_ = (np.asarray(proba, dtype=np.float32)
                       if proba is not None else None)
        self.n_classes_ = C
        return self

    def _best_split(self, Xb, node_idx, y_enc, feats, C, parent_imp, m):
        """Exact best split over the bin grid for the given feature subset."""
        k = int(feats.size)
        step = max(1, int(2_000_000 // max(k, 1)))

        hi = 0
        for s in range(0, m, step):
            chunk = Xb[np.ix_(node_idx[s:s + step], feats)]
            if chunk.size:
                hi = max(hi, int(chunk.max()))
        Beff = hi + 1
        if Beff < 2:
            return None

        counts = np.zeros(k * Beff * C, dtype=np.float64)
        base = (np.arange(k, dtype=np.int64) * Beff)[None, :]
        for s in range(0, m, step):
            sl = node_idx[s:s + step]
            cc = Xb[np.ix_(sl, feats)].astype(np.int64)
            flat = ((cc + base) * C + y_enc[sl][:, None]).ravel()
            counts += np.bincount(flat, minlength=k * Beff * C)
        counts = counts.reshape(k, Beff, C)

        cum = counts.cumsum(axis=1)[:, :-1, :]            # left counts
        nl = cum.sum(axis=-1)
        nr = m - nl
        cr = counts.sum(axis=1)[:, None, :] - cum         # right counts
        il = impurity_from_counts(cum, self.criterion)
        ir = impurity_from_counts(cr, self.criterion)
        weighted = (nl * il + nr * ir) / m
        valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
        gains = np.where(valid, parent_imp - weighted, -np.inf)

        pos = int(gains.argmax())
        gain = float(gains.flat[pos])
        if not np.isfinite(gain):
            return None
        fi, b = divmod(pos, Beff - 1)
        return int(feats[fi]), int(b), gain

    # ------------------------------------------------------------- prediction

    def apply(self, X):
        X = np.asarray(X, dtype=np.float32)
        cur = np.zeros(X.shape[0], dtype=np.int64)
        feat, thr = self.feature_, self.threshold_
        lf, rt = self.left_, self.right_
        while True:
            act = np.flatnonzero(feat[cur] >= 0)
            if act.size == 0:
                break
            nodes = cur[act]
            go_left = X[act, feat[nodes]] < thr[nodes]
            cur[act] = np.where(go_left, lf[nodes], rt[nodes])
        return cur

    def predict_encoded(self, X):
        return self.value_[self.apply(X)]

    def predict(self, X):
        return self.classes_[self.predict_encoded(X)]

    def predict_proba(self, X):
        leaves = self.apply(X)
        if self.proba_ is None:
            out = np.zeros((X.shape[0], self.n_classes_), dtype=np.float32)
            out[np.arange(X.shape[0]), self.value_[leaves]] = 1.0
            return out
        return self.proba_[leaves]

    # ---------------------------------------------------------------- summary

    @property
    def n_nodes_(self):
        return int(self.feature_.size)

    @property
    def n_leaves_(self):
        return int((self.feature_ < 0).sum())

    @property
    def depth_(self):
        return int(self.node_depth_.max())

    def describe(self):
        return {"n_nodes": self.n_nodes_, "n_leaves": self.n_leaves_,
                "depth": self.depth_, "criterion": self.criterion,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features}
