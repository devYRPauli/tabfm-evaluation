#!/usr/bin/env python3
"""BUG-4 frequency + mechanism: how often does a point collapse to uniform?

The worst point under one permutation went to exactly [0.5, 0.5] (uniform) despite
being a confident, far-from-boundary point. This probes how often that happens
across different context permutations, and whether NaN is involved.
"""
import numpy as np
import tabfm
from tabfm import TabFMClassifier

SEED = 0
D = 8
PERM_CTX = 200
N_TEST = 100


def make_linsep(rng, n, w):
    X = rng.normal(size=(n, D))
    y = (X @ w > 0.0).astype(int)
    return X, y


def main():
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
    rng = np.random.default_rng(SEED + 4)
    w = rng.normal(size=D)
    Xtr, ytr = make_linsep(rng, PERM_CTX, w)
    Xte, yte = make_linsep(rng, N_TEST, w)

    clf = TabFMClassifier(model=model, random_state=SEED)
    clf.fit(Xtr, ytr)
    base = np.asarray(clf.predict_proba(Xte))
    print("base: any NaN?", bool(np.isnan(base).any()),
          " proba value range: [%.5f, %.5f]" % (base.min(), base.max()))

    pr = np.random.default_rng(12345)
    for t in range(6):
        perm = pr.permutation(PERM_CTX)
        clf_r = TabFMClassifier(model=model, random_state=SEED)
        clf_r.fit(Xtr[perm], ytr[perm])
        proba = np.asarray(clf_r.predict_proba(Xte))
        per_point = np.max(np.abs(base - proba), axis=1)
        # a "collapse" = a point whose proba is (near) uniform 0.5 for binary
        near_uniform = np.sum(np.max(proba, axis=1) < 0.55)
        exact_half = np.sum(np.all(np.abs(proba - 0.5) < 1e-6, axis=1))
        print("perm %d: max_diff=%.4f  pts>0.3=%d  near_uniform(<0.55)=%d  exact_0.5=%d  NaN=%s"
              % (t, per_point.max(), int(np.sum(per_point > 0.3)),
                 int(near_uniform), int(exact_half), bool(np.isnan(proba).any())))


if __name__ == "__main__":
    main()
