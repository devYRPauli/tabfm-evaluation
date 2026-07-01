#!/usr/bin/env python3
"""BUG-4 investigation: why does context row order change predictions?

Controlled experiment (classification, JAX). Loads one model and reuses it.
Builds a learnable linearly-separable task with a shared labeling rule, then:

1. Control: predict twice on the SAME context order (expect zero diff).
2. Row-permute the context (X and y permuted together) at the default ensemble
   size and report whether the change is one outlier point or pervasive.
3. Sweep n_estimators to see if the effect comes from the ensemble wrapper.
4. Sweep context size to see if it correlates with row subsampling.

Prints a compact evidence table. Read the code path after, do not fix yet.
"""
import os
import numpy as np
import tabfm
from tabfm import TabFMClassifier

SEED = 0
D = 8
N_TEST = 50


def make_linsep(rng, n, w):
    X = rng.normal(size=(n, D))
    y = (X @ w > 0.0).astype(int)
    return X, y


def proba(model, n_estimators, Xtr, ytr, Xte):
    clf = TabFMClassifier(model=model, n_estimators=n_estimators, random_state=SEED)
    clf.fit(Xtr, ytr)
    return np.asarray(clf.predict_proba(Xte))


def summarize(base, other):
    per_point = np.max(np.abs(base - other), axis=1)  # max over classes per test row
    return {
        "max": float(np.max(per_point)),
        "mean": float(np.mean(per_point)),
        "n_gt_0.01": int(np.sum(per_point > 0.01)),
        "n_gt_0.10": int(np.sum(per_point > 0.10)),
        "n_gt_0.30": int(np.sum(per_point > 0.30)),
    }


def main():
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=D)

    # Fixed test set. A large context pool we can slice / permute.
    Xte, _ = make_linsep(rng, N_TEST, w)
    CTX = 200
    Xtr, ytr = make_linsep(rng, CTX, w)
    perm = rng.permutation(CTX)

    print("== control: same order twice (n_est=32) ==")
    b1 = proba(model, 32, Xtr, ytr, Xte)
    b2 = proba(model, 32, Xtr, ytr, Xte)
    print("   same-order diff:", summarize(b1, b2))

    print("== row permutation at n_est=32 (ctx=200) ==")
    p = proba(model, 32, Xtr[perm], ytr[perm], Xte)
    print("   perm diff:", summarize(b1, p))

    print("== n_estimators sweep (ctx=200) ==")
    for ne in [1, 2, 8, 32]:
        base = proba(model, ne, Xtr, ytr, Xte)
        permd = proba(model, ne, Xtr[perm], ytr[perm], Xte)
        s = summarize(base, permd)
        print("   n_est=%-3d max=%.4f mean=%.4f  >0.1:%d/%d >0.3:%d/%d"
              % (ne, s["max"], s["mean"], s["n_gt_0.10"], N_TEST, s["n_gt_0.30"], N_TEST))

    print("== context-size sweep (n_est=32) ==")
    for n in [20, 50, 100, 200]:
        cx, cy = make_linsep(np.random.default_rng(SEED + 100), n, w)
        pm = np.random.default_rng(SEED + 100).permutation(n)
        base = proba(model, 32, cx, cy, Xte)
        permd = proba(model, 32, cx[pm], cy[pm], Xte)
        s = summarize(base, permd)
        print("   ctx=%-3d max=%.4f mean=%.4f  >0.1:%d/%d"
              % (n, s["max"], s["mean"], s["n_gt_0.10"], N_TEST))


if __name__ == "__main__":
    main()
