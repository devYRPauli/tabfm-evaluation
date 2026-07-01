#!/usr/bin/env python3
"""BUG-4 reconciliation: why did phase2 report 0.5 but the sweep shows ~0.02?

Replicates the phase2 probe_permutation setup EXACTLY (D=8, PERM_CTX=200,
N_TEST=100, identical RNG draw order), then inspects the worst test point:
its base vs permuted proba, its predicted-class flip, and its distance to the
true decision boundary |X @ w|. Hypothesis: the large max-diff is a single
near-boundary point whose class flips under bf16 reordering, not a systemic
order dependence.
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

    # Exact phase2 probe_permutation RNG order.
    rng = np.random.default_rng(SEED + 4)
    w = rng.normal(size=D)
    Xtr, ytr = make_linsep(rng, PERM_CTX, w)
    Xte, yte = make_linsep(rng, N_TEST, w)

    clf = TabFMClassifier(model=model, random_state=SEED)
    clf.fit(Xtr, ytr)
    base = np.asarray(clf.predict_proba(Xte))

    row_perm = rng.permutation(PERM_CTX)
    clf_r = TabFMClassifier(model=model, random_state=SEED)
    clf_r.fit(Xtr[row_perm], ytr[row_perm])
    perm = np.asarray(clf_r.predict_proba(Xte))

    per_point = np.max(np.abs(base - perm), axis=1)
    idx = int(np.argmax(per_point))
    boundary_dist = np.abs(Xte @ w)  # 0 == exactly on the separating hyperplane

    print("max abs proba diff:", float(per_point.max()))
    print("points > 0.30:", int(np.sum(per_point > 0.30)),
          " > 0.10:", int(np.sum(per_point > 0.10)),
          " > 0.01:", int(np.sum(per_point > 0.01)), "of", N_TEST)
    print("worst point idx=%d" % idx)
    print("  base proba :", np.round(base[idx], 4).tolist())
    print("  perm proba :", np.round(perm[idx], 4).tolist())
    print("  |X@w| (boundary distance) at worst point: %.4f" % boundary_dist[idx])
    print("  median |X@w| over test set: %.4f" % float(np.median(boundary_dist)))
    order = np.argsort(-per_point)[:5]
    print("  top-5 points: per_point_diff, boundary_dist")
    for j in order:
        print("    idx=%-3d diff=%.4f  |X@w|=%.4f" % (j, per_point[j], boundary_dist[j]))


if __name__ == "__main__":
    main()
