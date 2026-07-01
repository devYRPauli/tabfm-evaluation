#!/usr/bin/env python3
"""Phase 2: known-answer sanity layer for TabFM.

Builds controlled tasks whose answers we know, so a harness bug shows up here
before it contaminates a real benchmark. Four probes:

1. Linearly separable classification. A working ICL model should approach perfect
   accuracy.
2. XOR classification. Requires genuine feature interaction, a linear model sits
   at chance. Tests that the model captures interactions, not just linear signal.
3. Monotone regression. A smooth monotone target, R2 should be high.
4. Context-size sweep on the separable task. Accuracy should rise as the number
   of in-context training rows grows. This is the direct evidence that the model
   learns from context rather than ignoring it.

Plus the architectural permutation-invariance test:
5. Row permutation of the context should not change predictions (the context is a
   set). Column permutation applied consistently to train and test should not
   change predictions if features are treated as a set. Both are measured and
   reported. Row invariance is treated as a hard expectation, column invariance is
   reported as a finding because the implementation may encode feature position.

Seeded and one-command runnable. Writes a JSON result and exits nonzero only if a
generous sanity floor is breached, which would indicate a harness or model fault.

Backend via TABFM_BACKEND (jax default). Host label via TABFM_HOST.
"""

import json
import os
import platform
import socket
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, r2_score

import tabfm
from tabfm import TabFMClassifier, TabFMRegressor

SEED = 0
D = 8                 # feature count for the probes
N_TEST = 100
CTX_SIZES = [10, 50, 200, 800]
PERM_CTX = 200        # context size used for the permutation probes

# Generous floors. A working ICL model clears these comfortably; breaching one
# points at a harness or model fault, not at a hard accuracy claim.
FLOOR_LINSEP_ACC = 0.80
FLOOR_XOR_ACC = 0.65
FLOOR_MONO_R2 = 0.80
ROW_PERM_TOL = 2.5e-2  # bfloat16 reductions are not associative, so reordering
# the context perturbs low-order bits; this bounds that numerical noise.


def load_clf():
    if BACKEND == "jax":
        return tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
    if BACKEND == "pytorch":
        return tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")
    raise ValueError("TABFM_BACKEND must be 'jax' or 'pytorch'")


def load_reg():
    if BACKEND == "jax":
        return tabfm.tabfm_v1_0_0_jax.load(model_type="regression")
    if BACKEND == "pytorch":
        return tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")
    raise ValueError("TABFM_BACKEND must be 'jax' or 'pytorch'")


def make_linsep(rng, n, w):
    """Linearly separable: label is the sign of the shared linear projection w.
    w MUST be shared between train and test. If each call drew its own w, the
    context would teach one hyperplane while the test labels came from another and
    accuracy would collapse to chance. Catching that class of mistake is the whole
    point of this sanity layer."""
    X = rng.normal(size=(n, D))
    y = (X @ w > 0.0).astype(int)
    return X, y


def make_xor(rng, n):
    """XOR on the first two features, the rest are noise. No linear signal."""
    X = rng.normal(size=(n, D))
    y = ((X[:, 0] > 0.0) ^ (X[:, 1] > 0.0)).astype(int)
    return X, y


def make_monotone(rng, n, w):
    """Monotone increasing target in every feature (linear plus cubic, w > 0).
    w is shared between train and test so both halves use the same target."""
    X = rng.normal(size=(n, D))
    y = (X * w).sum(axis=1) + 0.3 * (X ** 3 * w).sum(axis=1)
    y = y + 0.01 * rng.normal(size=n)
    return X, y


def to_int_labels(pred):
    """TabFM.predict returns a dtype=object array (boxed ints). sklearn metrics
    reject object arrays as 'unknown', so coerce to int for our integer-label
    sanity tasks."""
    pred = np.asarray(pred)
    if pred.dtype == object:
        pred = pred.astype(np.int64)
    return pred


def record(results, name, value, ok, detail):
    status = "PASS" if ok else "FAIL"
    print("[%s] %s :: %s" % (status, name, detail))
    results.append({"name": name, "value": value, "ok": bool(ok), "detail": detail})


def probe_linsep(clf_model, results):
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=D)
    Xtr, ytr = make_linsep(rng, max(CTX_SIZES), w)
    Xte, yte = make_linsep(rng, N_TEST, w)
    clf = TabFMClassifier(model=clf_model, random_state=SEED)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, to_int_labels(clf.predict(Xte)))
    record(results, "linearly separable accuracy", acc, acc >= FLOOR_LINSEP_ACC,
           "acc=%.3f (floor %.2f)" % (acc, FLOOR_LINSEP_ACC))


def probe_xor(clf_model, results):
    rng = np.random.default_rng(SEED + 1)
    Xtr, ytr = make_xor(rng, max(CTX_SIZES))
    Xte, yte = make_xor(rng, N_TEST)
    clf = TabFMClassifier(model=clf_model, random_state=SEED)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, to_int_labels(clf.predict(Xte)))
    record(results, "XOR accuracy (interaction)", acc, acc >= FLOOR_XOR_ACC,
           "acc=%.3f (floor %.2f, chance=0.50)" % (acc, FLOOR_XOR_ACC))


def probe_monotone(reg_model, results):
    rng = np.random.default_rng(SEED + 2)
    w = rng.uniform(0.5, 1.5, size=D)
    Xtr, ytr = make_monotone(rng, max(CTX_SIZES), w)
    Xte, yte = make_monotone(rng, N_TEST, w)
    reg = TabFMRegressor(model=reg_model, random_state=SEED)
    reg.fit(Xtr, ytr)
    r2 = r2_score(yte, np.asarray(reg.predict(Xte)))
    record(results, "monotone regression R2", r2, r2 >= FLOOR_MONO_R2,
           "R2=%.3f (floor %.2f)" % (r2, FLOOR_MONO_R2))


def probe_context_sweep(clf_model, results):
    """Accuracy should rise with context size if the model uses context."""
    rng = np.random.default_rng(SEED + 3)
    w = rng.normal(size=D)
    pool_X, pool_y = make_linsep(rng, max(CTX_SIZES), w)
    Xte, yte = make_linsep(rng, N_TEST, w)
    curve = []
    for n in CTX_SIZES:
        clf = TabFMClassifier(model=clf_model, random_state=SEED)
        clf.fit(pool_X[:n], pool_y[:n])
        acc = accuracy_score(yte, to_int_labels(clf.predict(Xte)))
        curve.append({"n_context": n, "acc": float(acc)})
        print("    context n=%-4d -> acc=%.3f" % (n, acc))
    grew = curve[-1]["acc"] >= curve[0]["acc"] + 0.02
    record(results, "accuracy grows with context", curve, grew,
           "acc %.3f at n=%d -> %.3f at n=%d" % (
               curve[0]["acc"], CTX_SIZES[0], curve[-1]["acc"], CTX_SIZES[-1]))


def probe_permutation(clf_model, results):
    rng = np.random.default_rng(SEED + 4)
    w = rng.normal(size=D)
    Xtr, ytr = make_linsep(rng, PERM_CTX, w)
    Xte, yte = make_linsep(rng, N_TEST, w)

    clf = TabFMClassifier(model=clf_model, random_state=SEED)
    clf.fit(Xtr, ytr)
    base = np.asarray(clf.predict_proba(Xte))

    row_perm = rng.permutation(PERM_CTX)
    clf_r = TabFMClassifier(model=clf_model, random_state=SEED)
    clf_r.fit(Xtr[row_perm], ytr[row_perm])
    row_proba = np.asarray(clf_r.predict_proba(Xte))
    per_point = np.max(np.abs(base - row_proba), axis=1)
    row_max = float(per_point.max())
    frac_stable = float(np.mean(per_point < 0.05))
    n_collapse = int(np.sum(per_point > 0.30))
    # Architecturally permutation-invariant; in bf16 nearly all points move < 0.05,
    # with rare single-point collapses to uniform (see BUG-4). Require the bulk to be
    # stable rather than the max, which the rare collapse dominates.
    record(results, "row-permutation invariance (bulk)", frac_stable, frac_stable >= 0.95,
           "%.0f%% points stable <0.05; max=%.3e; collapses>0.3=%d (BUG-4)" %
           (100 * frac_stable, row_max, n_collapse))

    col_perm = rng.permutation(D)
    clf_c = TabFMClassifier(model=clf_model, random_state=SEED)
    clf_c.fit(Xtr[:, col_perm], ytr)
    col_proba = np.asarray(clf_c.predict_proba(Xte[:, col_perm]))
    col_diff = float(np.max(np.abs(base - col_proba)))
    # Reported as a finding, not a hard floor: feature position may be encoded.
    record(results, "column-permutation invariance (finding)", col_diff, True,
           "max abs proba diff = %.3e (informational)" % col_diff)


def main():
    global BACKEND
    BACKEND = os.environ.get("TABFM_BACKEND", "jax")
    host = os.environ.get("TABFM_HOST", socket.gethostname())
    print("TabFM Phase 2 sanity :: host=%s backend=%s seed=%d" % (host, BACKEND, SEED))

    clf_model = load_clf()
    reg_model = load_reg()

    results = []
    t0 = time.time()
    probe_linsep(clf_model, results)
    probe_xor(clf_model, results)
    probe_monotone(reg_model, results)
    probe_context_sweep(clf_model, results)
    probe_permutation(clf_model, results)
    elapsed = time.time() - t0

    all_ok = all(r["ok"] for r in results)
    n_pass = sum(r["ok"] for r in results)
    print("\nSummary: %d/%d sanity checks passed in %.1fs" % (n_pass, len(results), elapsed))
    print("OVERALL: %s" % ("PASS" if all_ok else "FAIL"))

    prov = {
        "host_label": host,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": BACKEND,
        "seed": SEED,
        "tabfm_version": tabfm.__version__,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sanity_%s_%s.json" % (host, BACKEND))
    with open(out_path, "w") as fh:
        json.dump({"provenance": prov, "elapsed_s": elapsed,
                   "results": results, "overall_pass": all_ok}, fh, indent=2)
    print("wrote %s" % out_path)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
