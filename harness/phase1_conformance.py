#!/usr/bin/env python3
"""Phase 1: sklearn conformance and determinism check for TabFM.

Self-contained and seeded. Loads the TabFM classifier and regressor, fits on a
small synthetic task (fit only prepares transforms, it does not train the net),
and verifies the sklearn contract plus determinism under a fixed seed. Writes a
JSON result and exits nonzero if any hard check fails, so a single command tells
you whether the basics hold on this machine.

Backend is selected with TABFM_BACKEND (jax by default, or pytorch). Host label
for the result filename comes from TABFM_HOST or the system hostname.

Usage:
    python harness/phase1_conformance.py
    TABFM_BACKEND=pytorch TABFM_HOST=workstation python harness/phase1_conformance.py
"""

import json
import os
import platform
import socket
import sys
import time
from importlib import metadata

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import tabfm
from tabfm import TabFMClassifier, TabFMRegressor

SEED = 0
N_SAMPLES = 250
N_TEST = 50
N_FEATURES = 8
N_CLASSES = 3


def pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def load_model(model_type, backend):
    if backend == "jax":
        return tabfm.tabfm_v1_0_0_jax.load(model_type=model_type)
    if backend == "pytorch":
        return tabfm.tabfm_v1_0_0_pytorch.load(model_type=model_type)
    raise ValueError("TABFM_BACKEND must be 'jax' or 'pytorch', got %r" % backend)


def jax_devices():
    try:
        import jax
        return [str(d) for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        return ["jax-devices-error: %s" % exc]


def provenance(backend):
    return {
        "host_label": os.environ.get("TABFM_HOST", socket.gethostname()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": backend,
        "seed": SEED,
        "versions": {
            "tabfm": pkg_version("tabfm"),
            "jax": pkg_version("jax"),
            "jaxlib": pkg_version("jaxlib"),
            "flax": pkg_version("flax"),
            "torch": pkg_version("torch"),
            "numpy": pkg_version("numpy"),
            "scikit-learn": pkg_version("scikit-learn"),
        },
        "jax_devices": jax_devices() if backend == "jax" else None,
        "task": {
            "n_samples": N_SAMPLES,
            "n_test": N_TEST,
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
        },
    }


def check(name, ok, detail, checks):
    """Record a check, print a concise line, return ok."""
    status = "PASS" if ok else "FAIL"
    print("[%s] %s :: %s" % (status, name, detail))
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return ok


def run_classification(checks):
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=1,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        random_state=SEED,
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED, stratify=y
    )
    model = load_model("classification", BACKEND)

    clf = TabFMClassifier(model=model, random_state=SEED)
    clf.fit(Xtr, ytr)
    proba = np.asarray(clf.predict_proba(Xte))
    pred = np.asarray(clf.predict(Xte))

    check(
        "clf.predict shape",
        pred.shape == (N_TEST,),
        "got %s, want (%d,)" % (pred.shape, N_TEST),
        checks,
    )
    check(
        "clf.predict_proba shape",
        proba.shape == (N_TEST, N_CLASSES),
        "got %s, want (%d, %d)" % (proba.shape, N_TEST, N_CLASSES),
        checks,
    )
    has_classes = hasattr(clf, "classes_")
    check("clf.classes_ present", has_classes, str(getattr(clf, "classes_", None)), checks)
    if has_classes:
        check(
            "clf.classes_ matches labels",
            set(np.asarray(clf.classes_).tolist()) == set(np.unique(y).tolist()),
            "classes_=%s" % np.asarray(clf.classes_).tolist(),
            checks,
        )
    row_sums = proba.sum(axis=1)
    check(
        "proba rows sum to 1",
        np.allclose(row_sums, 1.0, atol=1e-4),
        "max abs dev from 1.0 = %.2e" % np.max(np.abs(row_sums - 1.0)),
        checks,
    )
    if has_classes:
        argmax_pred = np.asarray(clf.classes_)[np.argmax(proba, axis=1)]
        check(
            "predict equals argmax(proba)",
            np.array_equal(pred, argmax_pred),
            "%d / %d rows agree" % (int(np.sum(pred == argmax_pred)), N_TEST),
            checks,
        )

    # Determinism: a fresh classifier with the same seed and model.
    clf2 = TabFMClassifier(model=model, random_state=SEED)
    clf2.fit(Xtr, ytr)
    proba2 = np.asarray(clf2.predict_proba(Xte))
    max_diff = float(np.max(np.abs(proba - proba2)))
    check(
        "clf determinism (fixed seed)",
        np.array_equal(proba, proba2),
        "max abs diff across two runs = %.3e" % max_diff,
        checks,
    )
    return {"proba_determinism_max_abs_diff": max_diff}


def run_regression(checks):
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=5,
        noise=0.1,
        random_state=SEED,
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=N_TEST, random_state=SEED
    )
    model = load_model("regression", BACKEND)

    reg = TabFMRegressor(model=model, random_state=SEED)
    reg.fit(Xtr, ytr)
    pred = np.asarray(reg.predict(Xte))

    check(
        "reg.predict shape",
        pred.shape == (N_TEST,),
        "got %s, want (%d,)" % (pred.shape, N_TEST),
        checks,
    )
    check(
        "reg.predict finite",
        bool(np.all(np.isfinite(pred))),
        "min=%.3f max=%.3f" % (float(np.min(pred)), float(np.max(pred))),
        checks,
    )

    reg2 = TabFMRegressor(model=model, random_state=SEED)
    reg2.fit(Xtr, ytr)
    pred2 = np.asarray(reg2.predict(Xte))
    max_diff = float(np.max(np.abs(pred - pred2)))
    check(
        "reg determinism (fixed seed)",
        np.array_equal(pred, pred2),
        "max abs diff across two runs = %.3e" % max_diff,
        checks,
    )
    return {"pred_determinism_max_abs_diff": max_diff}


def main():
    global BACKEND
    BACKEND = os.environ.get("TABFM_BACKEND", "jax")
    prov = provenance(BACKEND)
    print("TabFM Phase 1 conformance :: host=%s backend=%s" % (prov["host_label"], BACKEND))
    print("versions: %s" % json.dumps(prov["versions"]))
    if prov["jax_devices"] is not None:
        print("jax devices: %s" % prov["jax_devices"])

    # TABFM_TASK selects which model(s) to load. On a 24 GB GPU only one full
    # TabFM model fits, so load one task at a time there (both is fine on CPU).
    task = os.environ.get("TABFM_TASK", "both")
    checks = []
    t0 = time.time()
    metrics = {}
    if task in ("both", "classification"):
        metrics.update(run_classification(checks))
    if task in ("both", "regression"):
        metrics.update(run_regression(checks))
    elapsed = time.time() - t0

    all_ok = all(c["ok"] for c in checks)
    n_pass = sum(c["ok"] for c in checks)
    print("\nSummary: %d/%d checks passed in %.1fs" % (n_pass, len(checks), elapsed))
    print("OVERALL: %s" % ("PASS" if all_ok else "FAIL"))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "phase1")
    os.makedirs(out_dir, exist_ok=True)
    task_suffix = "" if task == "both" else "_" + task
    out_path = os.path.join(out_dir, "conformance_%s_%s%s.json" % (prov["host_label"], BACKEND, task_suffix))
    with open(out_path, "w") as fh:
        json.dump({"provenance": prov, "elapsed_s": elapsed,
                   "checks": checks, "metrics": metrics, "overall_pass": all_ok},
                  fh, indent=2)
    print("wrote %s" % out_path)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
