#!/usr/bin/env python3
"""Phase 3: TabFM benchmark runner (one OpenML task/fold, one process).

Loads a TabArena Study 457 OpenML task/fold, loads the matching TabFM model
ONCE for the process (classification or regression, never both -- a single
process should only hold the model_type it needs), then runs both presets
against that one loaded model:

    default  -- TabFMClassifier(model=m) / TabFMRegressor(model=m)
    ensemble -- TabFMClassifier.ensemble(model=m) / TabFMRegressor.ensemble(model=m)

fit() does not train (TabFM is in-context learning); fit+predict is timed as
one unit per preset. Results are written to
results/phase3/<dataset>_fold<fold>_tabfm.json.

Args (positional argv, or env if argv omitted):
    TASK_ID FOLD DATASET MODEL_TYPE

Usage:
    python harness/phase3_tabfm.py 363685 0 maternal_health_risk classification
    TASK_ID=363685 FOLD=0 DATASET=maternal_health_risk MODEL_TYPE=classification \\
        python harness/phase3_tabfm.py

Must be run inside the JAX TabFM venv (~/tabfm-eval/.venv), under safe_run.sh.
"""

import json
import os
import platform
import sys
import time
from importlib import metadata

import numpy as np
import openml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_metrics as pm  # noqa: E402

import tabfm  # noqa: E402
from tabfm import TabFMClassifier, TabFMRegressor  # noqa: E402

SEED = int(os.environ.get("SEED", "0"))
np.random.seed(SEED)


def parse_args():
    def arg(i, name, default=None):
        if len(sys.argv) > i:
            return sys.argv[i]
        return os.environ.get(name, default)

    task_id = int(arg(1, "TASK_ID"))
    fold = int(arg(2, "FOLD", "0"))
    dataset = arg(3, "DATASET")
    model_type = arg(4, "MODEL_TYPE")
    if model_type not in ("classification", "regression"):
        raise ValueError("MODEL_TYPE must be 'classification' or 'regression'")
    return task_id, fold, dataset, model_type


def load_task_split(task_id, fold):
    task = openml.tasks.get_task(task_id)
    ds = task.get_dataset()
    X, y, cat_indicator, names = ds.get_data(target=ds.default_target_attribute)
    train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test, cat_indicator, names


def load_model(model_type):
    return tabfm.tabfm_v1_0_0_jax.load(model_type=model_type)


def build_estimator(preset, model_type, model):
    cls = TabFMClassifier if model_type == "classification" else TabFMRegressor
    if preset == "default":
        try:
            return cls(model=model, random_state=SEED)
        except TypeError:
            return cls(model=model)
    try:
        return cls.ensemble(model=model, random_state=SEED)
    except TypeError:
        return cls.ensemble(model=model)


def run_preset(preset, model_type, model, X_train, y_train, X_test, y_test):
    est = build_estimator(preset, model_type, model)
    t0 = time.time()
    est.fit(X_train, y_train)
    if model_type == "classification":
        proba = np.asarray(est.predict_proba(X_test))
        pred = np.asarray(est.predict(X_test))
        metrics = pm.classification_metrics(y_test, pred, proba, est.classes_)
    else:
        pred = np.asarray(est.predict(X_test))
        metrics = pm.regression_metrics(y_test, pred)
    elapsed = time.time() - t0
    print("  [%s] fit+predict %.1fs :: %s" % (preset, elapsed, metrics))
    return {"metrics": metrics, "fit_predict_s": elapsed}


def versions():
    v = {"python": platform.python_version(), "tabfm": tabfm.__version__}
    for pkg in ("openml", "scikit-learn", "numpy", "pandas"):
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = None
    return v


def main():
    task_id, fold, dataset, model_type = parse_args()
    print("TabFM Phase 3 :: task=%d fold=%d dataset=%s model_type=%s seed=%d" %
          (task_id, fold, dataset, model_type, SEED))

    X_train, X_test, y_train, y_test, cat_indicator, names = load_task_split(task_id, fold)
    model = load_model(model_type)

    results = {}
    for preset in ("default", "ensemble"):
        results[preset] = run_preset(preset, model_type, model, X_train, y_train, X_test, y_test)

    try:
        import jax
        device = str(jax.devices()[0])
    except Exception:
        device = None
    prov = {
        "task_id": task_id,
        "dataset": dataset,
        "fold": fold,
        "model_type": model_type,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "seed": SEED,
        "jax_device": device,
        "versions": versions(),
    }

    out_dir = os.environ.get("PHASE3_OUT_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "phase3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s_fold%d_tabfm.json" % (dataset, fold))
    with open(out_path, "w") as fh:
        json.dump({"provenance": prov, "results": results}, fh, indent=2)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
