#!/usr/bin/env python3
"""Phase 3: TabPFN backfill for one OpenML task/fold, one process.

Fills in the "tabpfn" key of an *existing* <dataset>_fold<fold>_baselines.json
without touching any of the other baselines (xgboost_light, xgboost_heavy,
random_forest, linear_floor) so it can be run standalone, in the foreground,
while those baselines run elsewhere with n_jobs=-1 -- no CPU contention, no
re-running work that already succeeded.

Loads the same task/fold split as phase3_tabfm.py / phase3_baselines.py
(openml.tasks.get_task, get_train_test_split_indices(repeat=0, fold=fold,
sample=0)), and scores TabPFN with the same metric definitions
(phase3_metrics.py) so the result is directly comparable to the other
baselines and to TabFM for that dataset/fold.

TabPFN gets raw data, no preprocessing (per its own docs, scaling/one-hot are
not effective for it -- same rationale as phase3_baselines.py). Classification
wraps label handling in a LabelEncoder round-trip so predict()/predict_proba()
report back in the original label space regardless of what TabPFNClassifier
does internally. Regression uses TabPFNRegressor directly.

Caps (same as phase3_baselines.py's TABPFN_MAX_ROWS/TABPFN_MAX_FEATURES):
datasets with n_train > 100_000 or n_features > 2_000 are recorded as
{"status": "unavailable", "reason": ...} without being fit.

A signal.alarm(300) wraps the whole fit+predict so one stuck dataset cannot
hang the run; on timeout the result is {"status": "timeout"}.

Args (positional argv, or env if argv omitted):
    TASK_ID FOLD DATASET MODEL_TYPE

Usage:
    TABPFN_TOKEN=... python harness/tabpfn_backfill.py 363621 0 blood-transfusion-service-center classification

Must be run inside the baselines venv (~/tabfm-eval/.venv-baselines), in the
FOREGROUND over an interactive-enough session (TabPFN's first network call
hangs when this process is backgrounded/detached from a TTY) -- do not run
under safe_run.sh or with a trailing '&'.
"""

import json
import os
import platform
import signal
import sys
import time
from importlib import metadata

import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_metrics as pm  # noqa: E402

from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: E402
from tabpfn.errors import TabPFNOutOfMemoryError  # noqa: E402

SEED = 0
np.random.seed(SEED)

TABPFN_MAX_ROWS = 100_000
TABPFN_MAX_FEATURES = 2_000
FIT_PREDICT_TIMEOUT_S = 300


class ModelTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ModelTimeout("tabpfn fit+predict exceeded %ds" % FIT_PREDICT_TIMEOUT_S)


class LabelEncodedTabPFNClassifier:
    """Round-trips y through a LabelEncoder so predict()/predict_proba() report
    back in the original label space, matching every other baseline's classes_
    / metric expectations (phase3_metrics.classification_metrics)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimator.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator.predict(X))]

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


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


def run_tabpfn(model_type, X_train, y_train, X_test, y_test, device="auto",
               ignore_pretraining_limits=False):
    """Fit+predict raw TabPFN, wrapped in a hard wall-clock timeout. Returns a
    results-shaped dict: {"metrics":..., "fit_predict_s":...},
    {"status": "timeout"}, or raises TabPFNOutOfMemoryError (caller retries on
    CPU -- the Studio's shared MPS pool can be under pressure from a
    concurrently running TabFM job). ignore_pretraining_limits bypasses
    TabPFN's own conservative ">1000 samples on CPU is slow" guard for the
    CPU fallback path -- this is a performance guard, not the documented
    size cap (TABPFN_MAX_ROWS/TABPFN_MAX_FEATURES) already checked by the
    caller before we get here."""
    if model_type == "classification":
        est = LabelEncodedTabPFNClassifier(TabPFNClassifier(
            random_state=SEED, device=device,
            ignore_pretraining_limits=ignore_pretraining_limits))
    else:
        est = TabPFNRegressor(
            random_state=SEED, device=device,
            ignore_pretraining_limits=ignore_pretraining_limits)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(FIT_PREDICT_TIMEOUT_S)
    try:
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
    except ModelTimeout:
        signal.alarm(0)
        print("  [tabpfn] TIMEOUT after %ds" % FIT_PREDICT_TIMEOUT_S)
        return {"status": "timeout"}
    finally:
        signal.alarm(0)
    print("  [tabpfn] fit+predict %.1fs :: %s" % (elapsed, metrics))
    return {"metrics": metrics, "fit_predict_s": elapsed}


def versions():
    v = {"python": platform.python_version()}
    for pkg in ("openml", "scikit-learn", "numpy", "pandas", "tabpfn"):
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = None
    return v


def main():
    task_id, fold, dataset, model_type = parse_args()
    print("TabPFN backfill :: task=%d fold=%d dataset=%s model_type=%s seed=%d" %
          (task_id, fold, dataset, model_type, SEED))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "phase3")
    out_path = os.path.join(out_dir, "%s_fold%d_baselines.json" % (dataset, fold))
    if not os.path.exists(out_path):
        print("  SKIP: %s does not exist -- run phase3_baselines.py first" % out_path)
        return

    X_train, X_test, y_train, y_test, cat_indicator, names = load_task_split(task_id, fold)
    n_train, n_features = len(X_train), X_train.shape[1]

    if n_train > TABPFN_MAX_ROWS or n_features > TABPFN_MAX_FEATURES:
        print("  [tabpfn] unavailable: exceeds TabPFN size cap (n_train=%d, n_features=%d)" %
              (n_train, n_features))
        tabpfn_result = {"status": "unavailable", "reason": "exceeds TabPFN size cap"}
    else:
        try:
            tabpfn_result = run_tabpfn(model_type, X_train, y_train, X_test, y_test)
        except TabPFNOutOfMemoryError as exc:
            print("  [tabpfn] OOM on default device (%r), retrying on CPU" % exc)
            try:
                tabpfn_result = run_tabpfn(model_type, X_train, y_train, X_test, y_test,
                                            device="cpu", ignore_pretraining_limits=True)
            except Exception as exc2:
                print("  [tabpfn] FAILED on CPU retry: %r" % exc2)
                tabpfn_result = {"status": "failed", "reason": repr(exc2)}
        except Exception as exc:
            print("  [tabpfn] FAILED: %r" % exc)
            tabpfn_result = {"status": "failed", "reason": repr(exc)}

    with open(out_path) as fh:
        doc = json.load(fh)
    doc.setdefault("results", {})["tabpfn"] = tabpfn_result
    doc.setdefault("provenance", {})["tabpfn_versions"] = versions()
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
