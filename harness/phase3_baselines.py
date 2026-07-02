#!/usr/bin/env python3
"""Phase 3: sklearn/xgboost/tabpfn baseline runner (one OpenML task/fold).

Loads the same task/fold split as phase3_tabfm.py and scores five baselines
with the same metric definitions (phase3_metrics.py), so the JSON outputs are
directly comparable to the TabFM result for that dataset/fold:

    xgboost_light  -- xgboost with library defaults
    xgboost_heavy  -- RandomizedSearchCV over xgboost (~20 iters, inner cv=3)
    random_forest  -- sklearn RandomForest(n_estimators=300)
    linear_floor   -- LogisticRegression (clf) / Ridge (reg)
    tabpfn         -- TabPFNClassifier/Regressor if importable, else recorded
                       as unavailable with a reason (missing dependency or a
                       dataset that exceeds TabPFN's documented size limits)

xgboost/random_forest/linear_floor/xgboost run through a sklearn
ColumnTransformer: one-hot (handle_unknown="ignore") for columns flagged
categorical by OpenML's cat_indicator, median-impute + standardize for the
rest. TabPFN is intentionally NOT run through that preprocessing -- per its
own docs, scaling and one-hot encoding are "not effective" for it and it is
built to consume raw numeric/categorical columns directly, the same raw
pandas frame TabFM gets.

XGBoost's sklearn wrapper is wrapped in a LabelEncodedClassifier for
classification tasks so string/categorical OpenML target labels (e.g.
"low risk"/"mid risk"/"high risk") round-trip safely regardless of the
installed xgboost version's own label handling; predict()/predict_proba()
are decoded back to the original label space before scoring so metrics see
the same label space as every other baseline.

Args (positional argv, or env if argv omitted):
    TASK_ID FOLD DATASET MODEL_TYPE

Usage:
    python harness/phase3_baselines.py 363685 0 maternal_health_risk classification

Must be run inside the baselines venv (~/tabfm-eval/.venv-baselines), under
safe_run.sh.
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
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_metrics as pm  # noqa: E402

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when tabpfn is absent
    TabPFNClassifier = TabPFNRegressor = None
    TABPFN_IMPORT_ERROR = repr(exc)

SEED = int(os.environ.get("SEED", "0"))
np.random.seed(SEED)

# Hard per-model wall-clock cap so no single baseline (a stuck TabPFN download, a
# runaway search) can stall an unattended overnight sweep. On timeout the model is
# recorded as failed and the run moves on.
MODEL_TIMEOUT_S = 1800


class ModelTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ModelTimeout("model exceeded %ds" % MODEL_TIMEOUT_S)

# TabPFN-3's documented recommended ceiling (README: "Usage & Compatibility ->
# Dataset Sizes"); above this it still runs with ignore_pretraining_limits=True
# but accuracy is not guaranteed, so we treat it as "too big" rather than force it.
TABPFN_MAX_ROWS = 100_000
TABPFN_MAX_FEATURES = 2_000

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
}


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Encodes y to 0..K-1 for fit, decodes predictions back to the original
    label space. Insulates a wrapped classifier from OpenML's raw string/
    categorical target labels regardless of that classifier's own label
    handling, and keeps classes_ / predict_proba column order consistent
    with every other baseline for phase3_metrics."""

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


def build_preprocessor(names, cat_indicator):
    cat_cols = [n for n, is_cat in zip(names, cat_indicator) if is_cat]
    num_cols = [n for n, is_cat in zip(names, cat_indicator) if not is_cat]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
    ])


def build_models(model_type, names, cat_indicator):
    models = {}
    if model_type == "classification":
        models["xgboost_light"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", LabelEncodedClassifier(xgb.XGBClassifier(random_state=SEED, eval_metric="logloss"))),
        ])
        heavy_search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=SEED, eval_metric="logloss"),
            param_distributions=XGB_PARAM_DIST, n_iter=20, cv=3, random_state=SEED, n_jobs=-1)
        models["xgboost_heavy"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", LabelEncodedClassifier(heavy_search)),
        ])
        models["random_forest"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ])
        models["linear_floor"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", LogisticRegression(max_iter=1000)),
        ])
    else:
        models["xgboost_light"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", xgb.XGBRegressor(random_state=SEED)),
        ])
        heavy_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=SEED),
            param_distributions=XGB_PARAM_DIST, n_iter=20, cv=3, random_state=SEED, n_jobs=-1)
        models["xgboost_heavy"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", heavy_search),
        ])
        models["random_forest"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", RandomForestRegressor(n_estimators=300, random_state=SEED)),
        ])
        models["linear_floor"] = Pipeline([
            ("pre", build_preprocessor(names, cat_indicator)),
            ("est", Ridge()),
        ])
    return models


def build_tabpfn(model_type, n_train, n_features):
    """Returns (estimator_or_None, unavailable_reason_or_None). Raw data only
    -- no ColumnTransformer, see module docstring."""
    if os.environ.get("RUN_TABPFN") != "1":
        return None, ("skipped: RUN_TABPFN!=1. TabPFN hangs on a C-level network "
                      "call when backgrounded in the batch sweep (signal timeout "
                      "cannot interrupt it); run as a separate TTY pass to backfill.")
    if TabPFNClassifier is None:
        return None, "import failed: %s" % TABPFN_IMPORT_ERROR
    if n_train > TABPFN_MAX_ROWS or n_features > TABPFN_MAX_FEATURES:
        return None, ("dataset exceeds TabPFN's recommended size (n_train=%d, "
                       "n_features=%d; limits %d/%d)" %
                       (n_train, n_features, TABPFN_MAX_ROWS, TABPFN_MAX_FEATURES))
    if model_type == "classification":
        return LabelEncodedClassifier(TabPFNClassifier(random_state=SEED)), None
    return TabPFNRegressor(random_state=SEED), None


def run_model(name, est, X_train, y_train, X_test, y_test, model_type):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MODEL_TIMEOUT_S)
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
    finally:
        signal.alarm(0)
    print("  [%s] fit+predict %.1fs :: %s" % (name, elapsed, metrics))
    return {"metrics": metrics, "fit_predict_s": elapsed}


def versions():
    v = {"python": platform.python_version(), "xgboost": xgb.__version__}
    for pkg in ("openml", "scikit-learn", "numpy", "pandas"):
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = None
    try:
        v["tabpfn"] = metadata.version("tabpfn")
    except Exception:
        v["tabpfn"] = None
    return v


def main():
    task_id, fold, dataset, model_type = parse_args()
    print("Baselines Phase 3 :: task=%d fold=%d dataset=%s model_type=%s seed=%d" %
          (task_id, fold, dataset, model_type, SEED))

    X_train, X_test, y_train, y_test, cat_indicator, names = load_task_split(task_id, fold)

    models = build_models(model_type, names, cat_indicator)
    tabpfn_est, tabpfn_reason = build_tabpfn(model_type, len(X_train), X_train.shape[1])
    if tabpfn_est is not None:
        models["tabpfn"] = tabpfn_est

    results = {}
    for name, est in models.items():
        try:
            results[name] = run_model(name, est, X_train, y_train, X_test, y_test, model_type)
        except Exception as exc:
            print("  [%s] FAILED: %s" % (name, exc))
            results[name] = {"status": "failed", "reason": repr(exc)}
    if tabpfn_reason is not None:
        print("  [tabpfn] unavailable: %s" % tabpfn_reason)
        results["tabpfn"] = {"status": "unavailable", "reason": tabpfn_reason}

    prov = {
        "task_id": task_id,
        "dataset": dataset,
        "fold": fold,
        "model_type": model_type,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "seed": SEED,
        "versions": versions(),
    }

    out_dir = os.environ.get("PHASE3_OUT_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "phase3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s_fold%d_baselines.json" % (dataset, fold))
    with open(out_path, "w") as fh:
        json.dump({"provenance": prov, "results": results}, fh, indent=2)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
