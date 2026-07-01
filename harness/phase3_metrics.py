#!/usr/bin/env python3
"""Phase 3: shared metric helpers for the TabFM vs. baselines benchmark harness.

Both phase3_tabfm.py and phase3_baselines.py import this module so every
estimator (TabFM presets and sklearn/xgboost/tabpfn baselines) is scored the
same way, on the same metric definitions, from the same result JSON shape.

Classification:
    accuracy  -- predicted labels coerced to the estimator's classes_ dtype
                 before sklearn.metrics.accuracy_score (TabFM.predict returns a
                 dtype=object array of boxed labels, which sklearn otherwise
                 rejects as an "unknown" target type).
    roc_auc   -- binary: positive class is classes_[1], score is proba[:, 1]
                 against y binarized against that positive class.
                 multiclass: sklearn's own multi_class="ovr" with labels=classes_
                 so proba columns and label set stay aligned.
                 Wrapped in try/except -- a degenerate fold (e.g. a class
                 missing from y_test) raises inside sklearn, and we record None
                 rather than crash the whole benchmark run.
    log_loss  -- sklearn.metrics.log_loss(y, proba, labels=classes_). Also
                 guarded: a single unstable fold should not take down the rest
                 of the run.

Regression:
    rmse -- sqrt(mean_squared_error)
    r2   -- r2_score
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score


def coerce_labels(y_pred, classes_):
    """TabFM.predict returns dtype=object; coerce to classes_' dtype so sklearn
    metrics see a proper typed array instead of raising on 'unknown' type."""
    y_pred = np.asarray(y_pred)
    target_dtype = np.asarray(classes_).dtype
    if y_pred.dtype != target_dtype:
        y_pred = y_pred.astype(target_dtype)
    return y_pred


def accuracy(y_true, y_pred, classes_):
    y_pred = coerce_labels(y_pred, classes_)
    return float(accuracy_score(y_true, y_pred))


def roc_auc(y_true, proba, classes_):
    """None on failure (e.g. a class absent from y_test in a small fold)."""
    try:
        classes_ = np.asarray(classes_)
        proba = np.asarray(proba)
        if len(classes_) == 2:
            y_bin = (np.asarray(y_true) == classes_[1]).astype(int)
            return float(roc_auc_score(y_bin, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes_))
    except Exception:
        return None


def log_loss_score(y_true, proba, classes_):
    """None on failure, same rationale as roc_auc."""
    try:
        return float(log_loss(y_true, proba, labels=classes_))
    except Exception:
        return None


def classification_metrics(y_true, y_pred, proba, classes_):
    return {
        "accuracy": accuracy(y_true, y_pred, classes_),
        "roc_auc": roc_auc(y_true, proba, classes_),
        "log_loss": log_loss_score(y_true, proba, classes_),
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
