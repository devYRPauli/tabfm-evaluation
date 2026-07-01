#!/usr/bin/env python3
"""Strong tuned-XGBoost baseline via Optuna, to replace the soft RandomizedSearch.

Critique addressed: the "heavy" XGBoost was a small RandomizedSearch that sometimes
overfit small folds, so "beats tuned trees" read as "beats my tuning budget". This
runs a real budget (default 100 trials, TPE) with proper inner cross-validation on
the training fold only (the OpenML test fold is held out), optimizing log loss for
classification and RMSE for regression, then refits the best config and scores the
held-out test fold. It merges an "xgboost_optuna" entry into the existing
<dataset>_fold<k>_baselines.json without touching the other baselines.

Args (positional or env): TASK_ID FOLD DATASET MODEL_TYPE
Env knobs: N_TRIALS (default 100), INNER_CV (default 3)
Run in the baselines venv (has xgboost, sklearn, openml, optuna).
"""
import json
import os
import sys
import time

import numpy as np
import openml
import optuna
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_metrics as pm  # noqa: E402

SEED = 0
N_TRIALS = int(os.environ.get("N_TRIALS", "100"))
INNER_CV = int(os.environ.get("INNER_CV", "3"))
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    def arg(i, name):
        return sys.argv[i] if len(sys.argv) > i else os.environ.get(name)
    task_id = int(arg(1, "TASK_ID"))
    fold = int(arg(2, "FOLD"))
    dataset = arg(3, "DATASET")
    model_type = arg(4, "MODEL_TYPE")
    return task_id, fold, dataset, model_type


def load_task_split(task_id, fold):
    task = openml.tasks.get_task(task_id)
    ds = task.get_dataset()
    X, y, cat, names = ds.get_data(target=ds.default_target_attribute)
    tr, te = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], cat, names


def preprocessor(names, cat):
    cat_cols = [n for n, c in zip(names, cat) if c]
    num_cols = [n for n, c in zip(names, cat) if not c]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
    ])


def search_and_fit(model_type, pre, X_train, y_train):
    is_clf = model_type == "classification"
    if is_clf:
        le = LabelEncoder().fit(y_train)
        y_enc = le.transform(y_train)
        classes = le.classes_
        cv = StratifiedKFold(INNER_CV, shuffle=True, random_state=SEED)
        scoring = "neg_log_loss"
        Est = xgb.XGBClassifier
        extra = {"eval_metric": "logloss"}
    else:
        y_enc = np.asarray(y_train, dtype=float)
        classes = None
        cv = KFold(INNER_CV, shuffle=True, random_state=SEED)
        scoring = "neg_root_mean_squared_error"
        Est = xgb.XGBRegressor
        extra = {}

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        )
        pipe = Pipeline([("pre", pre),
                         ("est", Est(random_state=SEED, n_jobs=-1, **params, **extra))])
        return cross_val_score(pipe, X_train, y_enc, cv=cv, scoring=scoring, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = Pipeline([("pre", pre),
                     ("est", Est(random_state=SEED, n_jobs=-1, **study.best_params, **extra))])
    best.fit(X_train, y_enc)
    return best, classes, study.best_params, study.best_value


def main():
    task_id, fold, dataset, model_type = parse_args()
    X_tr, X_te, y_tr, y_te, cat, names = load_task_split(task_id, fold)
    pre = preprocessor(names, cat)
    t0 = time.time()
    model, classes, best_params, best_cv = search_and_fit(model_type, pre, X_tr, y_tr)
    if model_type == "classification":
        proba = np.asarray(model.predict_proba(X_te))
        pred = classes[np.asarray(model.predict(X_te))]
        metrics = pm.classification_metrics(np.asarray(y_te), pred, proba, classes)
    else:
        pred = np.asarray(model.predict(X_te))
        metrics = pm.regression_metrics(np.asarray(y_te), pred)
    elapsed = time.time() - t0

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "phase3")
    path = os.path.join(out_dir, "%s_fold%d_baselines.json" % (dataset, fold))
    if os.path.exists(path):
        d = json.load(open(path))
    else:
        d = {"provenance": {"task_id": task_id, "dataset": dataset, "fold": fold,
                            "model_type": model_type, "seed": SEED}, "results": {}}
    d["results"]["xgboost_optuna"] = {"metrics": metrics, "fit_predict_s": elapsed,
                                      "n_trials": N_TRIALS, "inner_cv": INNER_CV,
                                      "best_params": best_params, "best_cv_score": best_cv}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(d, open(path, "w"), indent=2)
    print("[optuna] %s fold%d: %s (%.0fs, %d trials)" % (dataset, fold, metrics, elapsed, N_TRIALS))


if __name__ == "__main__":
    main()
