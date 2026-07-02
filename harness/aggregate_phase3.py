#!/usr/bin/env python3
"""Aggregate Phase 3 results into a fold-matched summary and a headline.

Correctness: for each dataset we only score on folds where BOTH a TabFM result and
a baselines result exist, so TabFM and the baselines are always compared over the
same folds (no averaging TabFM over 2 folds against a baseline over 1). Datasets
with zero common folds are listed under coverage gaps, not scored.

Reads results/phase3/*_tabfm.json and *_baselines.json, averages each model over
the common folds, decides a per-dataset winner (TabFM best preset vs best
baseline), writes results/phase3/SUMMARY.md, and prints a headline line last.
"""
import glob
import json
import os
import re
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "phase3")
TABFM_MODELS = ["default", "ensemble"]
BASE_MODELS = ["xgboost_light", "xgboost_heavy", "xgboost_optuna", "random_forest", "linear_floor", "tabpfn"]
TARGET = ["blood-transfusion-service-center", "credit-g", "maternal_health_risk",
          "students_dropout_and_academic_success", "churn", "MIC", "Bioresponse",
          "concrete_compressive_strength", "wine_quality", "houses", "diamonds",
          "SDSS17", "GiveMeSomeCredit"]


def parse(path):
    m = re.match(r"(.+)_fold(\d+)_(tabfm|baselines)\.json", os.path.basename(path))
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def collect():
    # per_fold[ds][fold]["tabfm"|"baselines"] = {model: metrics}; mtype[ds]
    per_fold = defaultdict(lambda: defaultdict(dict))
    mtype = {}
    for path in glob.glob(os.path.join(RES, "*.json")):
        ds, fold, kind = parse(path)
        if ds is None:
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        mtype[ds] = d.get("provenance", {}).get("model_type", "?")
        models = {m: r["metrics"] for m, r in d.get("results", {}).items()
                  if isinstance(r, dict) and r.get("metrics")}
        per_fold[ds][fold][kind] = models
    return per_fold, mtype


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def avg_over(folds_metrics, keys):
    return {k: mean([m.get(k) for m in folds_metrics]) for k in keys}


def fmt(v, nd=3):
    return ("%.*f" % (nd, v)) if isinstance(v, (int, float)) else "NA"


def main():
    per_fold, mtype = collect()
    lines = ["# Phase 3 results summary (fold-matched)", "",
             "TabFM vs baselines, averaged only over folds where BOTH a TabFM and a",
             "baselines result exist. Winner = TabFM best preset vs best baseline on the",
             "primary metric (accuracy for classification, R2 for regression).", ""]
    tabfm_win = tabfm_loss = tabfm_tie = 0
    clf_rows, reg_rows, gaps = [], [], []

    for ds in sorted(per_fold):
        is_clf = mtype.get(ds) == "classification"
        primary = "accuracy" if is_clf else "r2"
        keys = ["accuracy", "roc_auc", "log_loss"] if is_clf else ["rmse", "r2"]

        common = [f for f, kinds in per_fold[ds].items()
                  if "tabfm" in kinds and "baselines" in kinds]
        if not common:
            n_tf = sum("tabfm" in k for k in per_fold[ds].values())
            n_bl = sum("baselines" in k for k in per_fold[ds].values())
            gaps.append("%s: no common folds (tabfm=%d, baselines=%d)" % (ds, n_tf, n_bl))
            continue

        def model_avg(kind, name):
            fm = [per_fold[ds][f][kind][name] for f in common if name in per_fold[ds][f][kind]]
            return avg_over(fm, keys) if fm else None

        tabfm_scores = {m: model_avg("tabfm", m) for m in TABFM_MODELS}
        tabfm_scores = {m: v for m, v in tabfm_scores.items() if v}
        base_scores = {m: model_avg("baselines", m) for m in BASE_MODELS}
        base_scores = {m: v for m, v in base_scores.items() if v}

        def best(scores):
            c = [(v.get(primary), k, v) for k, v in scores.items() if v.get(primary) is not None]
            return max(c) if c else (None, None, None)

        tf_val, tf_name, tf_m = best(tabfm_scores)
        bl_val, bl_name, bl_m = best(base_scores)
        note = "" if len(common) == 3 else " (%d common fold(s))" % len(common)

        if tf_val is not None and bl_val is not None:
            verdict = "TabFM" if tf_val > bl_val else ("baseline" if tf_val < bl_val else "tie")
            tabfm_win += tf_val > bl_val
            tabfm_loss += tf_val < bl_val
            tabfm_tie += tf_val == bl_val
        else:
            verdict = "n/a"
        second = "roc_auc" if is_clf else "rmse"
        row = "| %s | %s %s | %s | %s %s | %s | %s |%s" % (
            ds, tf_name or "-", fmt(tf_val), fmt(tf_m.get(second) if tf_m else None),
            bl_name or "-", fmt(bl_val), fmt(bl_m.get(second) if bl_m else None), verdict, note)
        (clf_rows if is_clf else reg_rows).append(row)

    for ds in TARGET:
        if ds not in per_fold:
            gaps.append("%s: no results (attempted, did not complete: impractical at "
                        "scale or failed)" % ds)

    if clf_rows:
        lines += ["## Classification (primary: accuracy)", "",
                  "| dataset | TabFM best | TabFM auc | baseline best | base auc | winner |",
                  "|---|---|---|---|---|---|"] + clf_rows + [""]
    if reg_rows:
        lines += ["## Regression (primary: R2)", "",
                  "| dataset | TabFM best | TabFM rmse | baseline best | base rmse | winner |",
                  "|---|---|---|---|---|---|"] + reg_rows + [""]
    if gaps:
        lines += ["## Coverage gaps (not scored)", ""] + ["1. " + g for g in gaps] + [""]

    total = tabfm_win + tabfm_loss + tabfm_tie
    headline = ("TabFM sweep: won %d, lost %d, tied %d of %d fold-matched datasets; "
                "%d datasets unscored (coverage gaps). See results/phase3/SUMMARY.md"
                % (tabfm_win, tabfm_loss, tabfm_tie, total, len(gaps)))
    caveat = ("Note: this tally counts any nonzero margin on the primary metric as a "
              "decision. On a noise-aware reading the sub-0.005 margins (MIC, diamonds) "
              "are within seed/run variance and test-set granularity and should be read "
              "as ties, not wins (see results/phase3_seeds/SEED_VARIANCE.md); this does "
              "not change the direction of any result.")
    lines += ["## Headline", "", headline, "", caveat, ""]

    with open(os.path.join(RES, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(headline)


if __name__ == "__main__":
    main()
