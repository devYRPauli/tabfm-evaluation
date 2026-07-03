#!/bin/bash
# Critique #1 completion: baseline-side seed variance (Studio CPU). Re-runs the
# sklearn/xgboost/tabpfn baselines and the Optuna-tuned XGBoost at seeds 1 and 2
# on the small thin-margin datasets, matching the TabFM seed sweep, so the thin
# TabFM-vs-baseline margins can be judged against BOTH sides' run variance.
# Writes into results/phase3_seeds/seed<N>/ (canonical seed-0 untouched).
# Resumable: a (dataset,fold,seed) whose baselines file already has the optuna
# key is skipped. Runs under safe_run as the Studio memory backstop.
set -u

# Single-instance guard: refuse to start if another instance is alive. A stale
# second instance sharing this log is what caused an earlier premature-DONE race.
LOCK=/tmp/baseline_seed_sweep.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "baseline_seed_sweep already running (pid $(cat "$LOCK")); exiting"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

REPOROOT="$HOME/tabfm-eval/upstream/tabfm"
BASE_PY="$HOME/tabfm-eval/.venv-baselines/bin/python"
BASELINES="$HOME/tabfm-eval/harness/phase3_baselines.py"
OPTUNA="$HOME/tabfm-eval/harness/xgboost_optuna.py"
SAFE="$HOME/tabfm-eval/harness/safe_run.sh --mem-gb 48 --min-disk-gb 5 --swap-ceil-gb 20 --"
RESBASE="$HOME/tabfm-eval/results/phase3_seeds"

DATASETS='
MIC 363711 classification
concrete_compressive_strength 363625 regression
blood-transfusion-service-center 363621 classification
'

source "$HOME/.tabpfn_token" 2>/dev/null

echo "$DATASETS" | grep -v '^[[:space:]]*$' | while read -r name tid mtype; do
  [ -z "$name" ] && continue
  for seed in 1 2; do
    outdir="$RESBASE/seed$seed"
    mkdir -p "$outdir"
    for fold in 0 1 2; do
      base="$outdir/${name}_fold${fold}_baselines.json"
      # Done marker: MIC needs trees+optuna; others need only the trees file.
      if [ -f "$base" ]; then
        if [ "$name" != "MIC" ] || grep -q '"xgboost_optuna"' "$base"; then
          echo "[$(date +%H:%M:%S)] SKIP $name f$fold s$seed (done)"
          continue
        fi
      fi
      echo "[$(date +%H:%M:%S)] baselines $name f$fold s$seed"
      ( cd "$REPOROOT" && SEED=$seed PHASE3_OUT_DIR="$outdir" $SAFE "$BASE_PY" "$BASELINES" "$tid" "$fold" "$name" "$mtype" ) 2>&1 | tail -1
      # Optuna variance only matters for MIC: it is the only dataset with a thin
      # TabFM-vs-Optuna margin (+0.001). concrete/blood were thin vs TabPFN, so
      # their Optuna variance adds nothing to the crux and costs ~12 min/cell.
      if [ "$name" = "MIC" ]; then
        echo "[$(date +%H:%M:%S)] optuna $name f$fold s$seed"
        ( cd "$REPOROOT" && SEED=$seed PHASE3_OUT_DIR="$outdir" N_TRIALS=100 INNER_CV=3 $SAFE "$BASE_PY" "$OPTUNA" "$tid" "$fold" "$name" "$mtype" ) 2>&1 | tail -1
      fi
    done
  done
done
echo "BASELINE_SEED_DONE $(date +%H:%M:%S)"
