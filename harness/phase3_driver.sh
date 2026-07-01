#!/bin/bash
# Phase 3 full-matrix sweep driver. Runs from the MacBook (ssh orchestration only,
# no TabFM loads locally). Resumable: any (dataset, fold, runner) whose result JSON
# already exists in the local repo is skipped, so re-running continues where it
# left off after any interruption.
#
# Routing:
#   TabFM  -> try the workstation GPU (single 4090); on any non-zero exit (e.g. a
#             24 GB OOM on a large dataset) fall back to the Studio CPU under safe_run.
#   Baselines -> Studio CPU under safe_run, with the TabPFN token sourced.
# Results are rsynced back into the repo after every job so progress is durable.
#
# Usage: bash harness/phase3_driver.sh          (all datasets, folds 0 1 2)
set -u

REPO="/Users/yashrajpandey/tabfm-evaluation"
RES="$REPO/results/phase3"
FOLDS="0 1 2"
SAFE="~/tabfm-eval/harness/safe_run.sh --mem-gb 48 --min-disk-gb 5 --swap-ceil-gb 20 --"
JAX_PY="~/tabfm-eval/.venv/bin/python"
BASE_PY="~/tabfm-eval/.venv-baselines/bin/python"
TABFM="~/tabfm-eval/harness/phase3_tabfm.py"
BASELINES="~/tabfm-eval/harness/phase3_baselines.py"
REPOROOT="cd ~/tabfm-eval/upstream/tabfm"
mkdir -p "$RES"

# dataset  task_id  model_type  (ordered small -> large, GiveMeSomeCredit last)
DATASETS='
blood-transfusion-service-center 363621 classification
credit-g 363626 classification
maternal_health_risk 363685 classification
concrete_compressive_strength 363625 regression
MIC 363711 classification
Bioresponse 363620 classification
students_dropout_and_academic_success 363704 classification
churn 363623 classification
wine_quality 363708 regression
houses 363678 regression
diamonds 363631 regression
SDSS17 363699 classification
GiveMeSomeCredit 363673 classification
'

collect() {
  rsync -az macstudio:~/tabfm-eval/results/phase3/ "$RES/" 2>/dev/null
  rsync -az ubuntu:~/tabfm-eval/results/phase3/ "$RES/" 2>/dev/null
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

echo "$DATASETS" | grep -v '^[[:space:]]*$' | while read -r name task_id mtype; do
  [ -z "$name" ] && continue
  for fold in $FOLDS; do
    tabfm_json="$RES/${name}_fold${fold}_tabfm.json"
    base_json="$RES/${name}_fold${fold}_baselines.json"

    # ---- TabFM (GPU first, CPU fallback) ----
    if [ -f "$tabfm_json" ]; then
      log "SKIP tabfm $name fold$fold (exists)"
    else
      log "TabFM $name fold$fold -> GPU"
      if ssh -n -o BatchMode=yes ubuntu "$REPOROOT && CUDA_VISIBLE_DEVICES=0 $JAX_PY $TABFM $task_id $fold $name $mtype" \
           >/dev/null 2>&1; then
        log "  GPU ok"
      else
        log "  GPU failed (likely OOM), falling back to Studio CPU"
        ssh -n -o BatchMode=yes macstudio "$REPOROOT && $SAFE $JAX_PY $TABFM $task_id $fold $name $mtype" \
           >/dev/null 2>&1 && log "  CPU ok" || log "  CPU ALSO FAILED for $name fold$fold"
      fi
      collect
    fi

    # ---- Baselines (Studio CPU, token sourced) ----
    if [ -f "$base_json" ]; then
      log "SKIP baselines $name fold$fold (exists)"
    else
      log "Baselines $name fold$fold -> Studio CPU"
      ssh -n -o BatchMode=yes macstudio "source ~/.tabpfn_token 2>/dev/null; $REPOROOT && $SAFE $BASE_PY $BASELINES $task_id $fold $name $mtype" \
         >/dev/null 2>&1 && log "  baselines ok" || log "  baselines FAILED for $name fold$fold"
      collect
    fi
  done
done

log "sweep pass complete. Results in $RES"
echo "Coverage:"
ls -1 "$RES"/*.json 2>/dev/null | wc -l | awk '{print "  "$1" result files"}'
