#!/bin/bash
# Critique #1 multi-seed noise check. Re-runs TabFM at seeds 1 and 2 on the
# small thin-margin datasets whose seed-0 wins were inside 0.01 of a baseline,
# to measure TabFM's own run-to-run variance vs those margins. Seed 0 already
# lives in results/phase3 (canonical); seeds 1/2 write to results/phase3_seeds/
# seed<N>/ so the canonical results are never touched.
#
# TabFM only, GPU 1 only (workstation). Resumable: existing outputs are skipped.
# Diamonds is excluded on purpose: 54k rows OOMs the 24 GB GPU (see phase4).
set -u

REPOROOT=~/tabfm-eval/upstream/tabfm
PY=~/tabfm-eval/.venv/bin/python
TABFM=~/tabfm-eval/harness/phase3_tabfm.py
RESBASE=/home/fpt/tabfm-eval/results/phase3_seeds

# dataset  task_id  model_type
DATASETS='
MIC 363711 classification
concrete_compressive_strength 363625 regression
blood-transfusion-service-center 363621 classification
'

echo "$DATASETS" | grep -v '^[[:space:]]*$' | while read -r name tid mtype; do
  [ -z "$name" ] && continue
  for seed in 1 2; do
    outdir="$RESBASE/seed$seed"
    mkdir -p "$outdir"
    for fold in 0 1 2; do
      out="$outdir/${name}_fold${fold}_tabfm.json"
      if [ -f "$out" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $name fold$fold seed$seed (exists)"
        continue
      fi
      echo "[$(date +%H:%M:%S)] TabFM $name fold$fold seed$seed -> GPU1"
      ( cd "$REPOROOT" && CUDA_VISIBLE_DEVICES=1 SEED=$seed PHASE3_OUT_DIR="$outdir" \
          "$PY" "$TABFM" "$tid" "$fold" "$name" "$mtype" ) 2>&1 | tail -n 2
    done
  done
done
echo "SEED_SWEEP_DONE $(date +%H:%M:%S)"
