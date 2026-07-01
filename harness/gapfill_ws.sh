#!/bin/bash
# Workstation (Linux, 125 GB, 32 core) CPU gap-fill for the large datasets that
# OOM the 24 GB GPU. Runs TabFM jobs with memory-aware concurrency: it launches a
# new job only when a slot is free AND enough RAM is free, so it fills the box
# without OOM-killing. Forces the CPU backend (the venv is jax[cuda12], so hide the
# GPUs, which we reserve for Phase 4). Launch detached: nohup bash gapfill_ws.sh &
set -u
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
PY=$HOME/tabfm-eval/.venv/bin/python
H=$HOME/tabfm-eval/harness/phase3_tabfm.py
RES=$HOME/tabfm-eval/results/phase3
MAXC=2            # max concurrent heavy TabFM jobs
MIN_FREE_GB=40    # do not launch another unless this much RAM is free

# task_id|fold|name|model_type. Interleaved so a 150k job pairs with a smaller
# diamonds job rather than two 150k jobs stacking. GiveMeSomeCredit is the long pole.
JOBS='363673|0|GiveMeSomeCredit|classification
363631|1|diamonds|regression
363673|1|GiveMeSomeCredit|classification
363631|2|diamonds|regression
363673|2|GiveMeSomeCredit|classification'

free_gb(){ free -g | awk '/Mem:/{print $7}'; }
running(){ pgrep -f "[p]hase3_tabfm" | wc -l; }

echo "[$(date +%H:%M:%S)] workstation gap-fill start (maxc=$MAXC, min_free=${MIN_FREE_GB}GB)"
printf '%s\n' "$JOBS" | while IFS='|' read -r tid fold name mtype; do
  [ -z "$name" ] && continue
  [ -f "$RES/${name}_fold${fold}_tabfm.json" ] && { echo "skip $name f$fold (done)"; continue; }
  while [ "$(running)" -ge "$MAXC" ] || [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 20; done
  echo "[$(date +%H:%M:%S)] launch $name f$fold (free $(free_gb)GB, running $(running))"
  nohup $PY $H "$tid" "$fold" "$name" "$mtype" > "$HOME/tabfm-eval/gap_${name}_f${fold}.log" 2>&1 &
  sleep 40   # let it allocate before evaluating the next slot
done
wait
echo "[$(date +%H:%M:%S)] workstation gap-fill complete"
