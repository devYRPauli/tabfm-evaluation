#!/bin/bash
# Studio (macOS, 64 GB) CPU gap-fill for SDSS17. Conservative: ONE heavy TabFM job
# at a time under safe_run, because this is a Mac and over-packing RAM risks an OS
# restart. fold0 may already be running from an earlier manual launch; we wait for
# it, then do the remaining folds. Launch detached: nohup bash gapfill_studio.sh &
set -u
SAFE="$HOME/tabfm-eval/harness/safe_run.sh --mem-gb 52 --min-disk-gb 10 --swap-ceil-gb 30 --"
PY=$HOME/tabfm-eval/.venv/bin/python
H=$HOME/tabfm-eval/harness/phase3_tabfm.py
RES=$HOME/tabfm-eval/results/phase3

# Wait for any in-flight SDSS17 job (fold0) to finish first.
while pgrep -f "[p]hase3_tabfm.*SDSS17" >/dev/null 2>&1; do sleep 30; done

for fold in 0 1 2; do
  [ -f "$RES/SDSS17_fold${fold}_tabfm.json" ] && { echo "skip SDSS17 f$fold (done)"; continue; }
  echo "[$(date +%H:%M:%S)] SDSS17 fold$fold"
  $SAFE $PY $H 363699 "$fold" SDSS17 classification > "$HOME/tabfm-eval/gap_SDSS17_f${fold}.log" 2>&1
done
echo "[$(date +%H:%M:%S)] studio SDSS17 gap-fill complete"
