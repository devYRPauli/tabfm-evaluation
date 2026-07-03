#!/bin/bash
# Critique #3 completion: find the GPU context ceiling with XLA preallocation
# disabled. The main control run capped at n=10000 (still ~16.95 GB, ~7 GB
# headroom). This extends past 10k to locate the true OOM point. Each size runs
# in its own process so an OOM does not poison the next measurement. GPU 1 only.
set -u
PY="$HOME/tabfm-eval/.venv/bin/python"
TIMING="$HOME/tabfm-eval/harness/phase4_timing.py"
cd "$HOME/tabfm-eval" || exit 1

for n in 12000 15000 20000 30000; do
  out="results/phase4/ceiling_noprealloc_n${n}.json"
  if [ -f "$out" ]; then echo "[$(date +%H:%M:%S)] SKIP n=$n (exists)"; continue; fi
  echo "[$(date +%H:%M:%S)] ceiling n=$n (prealloc off, GPU1)"
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CONTEXT_SIZES=$n OUT="$out" \
    "$PY" "$TIMING" 2>&1 | grep -E "phase4\]" | tail -3
done
echo "CEILING_DONE $(date +%H:%M:%S)"
