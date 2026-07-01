#!/usr/bin/env python3
"""Phase 4: hardware timing harness for TabFM.

Measures single-predict latency (steady-state, excluding first-call JIT
compile) and peak GPU memory as a function of the number of in-context
training rows (n_train), at a fixed n_features/n_test, on synthetic
classification data.

For each n_train in the sweep:
    1. Generate synthetic classification data (sklearn make_classification,
       seed 0), n_train context rows + n_test query rows, n_features cols.
    2. Load the TabFM classification model fresh for this n (model is
       re-loaded per point to keep GPU memory accounting clean between
       points -- TabFM's JAX arrays otherwise persist on-device).
    3. fit() once (in-context, no training).
    4. One warmup predict() to trigger JIT compile (untimed).
    5. Time 3 predict() calls, take the median -> latency_s_median.
    6. Query nvidia-smi memory.used on the visible GPU right after the
       timed predicts -> peak_gpu_mem_mib for this point.

If a point raises (OOM or otherwise), it's recorded with status="OOM" (or
status="ERROR: <msg>" for a non-OOM exception) and the sweep continues to
the next n_train -- the goal is to find the GPU context-size ceiling, not
to stop at the first failure.

Env knobs:
    CONTEXT_SIZES   comma-separated list of n_train values, overrides the
                    default sweep [100, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    N_FEATURES      overrides the fixed feature count (default 20)
    N_TEST          overrides the fixed test-row count (default 100)
    OUT             overrides the output JSON path
                    (default results/phase4/timing_workstation-4090_gpu.json)

Usage:
    CUDA_VISIBLE_DEVICES=0 python harness/phase4_timing.py
    CUDA_VISIBLE_DEVICES=1 N_FEATURES=100 CONTEXT_SIZES=1000,5000 \\
        OUT=results/phase4/timing_workstation-4090_gpu1_features.json \\
        python harness/phase4_timing.py

Must be run inside the JAX TabFM venv (~/tabfm-eval/.venv).
"""

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
from sklearn.datasets import make_classification

# Measure the model's true working set, not XLA's preallocated pool. XLA
# preallocates a large fraction of the GPU by default, which makes nvidia-smi
# memory.used report a flat, misleading peak. Disable it unless the caller sets
# the env var explicitly (set XLA_PYTHON_CLIENT_PREALLOCATE=true to reproduce the
# original preallocated-pool measurement). This must run before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SEED = 0
DEFAULT_CONTEXT_SIZES = [100, 500, 1000, 2000, 5000, 10000, 20000, 40000]
N_WARMUP = 1
N_TIMED = 3


def parse_context_sizes():
    raw = os.environ.get("CONTEXT_SIZES")
    if not raw:
        return list(DEFAULT_CONTEXT_SIZES)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def gpu_mem_used_mib():
    """Query memory.used (MiB) on the CUDA device(s) visible to this process."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        lines = [int(l.strip()) for l in out.splitlines() if l.strip()]
        if visible not in (None, ""):
            idx = [int(v) for v in visible.split(",") if v.strip() != ""]
            lines = [lines[i] for i in idx if i < len(lines)]
        return max(lines) if lines else None
    except Exception:
        return None


def make_data(n_train, n_test, n_features, seed=SEED):
    n_samples = n_train + n_test
    n_informative = max(2, min(n_features, int(n_features * 0.5)))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=2,
        random_state=seed,
    )
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_test]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def run_point(n_train, n_features, n_test, model):
    from tabfm import TabFMClassifier

    X_train, y_train, X_test, _ = make_data(n_train, n_test, n_features)

    est = TabFMClassifier(model=model, random_state=SEED)
    est.fit(X_train, y_train)

    # Warmup: triggers JIT compile, excluded from timing.
    for _ in range(N_WARMUP):
        est.predict_proba(X_test)

    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        est.predict_proba(X_test)
        times.append(time.perf_counter() - t0)

    peak_mem = gpu_mem_used_mib()
    return statistics.median(times), peak_mem


def main():
    context_sizes = parse_context_sizes()
    n_features = int(os.environ.get("N_FEATURES", "20"))
    n_test = int(os.environ.get("N_TEST", "100"))
    out_path = os.environ.get(
        "OUT", "results/phase4/timing_workstation-4090_gpu.json"
    )

    import tabfm

    results = []
    ceiling_n = None
    for n_train in context_sizes:
        print(f"[phase4] n_train={n_train} n_features={n_features} n_test={n_test} ...",
              file=sys.stderr, flush=True)
        record = {
            "n_train": n_train,
            "n_features": n_features,
            "n_test": n_test,
            "latency_s_median": None,
            "peak_gpu_mem_mib": None,
            "status": "ok",
        }
        try:
            model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
            latency, peak_mem = run_point(n_train, n_features, n_test, model)
            record["latency_s_median"] = latency
            record["peak_gpu_mem_mib"] = peak_mem
            ceiling_n = n_train
            print(f"[phase4]   -> latency_s_median={latency:.4f} peak_gpu_mem_mib={peak_mem}",
                  file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: record & continue
            msg = str(exc)
            is_oom = "RESOURCE_EXHAUSTED" in msg or "OOM" in msg.upper() or "out of memory" in msg.lower()
            record["status"] = "OOM" if is_oom else f"ERROR: {msg[:200]}"
            record["peak_gpu_mem_mib"] = gpu_mem_used_mib()
            print(f"[phase4]   -> {record['status']}", file=sys.stderr, flush=True)
        results.append(record)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[phase4] wrote {out_path}", file=sys.stderr)
    if ceiling_n is not None:
        print(f"[phase4] largest n_train that fit: {ceiling_n}", file=sys.stderr)


if __name__ == "__main__":
    main()
