#!/usr/bin/env python3
"""Phase 4: hardware timing harness for TabFM PyTorch backend.

Measures single-predict latency (steady-state, excluding first-call JIT
compile) and peak GPU memory as a function of the number of in-context
training rows (n_train), at a fixed n_features/n_test, on synthetic
classification data.

Runs the PyTorch backend at current upstream main (activation chunking +
bfloat16 compute are on by default).

For each n_train in the sweep:
    1. Generate synthetic classification data (sklearn make_classification,
       seed 0), n_train context rows + n_test query rows, n_features cols.
    2. Load the TabFM classification model fresh for this n (model is
       re-loaded per point to keep GPU memory accounting clean between
       points -- TabFM's PyTorch tensors otherwise persist on-device).
    3. fit() once (in-context, no training).
    4. One warmup predict() to trigger JIT compile (untimed).
    5. Time 3 predict() calls, take the median -> latency_s_median.
    6. Query nvidia-smi memory.used on the visible GPU right after the
       timed predicts -> peak_gpu_mem_mib for this point. Also record
       peak_torch_alloc_mib and peak_torch_reserved_mib from PyTorch's
       memory tracking.

If a point raises (OOM or otherwise), it's recorded with status="OOM" (or
status="ERROR: <msg>" for a non-OOM exception) and the sweep continues to
the next n_train -- the goal is to find the GPU context-size ceiling, not
to stop at the first failure.

Env knobs:
    CONTEXT_SIZES   comma-separated list of n_train values, overrides the
                    default sweep [100, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    N_FEATURES      overrides the fixed feature count (default 20)
    N_TEST          overrides the fixed test-row count (default 100)
    DEVICE          torch device for the model (default "cuda"; the PyTorch
                    load() defaults to "cpu", so this must be set to run on GPU)
    OUT             overrides the output JSON path
                    (default results/phase4/timing_workstation-4090_gpu1_pytorch.json)

Usage:
    CUDA_VISIBLE_DEVICES=0 python harness/phase4_timing_pytorch.py
    CUDA_VISIBLE_DEVICES=1 N_FEATURES=100 CONTEXT_SIZES=1000,5000 \\
        OUT=results/phase4/timing_workstation-4090_gpu1_pytorch.json \\
        python harness/phase4_timing_pytorch.py

Must be run inside the PyTorch TabFM venv (~/tabfm-eval/.venv-pytorch).
"""

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
from sklearn.datasets import make_classification

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
    import torch

    X_train, y_train, X_test, _ = make_data(n_train, n_test, n_features)

    est = TabFMClassifier(model=model, random_state=SEED)
    est.fit(X_train, y_train)

    # Warmup: triggers JIT compile, excluded from timing.
    for _ in range(N_WARMUP):
        est.predict_proba(X_test)

    # Reset PyTorch memory tracking before timed predicts.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        est.predict_proba(X_test)
        times.append(time.perf_counter() - t0)

    peak_mem = gpu_mem_used_mib()
    
    # Record PyTorch memory metrics.
    peak_torch_alloc_mib = None
    peak_torch_reserved_mib = None
    if torch.cuda.is_available():
        peak_torch_alloc_mib = torch.cuda.max_memory_allocated() / 1024**2
        peak_torch_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
    
    return statistics.median(times), peak_mem, peak_torch_alloc_mib, peak_torch_reserved_mib


def main():
    context_sizes = parse_context_sizes()
    n_features = int(os.environ.get("N_FEATURES", "20"))
    n_test = int(os.environ.get("N_TEST", "100"))
    device = os.environ.get("DEVICE", "cuda")
    out_path = os.environ.get(
        "OUT", "results/phase4/timing_workstation-4090_gpu1_pytorch.json"
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
            "device": device,
            "latency_s_median": None,
            "peak_gpu_mem_mib": None,
            "peak_torch_alloc_mib": None,
            "peak_torch_reserved_mib": None,
            "status": "ok",
        }
        try:
            model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification", device=device)
            latency, peak_mem, torch_alloc, torch_reserved = run_point(n_train, n_features, n_test, model)
            record["latency_s_median"] = latency
            record["peak_gpu_mem_mib"] = peak_mem
            record["peak_torch_alloc_mib"] = torch_alloc
            record["peak_torch_reserved_mib"] = torch_reserved
            ceiling_n = n_train
            print(f"[phase4]   -> latency_s_median={latency:.4f} peak_gpu_mem_mib={peak_mem} "
                  f"torch_alloc={torch_alloc} torch_reserved={torch_reserved}",
                  file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: record & continue
            msg = str(exc)
            is_oom = (
                "CUDA out of memory" in msg
                or "OutOfMemoryError" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "OOM" in msg.upper()
                or "out of memory" in msg.lower()
            )
            record["status"] = "OOM" if is_oom else f"ERROR: {msg[:200]}"
            record["peak_gpu_mem_mib"] = gpu_mem_used_mib()
            print(f"[phase4]   -> {record['status']}", file=sys.stderr, flush=True)
        results.append(record)
        # Write incrementally so a mid-sweep interruption (shared GPU host)
        # still leaves the points completed so far on disk.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"[phase4] wrote {out_path}", file=sys.stderr)
    if ceiling_n is not None:
        print(f"[phase4] largest n_train that fit: {ceiling_n}", file=sys.stderr)


if __name__ == "__main__":
    main()
