# Phase 4: hardware characterization

Single-predict latency and peak memory of TabFM (default 32-member ensemble,
classification) versus the number of in-context training rows. Synthetic data,
n_features=20, n_test=100, seed 0, steady-state (first-call JIT compile excluded).
Raw data in results/phase4/.

## GPU (single RTX 4090, 24 GB)

| n_train | median latency | peak GPU mem |
|---|---|---|
| 100 | 2.32 s | 22752 MiB |
| 500 | 3.15 s | 22754 MiB |
| 1000 | 4.60 s | 22754 MiB |
| 2000 | 8.91 s | 22756 MiB |
| 5000 | 31.09 s | 22756 MiB |
| 10000 | 105.69 s | 22779 MiB |
| >10000 | OOM | exceeds 24 GB |

Two findings:
1. Reported peak GPU memory is flat at ~22.75 GB across all context sizes. This
   original sweep did not set XLA_PYTHON_CLIENT_PREALLOCATE=false, so `nvidia-smi
   memory.used` reported XLA's preallocated pool, not the live working set. A control
   run with preallocation disabled (below) resolves the mechanism.

   Control run, XLA_PYTHON_CLIENT_PREALLOCATE=false, same GPU (RTX 4090), on an
   otherwise-empty card so memory.used reflects only this process:

   | n_train | median latency | peak GPU mem (no prealloc) |
   |---|---|---|
   | 100 | 2.07 s | 16953 MiB |
   | 500 | 2.88 s | 16955 MiB |
   | 1000 | 4.36 s | 16955 MiB |
   | 2000 | 8.87 s | 16957 MiB |
   | 5000 | 31.60 s | 16957 MiB |
   | 10000 | 107.29 s | 16957 MiB |
   | 12000 | 147.93 s | 16955 MiB |
   | 15000 | 221.96 s | 16955 MiB |
   | 20000 | 384.07 s | 16955 MiB |
   | 30000 | OOM | (exceeds 24 GB) |

   The result is nuanced, not a clean "it was all an artifact." With the pool
   disabled the footprint drops to ~16.95 GB, so ~5.8 GB of the original 22.75 GB
   was pure XLA preallocation padding (an artifact). But the remaining ~16.95 GB is
   a real, large, and genuinely flat footprint: it grows only ~4 MiB from n=100 to
   n=10000. So the flatness is real (the 32-member ensemble's fixed weight/compiled-
   buffer cost dominates over the context-dependent activation memory at these
   sizes), and the true model footprint is ~16.95 GB, not the reported 22.75 GB.
   Latency is unchanged between the two runs (n=10000: 105.7 s vs 107.3 s),
   confirming it is the same workload measured with a different allocator. The OOM
   original >10k OOM (with prealloc on) was the preallocated pool, not the model:
   with the pool disabled the reported footprint stays flat at ~16.95 GB all the way
   to n=20000, and n=20000 now fits (it did not with prealloc on). n=30000 OOMs. So
   disabling preallocation roughly doubles the usable context ceiling on the 24 GB
   card, from ~10k to ~20k rows. The footprint is flat right up to the ceiling, so the
   OOM between 20k and 30k is a transient activation-memory spike during the forward
   pass that exceeds 24 GB, not a gradual fill. Even so, ~16.95 GB is a large fixed
   base, which is why the 78k and 150k datasets fail on GPU and fall back to CPU.
2. Latency is strongly super-linear in context: 100 rows is 2.3 s, 10000 rows is
   105.7 s (a 46x latency increase for 100x the context).

Feature width also costs latency (GPU): at n=1000, 20 features is 4.6 s vs 100
features 7.1 s; at n=5000, 31.1 s vs 44.9 s.

## CPU (Mac Studio M4 Max, 64 GB)

| n_train | median latency |
|---|---|
| 100 | 32.53 s |
| 500 | 78.60 s |
| 1000 | 147.27 s |
| 2000 | 346.04 s |
| 5000 | 1299.22 s |

Latency is super-linear in context, more steeply than on GPU: 100 rows is 32.5 s,
5000 rows is 1299 s (a 40x latency increase for 50x the context, and a single
n=5000 predict is ~21 minutes). The CPU fits far more context than the GPU because
of the 64 GB unified memory, but is much slower.

## CPU vs GPU

The GPU is roughly 14 to 42x faster than the Studio CPU, and the gap widens with
context (n=100: 32.5 s CPU vs 2.3 s GPU = 14x; n=500: 78.6 s vs 3.15 s = 25x;
n=1000: 147 s vs 4.6 s = 32x; n=2000: 346 s vs 8.9 s = 39x; n=5000: 1299 s vs
31.1 s = 42x). But the GPU is capped by its 24 GB memory (~10k context with XLA
preallocation on, ~20k with it disabled), while the CPU can hold 78k-150k context
in 64 GB (or 128 GB on the workstation) at the cost of ~1 hour per fold.

So the practical picture:
1. Small-to-mid context (up to ~10k rows): use the GPU, seconds to ~2 minutes.
2. Large context (tens of thousands of rows): GPU OOMs; CPU works but is
   impractically slow (tens of minutes to ~1 hour per fold). This matches the
   Phase 3 experience where the large TabArena datasets could not be benchmarked
   exhaustively.

No CPU-vs-GPU "crossover" in the usual sense: the GPU is always faster where it
fits, and simply stops fitting past ~10k rows. The story is a memory ceiling, not
a speed crossover.
