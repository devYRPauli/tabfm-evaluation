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

   The result is nuanced, not a clean "it was all an artifact." With the pool
   disabled the footprint drops to ~16.95 GB, so ~5.8 GB of the original 22.75 GB
   was pure XLA preallocation padding (an artifact). But the remaining ~16.95 GB is
   a real, large, and genuinely flat footprint: it grows only ~4 MiB from n=100 to
   n=10000. So the flatness is real (the 32-member ensemble's fixed weight/compiled-
   buffer cost dominates over the context-dependent activation memory at these
   sizes), and the true model footprint is ~16.95 GB, not the reported 22.75 GB.
   Latency is unchanged between the two runs (n=10000: 105.7 s vs 107.3 s),
   confirming it is the same workload measured with a different allocator. The OOM
   past ~10k rows is a real observed failure; since the working set is still flat at
   ~16.95 GB at n=10000 (about 7 GB of headroom on a 24 GB card), the OOM is a sharp
   context-activation increase somewhere past 10k, not the preallocated pool being
   exceeded. The >10k points were not measured with prealloc off, so the exact
   ceiling with the pool disabled is not characterized here. This large footprint is
   why the 78k and 150k datasets fail on GPU and fall back to CPU.
2. Latency is strongly super-linear in context: 100 rows is 2.3 s, 10000 rows is
   105.7 s (a 46x latency increase for 100x the context).

Feature width also costs latency (GPU): at n=1000, 20 features is 4.6 s vs 100
features 7.1 s; at n=5000, 31.1 s vs 44.9 s.

## CPU (Mac Studio M4 Max, 64 GB)

| n_train | median latency |
|---|---|
| 100 | 33.40 s |
| 500 | 76.37 s |
| 1000 | (computing) |

(Larger CPU points still running; table to be completed. The CPU fits far more
context than the GPU because of the 64 GB unified memory, but is much slower.)

## CPU vs GPU

The GPU is roughly 15 to 25x faster than the Studio CPU at these sizes (n=100:
33.4 s CPU vs 2.3 s GPU = 14x; n=500: 76.4 s vs 3.15 s = 24x). But the GPU is
capped at ~10k context by its 24 GB memory, while the CPU can hold 78k-150k
context in 64 GB (or 128 GB on the workstation) at the cost of ~1 hour per fold.

So the practical picture:
1. Small-to-mid context (up to ~10k rows): use the GPU, seconds to ~2 minutes.
2. Large context (tens of thousands of rows): GPU OOMs; CPU works but is
   impractically slow (tens of minutes to ~1 hour per fold). This matches the
   Phase 3 experience where the large TabArena datasets could not be benchmarked
   exhaustively.

No CPU-vs-GPU "crossover" in the usual sense: the GPU is always faster where it
fits, and simply stops fitting past ~10k rows. The story is a memory ceiling, not
a speed crossover.
