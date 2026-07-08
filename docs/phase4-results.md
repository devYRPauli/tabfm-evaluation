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

## PyTorch backend (upstream main: bf16 compute + activation chunking)

All of the above uses the JAX backend at our pinned commit b6ea70b. After the study,
a TabFM author noted the repo was updated to enable bf16 computation and activation
chunking "for speed and memory." Those two changes are PyTorch-backend-only and
post-date our pin (upstream commits cc56f13 and 99d72b7); the JAX backend already
computed in bfloat16 at b6ea70b, so the JAX accuracy and the numbers above are
unaffected. The genuinely-new datapoint is the PyTorch backend at current main.

Same 4090 (GPU 1), same sweep (n_features=20, n_test=100, seed 0), PyTorch backend at
upstream main with bf16 + chunking on by default. `torch alloc` is
`torch.cuda.max_memory_allocated` (device-authoritative live working set); `torch
reserved` is the caching-allocator reservation; `nvidia-smi` is total-on-card and noisy
on this shared 2-GPU host.

| n_train | median latency | torch alloc | torch reserved | nvidia-smi used |
|---|---|---|---|---|
| 100 | 1.84 s | 3187 MiB | 3404 MiB | 4675 MiB |
| 500 | 2.19 s | 3259 MiB | 3530 MiB | 4801 MiB |
| 1000 | 3.02 s | 3363 MiB | 3696 MiB | 4967 MiB |
| 2000 | 5.05 s | 3570 MiB | 4086 MiB | 5357 MiB |
| 5000 | 14.10 s | 3994 MiB | 4894 MiB | 5381 MiB |
| 10000 | 34.53 s | 4152 MiB | 5526 MiB | 6013 MiB |
| 20000 | 87.82 s | 5156 MiB | 7582 MiB | 8069 MiB |
| 40000 | 245.37 s | 7154 MiB | 11662 MiB | 12149 MiB |

The chunking is the story. Where the JAX backend sat at a ~16.95 GB flat footprint and
OOM'd past ~20k rows on the 24 GB card, the PyTorch backend's true working set scales
gently (3.2 GB at n=100 to 7.0 GB at n=40000) and n=40000 fits with ~12 GB of headroom.
The two memory metrics are not the same quantity: the JAX column is nvidia-smi (masked by
XLA preallocation), the PyTorch `torch alloc` column is device-authoritative, so this is
not a like-for-like allocation delta. The honest comparison is "flat ~16.95 GB, OOM at
~20-30k" (JAX pin) versus "3-7 GB, fits 40k" (PyTorch main).

Latency, same 4090, directly comparable (both are wall-clock predict medians):

| n_train | JAX (b6ea70b) | PyTorch (main) | speedup |
|---|---|---|---|
| 100 | 2.07 s | 1.84 s | 1.1x |
| 500 | 2.88 s | 2.19 s | 1.3x |
| 1000 | 4.36 s | 3.02 s | 1.4x |
| 2000 | 8.87 s | 5.05 s | 1.8x |
| 5000 | 31.60 s | 14.10 s | 2.2x |
| 10000 | 107.29 s | 34.53 s | 3.1x |
| 20000 | 384.07 s | 87.82 s | 4.4x |
| 40000 | OOM (>24 GB) | 245.37 s | - |

The PyTorch-main path is faster too, and the gap widens with context (roughly par at
n=100, ~4x at n=20000). This is a same-hardware comparison of two code states (older JAX
pin vs current chunked PyTorch main), not an inherent JAX-vs-PyTorch verdict. But for
anyone running TabFM on a single 24 GB GPU today, the practical takeaway is concrete: the
updated PyTorch backend fits roughly 2x the context and runs several times faster at large
context. Raw data: results/phase4/timing_workstation-4090_gpu1_pytorch.json.

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
