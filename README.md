# TabFM: an independent reproduction and hardware study

An honest, reproducible evaluation of Google's TabFM (a zero-shot foundation model
for tabular classification and regression), modeled on the turboquant-m1pro
evaluation. This is a standalone study. Negative results are reported plainly.
Everything here is driven from the source code as ground truth, not the blog.

Status: Phases 0-2 complete; Phase 3 (benchmark) and Phase 4 (hardware) in progress.
See docs/progress.md for the live state.

## What TabFM is (verified from the code)

1. A zero-shot tabular classifier and regressor via in-context learning. `fit` does
   not train the network; it prepares encoders and scalers, and the training rows
   become context for a single forward pass at predict time.
2. Two HF weight repos exist. `google/tabfm-1.0.0-jax` is the canonical Orbax
   checkpoint restored through flax.nnx; `google/tabfm-1.0.0-pytorch` is a converted
   pickled state_dict validated to match the JAX model within 1e-4. We use the JAX
   backend as primary. Details in docs/phase0-findings.md.
3. Two licenses: the GitHub code is Apache-2.0, the HF weights are
   `tabfm-non-commercial-v1.0`. They govern different artifacts.
4. Hard cap: `max_classes = 10`. Default forward pass uses a 32-member ensemble.

## Hardware (three machines)

1. MacBook Pro (M1 Pro, 16 GB): orchestration only. TabFM does not fit in 16 GB and
   an early attempt OOM-restarted the machine; see docs/safety-and-resource-limits.md.
2. Mac Studio (M4 Max, 64 GB): JAX CPU reference.
3. Workstation (Threadripper PRO 5955WX, 125 GB, 2x RTX 4090): single-GPU reference.
   Multi-GPU is unusable (see Bugs). Per-machine details in provenance/.

Pinned stack: Python 3.12, jax 0.10.2, flax 0.12.7, commit b6ea70b. Google's
requirements.txt is unsatisfiable on macOS arm64 (a torch+cpu pin); we install the
`.[jax]` extras and pin our own lock (env/).

## Results

### Phase 1: conformance (done)
sklearn contract holds and predictions are bit-exact deterministic under a fixed
seed, on both Studio CPU and single-4090 GPU; CPU and GPU agree numerically.

### Phase 2: known-answer sanity (done)
The model genuinely learns from context: XOR 1.00, accuracy rises 0.69 to 0.98 as
context grows, monotone regression R2 0.997. Architecturally permutation-invariant
over context rows (row order changes predictions only at bf16 numerical scale; see
BUG-4).

### Phase 3: TabArena benchmark (in progress)
Protocol: OpenML TabArena tasks (Study 457), official repeat-0 folds, 3 folds,
fit on raw features. Models: TabFM default and ensemble vs XGBoost (light and
heavy), random forest, a linear floor, and TabPFN. Metrics: accuracy, ROC AUC, log
loss (classification); RMSE, R2 (regression).

On every small-to-mid dataset it completed, TabFM won on the primary metric
against all baselines including TabPFN. The ranking is TabFM > TabPFN > tuned
trees. TabFM's edge over the tuned trees is clear (churn 0.979 vs 0.956 XGBoost,
maternal 0.877 vs 0.842 random forest); its edge over TabPFN is consistent but
thin (MIC 0.891 vs 0.889, concrete R2 0.950 vs 0.949, wine 0.548 vs 0.544).

One genuine failure: Bioresponse (1777 features), where TabFM could not run on GPU
or CPU while the baselines handled it (TabPFN won at 0.797). The two largest
datasets (SDSS17 78k, GiveMeSomeCredit 150k) could not be benchmarked exhaustively
because TabFM is impractically slow at that scale (see Phase 4). Full per-dataset
table: results/phase3/SUMMARY.md.

### Phase 4: hardware characterization (done, full tables in docs/phase4-results.md)
Single-predict latency vs in-context rows (32-member ensemble, 20 features):
1. GPU (24 GB 4090): 2.3 s at n=100 up to 105.7 s at n=10000, then OOM. Peak GPU
   memory is flat at ~22.75 GB (the ensemble model, not the context, fills the
   card), so a 24 GB GPU has a hard context ceiling near 10k rows.
2. CPU (Studio M4 Max): 33.4 s at n=100, 76.4 s at n=500. About 15 to 25x slower
   than GPU, but fits far larger context in 64 GB.
3. No speed crossover: the GPU is always faster where it fits and simply stops
   fitting past ~10k rows. Latency is super-linear in context on both. This is why
   the 78k-150k datasets OOM the GPU and take ~1 hour per fold on CPU.

## Conclusions

1. TabFM is a genuine small-to-mid-data champion. On the 10 TabArena datasets it
   completed (a few hundred to ~50k rows), it beat every baseline, including
   heavily tuned XGBoost, random forest, and TabPFN, on the primary metric. Its
   advantage over the tuned trees is clear; over TabPFN it is consistent but thin.
2. It does not generalize to every table. It failed outright on high-dimensional
   data (Bioresponse, 1777 features), where the tree baselines and TabPFN worked.
3. It is impractical at scale. A 24 GB GPU OOMs past roughly 10k context rows, and
   on CPU each 78k-150k-row fold takes about an hour, with latency super-linear in
   context. The blog's 150k upper bound runs but is not practical to use or
   benchmark exhaustively.
4. Cost caveat. Even where TabFM wins, it is 15 to 40x slower than the trees. The
   value is zero-shot convenience and accuracy on small-to-mid tables, not speed.

Net: the "small-data champion" framing holds and is sharpened. TabFM is an
excellent, tuning-free default for small-to-mid tabular problems, and not a
drop-in for large or high-dimensional data.

## Bugs found (docs/upstream-bugs.md, pending verify + dupe-check)

1. BUG-1 / BUG-2: multi-GPU sharding crashes through the public predict path
   (default batch_size on 2 GPUs, and the batch_size=32 workaround). Single GPU
   works; the second card is unusable out of the box.
2. BUG-3: `predict` returns a dtype=object array, which sklearn metrics reject.
3. BUG-4: predictions are not exactly invariant to context row order in bfloat16
   (usually ~1e-2, rarely a single point collapses to uniform). Architecturally
   invariant; a numerical-precision matter.

## Reproduce

Environment setup and the pinned locks (macOS-arm64-cpu and linux-x86_64-cuda) are
in env/. Everything is seeded (SEED=0), pinned to commit b6ea70b, and the results
JSONs are committed under results/.

Single-machine reproduction (portable, one box with the JAX env and, for baselines,
the baselines venv):
1. Phase 1: `python harness/phase1_conformance.py`
2. Phase 2: `python harness/phase2_sanity.py`
3. Phase 3 per dataset (task ids in docs/tabarena-tasks.md), for fold in 0 1 2:
   `python harness/phase3_tabfm.py <task_id> <fold> <name> <classification|regression>`
   `python harness/phase3_baselines.py <task_id> <fold> <name> <type>`
   (TabPFN backfill: `RUN_TABPFN=1 python harness/tabpfn_backfill.py ...`, foreground.)
4. Aggregate: `python harness/aggregate_phase3.py` -> results/phase3/SUMMARY.md
5. Phase 4 timing: `python harness/phase4_timing.py` (set CONTEXT_SIZES, N_FEATURES).

Note: `harness/phase3_driver.sh` and the `gapfill_*.sh` scripts are the specific
3-machine orchestration used for this run (they hardcode the `macstudio`/`ubuntu`
SSH aliases and route TabFM to the GPU with CPU fallback). They are not portable;
use the per-dataset commands above to reproduce on any single machine.

## Coverage and limitations (honest)

1. 9 of 13 target datasets are fully scored (3 folds, TabFM and all baselines on the
   same device, cuda:0). diamonds has 1 fold-matched comparison. maternal_health_risk
   fold0 ran on CPU while folds 1-2 ran on GPU (bf16 cross-device differences are
   ~1 sample; see BUG-4).
2. Not scored: Bioresponse (TabFM failed, high-dim), SDSS17 (78k) and GiveMeSomeCredit
   (150k) (TabFM impractically slow, did not complete). These are reported as findings,
   not hidden.
