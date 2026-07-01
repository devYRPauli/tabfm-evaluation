# TabFM: an independent reproduction and hardware study

An honest, reproducible evaluation of Google's [TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/),
a zero-shot foundation model for tabular classification and regression. This is a
standalone study in the spirit of a clean reproduction: the source code is treated
as ground truth over the blog post, every result is seeded and reproducible, and
negative results are reported plainly rather than hidden.

Model under test: [`google/tabfm-1.0.0-jax`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
(JAX backend), commit `b6ea70b` of [google-research/tabfm](https://github.com/google-research/tabfm).

## Key findings

1. TabFM is a strong small-to-mid-data model. On the datasets it fully completed (a
   few hundred to ~50k rows) it matched or beat every baseline on the primary metric,
   zero-shot. It is clearly ahead of the tuned trees; against TabPFN it is even, with
   margins (for example 0.891 vs 0.889) too small to separate from run variance at a
   single seed. Two caveats being closed: a multi-seed noise characterization and a
   stronger (Optuna) tree baseline are in progress. And this win rate is measured on
   the subset where TabFM was always most likely to win, since the harder datasets
   self-selected out by failing or timing out (see below). See
   [results/phase3/SUMMARY.md](results/phase3/SUMMARY.md).
2. It does not generalize to every table. It failed outright on high-dimensional
   data (Bioresponse, 1777 features) where the baselines worked.
3. It is impractical at scale. A 24 GB GPU OOMs past ~10k in-context rows, and on
   CPU each 78k-150k-row fold takes about an hour, with latency super-linear in
   context. See [docs/phase4-results.md](docs/phase4-results.md).
4. It is 15 to 40x slower than the trees even where it wins. The value is zero-shot
   convenience and accuracy on small-to-mid tables, not speed.
5. Four upstream bugs were found and documented for later PRs. See
   [docs/upstream-bugs.md](docs/upstream-bugs.md).

## What this repository contains

```
README.md                          This document, the complete picture
docs/                              Findings and write-ups
    phase0-findings.md             Weight format resolved (JAX vs pytorch), API, license, hard caps
    tabarena-tasks.md              The full 51-task TabArena suite and the 13-dataset subset used
    phase4-results.md              Hardware timing: latency and memory vs context, CPU vs GPU
    upstream-bugs.md               The four bugs found, with repros and proposed fixes
    multigpu-and-commit-pinning.md Multi-GPU sharding crash analysis and the commit-pin decision
    safety-and-resource-limits.md  The 16 GB OOM incident and the memory-safety layer built after it
    progress.md                    Per-phase status log
harness/                           All runnable code
    phase1_conformance.py          Phase 1: sklearn conformance and determinism check
    phase2_sanity.py               Phase 2: known-answer sanity layer (linsep, XOR, monotone, context)
    phase3_metrics.py              Shared metric definitions used by every estimator
    phase3_tabfm.py                Phase 3: TabFM default + ensemble on one OpenML task/fold
    phase3_baselines.py            Phase 3: XGBoost, random forest, linear floor, TabPFN
    tabpfn_backfill.py             TabPFN-only run that merges into existing baseline results
    aggregate_phase3.py            Fold-matched aggregation into results/phase3/SUMMARY.md
    phase4_timing.py               Phase 4: latency and peak memory vs context size
    safe_run.sh                    Memory watchdog: kills a job before it can OOM the machine
    sysmon.sh                      Lightweight health sampler (load, RAM, swap, disk)
    phase3_driver.sh               Three-machine orchestration used for this run (not portable)
    gapfill_ws.sh, gapfill_studio.sh, sweep_watcher.sh   Run-specific orchestration helpers
diagnostics/                       Controlled experiments
    bug4_row_permutation.py        Row-order sensitivity sweep (ensemble size, context size)
    bug4_reconcile.py              Worst-point inspection that reconciled the 0.5 outlier
    bug4_frequency.py              Frequency of the uniform-collapse across permutations
    tabpfn_smoke.py                Isolated TabPFN authentication and run test
results/                           Raw per-run JSON for every phase
    phase1/, phase2/, phase4/      Conformance, sanity, and timing results
    phase3/                        Per-dataset/fold TabFM and baseline results
    phase3/SUMMARY.md              The fold-matched head-to-head summary
env/                               Environment notes and pinned locks
    README.md                      Why Google's requirements.txt is unsatisfiable on macOS, and our locks
    lock-macos-arm64-cpu.txt       Full pinned lock for the Macs (JAX CPU)
    lock-linux-x86_64-cuda.txt     Full pinned lock for the workstation (jax[cuda12])
provenance/                        Exact chip, OS, and versions per machine
    machine-macbookpro-m1pro.md, machine-macstudio-m4max.md, machine-workstation-4090x2.md
```

## What TabFM is (verified from the code)

Full detail in [docs/phase0-findings.md](docs/phase0-findings.md). In short:

1. A zero-shot tabular classifier and regressor via in-context learning. `fit` does
   not train the network; the training rows become context for a single forward pass
   at predict time.
2. Two Hugging Face weight repos exist. `google/tabfm-1.0.0-jax` is the canonical
   Orbax checkpoint restored through flax.nnx; `google/tabfm-1.0.0-pytorch` is a
   converted state_dict validated to match within 1e-4. This study uses the JAX
   backend.
3. Two licenses: the GitHub code is Apache-2.0; the HF weights are
   `tabfm-non-commercial-v1.0`. They govern different artifacts.
4. Hard cap: `max_classes = 10`. The default forward pass is a 32-member ensemble.

## Method

1. Datasets: OpenML TabArena tasks (Study 457). The full 51-task suite and the chosen
   13-dataset subset (9 classification, 4 regression, spanning 748 to 150,000 rows)
   are in [docs/tabarena-tasks.md](docs/tabarena-tasks.md).
2. Protocol: official repeat-0 folds, 3 folds per dataset, fit on the raw OpenML
   features and labels (TabFM encodes internally; the baselines get a standard
   one-hot plus impute plus scale pipeline). Seed 0 throughout.
3. Models: TabFM default and its `.ensemble()` preset, versus XGBoost (light and a
   RandomizedSearchCV-tuned "heavy"), random forest, a logistic/ridge floor, and
   TabPFN. Metrics: accuracy, ROC AUC, log loss (classification); RMSE, R2 (regression).

## Environments and hardware

Three machines, one role each. Full provenance in [provenance/](provenance/).

1. MacBook Pro (M1 Pro, 16 GB): orchestration only. TabFM does not fit in 16 GB; an
   early attempt OOM-restarted the machine, which is why the safety layer exists (see
   [docs/safety-and-resource-limits.md](docs/safety-and-resource-limits.md)).
2. Mac Studio (M4 Max, 64 GB): JAX CPU reference.
3. Workstation (Threadripper PRO 5955WX, 125 GB, 2x RTX 4090): single-GPU reference
   (multi-GPU is unusable, see the bugs).

Pinned stack: Python 3.12, jax 0.10.2, flax 0.12.7, commit b6ea70b. Google's
`requirements.txt` is unsatisfiable on macOS arm64 (a torch+cpu pin); this study
installs the `.[jax]` extras and pins its own locks. See [env/README.md](env/README.md).

## Results

### Phase 1: conformance ([results/phase1/](results/phase1/))
The sklearn contract holds and predictions are bit-exact deterministic under a fixed
seed, on both Studio CPU and single-4090 GPU. CPU and GPU agree numerically.

### Phase 2: known-answer sanity ([results/phase2/](results/phase2/), [harness/phase2_sanity.py](harness/phase2_sanity.py))
The model genuinely learns from context: XOR 1.00, accuracy rises 0.69 to 0.98 as
context grows, monotone regression R2 0.997. It is architecturally permutation-invariant
over context rows (row order changes predictions only at bf16 numerical scale, see
[BUG-4](docs/upstream-bugs.md)).

### Phase 3: TabArena benchmark ([results/phase3/SUMMARY.md](results/phase3/SUMMARY.md))
Fold-matched (TabFM and baselines compared only over folds where both exist). Across
the fully-scored datasets, TabFM matched or beat every baseline on the primary metric.
Its edge over the tuned trees is clear (churn 0.979 vs 0.956 XGBoost, maternal 0.877 vs
0.842 random forest). Its edge over TabPFN is directionally consistent but within
single-seed run variance (MIC 0.891 vs 0.889, concrete R2 0.950 vs 0.949), so it should
not be read as a strict ranking yet. Caveat on the tree baseline: the "heavy" XGBoost is
a RandomizedSearch that sometimes overfit small folds, so "beats tuned trees" currently
means "beats this tuning budget"; an Optuna baseline is in progress. Two comparisons are
weak points, not clean wins: diamonds is a single fold-matched fold, and
maternal_health_risk mixes a CPU fold0 with GPU folds 1-2.

### Phase 4: hardware characterization ([docs/phase4-results.md](docs/phase4-results.md), [results/phase4/](results/phase4/))
Single-predict latency vs in-context rows (32-member ensemble, 20 features):

| n_train | GPU latency | GPU peak mem | CPU latency |
|---|---|---|---|
| 100 | 2.3 s | 22.75 GB | 33.4 s |
| 1000 | 4.6 s | 22.75 GB | (see docs) |
| 5000 | 31.1 s | 22.76 GB | (see docs) |
| 10000 | 105.7 s | 22.78 GB | - |
| >10000 | OOM | >24 GB | fits, slow |

Reported peak memory was flat at ~22.75 GB, but this run did NOT disable XLA
preallocation, so that figure most likely reflects XLA's preallocated pool rather
than the model's true working set. The OOM past ~10k rows is real (the job fails);
whether that limit is the model footprint or the preallocated pool being exceeded
is not yet established. A control run with XLA_PYTHON_CLIENT_PREALLOCATE=false is
pending. The GPU is ~15 to 25x
faster than CPU where it fits; there is no speed crossover, only a memory ceiling.

## Conclusions

TabFM is a strong, tuning-free default for small-to-mid tabular problems: on the
datasets it completed it matched or beat tuned trees and TabPFN. It is not a drop-in for
large or high-dimensional data: it fails on very wide tables and becomes impractically
slow past ~10k rows. The "small-data champion" framing holds; a strict model-vs-model
ranking against TabPFN does not yet, pending multi-seed noise characterization and a
stronger tree baseline.

## Bugs found

Logged in [docs/upstream-bugs.md](docs/upstream-bugs.md) (pending independent
verification and duplicate checks before filing):

1. BUG-1 / BUG-2: multi-GPU sharding crashes through the public predict path; only a
   single GPU is usable out of the box.
2. BUG-3: `predict` returns a dtype=object array, which sklearn metrics reject.
3. BUG-4: predictions are not exactly invariant to context row order in bfloat16 (a
   numerical-precision matter; architecturally the model is invariant).

## Reproducibility

Everything is seeded (SEED=0), pinned to commit b6ea70b, and the result JSONs are
committed under [results/](results/). Environment setup and both locks are in
[env/](env/).

Portable single-machine reproduction (one box with the JAX env, plus the baselines
venv for the baselines):

1. Phase 1: `python harness/phase1_conformance.py`
2. Phase 2: `python harness/phase2_sanity.py`
3. Phase 3, per dataset (task ids in [docs/tabarena-tasks.md](docs/tabarena-tasks.md)),
   for fold in 0 1 2:
   `python harness/phase3_tabfm.py <task_id> <fold> <name> <classification|regression>`
   and `python harness/phase3_baselines.py <task_id> <fold> <name> <type>`.
4. Aggregate: `python harness/aggregate_phase3.py` writes
   [results/phase3/SUMMARY.md](results/phase3/SUMMARY.md).
5. Phase 4: `python harness/phase4_timing.py` (set `CONTEXT_SIZES`, `N_FEATURES`).

The [`harness/phase3_driver.sh`](harness/phase3_driver.sh) and `gapfill_*.sh` scripts
are the specific three-machine orchestration used for this run (they hardcode the
`macstudio`/`ubuntu` SSH aliases and route TabFM to the GPU with CPU fallback). They
are not portable; use the per-dataset commands above on any single machine.

## Coverage and limitations (honest)

1. 9 of 13 target datasets are fully scored (3 folds, TabFM and all baselines on the
   same device). diamonds has 1 fold-matched comparison; maternal_health_risk fold0
   ran on CPU while folds 1-2 ran on GPU (bf16 cross-device differences are ~1 sample).
2. Not scored, reported as findings: Bioresponse (TabFM failed, high-dimensional),
   SDSS17 (78k) and GiveMeSomeCredit (150k) (TabFM impractically slow, did not complete).
3. The heavy XGBoost tuning occasionally underperformed the default on small folds
   (the RandomizedSearch overfit); this is reported as-is rather than tuned away.

## Credits and acknowledgments

This is an independent third-party evaluation. All credit for the model belongs to
its authors.

1. TabFM is by Google Research. It was introduced in the blog post "Introducing
   TabFM: A zero-shot foundation model for tabular data" (June 30, 2026) by
   Weihao Kong ([@weihaokong](https://github.com/weihaokong)) and Abhimanyu Das,
   Research Scientists at Google Research. Additional contributors to the
   [code repository](https://github.com/google-research/tabfm) include
   Erez Louidor ([@erzel](https://github.com/erzel)),
   Anna Eilering ([@ananci](https://github.com/ananci)),
   [@tamannarayan](https://github.com/tamannarayan), and
   [@tmacleod](https://github.com/tmacleod). Weights:
   [google/tabfm-1.0.0-jax and google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch).
2. The benchmark is [TabArena](https://github.com/autogluon/tabarena) (OpenML Study
   457), curated by the TabArena team, built on [OpenML](https://www.openml.org).
3. Baselines: [TabPFN](https://github.com/PriorLabs/TabPFN) by Prior Labs,
   [XGBoost](https://github.com/dmlc/xgboost), and
   [scikit-learn](https://scikit-learn.org). TabFM's own stack uses JAX and Flax.

If any author or maintainer wants a correction to how their work is described here,
please open an issue and it will be fixed.

## Licenses

The reproduction code in this repository is provided as-is for research use. TabFM's
own code is Apache-2.0; its weights are `tabfm-non-commercial-v1.0`. This study is a
non-commercial evaluation and does not redistribute the weights.
