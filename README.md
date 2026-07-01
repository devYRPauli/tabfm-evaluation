# TabFM: an independent reproduction and hardware study

An honest, reproducible evaluation of Google's [TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/),
a zero-shot foundation model for tabular classification and regression. This is a
standalone study in the spirit of a clean reproduction: the source code is treated
as ground truth over the blog post, every result is seeded and reproducible, and
negative results are reported plainly rather than hidden.

Model under test: [`google/tabfm-1.0.0-jax`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
(JAX backend), commit `b6ea70b` of [google-research/tabfm](https://github.com/google-research/tabfm).

## Key findings

1. TabFM is a genuine small-to-mid-data champion. On the 9 fully-scored TabArena
   datasets (a few hundred to ~50k rows) it beat every baseline, including heavily
   tuned XGBoost, random forest, and TabPFN, on the primary metric. Ranking:
   TabFM > TabPFN > tuned trees. See [results/phase3/SUMMARY.md](results/phase3/SUMMARY.md).
2. It does not generalize to every table. It failed outright on high-dimensional
   data (Bioresponse, 1777 features) where the baselines worked.
3. It is impractical at scale. A 24 GB GPU OOMs past ~10k in-context rows, and on
   CPU each 78k-150k-row fold takes about an hour, with latency super-linear in
   context. See [docs/phase4-results.md](docs/phase4-results.md).
4. It is 15 to 40x slower than the trees even where it wins. The value is zero-shot
   convenience and accuracy on small-to-mid tables, not speed.
5. Four upstream bugs were found and documented for later PRs. See
   [docs/upstream-bugs.md](docs/upstream-bugs.md).

## Repository structure

1. [`README.md`](README.md) - this document, the complete picture.
2. [`docs/`](docs/) - findings and write-ups:
   [phase0-findings.md](docs/phase0-findings.md) (weight format, API, license, hard caps),
   [tabarena-tasks.md](docs/tabarena-tasks.md) (the benchmark task list and the chosen subset),
   [phase4-results.md](docs/phase4-results.md) (hardware timing),
   [upstream-bugs.md](docs/upstream-bugs.md) (the four bugs),
   [multigpu-and-commit-pinning.md](docs/multigpu-and-commit-pinning.md),
   [safety-and-resource-limits.md](docs/safety-and-resource-limits.md),
   [progress.md](docs/progress.md) (per-phase status).
3. [`harness/`](harness/) - all runnable code: the phase harnesses, the shared
   metrics, the aggregator, and the orchestration and safety scripts (below).
4. [`diagnostics/`](diagnostics/) - the controlled experiments used to root-cause the
   bf16 permutation finding and to smoke-test TabPFN.
5. [`results/`](results/) - raw per-run JSON for every phase, plus
   [the fold-matched summary](results/phase3/SUMMARY.md).
6. [`env/`](env/) - environment notes and the pinned locks
   ([macOS arm64 CPU](env/lock-macos-arm64-cpu.txt),
   [Linux x86_64 CUDA](env/lock-linux-x86_64-cuda.txt)).
7. [`provenance/`](provenance/) - exact chip, OS, and versions for each of the three
   machines.

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
Fold-matched (TabFM and baselines compared only over folds where both exist). TabFM
won all 9 fully-scored datasets and the one fold-matched diamonds comparison. Its edge
over the tuned trees is clear (churn 0.979 vs 0.956 XGBoost, maternal 0.877 vs 0.842
random forest); its edge over TabPFN is consistent but thin (MIC 0.891 vs 0.889,
concrete R2 0.950 vs 0.949).

### Phase 4: hardware characterization ([docs/phase4-results.md](docs/phase4-results.md), [results/phase4/](results/phase4/))
Single-predict latency vs in-context rows (32-member ensemble, 20 features):

| n_train | GPU latency | GPU peak mem | CPU latency |
|---|---|---|---|
| 100 | 2.3 s | 22.75 GB | 33.4 s |
| 1000 | 4.6 s | 22.75 GB | (see docs) |
| 5000 | 31.1 s | 22.76 GB | (see docs) |
| 10000 | 105.7 s | 22.78 GB | - |
| >10000 | OOM | >24 GB | fits, slow |

Peak GPU memory is flat at ~22.75 GB (the ensemble model, not the context, fills the
card), so a 24 GB GPU has a hard context ceiling near 10k rows. The GPU is ~15 to 25x
faster than CPU where it fits; there is no speed crossover, only a memory ceiling.

## Conclusions

The "small-data champion" framing holds and is sharpened. TabFM is an excellent,
tuning-free default for small-to-mid tabular problems (it beat tuned trees and TabPFN
on every dataset it completed), but it is not a drop-in for large or high-dimensional
data: it fails on very wide tables and becomes impractically slow past ~10k rows.

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

## Licenses

The reproduction code in this repository is provided as-is for research use. TabFM's
own code is Apache-2.0; its weights are `tabfm-non-commercial-v1.0`. This study is a
non-commercial evaluation and does not redistribute the weights.
