# Progress

## Phase 0: provenance and environment. DONE
Three machines provisioned, validated, provenance in provenance/. JAX primary
backend. Two HF weight repos resolved (jax canonical, pytorch converted); see
docs/phase0-findings.md. Versions pinned jax 0.10.2 / flax 0.12.7 across machines;
env/lock-macos-arm64-cpu.txt. Commit pinned b6ea70b on all three (tabfm/src
identical to 443cbec); see docs/multigpu-and-commit-pinning.md. License split
documented.

## Phase 1: smoke and sklearn conformance. DONE (Studio CPU + workstation GPU)
Studio CPU: 10/10 checks, bit-exact deterministic. Workstation single-4090 GPU:
classification 7/7 + regression 3/3, bit-exact deterministic. CPU and GPU agree
numerically (proba dev 1.19e-7, identical regression range). GPU about 25s/model
vs about 160s for both on CPU. Results in results/phase1/.

Hardware finding: a single 24 GB GPU holds one TabFM model, not two. Loading both
the classification and regression models in one process OOMs the 4090, so the
harness loads one task at a time on GPU (TABFM_TASK).

## Phase 2: known-answer sanity layer. DONE on Studio, 6/6 green
All six checks pass on the Studio: linearly separable 0.990, XOR 1.000, monotone
R2 0.997, accuracy rises with context (0.69 at n=10 to 0.98 at n=800),
row-permutation invariance bulk (99% of points stable under 0.05), and
column-permutation informational. A harness bug was caught and fixed first (train
and test drew different labeling rules). BUG-4 root-caused (bf16 non-invariance,
architecturally invariant, rare single-point uniform collapse) and the check was
switched to a robust bulk-stability criterion. Results in results/phase2/.

## Phase 3: benchmark reproduction. SWEEP RUNNING (launched 2026-06-30 22:55)
Sweep runs in tmux `tabfm-sweep` on the MacBook under caffeinate. TabFM on the
workstation GPU with Studio-CPU fallback on OOM (Bioresponse fell back, as expected);
baselines on Studio CPU. TabPFN is SKIPPED (it hangs on a C-level network call when
backgrounded and the signal timeout cannot interrupt it); the token IS set on the
Studio at ~/.tabpfn_token, so backfill later with RUN_TABPFN=1 in a foreground TTY
pass of phase3_baselines.py per dataset.
RESUME if interrupted: `bash /Users/yashrajpandey/tabfm-evaluation/harness/phase3_driver.sh`
(skips any dataset/fold whose result JSON already exists). Watch:
`tail -f results/phase3/sweep.log`. Aggregate results/phase3/*.json in the morning.

Protocol grounded in the repo's tabarena examples: OpenML TabArena tasks (Study
457), official repeat-0 splits, fit on raw features and labels. Verified task list
in docs/tabarena-tasks.md.

Locked scope (user sign-off):
1. Datasets: 9 classification + 4 regression (the proposed subset plus the two
   stretch cases MIC 8-class and Bioresponse 1777-feature). 748 to 150,000 rows.
2. Folds: 3 per dataset (repeat-0, folds 0/1/2).
3. Models: TabFM default, TabFM.ensemble, XGBoost light + heavy, random forest,
   logistic/ridge floor, and TabPFN if it installs cleanly.
4. Metrics: accuracy, ROC AUC, log loss (classification); RMSE, R2 (regression);
   plus wall-clock (clean timing is Phase 4).
5. Accuracy on the Studio CPU under safe_run. GiveMeSomeCredit (150k) is the
   upper-claim stress case, run last with monitoring; an OOM there is a finding.

BUG-4 resolved: bf16 non-invariance to context row order (architecturally
invariant per code; usually ~1e-2, rarely one point collapses to uniform). Benign
for these set-level metrics. See docs/upstream-bugs.md.

## Phase 4: hardware characterization. NOT STARTED
Reframed after the MacBook incident to Studio CPU vs single-4090 GPU (the second
GPU is unusable per BUG-1/2, and the 16 GB MacBook is orchestration-only for
safety). See docs/safety-and-resource-limits.md.

## Safety
MacBook OOM-restart on 2026-06-30 diagnosed and remediated. All heavy runs go to
Studio and workstation under monitoring. harness/safe_run.sh + harness/sysmon.sh.
docs/safety-and-resource-limits.md.
