# Critique #1: TabFM multi-seed run variance

Re-ran TabFM (best of default/ensemble) at seeds 0, 1, 2 on the small
thin-margin datasets, same OpenML fold splits (fold is fixed by the task,
so only the model's random_state varies). Seed 0 is the canonical
results/phase3 run; seeds 1-2 in results/phase3_seeds/. GPU (single RTX 4090).

Per-fold TabFM primary metric by seed, and the spread across seeds:

| dataset | metric | fold | seed0 | seed1 | seed2 | std | range |
|---|---|---|---|---|---|---|---|
| MIC | accuracy | 0 | 0.8924 | 0.8924 | 0.8924 | 0.0000 | 0.0000 |
| MIC | accuracy | 1 | 0.8905 | 0.8905 | 0.8905 | 0.0000 | 0.0000 |
| MIC | accuracy | 2 | 0.8905 | 0.8922 | 0.8922 | 0.0008 | 0.0018 |
| concrete_compressive_strength | r2 | 0 | 0.9502 | 0.9510 | 0.9510 | 0.0004 | 0.0008 |
| concrete_compressive_strength | r2 | 1 | 0.9497 | 0.9496 | 0.9488 | 0.0004 | 0.0009 |
| concrete_compressive_strength | r2 | 2 | 0.9510 | 0.9493 | 0.9508 | 0.0008 | 0.0017 |
| blood-transfusion-service-center | accuracy | 0 | 0.7960 | 0.8000 | 0.8000 | 0.0019 | 0.0040 |
| blood-transfusion-service-center | accuracy | 1 | 0.8032 | 0.8032 | 0.8032 | 0.0000 | 0.0000 |
| blood-transfusion-service-center | accuracy | 2 | 0.8032 | 0.8032 | 0.8032 | 0.0000 | 0.0000 |

Fold-averaged per seed, then spread across the 3 seeds:

| dataset | metric | seed0 | seed1 | seed2 | seed-std | seed-range | thin vs (seed0) |
|---|---|---|---|---|---|---|---|
| MIC | accuracy | 0.8911 | 0.8917 | 0.8917 | 0.0003 | 0.0006 | TabPFN 0.889 / Optuna 0.890 |
| concrete_compressive_strength | r2 | 0.9503 | 0.9500 | 0.9502 | 0.0001 | 0.0003 | TabPFN 0.949 |
| blood-transfusion-service-center | accuracy | 0.8008 | 0.8021 | 0.8021 | 0.0006 | 0.0013 | TabPFN 0.793 / Optuna 0.787 |

## Reading

TabFM's own run-to-run variance is very small: seed-std <= 0.0006 on all three
datasets, and several folds are bit-identical across seeds (the 32-member
ensemble resampling barely perturbs the metric). So the thin seed-0 margins are
NOT a TabFM-noise artifact; TabFM sits stably where it sits.

But that does not upgrade the razor-thin wins. MIC (+0.002 vs TabPFN, +0.001 vs
Optuna) and concrete (+0.001 vs TabPFN) are within test-set granularity and the
baselines' own run variance, which was not characterized here (only TabFM was
re-seeded). They are reported as ties, not wins. blood-transfusion (+0.008 over
TabPFN) is comfortably outside TabFM's seed noise and is a more credible edge.

Limitation: this varies only the model seed on a fixed fold split; baseline-side
variance and data-resample variance are not measured.
