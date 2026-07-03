# Critique #1: seed variance (both sides) and thin-margin resolution

Re-ran TabFM at seeds 0/1/2 and the baselines (tree baselines all three
datasets; Optuna-XGBoost for MIC only, the sole thin TabFM-vs-Optuna margin)
on the same fixed OpenML fold splits. Only the model random_state varies.
Seed 0 = canonical results/phase3; seeds 1/2 = results/phase3_seeds/.

TabFM on GPU; baselines on the Studio CPU under safe_run. Fold-averaged score
per seed, then mean and population std across the available seeds.

## Per-method seed spread

| dataset | method | seed0 | seed1 | seed2 | mean | std |
|---|---|---|---|---|---|---|
| MIC | TabFM | 0.8911 | 0.8917 | 0.8917 | 0.8915 | 0.0003 |
| MIC | tabpfn | 0.8893 | - | - | 0.8893 | 0.0000 |
| MIC | xgboost_optuna | 0.8899 | 0.8882 | 0.8876 | 0.8886 | 0.0010 |
| MIC | xgboost_heavy | 0.8876 | 0.8864 | 0.8870 | 0.8870 | 0.0005 |
| MIC | random_forest | 0.8858 | 0.8864 | 0.8846 | 0.8856 | 0.0007 |
| concrete_compressive_strength | TabFM | 0.9503 | 0.9500 | 0.9502 | 0.9502 | 0.0001 |
| concrete_compressive_strength | tabpfn | 0.9491 | - | - | 0.9491 | 0.0000 |
| concrete_compressive_strength | xgboost_optuna | 0.9345 | - | - | 0.9345 | 0.0000 |
| concrete_compressive_strength | xgboost_heavy | 0.9365 | 0.9335 | 0.9307 | 0.9336 | 0.0024 |
| concrete_compressive_strength | random_forest | 0.9017 | 0.9022 | 0.9024 | 0.9021 | 0.0003 |
| blood-transfusion-service-center | TabFM | 0.8008 | 0.8021 | 0.8021 | 0.8017 | 0.0006 |
| blood-transfusion-service-center | tabpfn | 0.7928 | - | - | 0.7928 | 0.0000 |
| blood-transfusion-service-center | xgboost_optuna | 0.7874 | - | - | 0.7874 | 0.0000 |
| blood-transfusion-service-center | xgboost_heavy | 0.7754 | 0.7660 | 0.7781 | 0.7732 | 0.0051 |
| blood-transfusion-service-center | random_forest | 0.7446 | 0.7433 | 0.7366 | 0.7415 | 0.0035 |

Note: tabpfn seeds 1/2 could not be run in this environment (TabPFN hangs on a
C-level network call regardless of TTY: tested backgrounded, script-pty, and
ssh -t). Its seed-0 value is shown; its run variance is therefore unmeasured.

## Thin-margin resolution (TabFM vs comparator, seed-averaged)

| dataset | vs | margin | TabFM std | comparator std | reading |
|---|---|---|---|---|---|
| MIC | tabpfn | +0.0022 | 0.0003 | 0.0000 (seed0 only) | small edge; comparator var unmeasured |
| MIC | xgboost_optuna | +0.0029 | 0.0003 | 0.0010 | marginal edge (~2.8x noise, small in absolute terms) |
| concrete_compressive_strength | tabpfn | +0.0011 | 0.0001 | 0.0000 (seed0 only) | near-tie (margin ~ noise; comparator var unmeasured) |
| concrete_compressive_strength | xgboost_optuna | +0.0157 | 0.0001 | 0.0000 (seed0 only) | clear TabFM edge |
| blood-transfusion-service-center | tabpfn | +0.0089 | 0.0006 | 0.0000 (seed0 only) | clear TabFM edge |
| blood-transfusion-service-center | xgboost_optuna | +0.0143 | 0.0006 | 0.0000 (seed0 only) | clear TabFM edge |

## Reading

1. TabFM run-to-run variance is very small (std 0.0001-0.0006), smaller than
   every tree baseline (xgboost_heavy up to 0.005, random_forest up to 0.004).
   So TabFM is the more stable model, and its thin margins are not a TabFM-noise
   artifact.
2. The only two comparisons that stay within a few thousandths are concrete
   vs TabPFN (+0.0011) and MIC vs TabPFN (+0.0022). concrete vs TabPFN is a
   genuine near-tie. MIC's margins (+0.0022 vs TabPFN, +0.0029 vs Optuna) are
   small in absolute terms but sit a few times above the measured seed noise;
   Optuna on MIC actually got slightly worse across seeds (0.890 -> 0.889).
   These are best read as 'too close to call a robust win,' not as TabFM losses.
3. blood-transfusion (+0.0089 vs TabPFN) and concrete vs Optuna (+0.0157) are
   comfortably outside the noise: real TabFM edges.

Limitation: fixed fold splits, model-seed variance only (no data-resample
variance); and TabPFN seeds 1/2 are unmeasured (network hang), so the two
TabPFN margins rest on TabFM-side variance plus TabPFN's seed-0 point.
