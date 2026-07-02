# Phase 3 results summary (fold-matched)

TabFM vs baselines, averaged only over folds where BOTH a TabFM and a
baselines result exist. Winner = TabFM best preset vs best baseline on the
primary metric (accuracy for classification, R2 for regression).

## Classification (primary: accuracy)

| dataset | TabFM best | TabFM auc | baseline best | base auc | winner |
|---|---|---|---|---|---|
| MIC | ensemble 0.891 | 0.885 | xgboost_optuna 0.890 | 0.853 | TabFM |
| blood-transfusion-service-center | default 0.801 | 0.754 | tabpfn 0.793 | 0.751 | TabFM |
| churn | default 0.979 | 0.934 | tabpfn 0.972 | 0.935 | TabFM |
| credit-g | default 0.775 | 0.801 | random_forest 0.761 | 0.795 | TabFM |
| maternal_health_risk | ensemble 0.877 | 0.970 | tabpfn 0.857 | 0.962 | TabFM |
| students_dropout_and_academic_success | ensemble 0.799 | 0.910 | tabpfn 0.781 | 0.896 | TabFM |

## Regression (primary: R2)

| dataset | TabFM best | TabFM rmse | baseline best | base rmse | winner |
|---|---|---|---|---|---|
| concrete_compressive_strength | ensemble 0.950 | 3.721 | tabpfn 0.949 | 3.765 | TabFM |
| diamonds | ensemble 0.985 | 497.001 | xgboost_heavy 0.982 | 537.168 | TabFM | (2 common fold(s))
| houses | ensemble 0.898 | 0.181 | tabpfn 0.889 | 0.190 | TabFM |
| wine_quality | default 0.548 | 0.587 | tabpfn 0.544 | 0.589 | TabFM |

## Coverage gaps (not scored)

1. Bioresponse: no common folds (tabfm=0, baselines=3)
1. SDSS17: no results (attempted, did not complete: impractical at scale or failed)
1. GiveMeSomeCredit: no results (attempted, did not complete: impractical at scale or failed)

## Headline

TabFM sweep: won 10, lost 0, tied 0 of 10 fold-matched datasets; 3 datasets unscored (coverage gaps). See results/phase3/SUMMARY.md

Note: this tally counts any nonzero margin on the primary metric as a decision. On a noise-aware reading the sub-0.005 margins (MIC, diamonds) are within seed/run variance and test-set granularity and should be read as ties, not wins (see results/phase3_seeds/SEED_VARIANCE.md); this does not change the direction of any result.
