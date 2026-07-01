# TabArena Task List (verified)

Source: OpenML Study 457, "TabArena-v0.1 Suite"
(https://www.openml.org/api/v1/json/study/457), curated by the TabArena team and
linked from github.com/autogluon/tabarena. Each row was cross-checked against the
OpenML task API. 51 tasks total: 38 classification, 13 regression. Row range 748 to
150,000. No classification task exceeds TabFM's 10-class cap (max is MIC at 8).

## Full suite

| dataset | task_id | type | rows | features | classes |
|---|---|---|---|---|---|
| airfoil_self_noise | 363612 | reg | 1503 | 6 | - |
| Amazon_employee_access | 363613 | clf | 32769 | 10 | 2 |
| anneal | 363614 | clf | 898 | 39 | 5 |
| Another-Dataset-on-used-Fiat-500 | 363615 | reg | 1538 | 8 | - |
| APSFailure | 363616 | clf | 76000 | 171 | 2 |
| bank-marketing | 363618 | clf | 45211 | 14 | 2 |
| Bank_Customer_Churn | 363619 | clf | 10000 | 11 | 2 |
| Bioresponse | 363620 | clf | 3751 | 1777 | 2 |
| blood-transfusion-service-center | 363621 | clf | 748 | 5 | 2 |
| churn | 363623 | clf | 5000 | 20 | 2 |
| coil2000_insurance_policies | 363624 | clf | 9822 | 86 | 2 |
| concrete_compressive_strength | 363625 | reg | 1030 | 9 | - |
| credit-g | 363626 | clf | 1000 | 21 | 2 |
| credit_card_clients_default | 363627 | clf | 30000 | 24 | 2 |
| customer_satisfaction_in_airline | 363628 | clf | 129880 | 22 | 2 |
| diabetes | 363629 | clf | 768 | 9 | 2 |
| Diabetes130US | 363630 | clf | 71518 | 48 | 2 |
| diamonds | 363631 | reg | 53940 | 10 | - |
| E-CommereShippingData | 363632 | clf | 10999 | 11 | 2 |
| Fitness_Club | 363671 | clf | 1500 | 7 | 2 |
| Food_Delivery_Time | 363672 | reg | 45451 | 10 | - |
| GiveMeSomeCredit | 363673 | clf | 150000 | 11 | 2 |
| hazelnut-spread-contaminant-detection | 363674 | clf | 2400 | 31 | 2 |
| healthcare_insurance_expenses | 363675 | reg | 1338 | 7 | - |
| heloc | 363676 | clf | 10459 | 24 | 2 |
| hiva_agnostic | 363677 | clf | 3845 | 1618 | 3 |
| houses | 363678 | reg | 20640 | 9 | - |
| HR_Analytics_Job_Change_of_Data_Scientists | 363679 | clf | 19158 | 13 | 2 |
| in_vehicle_coupon_recommendation | 363681 | clf | 12684 | 25 | 2 |
| Is-this-a-good-customer | 363682 | clf | 1723 | 14 | 2 |
| jm1 | 363712 | clf | 10885 | 22 | 2 |
| kddcup09_appetency | 363683 | clf | 50000 | 213 | 2 |
| Marketing_Campaign | 363684 | clf | 2240 | 26 | 2 |
| maternal_health_risk | 363685 | clf | 1014 | 6 | 3 |
| miami_housing | 363686 | reg | 13776 | 16 | - |
| MIC | 363711 | clf | 1699 | 112 | 8 |
| NATICUSdroid | 363689 | clf | 7491 | 87 | 2 |
| online_shoppers_intention | 363691 | clf | 12330 | 18 | 2 |
| physiochemical_protein | 363693 | reg | 45730 | 10 | - |
| polish_companies_bankruptcy | 363694 | clf | 5910 | 65 | 2 |
| qsar-biodeg | 363696 | clf | 1054 | 42 | 2 |
| QSAR-TID-11 | 363697 | reg | 5742 | 1025 | - |
| QSAR_fish_toxicity | 363698 | reg | 907 | 7 | - |
| SDSS17 | 363699 | clf | 78053 | 12 | 3 |
| seismic-bumps | 363700 | clf | 2584 | 16 | 2 |
| splice | 363702 | clf | 3190 | 61 | 3 |
| students_dropout_and_academic_success | 363704 | clf | 4424 | 37 | 3 |
| superconductivity | 363705 | reg | 21263 | 82 | - |
| taiwanese_bankruptcy_prediction | 363706 | clf | 6819 | 95 | 2 |
| website_phishing | 363707 | clf | 1353 | 10 | 3 |
| wine_quality | 363708 | reg | 6497 | 13 | - |

## Proposed Phase 3 subset (awaiting sign-off)

Classification (7), spanning size, class count, and feature width:
1. blood-transfusion-service-center (363621) 748 rows, 2-class, 5 feat. Small numerical floor.
2. credit-g (363626) 1000 rows, 2-class, 21 feat. Small, categorical-heavy.
3. maternal_health_risk (363685) 1014 rows, 3-class, 6 feat. Repo-verified anchor.
4. students_dropout_and_academic_success (363704) 4424 rows, 3-class, 37 feat. Mid multiclass.
5. churn (363623) 5000 rows, 2-class, 20 feat. Mid.
6. SDSS17 (363699) 78053 rows, 3-class, 12 feat. Large multiclass.
7. GiveMeSomeCredit (363673) 150000 rows, 2-class, 11 feat. Largest, stress-tests the 150k upper claim.

Regression (4), spanning size:
1. concrete_compressive_strength (363625) 1030 rows, 9 feat. Small classic.
2. wine_quality (363708) 6497 rows, 13 feat. Mid.
3. houses (363678) 20640 rows, 9 feat. Large (California housing).
4. diamonds (363631) 53940 rows, 10 feat. Largest.

Optional stretch cases if we want harder tables:
1. MIC (363711) 1699 rows, 8-class, 112 feat. Highest class count in the suite.
2. Bioresponse (363620) 3751 rows, 2-class, 1777 feat. Exceeds the 500 feature soft cap, exercises feature subsampling.
