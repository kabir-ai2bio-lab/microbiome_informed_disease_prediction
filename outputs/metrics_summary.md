# Strict Binary AlzBiom-to-MiaGB Logistic Regression Benchmark

Positive class: Cognitive impaired. Negative class: Controls.

## Mapping Audit

- Exact matched genera: 154
- MiaGB-only genera padded as 0 in AlzBiom: 34
- AlzBiom-only genera padded as 0 in MiaGB: 1171
- Near-match audit candidates not used in model: 15

## AlzBiom 5-Fold Cross-Validation

| feature_set | n_features | AUROC mean±std | AUPRC mean±std | Accuracy mean±std | Balanced acc mean±std | Sensitivity mean±std | Specificity mean±std | F1 mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mw_top_5 | 5 | 0.706 +/- 0.064 | 0.773 +/- 0.041 | 0.640 +/- 0.048 | 0.604 +/- 0.040 | 0.832 +/- 0.126 | 0.375 +/- 0.106 | 0.725 +/- 0.058 |
| mw_top_15 | 15 | 0.662 +/- 0.051 | 0.728 +/- 0.033 | 0.606 +/- 0.044 | 0.578 +/- 0.042 | 0.756 +/- 0.098 | 0.400 +/- 0.102 | 0.688 +/- 0.051 |
| mw_top_20 | 20 | 0.660 +/- 0.060 | 0.730 +/- 0.040 | 0.617 +/- 0.069 | 0.588 +/- 0.071 | 0.769 +/- 0.096 | 0.406 +/- 0.127 | 0.698 +/- 0.061 |
| all_features | 1359 | 0.596 +/- 0.059 | 0.653 +/- 0.042 | 0.590 +/- 0.040 | 0.586 +/- 0.039 | 0.615 +/- 0.047 | 0.556 +/- 0.034 | 0.635 +/- 0.041 |

## Final Training and MiaGB External Testing

| feature_set | evaluation | n_features | auroc | auprc_average_precision | accuracy | balanced_accuracy | sensitivity_impaired_recall | specificity_controls_recall | precision_impaired | f1_impaired | confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mw_top_5 | alzbiom_final_training | 5 | 0.747 | 0.804 | 0.677 | 0.643 | 0.855 | 0.431 | 0.675 | 0.754 | TN=69, FP=91, FN=32, TP=189 |
| mw_top_5 | miagb_external_test | 5 | 0.525 | 0.402 | 0.405 | 0.502 | 0.859 | 0.145 | 0.365 | 0.512 | TN=45, FP=265, FN=25, TP=152 |
| mw_top_15 | alzbiom_final_training | 15 | 0.765 | 0.804 | 0.706 | 0.677 | 0.860 | 0.494 | 0.701 | 0.772 | TN=79, FP=81, FN=31, TP=190 |
| mw_top_15 | miagb_external_test | 15 | 0.533 | 0.397 | 0.433 | 0.529 | 0.881 | 0.177 | 0.380 | 0.531 | TN=55, FP=255, FN=21, TP=156 |
| mw_top_20 | alzbiom_final_training | 20 | 0.773 | 0.826 | 0.706 | 0.682 | 0.833 | 0.531 | 0.710 | 0.767 | TN=85, FP=75, FN=37, TP=184 |
| mw_top_20 | miagb_external_test | 20 | 0.517 | 0.367 | 0.429 | 0.524 | 0.870 | 0.177 | 0.377 | 0.526 | TN=55, FP=255, FN=23, TP=154 |
| all_features | alzbiom_final_training | 1359 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | TN=160, FP=0, FN=0, TP=221 |
| all_features | miagb_external_test | 1359 | 0.516 | 0.368 | 0.435 | 0.508 | 0.774 | 0.242 | 0.368 | 0.499 | TN=75, FP=235, FN=40, TP=137 |

## Selected Mann-Whitney Features

| feature_set | rank | feature | p_value | rank_biserial | mean_controls | mean_impaired |
| --- | --- | --- | --- | --- | --- | --- |
| mw_top_5 | 1 | Lacrimispora | 0.000 | -0.324 | 0.094 | 0.040 |
| mw_top_5 | 2 | Holdemania | 0.000 | -0.338 | 0.028 | 0.013 |
| mw_top_5 | 3 | Bacteroides | 0.000 | -0.351 | 8.851 | 5.563 |
| mw_top_5 | 4 | Lachnospira | 0.000 | -0.340 | 1.991 | 0.943 |
| mw_top_5 | 5 | GGB9064 | 0.000 | -0.236 | 0.019 | 0.003 |
| mw_top_15 | 1 | Lacrimispora | 0.000 | -0.324 | 0.094 | 0.040 |
| mw_top_15 | 2 | Holdemania | 0.000 | -0.338 | 0.028 | 0.013 |
| mw_top_15 | 3 | Bacteroides | 0.000 | -0.351 | 8.851 | 5.563 |
| mw_top_15 | 4 | Lachnospira | 0.000 | -0.340 | 1.991 | 0.943 |
| mw_top_15 | 5 | GGB9064 | 0.000 | -0.236 | 0.019 | 0.003 |
| mw_top_15 | 6 | GGB9614 | 0.000 | -0.254 | 0.033 | 0.014 |
| mw_top_15 | 7 | Veillonella | 0.000 | -0.251 | 0.084 | 0.061 |
| mw_top_15 | 8 | Blautia | 0.000 | 0.293 | 6.761 | 10.486 |
| mw_top_15 | 9 | Lawsonibacter | 0.000 | -0.280 | 0.134 | 0.095 |
| mw_top_15 | 10 | Roseburia | 0.000 | -0.282 | 2.993 | 1.839 |
| mw_top_15 | 11 | Butyricimonas | 0.000 | -0.248 | 0.074 | 0.035 |
| mw_top_15 | 12 | GGB3828 | 0.000 | -0.165 | 0.003 | 0.000 |
| mw_top_15 | 13 | Hydrogenoanaerobacterium | 0.000 | -0.249 | 0.029 | 0.015 |
| mw_top_15 | 14 | Odoribacter | 0.000 | -0.267 | 0.222 | 0.151 |
| mw_top_15 | 15 | Bilophila | 0.000 | -0.263 | 0.095 | 0.058 |
| mw_top_20 | 1 | Lacrimispora | 0.000 | -0.324 | 0.094 | 0.040 |
| mw_top_20 | 2 | Holdemania | 0.000 | -0.338 | 0.028 | 0.013 |
| mw_top_20 | 3 | Bacteroides | 0.000 | -0.351 | 8.851 | 5.563 |
| mw_top_20 | 4 | Lachnospira | 0.000 | -0.340 | 1.991 | 0.943 |
| mw_top_20 | 5 | GGB9064 | 0.000 | -0.236 | 0.019 | 0.003 |
| mw_top_20 | 6 | GGB9614 | 0.000 | -0.254 | 0.033 | 0.014 |
| mw_top_20 | 7 | Veillonella | 0.000 | -0.251 | 0.084 | 0.061 |
| mw_top_20 | 8 | Blautia | 0.000 | 0.293 | 6.761 | 10.486 |
| mw_top_20 | 9 | Lawsonibacter | 0.000 | -0.280 | 0.134 | 0.095 |
| mw_top_20 | 10 | Roseburia | 0.000 | -0.282 | 2.993 | 1.839 |
| mw_top_20 | 11 | Butyricimonas | 0.000 | -0.248 | 0.074 | 0.035 |
| mw_top_20 | 12 | GGB3828 | 0.000 | -0.165 | 0.003 | 0.000 |
| mw_top_20 | 13 | Hydrogenoanaerobacterium | 0.000 | -0.249 | 0.029 | 0.015 |
| mw_top_20 | 14 | Odoribacter | 0.000 | -0.267 | 0.222 | 0.151 |
| mw_top_20 | 15 | Bilophila | 0.000 | -0.263 | 0.095 | 0.058 |
| mw_top_20 | 16 | GGB33586 | 0.000 | -0.232 | 0.011 | 0.007 |
| mw_top_20 | 17 | GGB9063 | 0.000 | -0.193 | 0.018 | 0.006 |
| mw_top_20 | 18 | Oscillibacter | 0.000 | -0.268 | 0.880 | 0.524 |
| mw_top_20 | 19 | Clostridium | 0.000 | -0.262 | 1.391 | 1.139 |
| mw_top_20 | 20 | GGB9559 | 0.000 | -0.220 | 0.017 | 0.012 |
