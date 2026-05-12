# AlzBiom to MiaGB Genus Logistic Regression Benchmark

This benchmark trains a strict binary Logistic Regression model on AlzBiom and tests it on MiaGB.

The target is:

- Controls: `NC` in AlzBiom, or `MoCA >= 26` in MiaGB
- Cognitive impaired: `MCI` or `Dementia` in AlzBiom, or `MoCA < 26` in MiaGB

Column mapping is exact genus matching. For example:

```text
k__Bacteria|...|g__Akkermansia -> Akkermansia -> MiaGB Akkermansia
```

Missing genera are kept and padded with `0` on the missing dataset side.

The script runs Logistic Regression with:

- Mann-Whitney top 5 features
- Mann-Whitney top 15 features
- Mann-Whitney top 20 features
- all aligned genus features

It uses 5-fold cross-validation on AlzBiom and an external test on MiaGB.

## Run

```bash
/opt/homebrew/bin/python3 train_alzbiom_test_miagb.py
```

## Outputs

Aggregate results are saved in `outputs/`.

Sample-level input data, aligned matrices, and prediction tables are not committed to GitHub.
