#!/usr/bin/env python3
"""Strict binary AlzBiom-to-MiaGB logistic regression benchmark.

This script maps AlzBiom full taxonomy columns to genus names, aligns them
with MiaGB genus columns by exact name match, pads missing columns with zero,
and benchmarks Logistic Regression models trained on AlzBiom and tested on
MiaGB.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
LABEL_NAMES = {
    NEGATIVE_LABEL: "Controls",
    POSITIVE_LABEL: "Cognitive impaired",
}
FEATURE_RUNS = [("mw_top_5", 5), ("mw_top_15", 15), ("mw_top_20", 20), ("all_features", None)]


@dataclass(frozen=True)
class FeatureSelectionResult:
    features: list[str]
    table: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train strict-binary AlzBiom Logistic Regression models and test on MiaGB."
    )
    parser.add_argument(
        "--alz",
        default="AlzBiom_and_InHouse(genus).csv",
        type=Path,
        help="AlzBiom genus CSV path.",
    )
    parser.add_argument(
        "--mia",
        default="MiaGB(Genus).csv",
        type=Path,
        help="MiaGB genus CSV path.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        type=Path,
        help="Directory for generated outputs.",
    )
    return parser.parse_args()


def alz_column_to_genus(column: str) -> str:
    """Extract the genus name from an AlzBiom taxonomy column."""
    if "|g__" in column:
        return column.rsplit("|g__", 1)[1]
    if column.startswith("g__"):
        return column[3:]
    return column


def strict_alz_target(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "NC":
        return NEGATIVE_LABEL
    if text in {"MCI", "Dementia"}:
        return POSITIVE_LABEL
    return np.nan


def strict_mia_target(value: object) -> int:
    moca = float(value)
    return NEGATIVE_LABEL if moca >= 26 else POSITIVE_LABEL


def read_inputs(alz_path: Path, mia_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    alz = pd.read_csv(alz_path, low_memory=False)
    mia = pd.read_csv(mia_path, low_memory=False)
    return alz, mia


def build_feature_mapping(
    alz_columns: Iterable[str], mia_columns: Iterable[str]
) -> tuple[list[str], pd.DataFrame, dict[str, str]]:
    alz_raw_features = list(alz_columns)
    mia_features = list(mia_columns)
    alz_to_genus = {raw: alz_column_to_genus(raw) for raw in alz_raw_features}
    alz_genus_order = list(alz_to_genus.values())

    duplicates = pd.Series(alz_genus_order).value_counts()
    duplicate_names = duplicates[duplicates > 1]
    if not duplicate_names.empty:
        names = ", ".join(duplicate_names.index[:10])
        raise ValueError(f"AlzBiom genus names are not unique after mapping: {names}")

    alz_genus_set = set(alz_genus_order)
    mia_genus_set = set(mia_features)
    common = alz_genus_set & mia_genus_set

    rows: list[dict[str, object]] = []
    for raw in alz_raw_features:
        genus = alz_to_genus[raw]
        rows.append(
            {
                "mapping_type": "exact_match" if genus in common else "alz_only_padded_in_miagb",
                "alz_original_column": raw,
                "mapped_genus": genus,
                "mia_column": genus if genus in common else "",
                "used_in_model": True,
                "notes": "Exact genus match" if genus in common else "Absent in MiaGB; MiaGB value padded with 0",
            }
        )

    for genus in mia_features:
        if genus not in alz_genus_set:
            rows.append(
                {
                    "mapping_type": "mia_only_padded_in_alz",
                    "alz_original_column": "",
                    "mapped_genus": genus,
                    "mia_column": genus,
                    "used_in_model": True,
                    "notes": "Absent in AlzBiom; AlzBiom value padded with 0",
                }
            )

    near_matches = find_near_matches(sorted(mia_genus_set - alz_genus_set), sorted(alz_genus_set - mia_genus_set))
    for mia_genus, alz_genus, score in near_matches:
        rows.append(
            {
                "mapping_type": "near_match_audit_only",
                "alz_original_column": "",
                "mapped_genus": alz_genus,
                "mia_column": mia_genus,
                "used_in_model": False,
                "notes": f"Similarity {score:.3f}; not mapped because exact matching only",
            }
        )

    union_order = alz_genus_order + [genus for genus in mia_features if genus not in alz_genus_set]
    return union_order, pd.DataFrame(rows), alz_to_genus


def find_near_matches(mia_only: list[str], alz_only: list[str]) -> list[tuple[str, str, float]]:
    import difflib

    matches: list[tuple[str, str, float]] = []
    for mia_name in mia_only:
        for alz_name in difflib.get_close_matches(mia_name, alz_only, n=3, cutoff=0.78):
            score = difflib.SequenceMatcher(None, mia_name, alz_name).ratio()
            matches.append((mia_name, alz_name, score))
    return sorted(matches, key=lambda item: (-item[2], item[0], item[1]))


def align_features(
    alz: pd.DataFrame,
    mia: pd.DataFrame,
    union_order: list[str],
    alz_to_genus: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alz_feature_source = alz.iloc[:, 2:].copy()
    alz_feature_source = alz_feature_source.rename(columns=alz_to_genus)
    mia_feature_source = mia.iloc[:, 5:].copy()

    alz_aligned = alz_feature_source.reindex(columns=union_order, fill_value=0)
    mia_aligned = mia_feature_source.reindex(columns=union_order, fill_value=0)

    alz_aligned = alz_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mia_aligned = mia_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return alz_aligned, mia_aligned


def select_mann_whitney_features(
    x: pd.DataFrame, y: np.ndarray, top_k: int | None
) -> FeatureSelectionResult:
    if top_k is None:
        table = pd.DataFrame(
            {
                "feature": list(x.columns),
                "p_value": np.nan,
                "rank_biserial": np.nan,
                "abs_rank_biserial": np.nan,
                "mean_controls": x[y == NEGATIVE_LABEL].mean(axis=0).to_numpy(),
                "mean_impaired": x[y == POSITIVE_LABEL].mean(axis=0).to_numpy(),
            }
        )
        table["rank"] = np.arange(1, len(table) + 1)
        return FeatureSelectionResult(features=list(x.columns), table=table)

    y = np.asarray(y)
    controls = x[y == NEGATIVE_LABEL]
    impaired = x[y == POSITIVE_LABEL]
    n_controls = len(controls)
    n_impaired = len(impaired)

    rows: list[dict[str, float | str]] = []
    for feature in x.columns:
        control_values = controls[feature].to_numpy(dtype=float)
        impaired_values = impaired[feature].to_numpy(dtype=float)
        if np.all(control_values == control_values[0]) and np.all(impaired_values == impaired_values[0]):
            if control_values[0] == impaired_values[0]:
                p_value = 1.0
                rank_biserial = 0.0
            else:
                result = mannwhitneyu(impaired_values, control_values, alternative="two-sided", method="auto")
                p_value = float(result.pvalue)
                auc_like = float(result.statistic) / float(n_controls * n_impaired)
                rank_biserial = (2.0 * auc_like) - 1.0
        else:
            result = mannwhitneyu(impaired_values, control_values, alternative="two-sided", method="auto")
            p_value = float(result.pvalue)
            auc_like = float(result.statistic) / float(n_controls * n_impaired)
            rank_biserial = (2.0 * auc_like) - 1.0

        rows.append(
            {
                "feature": feature,
                "p_value": p_value,
                "rank_biserial": rank_biserial,
                "abs_rank_biserial": abs(rank_biserial),
                "mean_controls": float(np.mean(control_values)),
                "mean_impaired": float(np.mean(impaired_values)),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        by=["p_value", "abs_rank_biserial", "feature"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    table["rank"] = np.arange(1, len(table) + 1)
    return FeatureSelectionResult(features=table.head(top_k)["feature"].tolist(), table=table)


def fit_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_scaled, y_train)
    positive_index = list(model.classes_).index(POSITIVE_LABEL)
    prob = model.predict_proba(x_eval_scaled)[:, positive_index]
    pred = (prob >= 0.5).astype(int)
    return prob, pred, scaler, model


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    def safe_metric(func, *args, **kwargs) -> float:
        try:
            return float(func(*args, **kwargs))
        except ValueError:
            return float("nan")

    return {
        "n_samples": int(len(y_true)),
        "n_controls": int(np.sum(y_true == NEGATIVE_LABEL)),
        "n_impaired": int(np.sum(y_true == POSITIVE_LABEL)),
        "auroc": safe_metric(roc_auc_score, y_true, y_prob),
        "auprc_average_precision": safe_metric(average_precision_score, y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_impaired_recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "specificity_controls_recall": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision_impaired": float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_impaired": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metric_row(
    *,
    feature_set: str,
    evaluation: str,
    fold: int | str,
    n_features: int,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "feature_set": feature_set,
        "evaluation": evaluation,
        "fold": fold,
        "n_features": n_features,
        **metrics,
    }


def summarize_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "auprc_average_precision",
        "accuracy",
        "balanced_accuracy",
        "sensitivity_impaired_recall",
        "specificity_controls_recall",
        "precision_impaired",
        "f1_impaired",
    ]
    rows: list[dict[str, object]] = []
    for feature_set, group in fold_metrics.groupby("feature_set", sort=False):
        row: dict[str, object] = {
            "feature_set": feature_set,
            "evaluation": "alzbiom_5fold_cv",
            "n_features": int(group["n_features"].iloc[0]),
            "folds": int(group["fold"].nunique()),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cv_long(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n_samples",
        "n_controls",
        "n_impaired",
        "auroc",
        "auprc_average_precision",
        "accuracy",
        "balanced_accuracy",
        "sensitivity_impaired_recall",
        "specificity_controls_recall",
        "precision_impaired",
        "f1_impaired",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    rows: list[dict[str, object]] = []
    for feature_set, group in fold_metrics.groupby("feature_set", sort=False):
        for summary_name, values in [("mean", group[metric_columns].mean()), ("std", group[metric_columns].std(ddof=1))]:
            row: dict[str, object] = {
                "feature_set": feature_set,
                "evaluation": f"alzbiom_5fold_cv_{summary_name}",
                "fold": summary_name,
                "n_features": int(group["n_features"].iloc[0]),
                "folds": int(group["fold"].nunique()),
            }
            for column in metric_columns:
                row[column] = float(values[column])
            rows.append(row)
    return pd.DataFrame(rows)


def format_float(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format_float(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_metrics_summary(
    out_path: Path,
    cv_summary: pd.DataFrame,
    train_test_metrics: pd.DataFrame,
    mapping_report: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> None:
    cv_display = cv_summary.copy()
    cv_display["AUROC mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['auroc_mean'])} +/- {format_float(r['auroc_std'])}", axis=1
    )
    cv_display["AUPRC mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['auprc_average_precision_mean'])} +/- {format_float(r['auprc_average_precision_std'])}",
        axis=1,
    )
    cv_display["Accuracy mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['accuracy_mean'])} +/- {format_float(r['accuracy_std'])}", axis=1
    )
    cv_display["Balanced acc mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['balanced_accuracy_mean'])} +/- {format_float(r['balanced_accuracy_std'])}", axis=1
    )
    cv_display["Sensitivity mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['sensitivity_impaired_recall_mean'])} +/- {format_float(r['sensitivity_impaired_recall_std'])}",
        axis=1,
    )
    cv_display["Specificity mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['specificity_controls_recall_mean'])} +/- {format_float(r['specificity_controls_recall_std'])}",
        axis=1,
    )
    cv_display["F1 mean±std"] = cv_display.apply(
        lambda r: f"{format_float(r['f1_impaired_mean'])} +/- {format_float(r['f1_impaired_std'])}", axis=1
    )

    final_display = train_test_metrics.copy()
    final_display["confusion_matrix"] = final_display.apply(
        lambda r: f"TN={int(r['tn'])}, FP={int(r['fp'])}, FN={int(r['fn'])}, TP={int(r['tp'])}",
        axis=1,
    )

    mapping_counts = mapping_report["mapping_type"].value_counts().to_dict()
    feature_preview = selected_features[selected_features["feature_set"] != "all_features"].copy()
    feature_preview = feature_preview[feature_preview["rank"] <= 20]

    lines = [
        "# Strict Binary AlzBiom-to-MiaGB Logistic Regression Benchmark",
        "",
        "Positive class: Cognitive impaired. Negative class: Controls.",
        "",
        "## Mapping Audit",
        "",
        f"- Exact matched genera: {mapping_counts.get('exact_match', 0)}",
        f"- MiaGB-only genera padded as 0 in AlzBiom: {mapping_counts.get('mia_only_padded_in_alz', 0)}",
        f"- AlzBiom-only genera padded as 0 in MiaGB: {mapping_counts.get('alz_only_padded_in_miagb', 0)}",
        f"- Near-match audit candidates not used in model: {mapping_counts.get('near_match_audit_only', 0)}",
        "",
        "## AlzBiom 5-Fold Cross-Validation",
        "",
        markdown_table(
            cv_display,
            [
                "feature_set",
                "n_features",
                "AUROC mean±std",
                "AUPRC mean±std",
                "Accuracy mean±std",
                "Balanced acc mean±std",
                "Sensitivity mean±std",
                "Specificity mean±std",
                "F1 mean±std",
            ],
        ),
        "",
        "## Final Training and MiaGB External Testing",
        "",
        markdown_table(
            final_display,
            [
                "feature_set",
                "evaluation",
                "n_features",
                "auroc",
                "auprc_average_precision",
                "accuracy",
                "balanced_accuracy",
                "sensitivity_impaired_recall",
                "specificity_controls_recall",
                "precision_impaired",
                "f1_impaired",
                "confusion_matrix",
            ],
        ),
        "",
        "## Selected Mann-Whitney Features",
        "",
        markdown_table(
            feature_preview,
            ["feature_set", "rank", "feature", "p_value", "rank_biserial", "mean_controls", "mean_impaired"],
        ),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    alz, mia = read_inputs(args.alz, args.mia)
    union_order, mapping_report, alz_to_genus = build_feature_mapping(alz.columns[2:], mia.columns[5:])
    alz_x_all, mia_x_all = align_features(alz, mia, union_order, alz_to_genus)

    alz_targets = alz["Target"].map(strict_alz_target)
    labeled_mask = alz_targets.notna()
    alz_labeled = alz.loc[labeled_mask].reset_index(drop=True)
    alz_x_labeled = alz_x_all.loc[labeled_mask].reset_index(drop=True)
    y_alz = alz_targets.loc[labeled_mask].astype(int).to_numpy()

    mia_labeled = mia.reset_index(drop=True)
    mia_x = mia_x_all.reset_index(drop=True)
    y_mia = mia_labeled["MoCA"].map(strict_mia_target).astype(int).to_numpy()

    alz_audit = pd.concat(
        [
            alz_labeled[["clade_name", "Target"]].rename(columns={"clade_name": "sample_id", "Target": "original_target"}),
            pd.Series(y_alz, name="strict_binary_target"),
            alz_x_labeled,
        ],
        axis=1,
    )
    mia_audit = pd.concat(
        [
            mia_labeled[["Subject ID", "Group", "MoCA"]].rename(
                columns={"Subject ID": "sample_id", "Group": "original_group"}
            ),
            pd.Series(y_mia, name="strict_binary_target"),
            mia_x,
        ],
        axis=1,
    )

    mapping_report.to_csv(args.outdir / "genus_mapping_report.csv", index=False)
    alz_audit.to_csv(args.outdir / "aligned_alzbiom_genus.csv", index=False)
    mia_audit.to_csv(args.outdir / "aligned_miagb_genus.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows: list[dict[str, object]] = []
    selected_feature_rows: list[dict[str, object]] = []
    final_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set, top_k in FEATURE_RUNS:
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(alz_x_labeled, y_alz), start=1):
            x_train_full = alz_x_labeled.iloc[train_idx]
            y_train = y_alz[train_idx]
            x_valid_full = alz_x_labeled.iloc[valid_idx]
            y_valid = y_alz[valid_idx]

            selection = select_mann_whitney_features(x_train_full, y_train, top_k)
            features = selection.features
            valid_prob, valid_pred, _, _ = fit_predict(
                x_train_full[features], y_train, x_valid_full[features]
            )
            valid_metrics = compute_metrics(y_valid, valid_prob, valid_pred)
            cv_rows.append(
                metric_row(
                    feature_set=feature_set,
                    evaluation="alzbiom_cv_validation",
                    fold=fold_idx,
                    n_features=len(features),
                    metrics=valid_metrics,
                )
            )

        final_selection = select_mann_whitney_features(alz_x_labeled, y_alz, top_k)
        final_features = final_selection.features
        feature_table = final_selection.table.copy()
        feature_table.insert(0, "feature_set", feature_set)
        feature_table["selected"] = feature_table["feature"].isin(final_features)
        if top_k is None:
            selected_rows_for_run = feature_table
        else:
            selected_rows_for_run = feature_table[feature_table["selected"]].copy()
        selected_feature_rows.extend(selected_rows_for_run.to_dict("records"))

        train_prob, train_pred, scaler, model = fit_predict(
            alz_x_labeled[final_features], y_alz, alz_x_labeled[final_features]
        )
        train_metrics = compute_metrics(y_alz, train_prob, train_pred)
        final_metric_rows.append(
            metric_row(
                feature_set=feature_set,
                evaluation="alzbiom_final_training",
                fold="all",
                n_features=len(final_features),
                metrics=train_metrics,
            )
        )

        mia_scaled = scaler.transform(mia_x[final_features])
        positive_index = list(model.classes_).index(POSITIVE_LABEL)
        mia_prob = model.predict_proba(mia_scaled)[:, positive_index]
        mia_pred = (mia_prob >= 0.5).astype(int)
        mia_metrics = compute_metrics(y_mia, mia_prob, mia_pred)
        final_metric_rows.append(
            metric_row(
                feature_set=feature_set,
                evaluation="miagb_external_test",
                fold="all",
                n_features=len(final_features),
                metrics=mia_metrics,
            )
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "sample_id": mia_labeled["Subject ID"],
                    "moca": mia_labeled["MoCA"],
                    "original_group": mia_labeled["Group"],
                    "true_label": [LABEL_NAMES[int(label)] for label in y_mia],
                    "true_label_binary": y_mia,
                    "predicted_label": [LABEL_NAMES[int(label)] for label in mia_pred],
                    "predicted_label_binary": mia_pred,
                    "prob_cognitive_impaired": mia_prob,
                    "prob_controls": 1.0 - mia_prob,
                }
            )
        )

    cv_fold_metrics = pd.DataFrame(cv_rows)
    cv_summary = summarize_cv(cv_fold_metrics)
    train_test_metrics = pd.DataFrame(final_metric_rows)
    all_metrics = pd.concat([summarize_cv_long(cv_fold_metrics), train_test_metrics], ignore_index=True, sort=False)
    selected_features = pd.DataFrame(selected_feature_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    cv_fold_metrics.to_csv(args.outdir / "cv_fold_metrics.csv", index=False)
    all_metrics.to_csv(args.outdir / "logistic_regression_metrics.csv", index=False)
    selected_features.to_csv(args.outdir / "selected_features_by_run.csv", index=False)
    predictions.to_csv(args.outdir / "miagb_predictions_strict.csv", index=False)
    write_metrics_summary(
        args.outdir / "metrics_summary.md",
        cv_summary,
        train_test_metrics,
        mapping_report,
        selected_features,
    )

    print(f"Wrote outputs to {args.outdir.resolve()}")
    print(f"Labeled AlzBiom samples: {len(y_alz)}")
    print(f"MiaGB samples: {len(y_mia)}")
    print(f"Aligned features: {len(union_order)}")
    print(f"Exact matched genera: {(mapping_report['mapping_type'] == 'exact_match').sum()}")


if __name__ == "__main__":
    main()
