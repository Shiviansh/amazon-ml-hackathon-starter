"""Competition dataset audit with actionable, machine-readable findings.

Run this before feature engineering or fold creation. The audit never mutates
the supplied DataFrames. It checks schema compatibility, IDs, targets,
duplicates, train/test overlap, suspicious target leakage, missingness, and
univariate drift. Reports can be saved as JSON and a self-contained HTML file.

Typical use::

    report = audit_datasets(
        train,
        test,
        target_column="PRICE",
        id_columns=["PRODUCT_ID"],
        text_columns=["TITLE", "DESCRIPTION", "BULLET_POINTS", "BRAND"],
        categorical_columns=["PRODUCT_TYPE_ID"],
        task="regression",
    )
    write_json_report(report, "reports/audit.json")
    write_html_report(report, "reports/audit.html")
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.utils.multiclass import type_of_target

Severity = Literal["info", "warning", "error"]
Task = Literal["auto", "classification", "regression"]

VALID_TASKS = {"auto", "classification", "regression"}
SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


@dataclass(frozen=True)
class Finding:
    """One actionable result from the audit."""

    severity: Severity
    code: str
    message: str
    column: str | None = None
    train_value: Any = None
    test_value: Any = None
    recommendation: str | None = None


def _columns(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(value)
    if any(not isinstance(column, str) or not column for column in result):
        raise ValueError("Column names must be non-empty strings.")
    if len(result) != len(set(result)):
        raise ValueError("Column lists cannot contain duplicates.")
    return result


def _require_frame(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError(f"{name} cannot be empty.")
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name} has duplicate column names: {duplicates}.")


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    frame_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}.")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_ready(value: Any) -> Any:
    """Recursively convert NumPy/Pandas values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def _broad_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string_or_category"


def _semantic_type(
    series: pd.Series,
    forced_text: bool,
    forced_categorical: bool,
) -> str:
    if forced_text:
        return "text"
    if forced_categorical:
        return "categorical"
    broad = _broad_dtype(series)
    if broad != "string_or_category":
        return broad
    non_missing = series.dropna()
    if non_missing.empty:
        return "unknown"
    as_text = non_missing.astype(str)
    if float(as_text.str.len().median()) >= 40 or int(series.nunique(dropna=True)) > 100:
        return "text"
    return "categorical"


def _top_values(series: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    counts = series.astype("string").fillna("<NULL>").value_counts(dropna=False)
    total = len(series)
    return [
        {
            "value": str(value),
            "count": int(count),
            "fraction": float(count / total),
        }
        for value, count in counts.head(limit).items()
    ]


def _column_profile(
    series: pd.Series,
    forced_text: bool = False,
    forced_categorical: bool = False,
) -> dict[str, Any]:
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))
    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "semantic_type": _semantic_type(series, forced_text, forced_categorical),
        "rows": int(len(series)),
        "memory_bytes": int(series.memory_usage(index=False, deep=True)),
        "memory_mb": float(series.memory_usage(index=False, deep=True) / 1024**2),
        "missing": missing,
        "missing_rate": float(missing / len(series)),
        "unique": unique,
        "unique_rate_non_missing": float(unique / max(len(series) - missing, 1)),
    }

    if pd.api.types.is_numeric_dtype(series) and not forced_categorical:
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        profile["non_finite"] = int(np.sum(~np.isfinite(numeric) & ~pd.isna(numeric)))
        if finite.size:
            quantiles = np.quantile(finite, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
            profile["statistics"] = {
                "mean": _safe_float(np.mean(finite)),
                "std": _safe_float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "min": _safe_float(quantiles[0]),
                "p01": _safe_float(quantiles[1]),
                "p25": _safe_float(quantiles[2]),
                "median": _safe_float(quantiles[3]),
                "p75": _safe_float(quantiles[4]),
                "p99": _safe_float(quantiles[5]),
                "max": _safe_float(quantiles[6]),
            }
    else:
        profile["top_values"] = _top_values(series)
        if forced_text or profile["semantic_type"] == "text":
            text = series.astype("string")
            lengths = text.str.len()
            non_missing_lengths = lengths.dropna().to_numpy(dtype=float)
            blank = text.str.strip().eq("").fillna(False)
            profile["blank"] = int(blank.sum())
            profile["blank_rate"] = float(blank.mean())
            if non_missing_lengths.size:
                profile["length"] = {
                    "mean": _safe_float(np.mean(non_missing_lengths)),
                    "median": _safe_float(np.median(non_missing_lengths)),
                    "p95": _safe_float(np.quantile(non_missing_lengths, 0.95)),
                    "max": _safe_float(np.max(non_missing_lengths)),
                }
    return profile


def _memory_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Estimate safe dtype savings without mutating the source DataFrame."""
    index_bytes = int(frame.index.memory_usage(deep=True))
    current_columns = 0
    optimized_columns = 0
    opportunities: list[dict[str, Any]] = []

    for column in frame.columns:
        series = frame[column]
        current = int(series.memory_usage(index=False, deep=True))
        current_columns += current
        candidate = series
        suggested_dtype: str | None = None

        if pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series):
            candidate = pd.to_numeric(series, downcast="integer")
            suggested_dtype = str(candidate.dtype)
        elif pd.api.types.is_float_dtype(series):
            candidate = pd.to_numeric(series, downcast="float")
            suggested_dtype = str(candidate.dtype)
        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            non_missing = max(int(series.notna().sum()), 1)
            if int(series.nunique(dropna=True)) / non_missing <= 0.50:
                candidate = series.astype("category")
                suggested_dtype = "category"

        candidate_bytes = int(candidate.memory_usage(index=False, deep=True))
        if suggested_dtype == str(series.dtype) or candidate_bytes >= current * 0.95:
            candidate_bytes = current
            suggested_dtype = None
        optimized_columns += candidate_bytes
        if suggested_dtype is not None:
            opportunities.append({
                "column": column,
                "current_dtype": str(series.dtype),
                "suggested_dtype": suggested_dtype,
                "current_mb": float(current / 1024**2),
                "estimated_mb": float(candidate_bytes / 1024**2),
                "estimated_saving_mb": float((current - candidate_bytes) / 1024**2),
                "estimated_saving_fraction": float((current - candidate_bytes) / current),
            })

    current_total = index_bytes + current_columns
    optimized_total = index_bytes + optimized_columns
    opportunities.sort(key=lambda item: item["estimated_saving_mb"], reverse=True)
    return {
        "current_bytes": current_total,
        "current_mb": float(current_total / 1024**2),
        "estimated_optimized_bytes": optimized_total,
        "estimated_optimized_mb": float(optimized_total / 1024**2),
        "estimated_saving_bytes": current_total - optimized_total,
        "estimated_saving_mb": float((current_total - optimized_total) / 1024**2),
        "estimated_saving_fraction": float(
            (current_total - optimized_total) / max(current_total, 1)
        ),
        "index_mb": float(index_bytes / 1024**2),
        "opportunities": opportunities,
        "note": "Estimates only; validate float32 precision and category behavior before conversion.",
    }


def _sample_finite(series: pd.Series, limit: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size > limit:
        rng = np.random.default_rng(42)
        values = values[rng.choice(values.size, size=limit, replace=False)]
    return values


def _numeric_drift(train: pd.Series, test: pd.Series, sample_size: int) -> dict[str, Any]:
    left = _sample_finite(train, sample_size)
    right = _sample_finite(test, sample_size)
    if left.size == 0 or right.size == 0:
        return {"available": False, "reason": "no finite values in one dataset"}

    ks = ks_2samp(left, right, alternative="two-sided", method="auto")
    quantile_edges = np.unique(np.quantile(left, np.linspace(0.0, 1.0, 11)))
    psi: float | None = None
    if quantile_edges.size >= 3:
        edges = np.concatenate(([-np.inf], quantile_edges[1:-1], [np.inf]))
        left_counts = np.histogram(left, bins=edges)[0].astype(float)
        right_counts = np.histogram(right, bins=edges)[0].astype(float)
        epsilon = 1e-6
        left_share = np.clip(left_counts / left_counts.sum(), epsilon, None)
        right_share = np.clip(right_counts / right_counts.sum(), epsilon, None)
        psi = float(np.sum((right_share - left_share) * np.log(right_share / left_share)))

    return {
        "available": True,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": _safe_float(ks.pvalue),
        "psi": _safe_float(psi),
        "train_mean": _safe_float(np.mean(left)),
        "test_mean": _safe_float(np.mean(right)),
        "sampled_train": int(left.size),
        "sampled_test": int(right.size),
    }


def _categorical_drift(train: pd.Series, test: pd.Series) -> dict[str, Any]:
    left = train.astype("string").fillna("<NULL>")
    right = test.astype("string").fillna("<NULL>")
    left_share = left.value_counts(normalize=True, dropna=False)
    right_share = right.value_counts(normalize=True, dropna=False)
    categories = left_share.index.union(right_share.index)
    left_aligned = left_share.reindex(categories, fill_value=0.0)
    right_aligned = right_share.reindex(categories, fill_value=0.0)
    unseen = ~right.isin(set(left.unique()))
    return {
        "available": True,
        "total_variation": float(0.5 * np.abs(left_aligned - right_aligned).sum()),
        "unseen_test_rows": int(unseen.sum()),
        "unseen_test_rate": float(unseen.mean()),
        "train_unique": int(left.nunique()),
        "test_unique": int(right.nunique()),
    }


def _text_length_drift(train: pd.Series, test: pd.Series, sample_size: int) -> dict[str, Any]:
    left = train.astype("string").str.len()
    right = test.astype("string").str.len()
    result = _numeric_drift(left, right, sample_size)
    result["basis"] = "character_length"
    return result


def _sample_text(series: pd.Series, limit: int) -> pd.Series:
    values = series.dropna().astype(str)
    if len(values) > limit:
        values = values.sample(n=limit, random_state=42)
    return values


def _text_vocabulary_drift(
    train: pd.Series,
    test: pd.Series,
    sample_size: int,
) -> dict[str, Any]:
    """Estimate token OOV rates from deterministic row samples."""
    train_text = _sample_text(train, sample_size)
    test_text = _sample_text(test, sample_size)
    vocabulary: set[str] = set()
    train_token_count = 0
    for value in train_text:
        tokens = TOKEN_PATTERN.findall(value.casefold())
        train_token_count += len(tokens)
        vocabulary.update(tokens)

    test_token_count = 0
    oov_token_count = 0
    test_vocabulary: set[str] = set()
    for value in test_text:
        tokens = TOKEN_PATTERN.findall(value.casefold())
        test_token_count += len(tokens)
        test_vocabulary.update(tokens)
        oov_token_count += sum(token not in vocabulary for token in tokens)

    oov_unique = test_vocabulary - vocabulary
    return {
        "token_sampled_train_rows": int(len(train_text)),
        "token_sampled_test_rows": int(len(test_text)),
        "train_token_count": train_token_count,
        "test_token_count": test_token_count,
        "train_vocabulary_size": int(len(vocabulary)),
        "test_vocabulary_size": int(len(test_vocabulary)),
        "oov_token_count": oov_token_count,
        "oov_token_rate": float(oov_token_count / max(test_token_count, 1)),
        "oov_unique_token_count": int(len(oov_unique)),
        "oov_unique_token_rate": float(len(oov_unique) / max(len(test_vocabulary), 1)),
    }


def _add(
    findings: list[Finding],
    severity: Severity,
    code: str,
    message: str,
    **kwargs: Any,
) -> None:
    findings.append(Finding(severity, code, message, **kwargs))


def _id_audit(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    id_columns: tuple[str, ...],
    findings: list[Finding],
) -> dict[str, Any]:
    if not id_columns:
        return {"columns": [], "configured": False}
    _require_columns(train, id_columns, "train")
    if test is not None:
        _require_columns(test, id_columns, "test")

    result: dict[str, Any] = {"columns": list(id_columns), "configured": True}
    for name, frame in (("train", train), ("test", test)):
        if frame is None:
            continue
        missing = int(frame.loc[:, list(id_columns)].isna().any(axis=1).sum())
        duplicates = int(frame.duplicated(subset=list(id_columns), keep=False).sum())
        result[f"{name}_missing_rows"] = missing
        result[f"{name}_duplicate_rows"] = duplicates
        if missing:
            _add(
                findings,
                "error",
                "ID_MISSING",
                f"{name} has {missing} rows with a missing ID component.",
                train_value=missing if name == "train" else None,
                test_value=missing if name == "test" else None,
                recommendation="Repair or remove these rows before splitting and prediction.",
            )
        if duplicates:
            _add(
                findings,
                "error",
                "ID_DUPLICATE",
                f"{name} has {duplicates} rows participating in duplicate IDs.",
                train_value=duplicates if name == "train" else None,
                test_value=duplicates if name == "test" else None,
                recommendation="Determine whether these are repeated entities and group them during CV.",
            )

    if test is not None:
        train_ids = pd.util.hash_pandas_object(
            train.loc[:, list(id_columns)], index=False, categorize=True
        )
        test_ids = pd.util.hash_pandas_object(
            test.loc[:, list(id_columns)], index=False, categorize=True
        )
        overlap = int(test_ids.isin(set(train_ids.tolist())).sum())
        result["test_rows_with_train_id"] = overlap
        if overlap:
            _add(
                findings,
                "warning",
                "TRAIN_TEST_ID_OVERLAP",
                f"{overlap} test rows reuse an ID found in train.",
                test_value=overlap,
                recommendation="Check whether this is intentional; use entity-aware validation if it is.",
            )
    return result


def _target_audit(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_column: str | None,
    task: Task,
    id_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    findings: list[Finding],
) -> tuple[dict[str, Any], str | None]:
    if target_column is None:
        _add(
            findings,
            "warning",
            "TARGET_NOT_CONFIGURED",
            "No target column was configured; target quality and leakage checks were skipped.",
            recommendation="Pass target_column for a supervised competition audit.",
        )
        return {"configured": False}, None
    _require_columns(train, [target_column], "train")
    target = train[target_column]
    missing = int(target.isna().sum())
    unique = int(target.nunique(dropna=True))
    if missing:
        _add(
            findings,
            "error",
            "TARGET_MISSING",
            f"The training target has {missing} missing values.",
            column=target_column,
            train_value=missing,
            recommendation="Do not silently impute labels; remove or recover these rows explicitly.",
        )
    if unique < 2:
        _add(
            findings,
            "error",
            "TARGET_CONSTANT",
            "The target has fewer than two non-missing values.",
            column=target_column,
            train_value=unique,
        )

    resolved_task: str
    if task == "auto":
        try:
            inferred = type_of_target(target.dropna().to_numpy())
        except ValueError:
            inferred = "unknown"
        high_cardinality_numeric = (
            pd.api.types.is_numeric_dtype(target)
            and unique > max(20, int(0.05 * max(len(target) - missing, 1)))
        )
        resolved_task = (
            "classification"
            if inferred in {"binary", "multiclass"} and not high_cardinality_numeric
            else "regression"
        )
    else:
        resolved_task = task

    result: dict[str, Any] = {
        "configured": True,
        "column": target_column,
        "task": resolved_task,
        "missing": missing,
        "unique": unique,
    }

    if resolved_task == "regression":
        numeric = pd.to_numeric(target, errors="coerce")
        invalid = int((numeric.isna() & target.notna()).sum())
        infinite = int(np.isinf(numeric.dropna().to_numpy(dtype=float)).sum())
        result["non_numeric"] = invalid
        result["infinite"] = infinite
        if invalid or infinite:
            _add(
                findings,
                "error",
                "TARGET_NONFINITE",
                f"Regression target has {invalid} non-numeric and {infinite} infinite values.",
                column=target_column,
                recommendation="Clean invalid labels before computing folds or metrics.",
            )
        finite = numeric[np.isfinite(numeric)]
        if not finite.empty:
            result["statistics"] = _column_profile(finite)["statistics"]
    else:
        counts = target.value_counts(dropna=False)
        result["class_counts"] = [
            {"label": str(label), "count": int(count)}
            for label, count in counts.head(100).items()
        ]
        rare_count = int((counts < 5).sum())
        result["classes_with_fewer_than_5_rows"] = rare_count
        if rare_count:
            _add(
                findings,
                "warning",
                "RARE_TARGET_CLASSES",
                f"{rare_count} target classes have fewer than five rows.",
                column=target_column,
                train_value=rare_count,
                recommendation="Use rare-class pooling or a non-stratified fallback when creating folds.",
            )

    if test is not None and target_column in test.columns:
        populated = int(test[target_column].notna().sum())
        result["test_non_missing_target"] = populated
        severity: Severity = "error" if populated else "warning"
        _add(
            findings,
            severity,
            "TARGET_PRESENT_IN_TEST",
            f"The test data contains the target column with {populated} populated rows.",
            column=target_column,
            test_value=populated,
            recommendation="Confirm this is not leaked ground truth; remove it from model features.",
        )

    excluded = set(id_columns) | {target_column}
    categorical_mapping_checks: dict[str, Any] = {}
    for column in train.columns:
        if column in excluded:
            continue
        feature = train[column]
        comparable = target.notna() & feature.notna()
        if not comparable.any():
            continue
        left = feature[comparable].reset_index(drop=True)
        right = target[comparable].reset_index(drop=True)
        if np.array_equal(left.to_numpy(), right.to_numpy()):
            _add(
                findings,
                "error",
                "TARGET_CLONE_FEATURE",
                f"Feature '{column}' exactly reproduces the target where both are present.",
                column=column,
                recommendation="Exclude this leakage feature before any validation or training.",
            )
            continue
        if resolved_task == "regression" and pd.api.types.is_numeric_dtype(feature):
            numeric_feature = pd.to_numeric(feature, errors="coerce")
            numeric_target = pd.to_numeric(target, errors="coerce")
            valid = np.isfinite(numeric_feature) & np.isfinite(numeric_target)
            if int(valid.sum()) >= 20 and numeric_feature[valid].nunique() > 1:
                correlation = _safe_float(numeric_feature[valid].corr(numeric_target[valid]))
                if correlation is not None and abs(correlation) >= 0.995:
                    _add(
                        findings,
                        "warning",
                        "NEAR_PERFECT_TARGET_CORRELATION",
                        f"Feature '{column}' has Pearson correlation {correlation:.5f} with the target.",
                        column=column,
                        train_value=correlation,
                        recommendation="Verify that this feature is available unchanged at inference time.",
                    )
        elif resolved_task == "classification" and column in categorical_columns:
            mapping = pd.DataFrame({
                "feature": feature[comparable].astype("string"),
                "target": target[comparable].astype("string"),
            })
            grouped = mapping.groupby("feature", dropna=False)["target"].agg(
                rows="size", target_classes="nunique"
            )
            repeated = grouped[grouped["rows"] >= 2]
            repeated_rows = int(repeated["rows"].sum())
            pure_repeated_rows = int(
                repeated.loc[repeated["target_classes"] == 1, "rows"].sum()
            )
            purity = float(pure_repeated_rows / max(repeated_rows, 1))
            categorical_mapping_checks[column] = {
                "repeated_rows": repeated_rows,
                "pure_repeated_rows": pure_repeated_rows,
                "pure_mapping_rate": purity,
            }
            minimum_rows = max(20, int(0.10 * int(comparable.sum())))
            if repeated_rows >= minimum_rows and purity >= 0.98:
                _add(
                    findings,
                    "warning",
                    "CATEGORICAL_TARGET_MAPPING",
                    f"Repeated values in '{column}' map to one class {purity:.1%} of the time.",
                    column=column,
                    train_value=purity,
                    recommendation="Confirm this feature exists before prediction and test the mapping in leakage-safe folds.",
                )
    if categorical_mapping_checks:
        result["categorical_mapping_checks"] = categorical_mapping_checks
    return result, resolved_task


def _schema_audit(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_column: str | None,
    findings: list[Finding],
) -> dict[str, Any]:
    if test is None:
        return {"test_supplied": False}
    expected = set(train.columns) - ({target_column} if target_column else set())
    actual = set(test.columns) - ({target_column} if target_column else set())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        _add(
            findings,
            "error",
            "TEST_MISSING_COLUMNS",
            f"Test is missing {len(missing)} training feature columns: {missing}.",
            recommendation="Align train and test schemas before building features.",
        )
    if extra:
        _add(
            findings,
            "warning",
            "TEST_EXTRA_COLUMNS",
            f"Test has {len(extra)} extra columns: {extra}.",
            recommendation="Exclude unexplained test-only columns from the pipeline.",
        )

    mismatches: list[dict[str, str]] = []
    for column in sorted(expected & actual):
        train_kind = _broad_dtype(train[column])
        test_kind = _broad_dtype(test[column])
        if train_kind != test_kind:
            mismatches.append(
                {"column": column, "train_type": train_kind, "test_type": test_kind}
            )
            _add(
                findings,
                "warning",
                "DTYPE_MISMATCH",
                f"Column '{column}' is {train_kind} in train but {test_kind} in test.",
                column=column,
                train_value=str(train[column].dtype),
                test_value=str(test[column].dtype),
                recommendation="Apply one explicit parser/cast to both datasets.",
            )
    return {
        "test_supplied": True,
        "missing_from_test": missing,
        "extra_in_test": extra,
        "broad_dtype_mismatches": mismatches,
    }


def _duplicate_audit(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_column: str | None,
    id_columns: tuple[str, ...],
    findings: list[Finding],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, frame in (("train", train), ("test", test)):
        if frame is None:
            continue
        duplicate_rows = int(frame.duplicated(keep=False).sum())
        result[f"{name}_exact_duplicate_rows"] = duplicate_rows
        if duplicate_rows:
            _add(
                findings,
                "warning",
                "EXACT_DUPLICATE_ROWS",
                f"{name} has {duplicate_rows} rows participating in exact duplicates.",
                train_value=duplicate_rows if name == "train" else None,
                test_value=duplicate_rows if name == "test" else None,
                recommendation="Inspect duplicates and keep them in the same validation fold if retained.",
            )

    if test is None:
        return result
    excluded = set(id_columns)
    if target_column:
        excluded.add(target_column)
    feature_columns = sorted((set(train.columns) & set(test.columns)) - excluded)
    result["fingerprint_columns"] = feature_columns
    if feature_columns:
        train_hash = pd.util.hash_pandas_object(
            train.loc[:, feature_columns], index=False, categorize=True
        )
        test_hash = pd.util.hash_pandas_object(
            test.loc[:, feature_columns], index=False, categorize=True
        )
        train_duplicates = int(train_hash.duplicated(keep=False).sum())
        test_duplicates = int(test_hash.duplicated(keep=False).sum())
        overlap = int(test_hash.isin(set(train_hash.tolist())).sum())
        result["train_duplicate_feature_rows"] = train_duplicates
        result["test_duplicate_feature_rows"] = test_duplicates
        result["test_rows_matching_train_features"] = overlap
        if target_column and target_column in train.columns and train_duplicates:
            duplicate_keys = set(
                train_hash[train_hash.duplicated(keep=False)].tolist()
            )
            duplicate_frame = pd.DataFrame({
                "fingerprint": train_hash,
                "target": train[target_column].to_numpy(),
            })
            conflicts = (
                duplicate_frame[
                    duplicate_frame["fingerprint"].isin(duplicate_keys)
                ]
                .groupby("fingerprint", dropna=False)["target"]
                .nunique(dropna=False)
            )
            conflicting_keys = set(conflicts[conflicts > 1].index.tolist())
            conflicting_rows = int(train_hash.isin(conflicting_keys).sum())
            result["duplicate_feature_groups_with_conflicting_targets"] = int(
                len(conflicting_keys)
            )
            result["rows_in_conflicting_duplicate_groups"] = conflicting_rows
            if conflicting_rows:
                _add(
                    findings,
                    "warning",
                    "CONFLICTING_DUPLICATE_TARGETS",
                    f"{conflicting_rows} rows have duplicate features but disagreeing targets.",
                    column=target_column,
                    train_value=conflicting_rows,
                    recommendation="Keep each duplicate group within one fold and investigate label noise or hidden features.",
                )
        if train_duplicates:
            _add(
                findings,
                "warning",
                "DUPLICATE_TRAIN_FEATURES",
                f"{train_duplicates} training rows share an identical feature fingerprint.",
                train_value=train_duplicates,
                recommendation="Use the fingerprint as a CV group to prevent duplicate leakage.",
            )
        if overlap:
            _add(
                findings,
                "warning",
                "TRAIN_TEST_FEATURE_OVERLAP",
                f"{overlap} test rows exactly match a training feature fingerprint.",
                test_value=overlap,
                recommendation="Verify duplicates; never use target-derived aggregation across this boundary.",
            )
    return result


def audit_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    target_column: str | None = None,
    id_columns: str | Sequence[str] | None = None,
    text_columns: str | Sequence[str] | None = None,
    categorical_columns: str | Sequence[str] | None = None,
    task: Task = "auto",
    missing_rate_warning: float = 0.30,
    missing_rate_delta_warning: float = 0.10,
    drift_warning: float = 0.20,
    unseen_category_warning: float = 0.10,
    text_oov_warning: float = 0.15,
    drift_sample_size: int = 100_000,
    text_sample_size: int = 25_000,
) -> dict[str, Any]:
    """Audit train/test data and return a JSON-serializable report.

    Thresholds are investigation triggers, not automatic feature-removal rules.
    Numeric drift uses KS statistic and PSI; categorical drift uses total
    variation and the fraction of test rows containing unseen values.
    """
    _require_frame(train, "train")
    if test is not None:
        _require_frame(test, "test")
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}.")
    for name, value in {
        "missing_rate_warning": missing_rate_warning,
        "missing_rate_delta_warning": missing_rate_delta_warning,
        "drift_warning": drift_warning,
        "unseen_category_warning": unseen_category_warning,
        "text_oov_warning": text_oov_warning,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if not isinstance(drift_sample_size, int) or drift_sample_size < 100:
        raise ValueError("drift_sample_size must be an integer of at least 100.")
    if not isinstance(text_sample_size, int) or text_sample_size < 100:
        raise ValueError("text_sample_size must be an integer of at least 100.")

    ids = _columns(id_columns)
    texts = _columns(text_columns)
    categoricals = _columns(categorical_columns)
    _require_columns(train, texts, "train")
    _require_columns(train, categoricals, "train")
    if target_column in ids:
        raise ValueError("target_column cannot also be an ID column.")
    overlap = set(texts) & set(categoricals)
    if overlap:
        raise ValueError(
            f"Columns cannot be both text and categorical: {sorted(overlap)}."
        )

    findings: list[Finding] = []
    schema = _schema_audit(train, test, target_column, findings)
    id_report = _id_audit(train, test, ids, findings)
    target_report, resolved_task = _target_audit(
        train, test, target_column, task, ids, categoricals, findings
    )
    duplicates = _duplicate_audit(train, test, target_column, ids, findings)
    memory = {
        "train": _memory_audit(train),
        "test": _memory_audit(test) if test is not None else None,
    }

    train_profiles: dict[str, Any] = {}
    test_profiles: dict[str, Any] = {}
    drift: dict[str, Any] = {}
    common_columns = set(train.columns) & (set(test.columns) if test is not None else set())

    for column in train.columns:
        forced_text = column in texts
        forced_categorical = column in categoricals
        train_profile = _column_profile(
            train[column], forced_text, forced_categorical
        )
        train_profiles[column] = train_profile
        missing_rate = train_profile["missing_rate"]
        if missing_rate >= missing_rate_warning:
            _add(
                findings,
                "warning",
                "HIGH_MISSING_RATE",
                f"Column '{column}' is {missing_rate:.1%} missing in train.",
                column=column,
                train_value=missing_rate,
                recommendation="Add a missingness indicator and verify whether missingness carries meaning.",
            )
        if train_profile["unique"] <= 1:
            _add(
                findings,
                "warning",
                "CONSTANT_COLUMN",
                f"Column '{column}' has at most one non-missing value in train.",
                column=column,
                recommendation="Drop it unless its missingness itself is meaningful.",
            )

        if test is None or column not in common_columns or column == target_column:
            continue
        test_profile = _column_profile(
            test[column], forced_text, forced_categorical
        )
        test_profiles[column] = test_profile
        # Sequential or deliberately disjoint identifiers naturally have
        # different distributions. Their integrity is covered by _id_audit;
        # treating that difference as model drift only creates noise.
        if column in ids:
            continue
        missing_delta = abs(train_profile["missing_rate"] - test_profile["missing_rate"])
        if missing_delta >= missing_rate_delta_warning:
            _add(
                findings,
                "warning",
                "MISSINGNESS_DRIFT",
                f"Column '{column}' missing-rate difference is {missing_delta:.1%}.",
                column=column,
                train_value=train_profile["missing_rate"],
                test_value=test_profile["missing_rate"],
                recommendation="Validate missing-value features and preprocessing on held-out data.",
            )

        train_kind = train_profile["semantic_type"]
        test_kind = test_profile["semantic_type"]
        if train_kind == "numeric" and test_kind == "numeric":
            column_drift = _numeric_drift(train[column], test[column], drift_sample_size)
            drift[column] = column_drift
            if column_drift.get("available") and column_drift["ks_statistic"] >= drift_warning:
                _add(
                    findings,
                    "warning",
                    "NUMERIC_DRIFT",
                    f"Column '{column}' has KS drift {column_drift['ks_statistic']:.3f}.",
                    column=column,
                    train_value=column_drift.get("train_mean"),
                    test_value=column_drift.get("test_mean"),
                    recommendation="Inspect distributions and trust CV only if it reproduces this shift.",
                )
        elif forced_text or train_kind == "text" or test_kind == "text":
            column_drift = _text_length_drift(train[column], test[column], drift_sample_size)
            column_drift.update(
                _text_vocabulary_drift(
                    train[column],
                    test[column],
                    text_sample_size,
                )
            )
            drift[column] = column_drift
            if column_drift.get("available") and column_drift["ks_statistic"] >= drift_warning:
                _add(
                    findings,
                    "warning",
                    "TEXT_LENGTH_DRIFT",
                    f"Column '{column}' has text-length KS drift {column_drift['ks_statistic']:.3f}.",
                    column=column,
                    recommendation="Check truncation, cleaning, and source differences between train and test.",
                )
            if column_drift["oov_token_rate"] >= text_oov_warning:
                _add(
                    findings,
                    "warning",
                    "TEXT_VOCABULARY_DRIFT",
                    f"Column '{column}' has {column_drift['oov_token_rate']:.1%} OOV test tokens.",
                    column=column,
                    test_value=column_drift["oov_token_rate"],
                    recommendation="Use robust word/character features and preserve an unknown-token path.",
                )
        else:
            column_drift = _categorical_drift(train[column], test[column])
            drift[column] = column_drift
            if column_drift["total_variation"] >= drift_warning:
                _add(
                    findings,
                    "warning",
                    "CATEGORICAL_DRIFT",
                    f"Column '{column}' has total-variation drift {column_drift['total_variation']:.3f}.",
                    column=column,
                    recommendation="Use unknown-category handling and reproduce the shift in validation.",
                )
            if column_drift["unseen_test_rate"] >= unseen_category_warning:
                _add(
                    findings,
                    "warning",
                    "UNSEEN_TEST_CATEGORIES",
                    f"Column '{column}' has {column_drift['unseen_test_rate']:.1%} unseen test values.",
                    column=column,
                    test_value=column_drift["unseen_test_rate"],
                    recommendation="Reserve an explicit unknown category; never fit encoders on test labels.",
                )

    if test is not None:
        for column in test.columns:
            if column not in test_profiles:
                test_profiles[column] = _column_profile(
                    test[column],
                    column in texts,
                    column in categoricals,
                )

    findings.sort(
        key=lambda item: (-SEVERITY_ORDER[item.severity], item.code, item.column or "")
    )
    counts = {
        severity: sum(item.severity == severity for item in findings)
        for severity in ("error", "warning", "info")
    }
    status = "error" if counts["error"] else "warning" if counts["warning"] else "pass"
    report = {
        "audit_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "train_rows": int(len(train)),
            "train_columns": int(train.shape[1]),
            "test_rows": int(len(test)) if test is not None else None,
            "test_columns": int(test.shape[1]) if test is not None else None,
            "target_column": target_column,
            "task": resolved_task,
            "finding_counts": counts,
        },
        "configuration": {
            "id_columns": list(ids),
            "text_columns": list(texts),
            "categorical_columns": list(categoricals),
            "missing_rate_warning": missing_rate_warning,
            "missing_rate_delta_warning": missing_rate_delta_warning,
            "drift_warning": drift_warning,
            "unseen_category_warning": unseen_category_warning,
            "text_oov_warning": text_oov_warning,
            "drift_sample_size": drift_sample_size,
            "text_sample_size": text_sample_size,
        },
        "findings": [asdict(item) for item in findings],
        "schema": schema,
        "ids": id_report,
        "target": target_report,
        "duplicates": duplicates,
        "memory": memory,
        "profiles": {"train": train_profiles, "test": test_profiles},
        "drift": drift,
    }
    return _json_ready(report)


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write an audit report as strict UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a self-contained human-readable HTML audit report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    findings = report["findings"]
    finding_rows = [
        (
            item["severity"].upper(),
            item["code"],
            item.get("column") or "—",
            item["message"],
            item.get("recommendation") or "—",
        )
        for item in findings
    ] or [("PASS", "NO_FINDINGS", "—", "No configured checks raised a finding.", "—")]

    profile_rows: list[tuple[Any, ...]] = []
    for dataset, profiles in report["profiles"].items():
        for column, profile in profiles.items():
            profile_rows.append(
                (
                    dataset,
                    column,
                    profile["dtype"],
                    profile["semantic_type"],
                    profile["missing"],
                    f"{profile['missing_rate']:.2%}",
                    profile["unique"],
                    f"{profile['memory_mb']:.3f}",
                )
            )

    drift_rows: list[tuple[Any, ...]] = []
    for column, values in report["drift"].items():
        drift_rows.append(
            (
                column,
                values.get("basis", "values"),
                values.get("ks_statistic", "—"),
                values.get("psi", "—"),
                values.get("total_variation", "—"),
                values.get("unseen_test_rate", "—"),
                values.get("oov_token_rate", "—"),
            )
        )

    memory_rows: list[tuple[Any, ...]] = []
    for dataset, values in report["memory"].items():
        if values is None:
            continue
        memory_rows.append((
            dataset,
            f"{values['current_mb']:.2f}",
            f"{values['estimated_optimized_mb']:.2f}",
            f"{values['estimated_saving_mb']:.2f}",
            f"{values['estimated_saving_fraction']:.1%}",
        ))

    status = html.escape(str(report["status"]).upper())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset audit</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f5f7fb;color:#172033}}
main{{max-width:1200px;margin:32px auto;padding:0 20px}}h1{{margin-bottom:4px}}
.meta{{color:#61708a}}.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}}
.card{{background:white;border:1px solid #dfe5ef;border-radius:12px;padding:16px;min-width:150px;box-shadow:0 2px 8px #1720330d}}
.value{{font-size:24px;font-weight:700}}section{{background:white;border:1px solid #dfe5ef;border-radius:12px;padding:20px;margin:16px 0;overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #e7ebf2;vertical-align:top}}th{{background:#f8f9fc}}
.ERROR{{color:#b42318;font-weight:700}}.WARNING{{color:#b54708;font-weight:700}}.PASS{{color:#027a48}}
code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>Dataset audit: <span class="{status}">{status}</span></h1>
<div class="meta">Generated {html.escape(str(report['generated_at_utc']))} · audit v{html.escape(str(report['audit_version']))}</div>
<div class="cards">
<div class="card"><div>Train rows</div><div class="value">{summary['train_rows']:,}</div></div>
<div class="card"><div>Test rows</div><div class="value">{summary['test_rows'] if summary['test_rows'] is not None else '—'}</div></div>
<div class="card"><div>Errors</div><div class="value">{summary['finding_counts']['error']}</div></div>
<div class="card"><div>Warnings</div><div class="value">{summary['finding_counts']['warning']}</div></div>
<div class="card"><div>Train RAM</div><div class="value">{report['memory']['train']['current_mb']:.1f} MB</div></div>
</div>
<section><h2>Findings</h2>{_html_table(['Severity','Code','Column','Finding','Recommended action'], finding_rows)}</section>
<section><h2>Memory and downcasting</h2>{_html_table(['Dataset','Current MB','Estimated MB','Potential saving MB','Potential saving'], memory_rows)}</section>
<section><h2>Column profiles</h2>{_html_table(['Dataset','Column','dtype','Kind','Missing','Missing rate','Unique','RAM MB'], profile_rows)}</section>
<section><h2>Train–test drift</h2>{_html_table(['Column','Basis','KS','PSI','Total variation','Unseen test rate','Token OOV rate'], drift_rows)}</section>
</main></body></html>"""
    destination.write_text(document, encoding="utf-8")
    return destination


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be a .csv, .parquet, or .pq file.")


def _console_profile(report: dict[str, Any], dataset: str) -> str:
    profiles = report["profiles"][dataset]
    rows = [
        {
            "Column": column,
            "Dtype": profile["dtype"],
            "Kind": profile["semantic_type"],
            "Nulls": profile["missing"],
            "Null%": f"{profile['missing_rate']:.1%}",
            "Uniques": profile["unique"],
            "RAM_MB": f"{profile['memory_mb']:.3f}",
        }
        for column, profile in profiles.items()
    ]
    return pd.DataFrame(rows).to_string(index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ML competition train/test data.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--target-column")
    parser.add_argument("--id-columns", nargs="+")
    parser.add_argument("--text-columns", nargs="+")
    parser.add_argument("--categorical-columns", nargs="+")
    parser.add_argument("--task", choices=sorted(VALID_TASKS), default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("reports/audit.json"))
    parser.add_argument("--output-html", type=Path, default=Path("reports/audit.html"))
    parser.add_argument("--missing-rate-warning", type=float, default=0.30)
    parser.add_argument("--missing-rate-delta-warning", type=float, default=0.10)
    parser.add_argument("--drift-warning", type=float, default=0.20)
    parser.add_argument("--unseen-category-warning", type=float, default=0.10)
    parser.add_argument("--text-oov-warning", type=float, default=0.15)
    parser.add_argument("--drift-sample-size", type=int, default=100_000)
    parser.add_argument("--text-sample-size", type=int, default=25_000)
    parser.add_argument(
        "--fail-on",
        choices=["never", "error", "warning"],
        default="error",
        help="Return exit code 2 when this severity is present (default: error).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    train = _load_table(args.train)
    test = _load_table(args.test) if args.test else None
    report = audit_datasets(
        train,
        test,
        target_column=args.target_column,
        id_columns=args.id_columns,
        text_columns=args.text_columns,
        categorical_columns=args.categorical_columns,
        task=args.task,
        missing_rate_warning=args.missing_rate_warning,
        missing_rate_delta_warning=args.missing_rate_delta_warning,
        drift_warning=args.drift_warning,
        unseen_category_warning=args.unseen_category_warning,
        text_oov_warning=args.text_oov_warning,
        drift_sample_size=args.drift_sample_size,
        text_sample_size=args.text_sample_size,
    )
    write_json_report(report, args.output_json)
    write_html_report(report, args.output_html)

    counts = report["summary"]["finding_counts"]
    print(
        f"Audit status: {report['status'].upper()} | "
        f"errors={counts['error']} warnings={counts['warning']} info={counts['info']}"
    )
    for dataset in ("train", "test"):
        if report["profiles"][dataset]:
            memory = report["memory"][dataset]
            print(
                f"\n{dataset.upper()} PROFILE "
                f"({memory['current_mb']:.2f} MB; estimated optimized "
                f"{memory['estimated_optimized_mb']:.2f} MB)"
            )
            print(_console_profile(report, dataset))
    target = report["target"]
    if target.get("configured"):
        print(f"\nTARGET SUMMARY [{target['column']}] task={target['task']}")
        if "statistics" in target:
            print(pd.Series(target["statistics"]).to_string())
        elif "class_counts" in target:
            print(pd.DataFrame(target["class_counts"][:20]).to_string(index=False))
    print("\nFINDINGS")
    for item in report["findings"]:
        column = f" [{item['column']}]" if item.get("column") else ""
        print(f"{item['severity'].upper():7} {item['code']}{column}: {item['message']}")
    print(f"\nJSON report: {args.output_json}")
    print(f"HTML report: {args.output_html}")

    if args.fail_on == "warning" and (counts["warning"] or counts["error"]):
        return 2
    if args.fail_on == "error" and counts["error"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
