"""Residual profiling and failure diagnostics for OOF price predictions.

The analyzer joins genuine out-of-fold predictions to optional catalog
metadata strictly by ID, computes row-level error diagnostics, summarizes error
by business-relevant segments, and writes a compact self-contained HTML report.
It never fits on test predictions and never trusts input row order.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .metrics import smape
except ImportError:  # pragma: no cover - direct script execution
    from metrics import smape


ERROR_ANALYSIS_VERSION = "1.0"
RankMetric = Literal["smape", "absolute_error", "relative_error", "log_error"]
PRICE_TIER_LABELS = ("cheap", "mid", "luxury")
MISSING_LABEL = "<MISSING>"


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    worst_n: int = 50
    rank_by: RankMetric = "smape"
    min_group_size: int = 20
    price_quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    price_tier_cuts: tuple[float, float] | None = None
    brand_rare_max: int = 5
    brand_medium_max: int = 20
    text_short_max_words: int = 10
    text_medium_max_words: int = 30
    text_long_max_words: int = 100
    relative_error_floor: float = 1.0
    near_zero_prediction_threshold: float = 1.0
    top_groups_in_report: int = 25

    def __post_init__(self) -> None:
        if isinstance(self.worst_n, bool) or not isinstance(self.worst_n, int) or self.worst_n < 1:
            raise ValueError("worst_n must be a positive integer.")
        if self.rank_by not in {"smape", "absolute_error", "relative_error", "log_error"}:
            raise ValueError("rank_by must be smape, absolute_error, relative_error, or log_error.")
        if (
            isinstance(self.min_group_size, bool)
            or not isinstance(self.min_group_size, int)
            or self.min_group_size < 1
        ):
            raise ValueError("min_group_size must be a positive integer.")
        low_q, high_q = self.price_quantiles
        if not (0.0 < low_q < high_q < 1.0):
            raise ValueError("price_quantiles must satisfy 0 < low < high < 1.")
        if self.price_tier_cuts is not None:
            low, high = self.price_tier_cuts
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
                raise ValueError("price_tier_cuts must be finite, non-negative, and increasing.")
        if not (1 <= self.brand_rare_max < self.brand_medium_max):
            raise ValueError("Brand frequency thresholds must be positive and increasing.")
        if not (
            0 <= self.text_short_max_words
            < self.text_medium_max_words
            < self.text_long_max_words
        ):
            raise ValueError("Text length thresholds must be non-negative and increasing.")
        if not math.isfinite(self.relative_error_floor) or self.relative_error_floor <= 0.0:
            raise ValueError("relative_error_floor must be finite and positive.")
        if (
            not math.isfinite(self.near_zero_prediction_threshold)
            or self.near_zero_prediction_threshold < 0.0
        ):
            raise ValueError("near_zero_prediction_threshold must be finite and non-negative.")
        if (
            isinstance(self.top_groups_in_report, bool)
            or not isinstance(self.top_groups_in_report, int)
            or self.top_groups_in_report < 1
        ):
            raise ValueError("top_groups_in_report must be a positive integer.")


@dataclass
class ErrorAnalysisResult:
    rows: pd.DataFrame
    worst_errors: pd.DataFrame
    segment_tables: dict[str, pd.DataFrame]
    report: dict[str, Any]
    html_report: str


def _canonical_id(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        raise ValueError("IDs cannot contain missing values.")
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"number:{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("IDs cannot contain NaN or infinity.")
        return f"number:{int(number)}" if number.is_integer() else f"number:{number:.17g}"
    text = str(value).strip()
    if not text:
        raise ValueError("IDs cannot contain empty strings.")
    return f"text:{text}"


def _id_positions(values: Sequence[Any], *, label: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, value in enumerate(values):
        token = _canonical_id(value)
        if token in positions:
            raise ValueError(f"{label} contains duplicate ID {value!r}.")
        positions[token] = index
    return positions


_UNEVALUATED_TOKENS = frozenset(
    {"", "test", "holdout", "heldout", "held_out", "unassigned", "unused", "none", "null", "nan", "na", "n/a"}
)


def _evaluated_fold_mask(values: pd.Series, *, label: str) -> np.ndarray:
    series = values.reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_known = numeric.notna().to_numpy()
    if numeric_known.any() and np.isinf(numeric[numeric.notna()].to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains an infinite fold value.")
    missing = pd.isna(series).to_numpy()
    mask = ~missing
    mask[numeric_known] = numeric[numeric.notna()].to_numpy(dtype=float) >= 0.0
    for index in np.flatnonzero(~numeric_known & ~missing):
        token = str(series.iloc[index]).strip().casefold().replace("-", "_")
        if token in _UNEVALUATED_TOKENS:
            mask[index] = False
    if not np.any(mask):
        raise ValueError(f"{label} contains no evaluated OOF rows.")
    return mask


def _numeric(values: Any, *, label: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array.")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or infinity.")
    return result


def _row_smape(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = np.abs(target) + np.abs(prediction)
    result = np.zeros_like(denominator, dtype=np.float64)
    np.divide(
        2.0 * np.abs(prediction - target),
        denominator,
        out=result,
        where=denominator != 0.0,
    )
    return result


def _price_tiers(target: np.ndarray, config: ErrorAnalysisConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if config.price_tier_cuts is not None:
        low, high = config.price_tier_cuts
        source = "fixed"
    else:
        low, high = np.quantile(target, config.price_quantiles)
        source = "target_quantiles"
    if not low < high:
        tiers = np.full(len(target), "mid", dtype=object)
        return tiers, {
            "source": source,
            "low_cut": float(low),
            "high_cut": float(high),
            "degenerate": True,
        }
    tiers = np.select(
        [target <= low, target >= high],
        [PRICE_TIER_LABELS[0], PRICE_TIER_LABELS[2]],
        default=PRICE_TIER_LABELS[1],
    )
    return tiers.astype(object), {
        "source": source,
        "low_cut": float(low),
        "high_cut": float(high),
        "degenerate": False,
    }


def _brand_frequency_tier(values: pd.Series, config: ErrorAnalysisConfig) -> tuple[pd.Series, pd.Series]:
    clean = values.fillna(MISSING_LABEL).astype(str).str.strip().replace("", MISSING_LABEL)
    counts = clean.value_counts(dropna=False)

    def tier(brand: str) -> str:
        if brand == MISSING_LABEL:
            return "missing"
        count = int(counts[brand])
        if count == 1:
            return "singleton"
        if count <= config.brand_rare_max:
            return "rare"
        if count <= config.brand_medium_max:
            return "medium"
        return "frequent"

    return clean, clean.map(tier)


def _text_features(frame: pd.DataFrame, columns: Sequence[str], config: ErrorAnalysisConfig) -> pd.DataFrame:
    if not columns:
        combined = pd.Series([""] * len(frame), index=frame.index, dtype="string")
    else:
        combined = frame.loc[:, list(columns)].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    words = combined.str.findall(r"\b\w+\b").str.len().astype(int)
    characters = combined.str.len().astype(int)
    tiers = np.select(
        [
            words == 0,
            words <= config.text_short_max_words,
            words <= config.text_medium_max_words,
            words <= config.text_long_max_words,
        ],
        ["empty", "short", "medium", "long"],
        default="very_long",
    )
    return pd.DataFrame(
        {
            "combined_text": combined,
            "text_word_count": words,
            "text_char_count": characters,
            "text_length_tier": tiers,
        },
        index=frame.index,
    )


_MULTIPACK_RE = re.compile(
    r"\b(?:pack|set|case|bundle|lot)\s+of\s+\d+\b|\b\d+\s*[x×]\s*\d+\b|\b\d+\s*(?:pack|packs|pcs|pieces)\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"(?:₹|\$|€|£)|\b(?:rs\.?|inr|usd|eur|gbp)\b", re.IGNORECASE)
_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|litre|liter|oz|lb|cm|mm|m)\b",
    re.IGNORECASE,
)


def _segment_summary(
    rows: pd.DataFrame,
    segment_column: str,
    *,
    id_column: str,
    config: ErrorAnalysisConfig,
) -> pd.DataFrame:
    total_absolute_error = float(rows["absolute_error"].sum())
    records: list[dict[str, Any]] = []
    for value, group in rows.groupby(segment_column, dropna=False, observed=True, sort=False):
        count = len(group)
        worst_index = group["row_smape"].idxmax()
        row_errors = group["row_smape"].to_numpy(dtype=float)
        absolute = group["absolute_error"].to_numpy(dtype=float)
        signed = group["signed_error"].to_numpy(dtype=float)
        record = {
            "segment": str(value),
            "count": int(count),
            "share_rows": float(count / len(rows)),
            "target_mean": float(group["target"].mean()),
            "target_median": float(group["target"].median()),
            "prediction_mean": float(group["prediction"].mean()),
            "bias": float(signed.mean()),
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(signed ** 2))),
            "smape": float(row_errors.mean()),
            "median_row_smape": float(np.median(row_errors)),
            "p90_row_smape": float(np.quantile(row_errors, 0.90)),
            "p95_absolute_error": float(np.quantile(absolute, 0.95)),
            "smape_standard_error": (
                float(np.std(row_errors, ddof=1) / math.sqrt(count)) if count > 1 else None
            ),
            "share_total_absolute_error": (
                float(absolute.sum() / total_absolute_error) if total_absolute_error > 0.0 else 0.0
            ),
            "underprediction_rate": float(np.mean(signed < 0.0)),
            "eligible_for_ranking": bool(count >= config.min_group_size),
            "worst_id": rows.loc[worst_index, id_column],
        }
        records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(
        ["eligible_for_ranking", "smape", "count"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _failure_mode_summary(rows: pd.DataFrame, flags: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for flag in flags:
        selected = rows[rows[flag]]
        if selected.empty:
            records.append(
                {"failure_mode": flag, "count": 0, "prevalence": 0.0, "smape": None, "mae": None}
            )
            continue
        records.append(
            {
                "failure_mode": flag,
                "count": int(len(selected)),
                "prevalence": float(len(selected) / len(rows)),
                "smape": float(selected["row_smape"].mean()),
                "mae": float(selected["absolute_error"].mean()),
                "share_top_decile_errors": float(
                    np.mean(selected["row_smape"] >= rows["row_smape"].quantile(0.90))
                ),
            }
        )
    return pd.DataFrame(records).sort_values("smape", ascending=False, na_position="last")


def _overall_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    target = rows["target"].to_numpy(dtype=float)
    prediction = rows["prediction"].to_numpy(dtype=float)
    absolute = rows["absolute_error"].to_numpy(dtype=float)
    signed = rows["signed_error"].to_numpy(dtype=float)
    row_smape = rows["row_smape"].to_numpy(dtype=float)
    return {
        "rows": int(len(rows)),
        "smape": smape(target, prediction),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(signed ** 2))),
        "bias": float(signed.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "p90_row_smape": float(np.quantile(row_smape, 0.90)),
        "underprediction_rate": float(np.mean(signed < 0.0)),
        "overprediction_rate": float(np.mean(signed > 0.0)),
        "exact_prediction_rate": float(np.mean(signed == 0.0)),
        "within_10_percent_rate": float(np.mean(rows["relative_error"] <= 0.10)),
        "within_20_percent_rate": float(np.mean(rows["relative_error"] <= 0.20)),
    }


def analyze_errors(
    oof: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    id_column: str,
    target_column: str,
    prediction_column: str = "prediction",
    fold_column: str = "fold",
    category_columns: Sequence[str] = (),
    brand_column: str | None = None,
    text_columns: Sequence[str] = (),
    config: ErrorAnalysisConfig | None = None,
) -> ErrorAnalysisResult:
    """Create row-level and segmented diagnostics from evaluated OOF rows."""

    started = time.perf_counter()
    settings = config or ErrorAnalysisConfig()
    if not isinstance(oof, pd.DataFrame) or oof.empty:
        raise ValueError("oof must be a non-empty DataFrame.")
    core_columns = (id_column, target_column, prediction_column, fold_column)
    if any(not isinstance(column, str) or not column.strip() for column in core_columns):
        raise ValueError("Core column names must be non-empty strings.")
    if len(set(core_columns)) != len(core_columns):
        raise ValueError("ID, target, prediction, and fold columns must have distinct names.")
    categories = tuple(category_columns)
    texts = tuple(text_columns)
    metadata_roles = (*categories, *texts, *([brand_column] if brand_column else []))
    requested_metadata = tuple(dict.fromkeys(metadata_roles))
    if any(not isinstance(column, str) or not column.strip() for column in requested_metadata):
        raise ValueError("Metadata column names must be non-empty strings.")
    if len(metadata_roles) != len(set(metadata_roles)):
        raise ValueError("Metadata columns cannot be assigned to multiple analysis roles.")
    reserved = {id_column, target_column, prediction_column, fold_column}
    overlap = set(requested_metadata) & reserved
    if overlap:
        raise ValueError(f"Metadata analysis columns overlap reserved columns: {sorted(overlap)}.")
    required = {id_column, target_column, prediction_column}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF table is missing columns: {missing}.")

    base_columns = [id_column, target_column, prediction_column]
    if fold_column in oof.columns:
        base_columns.append(fold_column)
    base = oof.loc[:, base_columns].copy()
    original_rows = len(base)
    if fold_column in base.columns:
        mask = _evaluated_fold_mask(base[fold_column], label="OOF folds")
        base = base.loc[mask].reset_index(drop=True)
    dropped_rows = original_rows - len(base)
    oof_positions = _id_positions(base[id_column].tolist(), label="OOF IDs")
    reference_tokens = list(oof_positions)
    metadata_extra_rows = 0

    if requested_metadata:
        source = metadata if metadata is not None else oof
        if id_column not in source.columns:
            raise ValueError(f"Metadata is missing ID column {id_column!r}.")
        missing = sorted(set(requested_metadata) - set(source.columns))
        if missing:
            raise ValueError(f"Metadata is missing requested columns: {missing}.")
        metadata_positions = _id_positions(source[id_column].tolist(), label="metadata IDs")
        missing_ids = [token for token in reference_tokens if token not in metadata_positions]
        if missing_ids:
            raise ValueError(f"Metadata does not cover {len(missing_ids)} evaluated OOF IDs.")
        if metadata is not None:
            metadata_extra_rows = len(set(metadata_positions) - set(reference_tokens))
        aligned = source.iloc[[metadata_positions[token] for token in reference_tokens]].reset_index(drop=True)
        for column in requested_metadata:
            base[column] = aligned[column].to_numpy()

    target = _numeric(base[target_column], label="OOF target")
    prediction = _numeric(base[prediction_column], label="OOF prediction")
    if np.any(target < 0.0):
        raise ValueError("Price targets must be non-negative.")

    rows = pd.DataFrame(
        {
            id_column: base[id_column].to_numpy(),
            "target": target,
            "prediction": prediction,
        }
    )
    if fold_column in base.columns:
        rows[fold_column] = base[fold_column].to_numpy()
    for column in requested_metadata:
        rows[column] = base[column].to_numpy()

    rows["signed_error"] = prediction - target
    rows["absolute_error"] = np.abs(rows["signed_error"])
    rows["row_smape"] = _row_smape(target, prediction)
    rows["relative_error"] = rows["absolute_error"] / np.maximum(
        np.abs(target), settings.relative_error_floor
    )
    rows["log_error"] = np.log1p(np.maximum(prediction, 0.0)) - np.log1p(target)
    rows["absolute_log_error"] = np.abs(rows["log_error"])
    rows["prediction_to_target_ratio"] = np.divide(
        prediction,
        target,
        out=np.full(len(target), np.nan, dtype=float),
        where=target != 0.0,
    )
    price_tier, price_tier_metadata = _price_tiers(target, settings)
    rows["price_tier"] = price_tier

    if brand_column:
        clean_brand, brand_tier = _brand_frequency_tier(rows[brand_column], settings)
        rows[brand_column] = clean_brand
        rows["brand_frequency_tier"] = brand_tier
    text_frame = _text_features(rows, texts, settings)
    for column in text_frame:
        rows[column] = text_frame[column]

    combined_text = rows["combined_text"].fillna("").astype(str)
    rows["has_multipack_signal"] = combined_text.str.contains(_MULTIPACK_RE, regex=True)
    rows["has_currency_signal"] = combined_text.str.contains(_CURRENCY_RE, regex=True)
    rows["has_unit_signal"] = combined_text.str.contains(_UNIT_RE, regex=True)
    rows["severe_underprediction"] = (target > 0.0) & (prediction < 0.5 * target)
    rows["severe_overprediction"] = prediction > 2.0 * np.maximum(target, settings.relative_error_floor)
    rows["near_zero_prediction"] = prediction <= settings.near_zero_prediction_threshold
    rows["negative_prediction"] = prediction < 0.0

    segment_columns: dict[str, str] = {"price_tier": "price_tier", "text_length": "text_length_tier"}
    if fold_column in rows.columns:
        segment_columns["fold"] = fold_column
    if brand_column:
        segment_columns["brand"] = brand_column
        segment_columns["brand_frequency"] = "brand_frequency_tier"
    for column in categories:
        rows[column] = rows[column].fillna(MISSING_LABEL).astype(str).str.strip().replace("", MISSING_LABEL)
        segment_columns[f"category__{column}"] = column

    segment_tables = {
        name: _segment_summary(
            rows,
            column,
            id_column=id_column,
            config=settings,
        )
        for name, column in segment_columns.items()
    }
    failure_flags = (
        "severe_underprediction",
        "severe_overprediction",
        "near_zero_prediction",
        "negative_prediction",
        "has_multipack_signal",
        "has_currency_signal",
        "has_unit_signal",
    )
    segment_tables["failure_modes"] = _failure_mode_summary(rows, failure_flags)

    rank_column = {
        "smape": "row_smape",
        "absolute_error": "absolute_error",
        "relative_error": "relative_error",
        "log_error": "absolute_log_error",
    }[settings.rank_by]
    worst = rows.sort_values(
        [rank_column, "absolute_error", "absolute_log_error"],
        ascending=[False, False, False],
        kind="stable",
    ).head(min(settings.worst_n, len(rows))).copy()
    worst.insert(0, "error_rank", np.arange(1, len(worst) + 1))

    overall = _overall_metrics(rows)
    top_segments: dict[str, list[dict[str, Any]]] = {}
    for name, table in segment_tables.items():
        if name == "failure_modes" or table.empty:
            continue
        eligible = table[table["eligible_for_ranking"]]
        top_segments[name] = _json_ready(
            eligible.head(settings.top_groups_in_report).to_dict(orient="records")
        )
    warnings: list[str] = []
    if price_tier_metadata["degenerate"]:
        warnings.append("Price tiers are degenerate because the target has insufficient variation.")
    if not categories:
        warnings.append("No category columns were configured; category diagnostics are absent.")
    if brand_column is None:
        warnings.append("No brand column was configured; brand-frequency diagnostics are absent.")
    if not texts:
        warnings.append("No text columns were configured; text and packaging signals are unavailable.")

    report = {
        "error_analysis_version": ERROR_ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": asdict(settings),
        "overall": overall,
        "price_tiers": price_tier_metadata,
        "dropped_unevaluated_oof_rows": int(dropped_rows),
        "metadata_extra_rows": int(metadata_extra_rows),
        "segment_table_rows": {name: int(len(table)) for name, table in segment_tables.items()},
        "top_problem_segments": top_segments,
        "warnings": warnings,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    html_report = _render_html_report(report, worst, segment_tables, id_column, settings)
    return ErrorAnalysisResult(rows, worst, segment_tables, _json_ready(report), html_report)


def _render_html_report(
    report: Mapping[str, Any],
    worst: pd.DataFrame,
    segment_tables: Mapping[str, pd.DataFrame],
    id_column: str,
    config: ErrorAnalysisConfig,
) -> str:
    overall = report["overall"]
    cards = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in (
            ("OOF SMAPE", f"{overall['smape']:.6f}"),
            ("MAE", f"{overall['mae']:.3f}"),
            ("RMSE", f"{overall['rmse']:.3f}"),
            ("Bias", f"{overall['bias']:.3f}"),
            ("Within 20%", f"{overall['within_20_percent_rate']:.1%}"),
            ("Rows", f"{overall['rows']:,}"),
        )
    )
    warning_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in report["warnings"])
    sections: list[str] = []
    preferred = ["price_tier", "category", "brand_frequency", "text_length", "failure_modes", "fold"]
    ordered_names = sorted(
        segment_tables,
        key=lambda name: next(
            (index for index, prefix in enumerate(preferred) if name.startswith(prefix)),
            len(preferred),
        ),
    )
    for name in ordered_names:
        table = segment_tables[name].head(config.top_groups_in_report)
        sections.append(
            f"<section><h2>{html.escape(name.replace('__', ': ').replace('_', ' ').title())}</h2>"
            + table.to_html(index=False, border=0, classes="data", escape=True, float_format=lambda x: f"{x:.5g}")
            + "</section>"
        )
    worst_columns = [
        column
        for column in (
            "error_rank",
            id_column,
            "target",
            "prediction",
            "row_smape",
            "absolute_error",
            "relative_error",
            "price_tier",
            "brand_frequency_tier",
            "text_word_count",
            "has_multipack_signal",
            "has_currency_signal",
            "has_unit_signal",
            "combined_text",
        )
        if column in worst.columns
    ]
    worst_html = worst.loc[:, worst_columns].to_html(
        index=False,
        border=0,
        classes="data worst",
        escape=True,
        float_format=lambda x: f"{x:.6g}",
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OOF Error Analysis</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f7fb;color:#172033}}main{{max-width:1500px;margin:auto;padding:28px}}
h1{{margin:0 0 6px}}.muted{{color:#667085}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}}
.card,section{{background:#fff;border:1px solid #e3e7ef;border-radius:10px;padding:16px;box-shadow:0 1px 2px #1018280d}}.card span{{display:block;color:#667085;font-size:13px}}.card strong{{font-size:23px}}
section{{margin:18px 0;overflow:auto}}table.data{{border-collapse:collapse;width:100%;font-size:12px}}.data th,.data td{{border-bottom:1px solid #e8ebf0;padding:7px;text-align:right;vertical-align:top}}.data th:first-child,.data td:first-child{{text-align:left}}.worst td:last-child{{min-width:340px;text-align:left;max-width:650px}}
ul{{background:#fff6df;border:1px solid #f3d18a;border-radius:8px;padding:12px 30px}}</style></head>
<body><main><h1>OOF Residual & Failure Analysis</h1><p class="muted">Generated {html.escape(str(report['created_at_utc']))}; ranking={html.escape(config.rank_by)}</p>
<div class="cards">{cards}</div><h2>Warnings / coverage notes</h2><ul>{warning_html or '<li>None</li>'}</ul>
{''.join(sections)}<section><h2>Worst {len(worst)} prediction misses</h2>{worst_html}</section>
</main></body></html>"""


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input tables must be CSV or Parquet.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_from_files(
    oof_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    id_column: str,
    target_column: str,
    prediction_column: str = "prediction",
    fold_column: str = "fold",
    category_columns: Sequence[str] = (),
    brand_column: str | None = None,
    text_columns: Sequence[str] = (),
    config: ErrorAnalysisConfig | None = None,
) -> ErrorAnalysisResult:
    oof_file = Path(oof_path)
    metadata_file = Path(metadata_path) if metadata_path is not None else None
    result = analyze_errors(
        _load_table(oof_file),
        metadata=_load_table(metadata_file) if metadata_file is not None else None,
        id_column=id_column,
        target_column=target_column,
        prediction_column=prediction_column,
        fold_column=fold_column,
        category_columns=category_columns,
        brand_column=brand_column,
        text_columns=text_columns,
        config=config,
    )
    result.report["sources"] = {
        "oof": {"path": str(oof_file.resolve()), "sha256": _sha256_file(oof_file)},
        "metadata": (
            {"path": str(metadata_file.resolve()), "sha256": _sha256_file(metadata_file)}
            if metadata_file is not None else None
        ),
    }
    signature_payload = {
        "config": result.report["config"],
        "sources": result.report["sources"],
        "overall": result.report["overall"],
    }
    result.report["run_signature"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result.html_report = _render_html_report(
        result.report,
        result.worst_errors,
        result.segment_tables,
        id_column,
        config or ErrorAnalysisConfig(),
    )
    return result


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        if path.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=False)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame.to_parquet(temporary, index=False)
        else:
            raise ValueError("Output table must be CSV or Parquet.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_filename(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean or "segment"


def save_error_analysis(
    result: ErrorAnalysisResult,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_directory)
    outputs: dict[str, Path] = {
        "rows": destination / "error_rows.parquet",
        "worst_errors": destination / "worst_errors.csv",
        "report": destination / "error_analysis_report.json",
        "html": destination / "error_analysis_report.html",
    }
    used_segment_filenames: set[str] = set()
    for name in result.segment_tables:
        safe_name = _safe_filename(name)
        filename = f"segment_{safe_name}.csv"
        if filename.casefold() in used_segment_filenames:
            suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
            filename = f"segment_{safe_name}_{suffix}.csv"
        used_segment_filenames.add(filename.casefold())
        outputs[f"segment__{name}"] = destination / filename
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing error-analysis artifacts: {[str(path) for path in existing]}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_table(result.rows, outputs["rows"])
    _atomic_table(result.worst_errors, outputs["worst_errors"])
    _atomic_json(result.report, outputs["report"])
    _atomic_text(result.html_report, outputs["html"])
    for name, table in result.segment_tables.items():
        _atomic_table(table, outputs[f"segment__{name}"])
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile OOF residuals by catalog segment and emit worst-product diagnostics."
    )
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--category-columns", nargs="*", default=[])
    parser.add_argument("--brand-column")
    parser.add_argument("--text-columns", nargs="*", default=[])
    parser.add_argument("--worst-n", type=int, default=50)
    parser.add_argument(
        "--rank-by",
        choices=("smape", "absolute_error", "relative_error", "log_error"),
        default="smape",
    )
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--price-tier-cuts", nargs=2, type=float)
    parser.add_argument("--brand-rare-max", type=int, default=5)
    parser.add_argument("--brand-medium-max", type=int, default=20)
    parser.add_argument("--text-short-max-words", type=int, default=10)
    parser.add_argument("--text-medium-max-words", type=int, default=30)
    parser.add_argument("--text-long-max-words", type=int, default=100)
    parser.add_argument("--relative-error-floor", type=float, default=1.0)
    parser.add_argument("--near-zero-prediction-threshold", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ErrorAnalysisConfig(
        worst_n=args.worst_n,
        rank_by=args.rank_by,
        min_group_size=args.min_group_size,
        price_tier_cuts=tuple(args.price_tier_cuts) if args.price_tier_cuts else None,
        brand_rare_max=args.brand_rare_max,
        brand_medium_max=args.brand_medium_max,
        text_short_max_words=args.text_short_max_words,
        text_medium_max_words=args.text_medium_max_words,
        text_long_max_words=args.text_long_max_words,
        relative_error_floor=args.relative_error_floor,
        near_zero_prediction_threshold=args.near_zero_prediction_threshold,
    )
    result = analyze_from_files(
        args.oof,
        metadata_path=args.metadata,
        id_column=args.id_column,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        fold_column=args.fold_column,
        category_columns=args.category_columns,
        brand_column=args.brand_column,
        text_columns=args.text_columns,
        config=config,
    )
    outputs = save_error_analysis(result, args.output_dir, overwrite=args.overwrite)
    overall = result.report["overall"]
    print(
        f"OOF SMAPE={overall['smape']:.8f} | MAE={overall['mae']:.4f} | "
        f"bias={overall['bias']:.4f} | worst rows={len(result.worst_errors)}"
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
