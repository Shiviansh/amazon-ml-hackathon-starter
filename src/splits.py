"""Leakage-aware cross-validation split utilities for ML competitions.

The split should imitate how the hidden test set was created. This module
supports ordinary, stratified, grouped, stratified-grouped, and chronological
validation while preserving the original DataFrame index.

Typical use:

    folded = assign_folds(
        train,
        target_column="price",
        task="regression",
        strategy="kfold",
        n_splits=5,
        random_state=42,
    )

For repeated products or duplicate images, pass ``group_columns`` and use
``group`` or ``stratified_group``. For chronological validation, use ``time``
and consume folds through ``iter_fold_indices(..., temporal=True)`` so future
rows are never included in training.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.utils.multiclass import type_of_target

Task = Literal["auto", "classification", "regression"]
MissingGroupPolicy = Literal["unique", "together", "error"]
RareClassPolicy = Literal["pool", "fallback", "error"]
InfeasibleStratificationPolicy = Literal["warn", "error"]
Strategy = Literal[
    "auto",
    "kfold",
    "stratified",
    "group",
    "stratified_group",
    "time",
]

VALID_TASKS = {"auto", "classification", "regression"}
VALID_STRATEGIES = {
    "auto",
    "kfold",
    "stratified",
    "group",
    "stratified_group",
    "time",
}
VALID_MISSING_GROUP_POLICIES = {"unique", "together", "error"}
VALID_RARE_CLASS_POLICIES = {"pool", "fallback", "error"}
VALID_INFEASIBLE_POLICIES = {"warn", "error"}


class InfeasibleStratificationError(ValueError):
    """Raised when valid data cannot support the requested stratification."""


@dataclass(frozen=True)
class SplitMetadata:
    """Serializable record of how a fold assignment was produced."""

    requested_strategy: str
    strategy: str
    task: str
    n_splits: int
    random_state: int
    target_column: str | None
    group_columns: tuple[str, ...]
    time_column: str | None
    regression_bins: int
    missing_group_policy: str
    rare_class_policy: str
    infeasible_stratification: str
    fold_column: str
    n_rows: int
    n_unassigned: int


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")


def _normalize_columns(columns: str | Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return ()
    if isinstance(columns, str):
        return (columns,)
    normalized = tuple(columns)
    if not normalized or any(not isinstance(column, str) or not column for column in normalized):
        raise ValueError("group_columns must contain non-empty column names.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("group_columns contains duplicate column names.")
    return normalized


def _validate_common(frame: pd.DataFrame, n_splits: int, fold_column: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    if n_splits > len(frame):
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of rows ({len(frame)})."
        )
    if not isinstance(fold_column, str) or not fold_column:
        raise ValueError("fold_column must be a non-empty string.")
    if fold_column in frame.columns:
        raise ValueError(
            f"Fold column '{fold_column}' already exists. Remove or rename it explicitly."
        )


def _resolve_task(target: pd.Series | None, task: Task) -> Literal["classification", "regression"]:
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {sorted(VALID_TASKS)}.")
    if task != "auto":
        return task
    if target is None:
        return "regression"
    inferred = type_of_target(target.to_numpy())
    if inferred in {"binary", "multiclass"}:
        return "classification"
    if inferred == "continuous":
        return "regression"
    raise ValueError(
        f"Could not infer a supported task from target type '{inferred}'. "
        "Set task explicitly."
    )


def _resolve_strategy(
    strategy: Strategy,
    task: Literal["classification", "regression"],
    has_target: bool,
    has_groups: bool,
    has_time: bool,
) -> str:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {sorted(VALID_STRATEGIES)}."
        )
    if strategy != "auto":
        return strategy
    if has_time and has_groups:
        raise ValueError(
            "Automatic splitting is ambiguous when both time_column and "
            "group_columns are provided. Choose an explicit strategy after "
            "deciding whether chronological integrity or group isolation is primary."
        )
    if has_time:
        return "time"
    if has_groups:
        return "stratified_group" if has_target else "group"
    if has_target:
        return "stratified"
    return "kfold"


def build_group_labels(
    frame: pd.DataFrame,
    group_columns: str | Sequence[str],
    *,
    missing_group_policy: MissingGroupPolicy = "unique",
) -> pd.Series:
    """Build stable group IDs from one or more columns.

    ``missing_group_policy='unique'`` treats each row with any missing grouping
    key as an independent group, which is safest for unknown catalog data.
    ``'together'`` groups identical missing-key combinations, while ``'error'``
    requires complete grouping keys. Non-missing rows are hashed vectorially.
    """
    columns = _normalize_columns(group_columns)
    _require_columns(frame, columns)
    if missing_group_policy not in VALID_MISSING_GROUP_POLICIES:
        raise ValueError(
            "missing_group_policy must be one of: 'unique', 'together', or 'error'."
        )

    group_values = frame.loc[:, list(columns)]
    missing_rows = group_values.isna().any(axis=1).to_numpy()
    n_missing = int(missing_rows.sum())
    if missing_group_policy == "error" and n_missing:
        raise ValueError(
            f"Grouping columns contain missing values in {n_missing} rows."
        )

    hashes = pd.util.hash_pandas_object(
        group_values,
        index=False,
        categorize=True,
    )
    labels = hashes.astype("string").radd("g:")
    labels.index = frame.index
    labels.name = "_split_group"

    if missing_group_policy == "unique" and n_missing:
        positions = np.flatnonzero(missing_rows)
        unique_missing = pd.Series(positions, dtype="string").radd("m:").to_numpy()
        labels.iloc[positions] = unique_missing
    return labels.astype("string")


def _regression_strata(
    target: pd.Series,
    n_splits: int,
    requested_bins: int,
) -> pd.Series:
    """Create quantile bins with enough samples for stratified splitting."""
    if requested_bins < 2:
        raise ValueError("regression_bins must be at least 2.")
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any() or not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError("Regression target contains missing, non-numeric, or infinite values.")

    max_bins = min(requested_bins, int(numeric.nunique()), len(numeric) // n_splits)
    for n_bins in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(numeric, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        counts = bins.value_counts(dropna=False)
        if not bins.isna().any() and len(counts) >= 2 and int(counts.min()) >= n_splits:
            return bins.astype(np.int32)
    raise InfeasibleStratificationError(
        "Could not create regression strata with at least n_splits samples per bin. "
        "Use strategy='kfold' or reduce n_splits."
    )


def _classification_target(
    target: pd.Series,
    n_splits: int,
    rare_class_policy: RareClassPolicy,
) -> pd.Series | None:
    """Create integer strata, optionally pooling long-tail classes."""
    if target.isna().any():
        raise ValueError("Classification target contains missing values.")
    if rare_class_policy not in VALID_RARE_CLASS_POLICIES:
        raise ValueError(
            "rare_class_policy must be one of: 'pool', 'fallback', or 'error'."
        )

    counts = target.value_counts(dropna=False)
    rare = counts[counts < n_splits]
    encoded = pd.Series(
        pd.factorize(target, sort=False)[0],
        index=target.index,
        dtype="int32",
    )
    if rare.empty:
        return encoded

    preview = rare.head(10).to_dict()
    if rare_class_policy == "error":
        raise ValueError(
            "Every class needs at least n_splits rows for stratification. "
            f"Too-small classes: {preview}."
        )
    if rare_class_policy == "fallback":
        warnings.warn(
            "Rare classes cannot be individually stratified; falling back to a "
            f"non-stratified splitter. Examples: {preview}.",
            UserWarning,
            stacklevel=3,
        )
        return None

    rare_mask = target.isin(rare.index)
    pooled_count = int(rare_mask.sum())
    if pooled_count < n_splits:
        warnings.warn(
            "Rare-class pooling still has fewer rows than n_splits; falling back "
            f"to a non-stratified splitter. Examples: {preview}.",
            UserWarning,
            stacklevel=3,
        )
        return None

    encoded.loc[rare_mask] = int(encoded.max()) + 1
    if encoded.nunique() < 2:
        warnings.warn(
            "Rare-class pooling produced only one stratum; falling back to a "
            "non-stratified splitter.",
            UserWarning,
            stacklevel=3,
        )
        return None
    warnings.warn(
        f"Pooled {len(rare)} classes ({pooled_count} rows) into one stratum for "
        "fold assignment only; original target labels are unchanged.",
        UserWarning,
        stacklevel=3,
    )
    return encoded


def _stratification_labels(
    target: pd.Series,
    task: Literal["classification", "regression"],
    n_splits: int,
    regression_bins: int,
    rare_class_policy: RareClassPolicy,
) -> pd.Series | None:
    if task == "classification":
        return _classification_target(target, n_splits, rare_class_policy)
    return _regression_strata(target, n_splits, regression_bins)


def _validate_group_feasibility(
    groups: pd.Series,
    n_splits: int,
    stratification_labels: pd.Series | None = None,
    infeasible_stratification: InfeasibleStratificationPolicy = "warn",
) -> None:
    n_groups = int(groups.nunique(dropna=False))
    if n_groups < n_splits:
        raise ValueError(
            f"Only {n_groups} unique groups are available for {n_splits} folds."
        )
    if stratification_labels is None:
        return
    cross = pd.DataFrame({"label": stratification_labels, "group": groups})
    groups_per_label = cross.groupby("label", dropna=False)["group"].nunique()
    impossible = groups_per_label[groups_per_label < n_splits]
    if not impossible.empty:
        preview = impossible.head(10).to_dict()
        message = (
            "Perfect stratified-group balance is impossible because some strata "
            f"occur in fewer than n_splits groups: {preview}. The splitter can "
            "still produce the best approximate group-safe assignment."
        )
        if infeasible_stratification == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)


def assign_folds(
    frame: pd.DataFrame,
    *,
    target_column: str | None = None,
    task: Task = "auto",
    strategy: Strategy = "auto",
    group_columns: str | Sequence[str] | None = None,
    time_column: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    regression_bins: int = 20,
    missing_group_policy: MissingGroupPolicy = "unique",
    rare_class_policy: RareClassPolicy = "pool",
    infeasible_stratification: InfeasibleStratificationPolicy = "warn",
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a validated integer fold column.

    ``time`` uses expanding-window validation. Initial training-only rows are
    marked ``-1``. For temporal folds, call ``iter_fold_indices`` with
    ``temporal=True``; training with ``fold != validation_fold`` would leak
    future observations.
    """
    _validate_common(frame, n_splits, fold_column)
    if missing_group_policy not in VALID_MISSING_GROUP_POLICIES:
        raise ValueError(
            "missing_group_policy must be one of: 'unique', 'together', or 'error'."
        )
    if rare_class_policy not in VALID_RARE_CLASS_POLICIES:
        raise ValueError(
            "rare_class_policy must be one of: 'pool', 'fallback', or 'error'."
        )
    if infeasible_stratification not in VALID_INFEASIBLE_POLICIES:
        raise ValueError(
            "infeasible_stratification must be either 'warn' or 'error'."
        )
    group_columns_tuple = _normalize_columns(group_columns)
    required = list(group_columns_tuple)
    if target_column is not None:
        required.append(target_column)
    if time_column is not None:
        required.append(time_column)
    _require_columns(frame, required)

    target = frame[target_column] if target_column is not None else None
    resolved_task = _resolve_task(target, task)
    resolved_strategy = _resolve_strategy(
        strategy,
        resolved_task,
        target is not None,
        bool(group_columns_tuple),
        time_column is not None,
    )
    requested_strategy = resolved_strategy

    if resolved_strategy in {"stratified", "stratified_group"} and target is None:
        raise ValueError(f"strategy='{resolved_strategy}' requires target_column.")
    if resolved_strategy in {"group", "stratified_group"} and not group_columns_tuple:
        raise ValueError(f"strategy='{resolved_strategy}' requires group_columns.")
    if resolved_strategy == "time" and time_column is None:
        raise ValueError("strategy='time' requires time_column.")
    if resolved_strategy == "time" and group_columns_tuple:
        raise ValueError(
            "strategy='time' cannot also guarantee group isolation. Remove "
            "group_columns or choose an explicit group-based strategy."
        )

    result = frame.copy()
    folds = np.full(len(frame), -1, dtype=np.int16)
    positions = np.arange(len(frame))
    groups = (
        build_group_labels(
            frame,
            group_columns_tuple,
            missing_group_policy=missing_group_policy,
        )
        if group_columns_tuple
        else None
    )
    stratify = None
    if resolved_strategy in {"stratified", "stratified_group"}:
        assert target is not None
        try:
            stratify = _stratification_labels(
                target,
                resolved_task,
                n_splits,
                regression_bins,
                rare_class_policy,
            )
        except InfeasibleStratificationError:
            if infeasible_stratification == "error":
                raise
            warnings.warn(
                "Regression stratification is infeasible; falling back to a "
                "non-stratified splitter.",
                UserWarning,
                stacklevel=2,
            )
            stratify = None

        if stratify is None:
            resolved_strategy = (
                "group" if resolved_strategy == "stratified_group" else "kfold"
            )

    if resolved_strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(positions)
    elif resolved_strategy == "stratified":
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iterator = splitter.split(positions, stratify)
    elif resolved_strategy == "group":
        assert groups is not None
        _validate_group_feasibility(groups, n_splits)
        splitter = GroupKFold(n_splits=n_splits)
        split_iterator = splitter.split(positions, groups=groups)
    elif resolved_strategy == "stratified_group":
        assert groups is not None and stratify is not None
        _validate_group_feasibility(
            groups,
            n_splits,
            stratify,
            infeasible_stratification,
        )
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iterator = splitter.split(positions, stratify, groups)
    else:
        assert time_column is not None
        times = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
        if times.isna().any():
            raise ValueError(
                f"Time column '{time_column}' contains missing or unparseable values."
            )
        unique_times = np.sort(times.unique())
        if len(unique_times) <= n_splits:
            raise ValueError(
                f"Time splitting needs more than {n_splits} unique timestamps; "
                f"found {len(unique_times)}."
            )
        splitter = TimeSeriesSplit(n_splits=n_splits)
        chronological_splits = splitter.split(unique_times)
        split_iterator = (
            (
                np.flatnonzero(times.isin(unique_times[train_times]).to_numpy()),
                np.flatnonzero(times.isin(unique_times[valid_times]).to_numpy()),
            )
            for train_times, valid_times in chronological_splits
        )

    for fold_id, (_, valid_positions) in enumerate(split_iterator):
        if np.any(folds[valid_positions] != -1):
            raise RuntimeError("A row was assigned to more than one validation fold.")
        folds[valid_positions] = fold_id

    result[fold_column] = pd.Series(folds, index=result.index, dtype="int16")
    validate_fold_assignment(
        result,
        fold_column=fold_column,
        n_splits=n_splits,
        group_columns=group_columns_tuple or None,
        missing_group_policy=missing_group_policy,
        temporal=resolved_strategy == "time",
    )
    result.attrs["split_metadata"] = asdict(
        SplitMetadata(
            requested_strategy=requested_strategy,
            strategy=resolved_strategy,
            task=resolved_task,
            n_splits=n_splits,
            random_state=random_state,
            target_column=target_column,
            group_columns=group_columns_tuple,
            time_column=time_column,
            regression_bins=regression_bins,
            missing_group_policy=missing_group_policy,
            rare_class_policy=rare_class_policy,
            infeasible_stratification=infeasible_stratification,
            fold_column=fold_column,
            n_rows=len(result),
            n_unassigned=int(np.sum(folds == -1)),
        )
    )
    return result


def validate_fold_assignment(
    frame: pd.DataFrame,
    *,
    fold_column: str = "fold",
    n_splits: int | None = None,
    group_columns: str | Sequence[str] | None = None,
    missing_group_policy: MissingGroupPolicy = "unique",
    temporal: bool = False,
) -> None:
    """Raise ``ValueError`` if a fold assignment is incomplete or leaky."""
    _require_columns(frame, [fold_column])
    fold_values = frame[fold_column]
    if fold_values.isna().any():
        raise ValueError("Fold assignment contains missing values.")
    numeric = pd.to_numeric(fold_values, errors="coerce")
    if numeric.isna().any() or not np.all(numeric == np.floor(numeric)):
        raise ValueError("Fold values must be integers.")
    folds = numeric.astype(int)
    if (folds < -1).any():
        raise ValueError("Fold values below -1 are invalid.")
    if not temporal and (folds == -1).any():
        raise ValueError("Non-temporal fold assignments cannot contain -1.")

    observed = sorted(int(value) for value in folds[folds >= 0].unique())
    expected_count = n_splits if n_splits is not None else len(observed)
    expected = list(range(expected_count))
    if observed != expected:
        raise ValueError(f"Expected validation folds {expected}, observed {observed}.")
    for fold_id in expected:
        if int((folds == fold_id).sum()) == 0:
            raise ValueError(f"Fold {fold_id} has no validation rows.")

    columns = _normalize_columns(group_columns)
    if columns:
        groups = build_group_labels(
            frame,
            columns,
            missing_group_policy=missing_group_policy,
        )
        group_fold_counts = pd.DataFrame({"group": groups, "fold": folds})
        # Ignore temporal warm-up (-1); those rows are training-only by design.
        group_fold_counts = group_fold_counts[group_fold_counts["fold"] >= 0]
        leaking = group_fold_counts.groupby("group")["fold"].nunique()
        if (leaking > 1).any():
            examples = leaking[leaking > 1].index[:5].tolist()
            raise ValueError(
                "Group leakage detected: some groups occur in multiple validation folds. "
                f"Example hashes: {examples}."
            )


def iter_fold_indices(
    folds: Sequence[int] | pd.Series | np.ndarray,
    *,
    temporal: bool = False,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold_id, train_positions, validation_positions)``.

    For ordinary CV, training rows are all rows outside the validation fold.
    For temporal CV, training rows have an earlier fold ID, including ``-1``
    warm-up rows; this prevents training on the future.
    """
    values = np.asarray(folds)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("folds must be a non-empty 1-D sequence.")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("folds must contain integer values.")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("folds must contain finite integer values.")
    values = values.astype(int)
    fold_ids = sorted(int(value) for value in np.unique(values) if value >= 0)
    if fold_ids != list(range(len(fold_ids))):
        raise ValueError(f"Validation fold IDs must be contiguous from zero; got {fold_ids}.")

    positions = np.arange(values.size)
    for fold_id in fold_ids:
        valid = positions[values == fold_id]
        train = positions[values < fold_id] if temporal else positions[values != fold_id]
        if train.size == 0 or valid.size == 0:
            raise ValueError(f"Fold {fold_id} has an empty train or validation partition.")
        yield fold_id, train, valid


def fold_summary(
    frame: pd.DataFrame,
    *,
    fold_column: str = "fold",
    target_column: str | None = None,
) -> pd.DataFrame:
    """Return fold sizes and optional numeric target statistics."""
    required = [fold_column] + ([target_column] if target_column else [])
    _require_columns(frame, required)
    grouped = frame.groupby(fold_column, dropna=False)
    summary = grouped.size().rename("rows").to_frame()
    summary["fraction"] = summary["rows"] / len(frame)
    if target_column is not None and pd.api.types.is_numeric_dtype(frame[target_column]):
        summary["target_mean"] = grouped[target_column].mean()
        summary["target_std"] = grouped[target_column].std()
        summary["target_min"] = grouped[target_column].min()
        summary["target_max"] = grouped[target_column].max()
    return summary.reset_index()


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be a .csv, .parquet, or .pq file.")


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Output must be a .csv, .parquet, or .pq file.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-aware CV folds.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-column")
    parser.add_argument("--id-column")
    parser.add_argument("--task", choices=sorted(VALID_TASKS), default="auto")
    parser.add_argument("--strategy", choices=sorted(VALID_STRATEGIES), default="auto")
    parser.add_argument("--group-columns", nargs="+")
    parser.add_argument("--time-column")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--regression-bins", type=int, default=20)
    parser.add_argument(
        "--missing-group-policy",
        choices=sorted(VALID_MISSING_GROUP_POLICIES),
        default="unique",
    )
    parser.add_argument(
        "--rare-class-policy",
        choices=sorted(VALID_RARE_CLASS_POLICIES),
        default="pool",
    )
    parser.add_argument(
        "--infeasible-stratification",
        choices=sorted(VALID_INFEASIBLE_POLICIES),
        default="warn",
    )
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Write every input column instead of only ID/group/time/target/fold columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame = _load_table(args.input)
    if args.id_column:
        _require_columns(frame, [args.id_column])
        if frame[args.id_column].isna().any() or not frame[args.id_column].is_unique:
            raise ValueError("id_column must be non-missing and unique.")

    folded = assign_folds(
        frame,
        target_column=args.target_column,
        task=args.task,
        strategy=args.strategy,
        group_columns=args.group_columns,
        time_column=args.time_column,
        n_splits=args.n_splits,
        random_state=args.random_state,
        regression_bins=args.regression_bins,
        missing_group_policy=args.missing_group_policy,
        rare_class_policy=args.rare_class_policy,
        infeasible_stratification=args.infeasible_stratification,
        fold_column=args.fold_column,
    )
    metadata = dict(folded.attrs["split_metadata"])

    if args.full_data:
        output = folded
    else:
        keep = [args.id_column] if args.id_column else []
        keep += list(args.group_columns or [])
        keep += [args.time_column] if args.time_column else []
        keep += [args.target_column] if args.target_column else []
        keep += [args.fold_column]
        output = folded.loc[:, list(dict.fromkeys(column for column in keep if column))]

    _save_table(output, args.output)
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(fold_summary(folded, fold_column=args.fold_column, target_column=args.target_column).to_string(index=False))
    print(f"\nSaved folds: {args.output}")
    print(f"Saved metadata: {metadata_path}")
    if metadata["n_unassigned"]:
        print(
            f"Temporal warm-up rows marked -1: {metadata['n_unassigned']}. "
            "Use iter_fold_indices(..., temporal=True)."
        )


if __name__ == "__main__":
    main()
