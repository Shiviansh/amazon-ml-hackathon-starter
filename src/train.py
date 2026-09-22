"""Fold-aware, leakage-safe baseline training for ML competitions.

The default pipeline is designed for Amazon-style catalog regression:

* word and character TF-IDF over configured text columns,
* one-hot encoding for categorical columns,
* median imputation and scaling for numeric columns,
* Ridge regression on ``log1p(target)`` or optional LightGBM,
* out-of-fold predictions, averaged test predictions, metrics, and metadata.

Every preprocessor is fitted inside its fold. Raw data is never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Support both ``python -m src.train`` and ``python src/train.py``.
    from .metrics import METRIC_REGISTRY, evaluate_metric
    from .models import (
        FoldModelStore,
        ModelConfig,
        UnifiedModel,
        create_model,
        gpu_backend_preflight,
        smooth_log_mape_objective,
        smooth_log_smape_objective,
        smooth_raw_mape_objective,
        smooth_raw_smape_objective,
    )
    from .splits import assign_folds, iter_fold_indices, validate_fold_assignment
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from metrics import METRIC_REGISTRY, evaluate_metric
    from models import (
        FoldModelStore,
        ModelConfig,
        UnifiedModel,
        create_model,
        gpu_backend_preflight,
        smooth_log_mape_objective,
        smooth_log_smape_objective,
        smooth_raw_mape_objective,
        smooth_raw_smape_objective,
    )
    from splits import assign_folds, iter_fold_indices, validate_fold_assignment

Task = Literal["regression", "classification"]
ModelName = Literal["linear", "lightgbm", "catboost", "xgboost"]
DeviceName = Literal["auto", "cpu", "gpu"]
TargetTransform = Literal["none", "log1p"]

REGRESSION_METRICS = {"smape", "smape_percent", "mape", "mape_percent", "mae", "rmse"}
CLASSIFICATION_METRICS = {"accuracy", "f1_micro", "f1_macro", "f1_weighted"}


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable training configuration."""

    target_column: str
    id_column: str
    text_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    task: Task = "regression"
    metric: str = "smape"
    model: ModelName = "linear"
    device: DeviceName = "cpu"
    target_transform: TargetTransform = "log1p"
    fold_column: str = "fold"
    n_splits: int = 5
    random_state: int = 42
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    min_df: int = 2
    word_max_features: int = 120_000
    char_max_features: int = 120_000
    use_word_tfidf: bool = True
    use_char_tfidf: bool = True
    lightgbm_text_max_features_per_block: int = 20_000
    linear_strength: float = 4.0
    lgbm_estimators: int = 2_000
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 31
    lgbm_early_stopping_rounds: int = 100
    model_params: Mapping[str, Any] = field(default_factory=dict)
    prediction_floor: float | None = 0.0
    prediction_ceiling: float | None = None
    temporal: bool = False
    objective: str = "auto"
    extract_domain_features: bool = False


@dataclass
class TrainingResult:
    """Predictions and metadata produced by cross-validation."""

    oof: pd.DataFrame
    test_predictions: pd.DataFrame
    report: dict[str, Any]
    models: list[tuple[Any, Any]] = field(default_factory=list, repr=False)


def _column_tuple(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(value)
    if any(not isinstance(column, str) or not column for column in result):
        raise ValueError("Feature column names must be non-empty strings.")
    if len(result) != len(set(result)):
        raise ValueError("Feature column lists cannot contain duplicates.")
    return result


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_ready(item) for item in value]
    return str(value)


def validate_config(config: TrainingConfig) -> None:
    """Reject ambiguous or unsafe training configurations early."""
    if config.task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")
    if config.model not in {"linear", "lightgbm", "catboost", "xgboost"}:
        raise ValueError("model must be linear, lightgbm, catboost, or xgboost.")
    if config.device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu.")
    if config.model == "linear" and config.device == "gpu":
        raise ValueError("The linear Ridge/logistic backend is CPU-only.")
    if config.metric not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{config.metric}'.")
    valid_metrics = REGRESSION_METRICS if config.task == "regression" else CLASSIFICATION_METRICS
    if config.metric not in valid_metrics:
        raise ValueError(f"Metric '{config.metric}' is incompatible with task='{config.task}'.")
    if config.objective not in {
        "auto", "smape", "mape", "huber", "l1", "l2", "regression", "mae"
    }:
        raise ValueError(f"Unknown objective '{config.objective}'.")
    if config.task == "classification" and config.target_transform != "none":
        raise ValueError("Classification requires target_transform='none'.")
    if config.target_transform not in {"none", "log1p"}:
        raise ValueError("target_transform must be 'none' or 'log1p'.")
    if not isinstance(config.n_splits, int) or config.n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    if not isinstance(config.min_df, int) or config.min_df < 1:
        raise ValueError("min_df must be an integer of at least 1.")
    if config.word_ngram_max < 1:
        raise ValueError("word_ngram_max must be at least 1.")
    if config.char_ngram_min < 1 or config.char_ngram_max < config.char_ngram_min:
        raise ValueError("Invalid character n-gram range.")
    if config.word_max_features < 1 or config.char_max_features < 1:
        raise ValueError("TF-IDF feature limits must be positive.")
    if config.lightgbm_text_max_features_per_block < 1:
        raise ValueError("lightgbm_text_max_features_per_block must be positive.")
    if config.linear_strength <= 0:
        raise ValueError("linear_strength must be positive.")
    if not isinstance(config.model_params, Mapping):
        raise ValueError("model_params must be a mapping.")
    if any(not isinstance(key, str) or not key.strip() for key in config.model_params):
        raise ValueError("model_params keys must be non-empty strings.")
    controlled_model_params = {
        "C", "alpha", "random_state", "random_seed", "seed", "n_jobs",
        "thread_count", "task_type", "device", "device_type",
        "early_stopping_rounds", "n_estimators", "iterations", "max_iter",
        "learning_rate", "objective", "eval_metric", "metric", "num_class",
    }
    controlled_overlap = controlled_model_params & set(config.model_params)
    if controlled_overlap:
        raise ValueError(
            "model_params cannot override controlled settings: "
            f"{sorted(controlled_overlap)}."
        )
    if config.prediction_floor is not None and config.prediction_ceiling is not None:
        if config.prediction_floor > config.prediction_ceiling:
            raise ValueError("prediction_floor cannot exceed prediction_ceiling.")

    feature_groups = {
        "text": set(config.text_columns),
        "categorical": set(config.categorical_columns),
        "numeric": set(config.numeric_columns),
    }
    if not any(feature_groups.values()):
        raise ValueError("Configure at least one text, categorical, or numeric feature.")
    if (
        config.text_columns
        and not config.use_word_tfidf
        and not config.use_char_tfidf
        and not config.categorical_columns
        and not config.numeric_columns
    ):
        raise ValueError("All configured features are disabled.")
    names = list(feature_groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = feature_groups[left] & feature_groups[right]
            if overlap:
                raise ValueError(
                    f"Columns cannot belong to both {left} and {right}: {sorted(overlap)}."
                )
    forbidden = {config.target_column, config.fold_column}
    selected = set().union(*feature_groups.values())
    overlap = selected & forbidden
    if overlap:
        raise ValueError(f"Target/fold columns cannot be model features: {sorted(overlap)}.")


def _validate_frames(train: pd.DataFrame, test: pd.DataFrame, config: TrainingConfig) -> None:
    if not isinstance(train, pd.DataFrame) or train.empty:
        raise ValueError("train must be a non-empty pandas DataFrame.")
    if not isinstance(test, pd.DataFrame) or test.empty:
        raise ValueError("test must be a non-empty pandas DataFrame.")
    required_train = {
        config.id_column,
        config.target_column,
        *config.text_columns,
        *config.categorical_columns,
        *config.numeric_columns,
    }
    required_test = required_train - {config.target_column}
    missing_train = sorted(required_train - set(train.columns))
    missing_test = sorted(required_test - set(test.columns))
    if missing_train:
        raise ValueError(f"Train is missing required columns: {missing_train}.")
    if missing_test:
        raise ValueError(f"Test is missing required columns: {missing_test}.")
    for name, frame in (("train", train), ("test", test)):
        if frame[config.id_column].isna().any():
            raise ValueError(f"{name} ID column contains missing values.")
        if not frame[config.id_column].is_unique:
            raise ValueError(f"{name} ID column must be unique.")
    if train[config.target_column].isna().any():
        raise ValueError("Training target contains missing values.")
    if set(train[config.id_column]) & set(test[config.id_column]):
        raise ValueError("Train and test IDs overlap; confirm the data boundary before training.")

    if config.task == "regression":
        target = pd.to_numeric(train[config.target_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(target)):
            raise ValueError("Regression target must contain only finite numeric values.")
        if config.target_transform == "log1p" and np.any(target <= -1.0):
            raise ValueError("log1p target transform requires every target to be greater than -1.")
    elif train[config.target_column].nunique(dropna=False) < 2:
        raise ValueError("Classification target must contain at least two classes.")


def _canonical_join_id(value: Any) -> str:
    """Create a type-aware ID token for fail-closed auxiliary joins."""
    if value is None:
        raise ValueError("Auxiliary feature IDs must be non-missing.")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("Auxiliary feature IDs must be non-missing.")
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Auxiliary feature IDs must be finite.")
        return f"float:{numeric.hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _align_auxiliary_rows(
    auxiliary: pd.DataFrame,
    target: pd.DataFrame,
    *,
    id_column: str,
    label: str,
) -> pd.DataFrame:
    if id_column not in auxiliary.columns:
        if "id" in auxiliary.columns and id_column != "id":
            auxiliary = auxiliary.rename(columns={"id": id_column})
        else:
            raise ValueError(f"{label} is missing ID column '{id_column}'.")
    if auxiliary.empty:
        raise ValueError(f"{label} must be non-empty.")
    aux_tokens = [_canonical_join_id(value) for value in auxiliary[id_column]]
    target_tokens = [_canonical_join_id(value) for value in target[id_column]]
    if len(set(aux_tokens)) != len(aux_tokens):
        raise ValueError(f"{label} IDs must be unique.")
    if len(set(target_tokens)) != len(target_tokens):
        raise ValueError("Raw data IDs must be unique before auxiliary joins.")
    if set(aux_tokens) != set(target_tokens):
        missing = len(set(target_tokens) - set(aux_tokens))
        extra = len(set(aux_tokens) - set(target_tokens))
        raise ValueError(
            f"{label} ID set mismatch: missing={missing}, extra={extra}."
        )
    positions = {token: index for index, token in enumerate(aux_tokens)}
    return auxiliary.iloc[[positions[token] for token in target_tokens]].reset_index(drop=True)


def attach_auxiliary_feature_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_auxiliary: pd.DataFrame,
    test_auxiliary: pd.DataFrame,
    config: TrainingConfig,
    *,
    prefix: str,
    label: str,
    folds: pd.DataFrame | None = None,
    allow_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Strictly align and attach a numeric train/test auxiliary feature pair.

    IDs are matched by immutable typed tokens, never row position. Train/test
    schemas must agree exactly. OOF target-encoded columns require the same
    supplied fold table and a matching fold column in the auxiliary train file.
    """
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("Auxiliary feature prefix must be alphanumeric/underscore.")
    train_aligned = _align_auxiliary_rows(
        train_auxiliary, train, id_column=config.id_column, label=f"{label} train"
    )
    test_aligned = _align_auxiliary_rows(
        test_auxiliary, test, id_column=config.id_column, label=f"{label} test"
    )

    forbidden = {config.id_column, config.fold_column}
    if config.target_column in train_aligned.columns or config.target_column in test_aligned.columns:
        raise ValueError(f"{label} must not contain target column '{config.target_column}'.")
    train_columns = [column for column in train_aligned.columns if column not in forbidden]
    test_columns = [column for column in test_aligned.columns if column not in forbidden]
    if not train_columns:
        raise ValueError(f"{label} contains no usable feature columns.")
    if set(train_columns) != set(test_columns):
        raise ValueError(
            f"{label} train/test schemas differ: "
            f"train_only={sorted(set(train_columns) - set(test_columns))}, "
            f"test_only={sorted(set(test_columns) - set(train_columns))}."
        )
    test_aligned = test_aligned[[config.id_column, *test_columns]].reindex(
        columns=[config.id_column, *train_columns]
    )

    target_encoded = [column for column in train_columns if column.startswith("te_")]
    if target_encoded:
        if folds is None or config.fold_column not in train_aligned.columns:
            raise ValueError(
                f"{label} contains target-encoded columns but no verifiable external folds."
            )
        folds_aligned = _align_auxiliary_rows(
            folds[[config.id_column, config.fold_column]],
            train,
            id_column=config.id_column,
            label="fold table",
        )
        if not np.array_equal(
            train_aligned[config.fold_column].to_numpy(),
            folds_aligned[config.fold_column].to_numpy(),
        ):
            raise ValueError(f"{label} fold assignments do not match the supplied fold table.")

    renamed_columns = tuple(f"{prefix}__{column}" for column in train_columns)
    collisions = (set(renamed_columns) & set(train.columns)) | (
        set(renamed_columns) & set(test.columns)
    )
    if collisions:
        raise ValueError(f"{label} feature names collide with raw data: {sorted(collisions)}.")

    train_data: dict[str, np.ndarray] = {}
    test_data: dict[str, np.ndarray] = {}
    for original, renamed in zip(train_columns, renamed_columns):
        for source, destination, split in (
            (train_aligned, train_data, "train"),
            (test_aligned, test_data, "test"),
        ):
            numeric = pd.to_numeric(source[original], errors="coerce")
            invalid = source[original].notna() & numeric.isna()
            if invalid.any():
                raise ValueError(f"{label} {split} column '{original}' is not numeric.")
            values = numeric.to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError(f"{label} {split} column '{original}' contains infinity.")
            if not allow_missing and np.isnan(values).any():
                raise ValueError(f"{label} {split} column '{original}' contains missing values.")
            destination[renamed] = numeric.to_numpy()

    # Construct each block once; repeated DataFrame.insert is prohibitively
    # fragmented for 384/768/1024-dimensional embeddings.
    train_values = pd.DataFrame(train_data, index=train.index)
    test_values = pd.DataFrame(test_data, index=test.index)
    joined_train = pd.concat([train.reset_index(drop=True), train_values.reset_index(drop=True)], axis=1)
    joined_test = pd.concat([test.reset_index(drop=True), test_values.reset_index(drop=True)], axis=1)
    return joined_train, joined_test, renamed_columns


def attach_folds(
    train: pd.DataFrame,
    folds: pd.DataFrame | None,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Return train with validated folds, joining an external fold table by ID."""
    if folds is None:
        if config.fold_column in train.columns:
            result = train.copy()
        else:
            if config.temporal:
                raise ValueError(
                    "temporal=True requires a precomputed temporal fold column or fold table."
                )
            result = assign_folds(
                train,
                target_column=config.target_column,
                task=config.task,
                strategy="stratified",
                n_splits=config.n_splits,
                random_state=config.random_state,
                fold_column=config.fold_column,
            )
    else:
        required = {config.id_column, config.fold_column}
        missing = sorted(required - set(folds.columns))
        if missing:
            raise ValueError(f"Fold table is missing required columns: {missing}.")
        if folds[config.id_column].isna().any() or not folds[config.id_column].is_unique:
            raise ValueError("Fold table IDs must be non-missing and unique.")
        if config.fold_column in train.columns:
            raise ValueError(
                f"Train already contains '{config.fold_column}'; do not also supply a fold table."
            )
        fold_subset = folds[[config.id_column, config.fold_column]]
        result = train.merge(
            fold_subset,
            on=config.id_column,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if result[config.fold_column].isna().any():
            missing_count = int(result[config.fold_column].isna().sum())
            raise ValueError(f"Fold table does not cover {missing_count} training IDs.")
        extra_ids = int((~folds[config.id_column].isin(train[config.id_column])).sum())
        if extra_ids:
            raise ValueError(f"Fold table contains {extra_ids} IDs not present in train.")

    validate_fold_assignment(
        result,
        fold_column=config.fold_column,
        n_splits=config.n_splits,
        temporal=config.temporal,
    )
    return result


def build_feature_frame(frame: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Create model inputs without learning statistics or mutating raw data."""
    feature_data: dict[str, Any] = {}
    if config.text_columns:
        combined = pd.Series("", index=frame.index, dtype="string")
        for column in config.text_columns:
            marker = f" __FIELD_{column.upper()}__ "
            combined = combined + marker + frame[column].fillna("").astype("string")
        feature_data["__combined_text"] = combined.str.strip().fillna("")
    for column in config.categorical_columns:
        feature_data[column] = frame[column].astype("string").fillna("<MISSING>")
    for column in config.numeric_columns:
        feature_data[column] = pd.to_numeric(frame[column], errors="coerce")
    # Build once to avoid a fragmented DataFrame when hundreds of embedding
    # dimensions are configured as numeric features.
    output = pd.DataFrame(feature_data, index=frame.index)
    return output.reset_index(drop=True)


def build_preprocessor(config: TrainingConfig) -> ColumnTransformer:
    """Build an unfitted sparse preprocessing graph."""
    transformers: list[tuple[str, Any, Any]] = []
    word_limit = config.word_max_features
    char_limit = config.char_max_features
    if config.model == "lightgbm":
        word_limit = min(word_limit, config.lightgbm_text_max_features_per_block)
        char_limit = min(char_limit, config.lightgbm_text_max_features_per_block)
    if config.text_columns and config.use_word_tfidf:
        transformers.append((
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, config.word_ngram_max),
                    min_df=config.min_df,
                    max_features=word_limit,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                "__combined_text",
            ))
    if config.text_columns and config.use_char_tfidf:
        transformers.append((
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(config.char_ngram_min, config.char_ngram_max),
                    min_df=config.min_df,
                    max_features=char_limit,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                "__combined_text",
            ))
    if config.categorical_columns:
        transformers.append((
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=config.min_df,
                dtype=np.float32,
            ),
            list(config.categorical_columns),
        ))
    if config.numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]),
            list(config.numeric_columns),
        ))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=-1,
        verbose_feature_names_out=True,
    )


def resolve_training_device(config: TrainingConfig) -> tuple[str, dict[str, Any]]:
    """Resolve auto/cpu/gpu without ever silently downgrading explicit GPU."""

    requested = config.device
    if config.model == "linear":
        if requested == "gpu":
            raise RuntimeError("model='linear' cannot use GPU training.")
        return "cpu", {
            "requested": requested,
            "resolved": "cpu",
            "probe_success": False,
            "reason": "The linear Ridge/logistic backend is CPU-only.",
        }
    if requested == "cpu":
        return "cpu", {
            "requested": requested,
            "resolved": "cpu",
            "probe_success": None,
            "reason": "CPU was explicitly requested.",
        }
    sparse_input = bool(config.text_columns or config.categorical_columns)
    available, reason = gpu_backend_preflight(config.model, sparse_input=sparse_input)
    if requested == "gpu" and not available:
        raise RuntimeError(
            f"GPU training was explicitly requested for {config.model}, but its installed "
            f"backend cannot execute on GPU. Probe error: {reason}"
        )
    resolved = "gpu" if available else "cpu"
    return resolved, {
        "requested": requested,
        "resolved": resolved,
        "probe_success": available,
        "reason": reason if available else f"Auto fallback to CPU: {reason}",
    }


def _make_model(config: TrainingConfig, fold_id: int) -> Any:
    seed = config.random_state + fold_id
    if config.model == "linear":
        if config.task == "regression":
            return Ridge(alpha=config.linear_strength, solver="lsqr")
        return LogisticRegression(
            C=1.0 / config.linear_strength,
            solver="saga",
            max_iter=2_000,
            random_state=seed,
        )

    resolved_obj = config.objective
    if resolved_obj == "auto":
        if config.metric in {"smape", "smape_percent"}:
            resolved_obj = "smape"
        elif config.metric in {"mape", "mape_percent"}:
            resolved_obj = "mape"
        elif config.metric in {"mae"}:
            resolved_obj = "l1"
        else:
            resolved_obj = "l1"

    if config.model in {"catboost", "xgboost"}:
        depth = max(2, min(12, int(round(math.log2(config.lgbm_num_leaves)))))
        parameters = {"max_depth": depth} if config.model == "xgboost" else {"depth": depth}
        parameters.update(dict(config.model_params))
        return create_model(
            ModelConfig(
                engine=config.model,
                task=config.task,
                objective=resolved_obj if config.task == "regression" else None,
                random_state=seed,
                n_jobs=-1,
                device=config.device,
                early_stopping_rounds=config.lgbm_early_stopping_rounds,
                max_iterations=config.lgbm_estimators,
                learning_rate=config.lgbm_learning_rate,
                params=parameters,
            )
        )

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - dependency is installed here.
        raise RuntimeError(
            "model='lightgbm' requires lightgbm. Install requirements.txt first."
        ) from exc
    common = {
        "n_estimators": config.lgbm_estimators,
        "learning_rate": config.lgbm_learning_rate,
        "num_leaves": config.lgbm_num_leaves,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "device_type": "gpu" if config.device == "gpu" else "cpu",
    }
    if config.device == "gpu":
        common["max_bin"] = 63
    common.update(dict(config.model_params))
    if config.task == "regression":
        if resolved_obj == "smape":
            lgb_obj = (
                smooth_log_smape_objective
                if config.target_transform == "log1p"
                else smooth_raw_smape_objective
            )
        elif resolved_obj == "mape":
            lgb_obj = (
                smooth_log_mape_objective
                if config.target_transform == "log1p"
                else smooth_raw_mape_objective
            )
        elif resolved_obj == "huber":
            lgb_obj = "huber"
            common.setdefault("huber_alpha", 0.8)
        elif resolved_obj in {"l1", "mae"}:
            lgb_obj = "regression_l1"
        else:
            lgb_obj = "regression"
        return lgb.LGBMRegressor(objective=lgb_obj, metric="None", **common)
    return lgb.LGBMClassifier(**common)


def _transform_target(values: np.ndarray, config: TrainingConfig) -> np.ndarray:
    if config.task == "regression":
        numeric = values.astype(np.float64)
        return np.log1p(numeric) if config.target_transform == "log1p" else numeric
    return values


def _restore_predictions(values: np.ndarray, config: TrainingConfig) -> np.ndarray:
    result = np.asarray(values)
    if config.task == "regression":
        result = result.astype(np.float64)
        if config.target_transform == "log1p":
            result = np.expm1(result)
        if config.prediction_floor is not None or config.prediction_ceiling is not None:
            result = np.clip(
                result,
                config.prediction_floor if config.prediction_floor is not None else -np.inf,
                config.prediction_ceiling if config.prediction_ceiling is not None else np.inf,
            )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Model produced a non-finite regression prediction.")
    return result


def _inverse_regression_target(values: np.ndarray, config: TrainingConfig) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return np.expm1(result) if config.target_transform == "log1p" else result


def _make_lgbm_eval_metric(config: TrainingConfig) -> Any:
    """Evaluate LightGBM on the official metric in the original target scale."""
    if config.task != "regression":
        return None

    def competition_metric(y_true_fit: np.ndarray, y_pred_fit: np.ndarray) -> tuple[str, float, bool]:
        y_true = _inverse_regression_target(y_true_fit, config)
        y_pred = _restore_predictions(y_pred_fit, config)
        score = evaluate_metric(config.metric, y_true, y_pred)
        return config.metric, float(score), bool(METRIC_REGISTRY[config.metric].greater_is_better)

    competition_metric.__name__ = f"competition_{config.metric}"
    return competition_metric


def _fit_model(
    model: Any,
    x_train: sparse.spmatrix | np.ndarray,
    y_train: np.ndarray,
    x_valid: sparse.spmatrix | np.ndarray,
    y_valid: np.ndarray,
    config: TrainingConfig,
) -> None:
    if isinstance(model, UnifiedModel):
        model.fit(
            x_train,
            y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        return
    if config.model != "lightgbm":
        model.fit(x_train, y_train)
        return
    import lightgbm as lgb

    callbacks = [
        lgb.early_stopping(
            config.lgbm_early_stopping_rounds,
            first_metric_only=True,
            verbose=False,
        ),
        lgb.log_evaluation(period=0),
    ]
    fit_kwargs: dict[str, Any] = {
        "eval_X": x_valid,
        "eval_y": y_valid,
        "callbacks": callbacks,
    }
    custom_metric = _make_lgbm_eval_metric(config)
    if custom_metric is not None:
        fit_kwargs["eval_metric"] = custom_metric
    model.fit(
        x_train,
        y_train,
        **fit_kwargs,
    )


def _aggregate_regression_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    if not predictions:
        raise RuntimeError("No fold produced test predictions.")
    return np.mean(np.vstack(predictions).astype(np.float64), axis=0)


def _aligned_probabilities(model: Any, features: Any, class_labels: list[Any]) -> np.ndarray:
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise RuntimeError("Classification models must expose predict_proba and classes_.")
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    aligned = np.zeros((raw.shape[0], len(class_labels)), dtype=np.float64)
    model_classes = list(model.classes_)
    for target_index, label in enumerate(class_labels):
        matches = [index for index, value in enumerate(model_classes) if value == label]
        if matches:
            aligned[:, target_index] = raw[:, matches[0]]
    row_sums = aligned.sum(axis=1)
    if not np.all(np.isfinite(aligned)) or np.any(row_sums <= 0.0):
        raise RuntimeError("Model produced invalid class probabilities.")
    return aligned / row_sums[:, None]


def _run_signature(
    folded: pd.DataFrame,
    test: pd.DataFrame,
    config: TrainingConfig,
) -> str:
    """Fingerprint configuration, relevant data, IDs, targets, and folds."""
    digest = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    )
    train_columns = [
        config.id_column,
        config.target_column,
        config.fold_column,
        *config.text_columns,
        *config.categorical_columns,
        *config.numeric_columns,
    ]
    test_columns = [
        config.id_column,
        *config.text_columns,
        *config.categorical_columns,
        *config.numeric_columns,
    ]
    for frame, columns in ((folded, train_columns), (test, test_columns)):
        hashes = pd.util.hash_pandas_object(
            frame.loc[:, list(dict.fromkeys(columns))],
            index=False,
            categorize=True,
        ).to_numpy(dtype=np.uint64)
        digest.update(hashes.tobytes())
        digest.update(str(frame.shape).encode("ascii"))
    return digest.hexdigest()


def train_cross_validated(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: TrainingConfig,
    *,
    folds: pd.DataFrame | None = None,
    model_directory: str | Path | None = None,
    retain_models: bool = False,
    overwrite_models: bool = False,
    resume: bool = False,
) -> TrainingResult:
    """Fit one leakage-safe model per fold and return OOF/test predictions."""
    validate_config(config)
    requested_device = config.device
    resolved_device, device_report = resolve_training_device(config)
    config = replace(config, device=resolved_device)
    if config.extract_domain_features and config.text_columns:
        try:
            from .features import extract_deterministic_domain_features
        except ImportError:
            from features import extract_deterministic_domain_features

        t_col = next(
            (c for c in ("TITLE", "title") if c in train.columns),
            config.text_columns[0] if config.text_columns else "TITLE",
        )
        p_col = next(
            (c for c in ("PACK_SIZE", "pack_size", "PACK", "pack") if c in train.columns),
            "PACK_SIZE",
        )
        d_col = next(
            (c for c in ("DESCRIPTION", "description", "desc") if c in train.columns),
            config.text_columns[1] if len(config.text_columns) > 1 else "DESCRIPTION",
        )
        dom_tr = extract_deterministic_domain_features(
            train, title_column=t_col, pack_column=p_col, desc_column=d_col
        )
        dom_te = extract_deterministic_domain_features(
            test, title_column=t_col, pack_column=p_col, desc_column=d_col
        )
        added_cols: list[str] = []
        for col in dom_tr.columns:
            new_col = f"dom__{col}"
            train = train.assign(**{new_col: dom_tr[col].values})
            test = test.assign(**{new_col: dom_te[col].values})
            added_cols.append(new_col)
        config = replace(
            config,
            numeric_columns=tuple(
                dict.fromkeys([*config.numeric_columns, *added_cols])
            ),
        )

    _validate_frames(train, test, config)
    folded = attach_folds(train, folds, config)
    feature_train = build_feature_frame(folded, config)
    feature_test = build_feature_frame(test, config)
    target = folded[config.target_column].to_numpy()
    fold_values = folded[config.fold_column].to_numpy(dtype=int)
    run_signature = _run_signature(folded, test, config)
    class_labels = list(pd.unique(target)) if config.task == "classification" else []
    probability_columns = [f"probability_{index}" for index in range(len(class_labels))]

    if config.task == "regression":
        oof_values = np.full(len(folded), np.nan, dtype=np.float64)
        oof_probabilities = None
    else:
        oof_values = np.full(len(folded), None, dtype=object)
        oof_probabilities = np.full(
            (len(folded), len(class_labels)), np.nan, dtype=np.float64
        )

    test_by_fold: list[np.ndarray] = []
    test_probabilities_by_fold: list[np.ndarray] = []
    fold_reports: list[dict[str, Any]] = []
    fitted_models: list[tuple[Any, Any]] = []
    feature_counts: list[int] = []
    started = time.perf_counter()
    model_path = Path(model_directory) if model_directory is not None else None
    model_store = FoldModelStore(model_path) if model_path is not None else None
    if resume and model_path is None:
        raise ValueError("resume=True requires model_directory checkpoints.")
    if resume and overwrite_models:
        raise ValueError("resume and overwrite_models cannot both be enabled.")
    if model_path is not None:
        existing_models = sorted(model_path.glob("fold_*.joblib")) if model_path.exists() else []
        legacy_or_unsigned = [
            path for path in existing_models
            if not re.fullmatch(r"fold_\d{3}\.joblib", path.name)
            or not path.with_suffix(path.suffix + ".metadata.json").is_file()
        ]
        if legacy_or_unsigned:
            raise ValueError(
                "Model directory contains legacy or unsigned checkpoints that cannot be "
                f"safely resumed: {[str(path) for path in legacy_or_unsigned]}."
            )
        if existing_models and not overwrite_models and not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing fold models: {[str(path) for path in existing_models]}."
            )
        model_path.mkdir(parents=True, exist_ok=True)

    for fold_id, train_positions, valid_positions in iter_fold_indices(
        fold_values,
        temporal=config.temporal,
    ):
        fold_started = time.perf_counter()
        checkpoint_path = model_store.path_for(fold_id) if model_store else None
        if resume and checkpoint_path is not None and checkpoint_path.exists():
            try:
                bundle = model_store.load_inference_bundle(
                    fold_id,
                    run_signature,
                    expected_valid_positions=valid_positions,
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"Checkpoint configuration/data mismatch for fold {fold_id}: {exc}"
                ) from exc
            valid_prediction = np.asarray(bundle.valid_prediction)
            test_prediction = np.asarray(bundle.test_prediction)
            if len(valid_prediction) != len(valid_positions) or len(test_prediction) != len(test):
                raise ValueError(f"Checkpoint prediction shape mismatch for fold {fold_id}.")
            oof_values[valid_positions] = valid_prediction
            test_by_fold.append(test_prediction)
            if config.task == "classification":
                valid_probability = np.asarray(
                    bundle.valid_probabilities, dtype=np.float64
                )
                test_probability = np.asarray(
                    bundle.test_probabilities, dtype=np.float64
                )
                assert oof_probabilities is not None
                if valid_probability.shape != (len(valid_positions), len(class_labels)):
                    raise ValueError(f"Checkpoint probability shape mismatch for fold {fold_id}.")
                oof_probabilities[valid_positions] = valid_probability
                test_probabilities_by_fold.append(test_probability)
            fold_report = dict(bundle.fold_report)
            fold_report["resumed"] = True
            fold_reports.append(fold_report)
            feature_counts.append(int(fold_report["features"]))
            if retain_models:
                fitted_models.append((bundle.preprocessor, bundle.model))
            print(
                f"Fold {fold_id}: resumed | {config.metric}={fold_report['score']:.6f} | "
                f"features={fold_report['features']:,}"
            )
            continue

        preprocessor = build_preprocessor(config)
        x_train = preprocessor.fit_transform(feature_train.iloc[train_positions])
        x_valid = preprocessor.transform(feature_train.iloc[valid_positions])
        x_test = preprocessor.transform(feature_test)
        if x_train.shape[1] == 0:
            raise RuntimeError("Preprocessing produced zero model features.")
        feature_counts.append(int(x_train.shape[1]))

        y_train = _transform_target(target[train_positions], config)
        y_valid_fit = _transform_target(target[valid_positions], config)
        model = _make_model(config, fold_id)
        _fit_model(model, x_train, y_train, x_valid, y_valid_fit, config)

        if config.task == "classification":
            valid_probability = _aligned_probabilities(model, x_valid, class_labels)
            test_probability = _aligned_probabilities(model, x_test, class_labels)
            label_array = np.asarray(class_labels, dtype=object)
            valid_prediction = label_array[np.argmax(valid_probability, axis=1)]
            test_prediction = label_array[np.argmax(test_probability, axis=1)]
            assert oof_probabilities is not None
            oof_probabilities[valid_positions] = valid_probability
            test_probabilities_by_fold.append(test_probability)
        else:
            valid_probability = None
            test_probability = None
            valid_prediction = _restore_predictions(model.predict(x_valid), config)
            test_prediction = _restore_predictions(model.predict(x_test), config)
        oof_values[valid_positions] = valid_prediction
        test_by_fold.append(test_prediction)
        fold_score = evaluate_metric(
            config.metric,
            target[valid_positions],
            valid_prediction,
        )
        fold_report = {
            "fold": fold_id,
            "train_rows": int(len(train_positions)),
            "validation_rows": int(len(valid_positions)),
            "features": int(x_train.shape[1]),
            "metric": config.metric,
            "score": float(fold_score),
            "seconds": float(time.perf_counter() - fold_started),
            "best_iteration": _json_ready(getattr(model, "best_iteration_", None)),
            "early_stopping_metric": (
                config.metric
                if config.model == "lightgbm"
                else "backend_default_original_fit_scale"
                if config.model in {"catboost", "xgboost"}
                else None
            ),
            "device": resolved_device,
            "resumed": False,
        }
        fold_reports.append(fold_report)
        print(
            f"Fold {fold_id}: {config.metric}={fold_score:.6f} | "
            f"train={len(train_positions):,} valid={len(valid_positions):,} "
            f"features={x_train.shape[1]:,} time={fold_report['seconds']:.1f}s"
        )

        if model_path is not None:
            assert model_store is not None
            model_store.save_inference_bundle(
                fold_id,
                run_signature,
                feature_count=int(x_train.shape[1]),
                valid_positions=valid_positions,
                preprocessor=preprocessor,
                model=model,
                training_config=asdict(config),
                fold_report=fold_report,
                class_labels=class_labels,
                valid_prediction=valid_prediction,
                test_prediction=test_prediction,
                valid_probabilities=valid_probability,
                test_probabilities=test_probability,
                provenance={"producer": "src.train", "training_version": "1.2"},
                overwrite=overwrite_models,
            )
        if retain_models:
            fitted_models.append((preprocessor, model))
        del x_train, x_valid, x_test
        if not retain_models:
            del preprocessor, model
        gc.collect()

    evaluated = fold_values >= 0
    if config.task == "regression":
        if np.isnan(oof_values[evaluated]).any():
            raise RuntimeError("Some validation rows did not receive OOF predictions.")
    elif any(value is None for value in oof_values[evaluated]):
        raise RuntimeError("Some validation rows did not receive OOF predictions.")
    if config.task == "classification":
        assert oof_probabilities is not None
        if not np.all(np.isfinite(oof_probabilities[evaluated])):
            raise RuntimeError("Some validation rows have missing class probabilities.")

    overall_score = evaluate_metric(
        config.metric,
        target[evaluated],
        oof_values[evaluated],
    )
    if config.task == "regression":
        test_prediction = _aggregate_regression_predictions(test_by_fold)
        averaged_test_probabilities = None
    else:
        if not test_probabilities_by_fold:
            raise RuntimeError("No fold produced classification probabilities.")
        averaged_test_probabilities = np.mean(
            np.stack(test_probabilities_by_fold, axis=0), axis=0
        )
        label_array = np.asarray(class_labels, dtype=object)
        test_prediction = label_array[np.argmax(averaged_test_probabilities, axis=1)]
    oof = pd.DataFrame({
        config.id_column: folded[config.id_column].to_numpy(),
        config.target_column: target,
        config.fold_column: fold_values,
        "prediction": oof_values,
    })
    test_predictions = pd.DataFrame({
        config.id_column: test[config.id_column].to_numpy(),
        "prediction": test_prediction,
    })
    if config.task == "classification":
        assert oof_probabilities is not None and averaged_test_probabilities is not None
        for index, column in enumerate(probability_columns):
            oof[column] = oof_probabilities[:, index]
            test_predictions[column] = averaged_test_probabilities[:, index]
    report = {
        "training_version": "1.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": asdict(config),
        "device": device_report,
        "overall_metric": config.metric,
        "overall_score": float(overall_score),
        "greater_is_better": bool(METRIC_REGISTRY[config.metric].greater_is_better),
        "folds": fold_reports,
        "score_mean": float(np.mean([item["score"] for item in fold_reports])),
        "score_std": float(np.std([item["score"] for item in fold_reports], ddof=1))
        if len(fold_reports) > 1
        else 0.0,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "oof_rows": int(evaluated.sum()),
        "feature_count_min": int(min(feature_counts)),
        "feature_count_max": int(max(feature_counts)),
        "run_signature": run_signature,
        "resumed_folds": int(sum(bool(item.get("resumed")) for item in fold_reports)),
        "class_probability_columns": [
            {"column": column, "class_label": _json_ready(label)}
            for column, label in zip(probability_columns, class_labels)
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return TrainingResult(oof, test_predictions, _json_ready(report), fitted_models)


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be a .csv, .parquet, or .pq file.")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".csv":
        frame.to_csv(temporary, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(temporary, index=False)
    else:
        raise ValueError("Output table must be .csv, .parquet, or .pq.")
    os.replace(temporary, path)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_training_result(
    result: TrainingResult,
    output_directory: str | Path,
    config: TrainingConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write OOF, test predictions, submission, and run metadata."""
    destination = Path(output_directory)
    known_outputs = {
        "oof": destination / "oof.parquet",
        "test_predictions": destination / "test_predictions.parquet",
        "submission": destination / "submission.csv",
        "report": destination / "training_report.json",
    }
    existing = [path for path in known_outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing run artifacts: {[str(path) for path in existing]}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    _write_table(result.oof, known_outputs["oof"])
    _write_table(result.test_predictions, known_outputs["test_predictions"])
    submission = result.test_predictions.rename(columns={"prediction": config.target_column})
    _write_table(submission, known_outputs["submission"])
    _write_json(result.report, known_outputs["report"])
    return known_outputs


def _existing_run_artifacts(output_directory: Path) -> list[Path]:
    candidates = [
        output_directory / "oof.parquet",
        output_directory / "test_predictions.parquet",
        output_directory / "submission.csv",
        output_directory / "training_report.json",
    ]
    if (output_directory / "models").exists():
        candidates.extend(sorted((output_directory / "models").glob("fold_*.joblib")))
    return [path for path in candidates if path.exists()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe CV baseline.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--text-columns", nargs="*", default=[])
    parser.add_argument("--categorical-columns", nargs="*", default=[])
    parser.add_argument("--numeric-columns", nargs="*", default=[])
    parser.add_argument(
        "--engineered-features",
        nargs=2,
        type=Path,
        metavar=("TRAIN_FEATURES", "TEST_FEATURES"),
        help="Numeric engineered train/test tables joined strictly by ID.",
    )
    parser.add_argument(
        "--embeddings",
        nargs=2,
        type=Path,
        metavar=("TRAIN_EMBEDDINGS", "TEST_EMBEDDINGS"),
        help="Dense embedding train/test tables joined strictly by ID.",
    )
    parser.add_argument(
        "--image-embeddings",
        nargs=2,
        type=Path,
        metavar=("TRAIN_IMAGE_EMBEDDINGS", "TEST_IMAGE_EMBEDDINGS"),
        help="CLIP/DINOv2 image embedding train/test tables joined strictly by ID.",
    )
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--metric", choices=sorted(METRIC_REGISTRY), default="smape")
    parser.add_argument(
        "--model",
        choices=["linear", "lightgbm", "catboost", "xgboost"],
        default="linear",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help=(
            "Training device. auto performs a real backend probe and uses GPU only "
            "when it works; explicit gpu never silently falls back to CPU."
        ),
    )
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=120_000)
    parser.add_argument("--char-max-features", type=int, default=120_000)
    parser.add_argument("--no-word-tfidf", action="store_true")
    parser.add_argument("--no-char-tfidf", action="store_true")
    parser.add_argument("--lightgbm-text-max-features-per-block", type=int, default=20_000)
    parser.add_argument("--linear-strength", type=float, default=4.0)
    parser.add_argument("--lgbm-estimators", type=int, default=2_000)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--model-params-json",
        type=Path,
        help=(
            "JSON object of backend-specific parameters, such as a "
            "best_params_<engine>.json artifact produced by src/tune.py."
        ),
    )
    parser.add_argument(
        "--objective",
        default="auto",
        choices=["auto", "smape", "mape", "huber", "l1", "l2", "regression", "mae"],
        help="Loss objective for tree learners ('auto' dynamically selects based on metric).",
    )
    parser.add_argument(
        "--extract-domain-features",
        action="store_true",
        help="Extract deterministic pack, unit, and dimension features from text columns.",
    )
    parser.add_argument("--prediction-floor", type=float, default=0.0)
    parser.add_argument("--prediction-ceiling", type=float)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated fold checkpoints created with --save-models.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    if args.resume and not args.save_models:
        raise ValueError("--resume requires --save-models.")
    existing = _existing_run_artifacts(args.output_dir)
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"Output run already contains artifacts: {[str(path) for path in existing]}. "
            "Choose another --output-dir or pass --overwrite."
        )
    model_params: dict[str, Any] = {}
    tuned_learning_rate = args.lgbm_learning_rate
    tuned_estimators = args.lgbm_estimators
    tuned_early_stopping = args.lgbm_early_stopping_rounds
    if args.model_params_json is not None:
        payload = json.loads(args.model_params_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--model-params-json must contain a JSON object.")
        # Tuner artifacts contain both metadata and the directly usable mapping.
        candidate = payload.get("model_params", payload)
        if not isinstance(candidate, dict):
            raise ValueError("The model_params entry must be a JSON object.")
        model_params = candidate
        artifact_engine = payload.get("engine")
        if artifact_engine is not None and artifact_engine != args.model:
            raise ValueError(
                f"Tuning artifact engine={artifact_engine!r} does not match --model={args.model!r}."
            )
        tuned_learning_rate = float(payload.get("learning_rate", tuned_learning_rate))
        tuned_estimators = int(payload.get("max_iterations", tuned_estimators))
        tuned_early_stopping = int(
            payload.get("early_stopping_rounds", tuned_early_stopping)
        )
    config = TrainingConfig(
        target_column=args.target_column,
        id_column=args.id_column,
        text_columns=_column_tuple(args.text_columns),
        categorical_columns=_column_tuple(args.categorical_columns),
        numeric_columns=_column_tuple(args.numeric_columns),
        task=args.task,
        metric=args.metric,
        model=args.model,
        device=args.device,
        target_transform=args.target_transform,
        fold_column=args.fold_column,
        n_splits=args.n_splits,
        random_state=args.random_state,
        word_ngram_max=args.word_ngram_max,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        use_word_tfidf=not args.no_word_tfidf,
        use_char_tfidf=not args.no_char_tfidf,
        lightgbm_text_max_features_per_block=args.lightgbm_text_max_features_per_block,
        linear_strength=args.linear_strength,
        lgbm_estimators=tuned_estimators,
        lgbm_learning_rate=tuned_learning_rate,
        lgbm_num_leaves=args.lgbm_num_leaves,
        lgbm_early_stopping_rounds=tuned_early_stopping,
        model_params=model_params,
        prediction_floor=args.prediction_floor,
        prediction_ceiling=args.prediction_ceiling,
        temporal=args.temporal,
        objective=args.objective,
        extract_domain_features=args.extract_domain_features,
    )
    train = _load_table(args.train)
    test = _load_table(args.test)
    folds = _load_table(args.folds) if args.folds else None
    auxiliary_report: list[dict[str, Any]] = []
    auxiliary_specs = (
        ("engineered_features", args.engineered_features, "eng", True),
        ("embeddings", args.embeddings, "emb", False),
        ("image_embeddings", args.image_embeddings, "img", False),
    )
    for label, paths_pair, prefix, allow_missing in auxiliary_specs:
        if paths_pair is None:
            continue
        train_path, test_path = paths_pair
        train_auxiliary = _load_table(train_path)
        test_auxiliary = _load_table(test_path)
        train, test, added_columns = attach_auxiliary_feature_pair(
            train,
            test,
            train_auxiliary,
            test_auxiliary,
            config,
            prefix=prefix,
            label=label,
            folds=folds,
            allow_missing=allow_missing,
        )
        config = replace(
            config,
            numeric_columns=tuple([*config.numeric_columns, *added_columns]),
        )
        auxiliary_report.append({
            "name": label,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "feature_count": len(added_columns),
            "feature_prefix": f"{prefix}__",
        })
    model_directory = args.output_dir / "models" if args.save_models else None
    result = train_cross_validated(
        train,
        test,
        config,
        folds=folds,
        model_directory=model_directory,
        overwrite_models=args.overwrite,
        resume=args.resume,
    )
    result.report["auxiliary_features"] = auxiliary_report
    paths = save_training_result(
        result,
        args.output_dir,
        config,
        overwrite=args.overwrite or args.resume,
    )
    print(
        f"\nOOF {config.metric}: {result.report['overall_score']:.6f} | "
        f"fold mean={result.report['score_mean']:.6f} "
        f"std={result.report['score_std']:.6f}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
