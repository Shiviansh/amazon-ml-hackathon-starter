"""Fold-safe Optuna tuning for XGBoost and CatBoost price models.

The expensive feature preprocessing graph is fitted once per CV training fold,
never on validation rows, and cached on disk. Optuna trials then reuse those
immutable fold matrices while each candidate model is still trained and scored
independently on every fold. The objective is the repository's official metric
on concatenated out-of-fold predictions in the original target scale.

GPU requests are fail-closed: a real backend fit must pass before a long study
starts. CatBoost's GPU parameter space intentionally omits ``rsm`` because that
backend supports it only for pairwise ranking losses, not price regression.
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
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

try:  # Support ``python -m src.tune`` and ``python src/tune.py``.
    from .metrics import METRIC_REGISTRY, evaluate_metric
    from .models import ModelConfig, create_model, gpu_backend_preflight
    from .splits import iter_fold_indices
    from .train import (
        REGRESSION_METRICS,
        TrainingConfig,
        _restore_predictions,
        _transform_target,
        attach_folds,
        build_feature_frame,
        build_preprocessor,
        validate_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from metrics import METRIC_REGISTRY, evaluate_metric
    from models import ModelConfig, create_model, gpu_backend_preflight
    from splits import iter_fold_indices
    from train import (
        REGRESSION_METRICS,
        TrainingConfig,
        _restore_predictions,
        _transform_target,
        attach_folds,
        build_feature_frame,
        build_preprocessor,
        validate_config,
    )


TUNING_VERSION = "1.0"
EngineName = Literal["xgboost", "catboost"]
DeviceName = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True)
class TuningConfig:
    """Validated search/runtime controls shared by each backend study."""

    engines: tuple[EngineName, ...] = ("xgboost", "catboost")
    n_trials: int = 50
    timeout_seconds: float | None = None
    device: DeviceName = "gpu"
    max_iterations: int = 3_000
    early_stopping_rounds: int = 150
    sampler_seed: int = 42
    startup_trials: int = 10
    pruner_warmup_folds: int = 1
    n_jobs_trials: int = 1
    backend_n_jobs: int = -1
    study_name: str = "amazon_ml_price"
    storage: str | None = None
    load_if_exists: bool = True
    show_progress_bar: bool = False

    def __post_init__(self) -> None:
        if not self.engines or len(self.engines) != len(set(self.engines)):
            raise ValueError("engines must be a non-empty sequence without duplicates.")
        if any(engine not in {"xgboost", "catboost"} for engine in self.engines):
            raise ValueError("Only xgboost and catboost tuning are supported.")
        for name, value, minimum in (
            ("n_trials", self.n_trials, 1),
            ("max_iterations", self.max_iterations, 1),
            ("startup_trials", self.startup_trials, 0),
            ("pruner_warmup_folds", self.pruner_warmup_folds, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if (
            isinstance(self.early_stopping_rounds, bool)
            or not isinstance(self.early_stopping_rounds, int)
            or self.early_stopping_rounds < 0
        ):
            raise ValueError("early_stopping_rounds must be an integer >= 0.")
        if self.timeout_seconds is not None:
            if (
                isinstance(self.timeout_seconds, bool)
                or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0
            ):
                raise ValueError("timeout_seconds must be finite and positive.")
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be auto, cpu, or gpu.")
        if isinstance(self.sampler_seed, bool) or not isinstance(self.sampler_seed, int):
            raise ValueError("sampler_seed must be an integer.")
        if not isinstance(self.n_jobs_trials, int) or self.n_jobs_trials < 1:
            raise ValueError("n_jobs_trials must be a positive integer.")
        if self.device == "gpu" and self.n_jobs_trials != 1:
            raise ValueError("Use n_jobs_trials=1 on one GPU to avoid VRAM contention and OOMs.")
        if (
            isinstance(self.backend_n_jobs, bool)
            or not isinstance(self.backend_n_jobs, int)
            or self.backend_n_jobs == 0
        ):
            raise ValueError("backend_n_jobs must be a non-zero integer.")
        if not isinstance(self.study_name, str) or not self.study_name.strip():
            raise ValueError("study_name must be a non-empty string.")


@dataclass
class TuningResult:
    """Serializable study outputs plus live Optuna study objects."""

    report: dict[str, Any]
    trials: pd.DataFrame
    best_parameters: dict[str, dict[str, Any]]
    studies: dict[str, Any] = field(default_factory=dict, repr=False)

    def close(self) -> None:
        """Release persistent Optuna database sessions and Windows file handles."""

        for study in self.studies.values():
            storage = getattr(study, "_storage", None)
            if storage is None:
                continue
            remove_session = getattr(storage, "remove_session", None)
            if callable(remove_session):
                remove_session()
            backend = getattr(storage, "_backend", storage)
            backend_remove = getattr(backend, "remove_session", None)
            if callable(backend_remove) and backend is not storage:
                backend_remove()
            engine = getattr(backend, "engine", None)
            if engine is not None and hasattr(engine, "dispose"):
                engine.dispose()

    def __enter__(self) -> "TuningResult":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class _CachedFold:
    fold_id: int
    train_matrix: Path
    valid_matrix: Path
    train_target: Path
    valid_target_fit: Path
    valid_target_raw: Path
    is_sparse: bool
    train_rows: int
    valid_rows: int
    feature_count: int


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - guarded by requirements
        raise RuntimeError(
            "Hyperparameter tuning requires Optuna. Run: pip install -r requirements.txt"
        ) from exc
    return optuna


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_ready(item) for item in value]
    return str(value)


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


def attach_tuning_feature_table(
    train: pd.DataFrame,
    features: pd.DataFrame,
    config: TrainingConfig,
    *,
    prefix: str,
    expected_folds: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Strictly ID-align one numeric feature table for tuning.

    Target-encoded columns (``te_*``) are accepted only when the feature table's
    fold column exactly matches the externally supplied folds or the fold column
    already present in ``train``.
    """

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("Feature prefix must start with a letter and be alphanumeric/underscore.")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Feature table must be a non-empty DataFrame.")
    if config.id_column not in train.columns or config.id_column not in features.columns:
        raise ValueError(f"Both tables must contain ID column {config.id_column!r}.")
    if config.target_column in features.columns:
        raise ValueError("Auxiliary feature tables must not contain the target column.")

    train_tokens = [_canonical_id(value) for value in train[config.id_column]]
    feature_tokens = [_canonical_id(value) for value in features[config.id_column]]
    if len(set(train_tokens)) != len(train_tokens):
        raise ValueError("Training IDs must be unique.")
    if len(set(feature_tokens)) != len(feature_tokens):
        raise ValueError("Feature-table IDs must be unique.")
    if set(train_tokens) != set(feature_tokens):
        missing = len(set(train_tokens) - set(feature_tokens))
        extra = len(set(feature_tokens) - set(train_tokens))
        raise ValueError(f"Feature-table ID set mismatch: missing={missing}, extra={extra}.")
    positions = {token: index for index, token in enumerate(feature_tokens)}
    aligned = features.iloc[[positions[token] for token in train_tokens]].reset_index(drop=True)

    value_columns = [
        column for column in aligned.columns if column not in {config.id_column, config.fold_column}
    ]
    if not value_columns:
        raise ValueError("Feature table contains no usable feature columns.")
    target_encoded = [column for column in value_columns if column.startswith("te_")]
    if target_encoded:
        if config.fold_column not in aligned.columns:
            raise ValueError("Target-encoded features require a fold column in the feature table.")
        fold_source = expected_folds
        if fold_source is None and config.fold_column in train.columns:
            fold_source = train[[config.id_column, config.fold_column]]
        if fold_source is None:
            raise ValueError("Target-encoded features require externally verifiable folds.")
        if config.id_column not in fold_source or config.fold_column not in fold_source:
            raise ValueError("Expected folds are missing ID/fold columns.")
        fold_tokens = [_canonical_id(value) for value in fold_source[config.id_column]]
        if len(set(fold_tokens)) != len(fold_tokens) or set(fold_tokens) != set(train_tokens):
            raise ValueError("Expected-fold IDs must uniquely match training IDs.")
        fold_positions = {token: index for index, token in enumerate(fold_tokens)}
        expected = fold_source.iloc[[fold_positions[token] for token in train_tokens]][
            config.fold_column
        ].to_numpy()
        if not np.array_equal(aligned[config.fold_column].to_numpy(), expected):
            raise ValueError("Feature-table folds do not match the expected CV folds.")

    output_data: dict[str, np.ndarray] = {}
    added: list[str] = []
    for column in value_columns:
        numeric = pd.to_numeric(aligned[column], errors="coerce")
        invalid = aligned[column].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Feature column {column!r} must be numeric and finite when present.")
        renamed = f"{prefix}__{column}"
        if renamed in train.columns or renamed in output_data:
            raise ValueError(f"Feature name collision: {renamed!r}.")
        output_data[renamed] = numeric.to_numpy()
        added.append(renamed)
    block = pd.DataFrame(output_data, index=train.index)
    joined = pd.concat([train.reset_index(drop=True), block.reset_index(drop=True)], axis=1)
    return joined, tuple(added)


def _validate_inputs(train: pd.DataFrame, config: TrainingConfig) -> None:
    if not isinstance(train, pd.DataFrame) or train.empty:
        raise ValueError("train must be a non-empty DataFrame.")
    if config.task != "regression":
        raise ValueError("This price tuner currently supports regression only.")
    if config.metric not in REGRESSION_METRICS:
        raise ValueError(f"Metric {config.metric!r} is not a regression metric.")
    required = {
        config.id_column,
        config.target_column,
        *config.text_columns,
        *config.categorical_columns,
        *config.numeric_columns,
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}.")
    if train[config.id_column].isna().any() or not train[config.id_column].is_unique:
        raise ValueError("Training IDs must be non-missing and unique.")
    target = pd.to_numeric(train[config.target_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Regression target must be finite and numeric.")
    if config.target_transform == "log1p" and np.any(target <= -1.0):
        raise ValueError("log1p target transform requires every target to be greater than -1.")


def _cache_signature(folded: pd.DataFrame, config: TrainingConfig) -> str:
    payload = asdict(config)
    # Backend parameters do not affect feature preprocessing.
    payload.pop("model_params", None)
    payload.pop("device", None)
    payload.pop("lgbm_estimators", None)
    payload.pop("lgbm_learning_rate", None)
    payload.pop("lgbm_num_leaves", None)
    payload.pop("lgbm_early_stopping_rounds", None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    columns = list(dict.fromkeys([
        config.id_column,
        config.target_column,
        config.fold_column,
        *config.text_columns,
        *config.categorical_columns,
        *config.numeric_columns,
    ]))
    hashes = pd.util.hash_pandas_object(
        folded.loc[:, columns], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    digest.update(hashes.tobytes())
    digest.update(str(folded.shape).encode("ascii"))
    return digest.hexdigest()


def _save_matrix(matrix: Any, path_without_suffix: Path) -> tuple[Path, bool]:
    if sparse.issparse(matrix):
        path = path_without_suffix.with_suffix(".npz")
        sparse.save_npz(path, sparse.csr_matrix(matrix), compressed=True)
        return path, True
    path = path_without_suffix.with_suffix(".npy")
    np.save(path, np.asarray(matrix), allow_pickle=False)
    return path, False


def _load_matrix(path: Path, is_sparse: bool) -> Any:
    return sparse.load_npz(path) if is_sparse else np.load(path, mmap_mode="r")


def _entry_from_record(root: Path, record: Mapping[str, Any]) -> _CachedFold:
    entry = _CachedFold(
        fold_id=int(record["fold_id"]),
        train_matrix=root / str(record["train_matrix"]),
        valid_matrix=root / str(record["valid_matrix"]),
        train_target=root / str(record["train_target"]),
        valid_target_fit=root / str(record["valid_target_fit"]),
        valid_target_raw=root / str(record["valid_target_raw"]),
        is_sparse=bool(record["is_sparse"]),
        train_rows=int(record["train_rows"]),
        valid_rows=int(record["valid_rows"]),
        feature_count=int(record["feature_count"]),
    )
    for path in (
        entry.train_matrix,
        entry.valid_matrix,
        entry.train_target,
        entry.valid_target_fit,
        entry.valid_target_raw,
    ):
        if not path.is_file():
            raise RuntimeError(f"Fold cache is incomplete; missing {path}.")
    return entry


def _prepare_fold_cache(
    train: pd.DataFrame,
    folds: pd.DataFrame | None,
    config: TrainingConfig,
    cache_parent: Path,
) -> tuple[list[_CachedFold], dict[str, Any]]:
    folded = attach_folds(train, folds, config)
    signature = _cache_signature(folded, config)
    root = cache_parent / signature[:20]
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature or manifest.get("status") != "complete":
            raise RuntimeError(f"Fold cache manifest is invalid: {manifest_path}.")
        entries = [_entry_from_record(root, item) for item in manifest["folds"]]
        manifest["reused"] = True
        return entries, manifest
    if root.exists():
        raise RuntimeError(f"Incomplete fold cache exists at {root}; remove that exact directory.")

    root.mkdir(parents=True, exist_ok=False)
    feature_frame = build_feature_frame(folded, config)
    target_raw = pd.to_numeric(folded[config.target_column], errors="raise").to_numpy(dtype=float)
    fold_values = folded[config.fold_column].to_numpy(dtype=int)
    records: list[dict[str, Any]] = []
    try:
        for fold_id, train_positions, valid_positions in iter_fold_indices(
            fold_values, temporal=config.temporal
        ):
            preprocessor = build_preprocessor(config)
            x_train = preprocessor.fit_transform(feature_frame.iloc[train_positions])
            x_valid = preprocessor.transform(feature_frame.iloc[valid_positions])
            if x_train.shape[1] == 0 or x_train.shape[1] != x_valid.shape[1]:
                raise RuntimeError(f"Fold {fold_id} preprocessing produced an invalid feature matrix.")
            train_matrix, train_sparse = _save_matrix(x_train, root / f"fold_{fold_id}_train")
            valid_matrix, valid_sparse = _save_matrix(x_valid, root / f"fold_{fold_id}_valid")
            if train_sparse != valid_sparse:
                raise RuntimeError("Train/validation matrix storage formats differ unexpectedly.")
            y_train_fit = _transform_target(target_raw[train_positions], config)
            y_valid_raw = target_raw[valid_positions]
            y_valid_fit = _transform_target(y_valid_raw, config)
            train_target = root / f"fold_{fold_id}_train_target.npy"
            valid_target_fit = root / f"fold_{fold_id}_valid_target_fit.npy"
            valid_target_raw = root / f"fold_{fold_id}_valid_target_raw.npy"
            np.save(train_target, y_train_fit, allow_pickle=False)
            np.save(valid_target_fit, y_valid_fit, allow_pickle=False)
            np.save(valid_target_raw, y_valid_raw, allow_pickle=False)
            records.append({
                "fold_id": int(fold_id),
                "train_matrix": train_matrix.name,
                "valid_matrix": valid_matrix.name,
                "train_target": train_target.name,
                "valid_target_fit": valid_target_fit.name,
                "valid_target_raw": valid_target_raw.name,
                "is_sparse": train_sparse,
                "train_rows": int(len(train_positions)),
                "valid_rows": int(len(valid_positions)),
                "feature_count": int(x_train.shape[1]),
            })
            del preprocessor, x_train, x_valid
            gc.collect()
        manifest = {
            "status": "complete",
            "signature": signature,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(folded)),
            "evaluated_rows": int(np.sum(fold_values >= 0)),
            "folds": records,
            "reused": False,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return [_entry_from_record(root, item) for item in records], manifest


def _resolve_device(engine: EngineName, requested: DeviceName, sparse_input: bool) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "CPU explicitly requested."
    available, reason = gpu_backend_preflight(engine, sparse_input=sparse_input)
    if requested == "gpu" and not available:
        raise RuntimeError(f"GPU tuning was explicitly requested for {engine}: {reason}")
    return ("gpu", reason) if available else ("cpu", f"Auto fallback to CPU: {reason}")


def _suggest_search_parameters(trial: Any, engine: EngineName, device: str) -> dict[str, Any]:
    search: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 64.0, log=True),
    }
    if engine == "xgboost":
        search.update({
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        })
    else:
        search["random_strength"] = trial.suggest_float(
            "random_strength", 1e-3, 10.0, log=True
        )
        if device == "cpu":
            # CatBoost calls column subsampling rsm. GPU regression rejects it.
            search["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )
    return search


def _to_backend_parameters(
    engine: EngineName,
    search: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    if engine == "xgboost":
        return {
            "max_depth": int(search["max_depth"]),
            "colsample_bytree": float(search["colsample_bytree"]),
            "subsample": float(search["subsample"]),
            "reg_alpha": float(search["reg_alpha"]),
            "reg_lambda": float(search["reg_lambda"]),
            "min_child_weight": float(search["min_child_weight"]),
            "gamma": float(search["gamma"]),
            "max_bin": int(search["max_bin"]),
        }
    params: dict[str, Any] = {
        "depth": int(search["max_depth"]),
        "grow_policy": "Depthwise",
        "bootstrap_type": "Bernoulli",
        "subsample": float(search["subsample"]),
        "l2_leaf_reg": float(search["reg_lambda"]),
        "min_data_in_leaf": max(1, int(round(float(search["min_child_weight"])))),
        "random_strength": float(search["random_strength"]),
        "loss_function": "RMSE",
    }
    if device == "cpu":
        params["rsm"] = float(search["colsample_bytree"])
    return params


def _make_objective(
    engine: EngineName,
    device: str,
    fold_cache: Sequence[_CachedFold],
    training: TrainingConfig,
    tuning: TuningConfig,
) -> Any:
    optuna = _import_optuna()
    greater_is_better = METRIC_REGISTRY[training.metric].greater_is_better

    def objective(trial: Any) -> float:
        started = time.perf_counter()
        search = _suggest_search_parameters(trial, engine, device)
        model_params = _to_backend_parameters(engine, search, device)
        all_target: list[np.ndarray] = []
        all_prediction: list[np.ndarray] = []
        fold_scores: list[float] = []
        best_iterations: list[int | None] = []
        for step, fold in enumerate(fold_cache):
            x_train = _load_matrix(fold.train_matrix, fold.is_sparse)
            x_valid = _load_matrix(fold.valid_matrix, fold.is_sparse)
            y_train = np.load(fold.train_target, mmap_mode="r")
            y_valid_fit = np.load(fold.valid_target_fit, mmap_mode="r")
            y_valid_raw = np.load(fold.valid_target_raw, mmap_mode="r")
            model = create_model(ModelConfig(
                engine=engine,
                task="regression",
                random_state=training.random_state + trial.number * 1009 + fold.fold_id,
                n_jobs=tuning.backend_n_jobs,
                device=device,
                early_stopping_rounds=tuning.early_stopping_rounds,
                eval_metric="RMSE" if engine == "catboost" else "mae",
                max_iterations=tuning.max_iterations,
                learning_rate=float(search["learning_rate"]),
                params=model_params,
            ))
            model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid_fit)
            prediction = _restore_predictions(model.predict(x_valid), training)
            raw_target = np.asarray(y_valid_raw, dtype=float)
            fold_score = evaluate_metric(training.metric, raw_target, prediction)
            fold_scores.append(float(fold_score))
            best_iterations.append(model.best_iteration_)
            all_target.append(raw_target.copy())
            all_prediction.append(np.asarray(prediction, dtype=float))
            cumulative_score = evaluate_metric(
                training.metric,
                np.concatenate(all_target),
                np.concatenate(all_prediction),
            )
            trial.report(float(cumulative_score), step=step)
            del model, x_train, x_valid, y_train, y_valid_fit, y_valid_raw, prediction
            gc.collect()
            if step + 1 >= tuning.pruner_warmup_folds and trial.should_prune():
                trial.set_user_attr("fold_scores", fold_scores)
                trial.set_user_attr("best_iterations", best_iterations)
                trial.set_user_attr("resolved_model_params", model_params)
                raise optuna.TrialPruned(f"Pruned after fold {fold.fold_id}.")
        score = evaluate_metric(
            training.metric,
            np.concatenate(all_target),
            np.concatenate(all_prediction),
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("resolved_model_params", model_params)
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - started)
        trial.set_user_attr("greater_is_better", greater_is_better)
        return float(score)

    return objective


def _trial_rows(engine: str, study: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        duration = trial.duration.total_seconds() if trial.duration is not None else None
        row: dict[str, Any] = {
            "engine": engine,
            "trial": int(trial.number),
            "state": trial.state.name,
            "value": float(trial.value) if trial.value is not None else None,
            "duration_seconds": duration,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        fold_scores = trial.user_attrs.get("fold_scores", [])
        for index, score in enumerate(fold_scores):
            row[f"fold_{index}_score"] = score
        rows.append(row)
    return rows


def tune_hyperparameters(
    train: pd.DataFrame,
    training_config: TrainingConfig,
    tuning_config: TuningConfig | None = None,
    *,
    folds: pd.DataFrame | None = None,
    cache_directory: str | Path | None = None,
) -> TuningResult:
    """Run one independent Optuna study per requested tree backend."""

    optuna = _import_optuna()
    tuning = tuning_config or TuningConfig()
    # Validate the base feature/target contract using a supported tree backend.
    cache_training = replace(
        training_config,
        model="xgboost",
        device="cpu",
        model_params={},
    )
    validate_config(cache_training)
    _validate_inputs(train, cache_training)
    if tuning.pruner_warmup_folds > cache_training.n_splits:
        raise ValueError("pruner_warmup_folds cannot exceed n_splits.")
    if tuning.n_jobs_trials > 1 and tuning.device in {"auto", "gpu"}:
        raise ValueError("Parallel trials are unsafe when GPU use is possible; use n_jobs_trials=1.")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if cache_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="amazon_ml_tune_")
        cache_parent = Path(temporary.name)
    else:
        cache_parent = Path(cache_directory)
        cache_parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        fold_cache, cache_report = _prepare_fold_cache(
            train, folds, cache_training, cache_parent
        )
        sparse_input = any(item.is_sparse for item in fold_cache)
        studies: dict[str, Any] = {}
        engine_reports: dict[str, Any] = {}
        best_parameters: dict[str, dict[str, Any]] = {}
        trial_records: list[dict[str, Any]] = []
        direction = "maximize" if METRIC_REGISTRY[cache_training.metric].greater_is_better else "minimize"
        for engine in tuning.engines:
            device, device_reason = _resolve_device(engine, tuning.device, sparse_input)
            contract_payload = {
                "tuning_version": TUNING_VERSION,
                "engine": engine,
                "device": device,
                "max_iterations": tuning.max_iterations,
                "early_stopping_rounds": tuning.early_stopping_rounds,
                "sampler_seed": tuning.sampler_seed,
                "startup_trials": tuning.startup_trials,
                "pruner_warmup_folds": tuning.pruner_warmup_folds,
                "backend_n_jobs": tuning.backend_n_jobs,
                "target_transform": cache_training.target_transform,
                "prediction_floor": cache_training.prediction_floor,
                "prediction_ceiling": cache_training.prediction_ceiling,
            }
            study_contract = hashlib.sha256(
                json.dumps(contract_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            sampler = optuna.samplers.TPESampler(
                seed=tuning.sampler_seed,
                n_startup_trials=tuning.startup_trials,
                multivariate=True,
            )
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=tuning.startup_trials,
                n_warmup_steps=max(0, tuning.pruner_warmup_folds - 1),
            )
            study = optuna.create_study(
                study_name=f"{tuning.study_name}_{engine}",
                storage=tuning.storage,
                sampler=sampler,
                pruner=pruner,
                direction=direction,
                load_if_exists=tuning.load_if_exists,
            )
            existing_signature = study.user_attrs.get("fold_cache_signature")
            if study.trials and existing_signature != cache_report["signature"]:
                raise ValueError(
                    f"Refusing to resume study {study.study_name!r} on different data/folds. "
                    "Use a new --study-name or matching inputs."
                )
            existing_metric = study.user_attrs.get("metric")
            if study.trials and existing_metric != cache_training.metric:
                raise ValueError(
                    f"Refusing to resume study {study.study_name!r} with metric "
                    f"{cache_training.metric!r}; existing metric is {existing_metric!r}."
                )
            existing_contract = study.user_attrs.get("study_contract")
            if study.trials and existing_contract != study_contract:
                raise ValueError(
                    f"Refusing to resume study {study.study_name!r} with changed search/runtime "
                    "settings. Use a new --study-name."
                )
            study.set_user_attr("tuning_version", TUNING_VERSION)
            study.set_user_attr("fold_cache_signature", cache_report["signature"])
            study.set_user_attr("metric", cache_training.metric)
            study.set_user_attr("device", device)
            study.set_user_attr("study_contract", study_contract)
            existing_trials = len(study.trials)
            study.optimize(
                _make_objective(engine, device, fold_cache, cache_training, tuning),
                n_trials=tuning.n_trials,
                timeout=tuning.timeout_seconds,
                n_jobs=tuning.n_jobs_trials,
                gc_after_trial=True,
                show_progress_bar=tuning.show_progress_bar,
            )
            completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
            if not completed:
                raise RuntimeError(f"Study {engine} completed no successful trials.")
            best = study.best_trial
            search_params = dict(best.params)
            model_params = _to_backend_parameters(engine, search_params, device)
            artifact = {
                "engine": engine,
                "metric": cache_training.metric,
                "best_value": float(best.value),
                "best_trial": int(best.number),
                "device": device,
                "learning_rate": float(search_params["learning_rate"]),
                "max_iterations": tuning.max_iterations,
                "early_stopping_rounds": tuning.early_stopping_rounds,
                "search_params": _json_ready(search_params),
                "model_params": _json_ready(model_params),
                "fold_scores": _json_ready(best.user_attrs.get("fold_scores", [])),
                "best_iterations": _json_ready(best.user_attrs.get("best_iterations", [])),
            }
            best_parameters[engine] = artifact
            states: dict[str, int] = {}
            for trial in study.trials:
                states[trial.state.name] = states.get(trial.state.name, 0) + 1
            engine_reports[engine] = {
                **artifact,
                "study_name": study.study_name,
                "device_reason": device_reason,
                "trial_states": states,
                "trials_before_this_run": existing_trials,
                "catboost_gpu_limitations": (
                    ["reg_alpha has no CatBoost equivalent", "rsm/colsample is invalid for GPU regression"]
                    if engine == "catboost" and device == "gpu"
                    else []
                ),
                "training_overrides": {
                    "model": engine,
                    "device": device,
                    "lgbm_estimators": tuning.max_iterations,
                    "lgbm_learning_rate": float(search_params["learning_rate"]),
                    "lgbm_early_stopping_rounds": tuning.early_stopping_rounds,
                    "model_params": _json_ready(model_params),
                },
            }
            studies[engine] = study
            trial_records.extend(_trial_rows(engine, study))

        report = {
            "tuning_version": TUNING_VERSION,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metric": cache_training.metric,
            "direction": direction,
            "training_config": _json_ready(asdict(cache_training)),
            "tuning_config": _json_ready(asdict(tuning)),
            "cache": _json_ready(cache_report),
            "engines": engine_reports,
            "elapsed_seconds": float(time.perf_counter() - started),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "optuna": getattr(optuna, "__version__", None),
            },
        }
        return TuningResult(
            report=_json_ready(report),
            trials=pd.DataFrame(trial_records),
            best_parameters=best_parameters,
            studies=studies,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        path,
    )


def save_tuning_result(
    result: TuningResult,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_directory)
    outputs: dict[str, Path] = {
        "report": destination / "tuning_report.json",
        "trials": destination / "tuning_trials.csv",
    }
    for engine in result.best_parameters:
        outputs[f"best__{engine}"] = destination / f"best_params_{engine}.json"
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite tuning artifacts: {[str(p) for p in existing]}.")
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(result.report, outputs["report"])
    csv_temporary = outputs["trials"].with_name(outputs["trials"].name + ".tmp")
    try:
        result.trials.to_csv(csv_temporary, index=False)
        os.replace(csv_temporary, outputs["trials"])
    finally:
        csv_temporary.unlink(missing_ok=True)
    for engine, payload in result.best_parameters.items():
        _atomic_json(payload, outputs[f"best__{engine}"])
    return outputs


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fold-safe Optuna tuning for GPU XGBoost and CatBoost price regressors."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--text-columns", nargs="*", default=[])
    parser.add_argument("--categorical-columns", nargs="*", default=[])
    parser.add_argument("--numeric-columns", nargs="*", default=[])
    parser.add_argument("--engineered-features", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--image-embeddings", type=Path)
    parser.add_argument("--metric", choices=sorted(REGRESSION_METRICS), default="smape")
    parser.add_argument("--target-transform", choices=("none", "log1p"), default="log1p")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=120_000)
    parser.add_argument("--char-max-features", type=int, default=120_000)
    parser.add_argument("--no-word-tfidf", action="store_true")
    parser.add_argument("--no-char-tfidf", action="store_true")
    parser.add_argument("--prediction-floor", type=float, default=0.0)
    parser.add_argument("--prediction-ceiling", type=float)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--engines", nargs="+", choices=("xgboost", "catboost"), default=["xgboost", "catboost"])
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--trials", type=int, default=50, help="New trials per engine.")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-iterations", type=int, default=3_000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--pruner-warmup-folds", type=int, default=1)
    parser.add_argument("--study-name", default="amazon_ml_price")
    parser.add_argument("--storage", help="Optuna storage URL, e.g. sqlite:///reports/tuning.db")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    expected_outputs = [
        args.output_dir / "tuning_report.json",
        args.output_dir / "tuning_trials.csv",
        *(args.output_dir / f"best_params_{engine}.json" for engine in args.engines),
    ]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Refusing to start a long tuning run because output artifacts exist: "
            f"{[str(path) for path in existing_outputs]}. Pass --overwrite to resume/update."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage
    if storage is None:
        database = (args.output_dir / "optuna_studies.db").resolve().as_posix()
        storage = f"sqlite:///{database}"
    train = _load_table(args.train)
    folds = _load_table(args.folds) if args.folds else None
    config = TrainingConfig(
        target_column=args.target_column,
        id_column=args.id_column,
        text_columns=tuple(args.text_columns),
        categorical_columns=tuple(args.categorical_columns),
        numeric_columns=tuple(args.numeric_columns),
        task="regression",
        metric=args.metric,
        model="xgboost",
        device="cpu",
        target_transform=args.target_transform,
        fold_column=args.fold_column,
        n_splits=args.n_splits,
        random_state=args.random_state,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        use_word_tfidf=not args.no_word_tfidf,
        use_char_tfidf=not args.no_char_tfidf,
        prediction_floor=args.prediction_floor,
        prediction_ceiling=args.prediction_ceiling,
        temporal=args.temporal,
    )
    auxiliary_sources: list[dict[str, Any]] = []
    numeric_columns = list(config.numeric_columns)
    for label, path, prefix in (
        ("engineered_features", args.engineered_features, "eng"),
        ("embeddings", args.embeddings, "emb"),
        ("image_embeddings", args.image_embeddings, "img"),
    ):
        if path is None:
            continue
        train, added = attach_tuning_feature_table(
            train,
            _load_table(path),
            config,
            prefix=prefix,
            expected_folds=folds,
        )
        numeric_columns.extend(added)
        config = replace(config, numeric_columns=tuple(numeric_columns))
        auxiliary_sources.append({
            "name": label,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "features": len(added),
        })
    tuning = TuningConfig(
        engines=tuple(args.engines),
        n_trials=args.trials,
        timeout_seconds=args.timeout_seconds,
        device=args.device,
        max_iterations=args.max_iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        sampler_seed=args.random_state,
        startup_trials=args.startup_trials,
        pruner_warmup_folds=args.pruner_warmup_folds,
        study_name=args.study_name,
        storage=storage,
        load_if_exists=not args.no_resume,
        show_progress_bar=args.show_progress,
    )
    result = tune_hyperparameters(
        train,
        config,
        tuning,
        folds=folds,
        cache_directory=args.cache_dir,
    )
    result.report["sources"] = {
        "train": {"path": str(args.train.resolve()), "sha256": _sha256_file(args.train)},
        "folds": (
            {"path": str(args.folds.resolve()), "sha256": _sha256_file(args.folds)}
            if args.folds else None
        ),
        "auxiliary": auxiliary_sources,
    }
    try:
        outputs = save_tuning_result(result, args.output_dir, overwrite=args.overwrite)
        for engine, best in result.best_parameters.items():
            print(
                f"{engine}: best {args.metric}={best['best_value']:.8f} "
                f"trial={best['best_trial']} device={best['device']}"
            )
        for name, path in outputs.items():
            print(f"{name}: {path}")
    finally:
        result.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
