"""Unified, defensive model interfaces for competition training.

Supported estimator families:

* Ridge regression / logistic linear classification
* LightGBM
* CatBoost
* XGBoost

The public :class:`UnifiedModel` normalizes validation, class encoding,
probability ordering, early stopping, best-iteration reporting, and prediction
shape checks. :class:`FoldModelStore` adds atomic, SHA-256-authenticated fold
serialization with provenance validation before joblib deserialization.

This module expects an already numeric two-dimensional feature matrix. Text,
categorical, image, and missing-value preprocessing remain the responsibility
of the fold-local pipeline in ``train.py``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder


MODELS_VERSION = "1.0"
FOLD_CHECKPOINT_VERSION = 1
INFERENCE_BUNDLE_VERSION = 1
EngineName = Literal["ridge", "lightgbm", "catboost", "xgboost"]
TaskName = Literal["regression", "classification"]
DeviceName = Literal["cpu", "gpu"]
TREE_ENGINES = frozenset({"lightgbm", "catboost", "xgboost"})


def smooth_log_smape_objective(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth pseudo-gradient and positive curvature proxy for log-space SMAPE.

    This is a training surrogate, not an exact derivative of the official SMAPE.
    The returned Hessian-like values are a positive curvature heuristic used by
    second-order tree learners; they are not the mathematical second derivative
    of the returned gradient. Validate this objective against the official metric.
    """
    delta = y_pred - y_true
    th = np.tanh(np.abs(delta) / 2.0)
    sech2 = np.maximum(1e-4, 1.0 - th ** 2)
    soft_sign = np.tanh(delta / 0.05)
    grad = soft_sign * sech2
    hess = np.maximum(1e-3, sech2 + 0.01)
    return grad, hess


def smooth_raw_smape_objective(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SMAPE-inspired pseudo-gradient and positive curvature heuristic on raw scale."""
    delta = y_pred - y_true
    denom = np.maximum(1e-4, np.abs(y_pred) + np.abs(y_true))
    soft_abs = np.sqrt(delta ** 2 + 1e-4)
    grad = 2.0 * (delta / soft_abs) / denom
    hess = np.maximum(1e-3, 2.0 / (denom * soft_abs))
    return grad, hess


def smooth_log_mape_objective(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """MAPE-inspired pseudo-gradient and positive curvature heuristic in log space."""
    delta = np.clip(y_pred - y_true, -8.0, 8.0)
    exp_delta = np.exp(delta)
    soft_sign = np.tanh(delta / 0.05)
    grad = soft_sign * exp_delta
    hess = np.maximum(1e-3, exp_delta + 0.01)
    return grad, hess


def smooth_raw_mape_objective(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """MAPE-inspired pseudo-gradient and positive curvature heuristic on raw scale."""
    delta = y_pred - y_true
    denom = np.maximum(1e-4, np.abs(y_true))
    soft_sign = np.tanh(delta / 0.05)
    grad = soft_sign / denom
    sech2 = np.maximum(1e-4, 1.0 - soft_sign ** 2)
    hess = np.maximum(1e-3, sech2 / (0.05 * denom))
    return grad, hess


@dataclass(frozen=True)
class ModelConfig:
    engine: EngineName = "ridge"
    task: TaskName = "regression"
    objective: str | None = None
    random_state: int = 42
    n_jobs: int = -1
    device: DeviceName = "cpu"
    early_stopping_rounds: int = 100
    eval_metric: str | None = None
    ridge_alpha: float = 4.0
    max_iterations: int = 2_000
    learning_rate: float = 0.03
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in {"ridge", "lightgbm", "catboost", "xgboost"}:
            raise ValueError("Unsupported engine.")
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification.")
        if self.objective is not None and self.objective not in {
            "default", "smape", "mape", "huber", "l1", "l2", "regression", "mae"
        }:
            raise ValueError(f"Unsupported objective: {self.objective!r}")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu.")
        if self.engine == "ridge" and self.device != "cpu":
            raise ValueError("Ridge/logistic models support only device='cpu'.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer.")
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero integer.")
        if self.early_stopping_rounds < 0:
            raise ValueError("early_stopping_rounds must be >= 0.")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(self.params, Mapping):
            raise ValueError("params must be a mapping.")
        reserved = {
            "C",
            "alpha",
            "random_state",
            "random_seed",
            "seed",
            "n_jobs",
            "thread_count",
            "task_type",
            "device",
            "device_type",
            "early_stopping_rounds",
            "n_estimators",
            "iterations",
            "max_iter",
            "learning_rate",
            "objective",
            "eval_metric",
            "metric",
            "num_class",
        }
        overlap = reserved & set(self.params)
        if overlap:
            raise ValueError(
                f"Model params cannot override controlled settings: {sorted(overlap)}"
            )


@dataclass
class InferenceFoldBundle:
    """Authenticated fold model plus everything required for raw inference."""

    fold_id: int
    run_signature: str
    feature_count: int
    valid_positions: np.ndarray
    preprocessor: Any
    model: Any
    training_config: dict[str, Any]
    fold_report: dict[str, Any]
    class_labels: list[Any]
    valid_prediction: np.ndarray | None = None
    test_prediction: np.ndarray | None = None
    valid_probabilities: np.ndarray | None = None
    test_probabilities: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _dependency_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def backend_status() -> dict[str, dict[str, Any]]:
    """Return installed/available versions without importing heavy libraries."""

    packages = {
        "ridge": "scikit-learn",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "xgboost": "xgboost",
    }
    return {
        engine: {
            "available": (version := _dependency_version(package)) is not None,
            "package": package,
            "version": version,
        }
        for engine, package in packages.items()
    }


def _import_backend(engine: str) -> Any:
    module_name = {"lightgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost"}[engine]
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"engine={engine!r} requires the optional package {module_name!r}. "
            "Install requirements.txt before using this backend."
        ) from exc


@functools.lru_cache(maxsize=None)
def gpu_backend_preflight(engine: str, sparse_input: bool = False) -> tuple[bool, str]:
    """Run a tiny real GPU fit, optionally with CSR input, before long training.

    Backend/package versions are not reliable capability indicators. This probe
    checks the installed binary, driver, and the relevant dense/sparse path.
    Results are cached for the life of the Python process.
    """

    if engine not in TREE_ENGINES:
        return False, f"engine={engine!r} has no GPU tree learner."
    rng = np.random.default_rng(917)
    dense = rng.normal(size=(128, 8)).astype(np.float32)
    features: Any = sparse.csr_matrix(dense) if sparse_input else dense
    target = 2.0 * dense[:, 0] - dense[:, 1]
    try:
        module = _import_backend(engine)
        if engine == "lightgbm":
            estimator = module.LGBMRegressor(
                n_estimators=2,
                max_bin=63,
                device_type="gpu",
                verbosity=-1,
            )
        elif engine == "catboost":
            estimator = module.CatBoostRegressor(
                iterations=2,
                depth=3,
                task_type="GPU",
                verbose=False,
                allow_writing_files=False,
            )
        else:
            estimator = module.XGBRegressor(
                n_estimators=2,
                max_depth=3,
                tree_method="hist",
                device="cuda",
                verbosity=0,
            )
        estimator.fit(features, target)
    except Exception as exc:
        matrix_kind = "CSR sparse" if sparse_input else "dense"
        return False, f"{engine} {matrix_kind} GPU probe failed: {type(exc).__name__}: {exc}"
    matrix_kind = "CSR sparse" if sparse_input else "dense"
    return True, f"{engine} completed a real {matrix_kind} GPU training probe."


def _dense_has_invalid(matrix: np.ndarray, *, allow_nan: bool) -> bool:
    """Check large dense arrays in bounded-memory blocks."""

    flat = matrix.reshape(-1)
    block_size = 1_000_000
    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        if allow_nan:
            if np.isinf(block).any():
                return True
        elif not np.isfinite(block).all():
            return True
    return False


def _validate_matrix(
    values: Any,
    *,
    label: str,
    expected_features: int | None = None,
    allow_nan: bool = False,
) -> sparse.spmatrix | np.ndarray:
    if sparse.issparse(values):
        matrix = values
        if matrix.ndim != 2:
            raise ValueError(f"{label} must be two-dimensional.")
        if matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(f"{label} must contain at least one row and feature.")
        if matrix.data.size:
            invalid = (
                np.isinf(matrix.data).any()
                if allow_nan
                else not np.isfinite(matrix.data).all()
            )
            if invalid:
                message = "infinity" if allow_nan else "NaN or infinity"
                raise ValueError(f"{label} contains {message}.")
    else:
        try:
            matrix = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a numeric matrix.") from exc
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(f"{label} must be a non-empty two-dimensional matrix.")
        if not np.issubdtype(matrix.dtype, np.number):
            try:
                matrix = matrix.astype(np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must contain numeric values.") from exc
        if _dense_has_invalid(matrix, allow_nan=allow_nan):
            message = "infinity" if allow_nan else "NaN or infinity"
            raise ValueError(f"{label} contains {message}.")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f"{label} has {matrix.shape[1]} features; expected {expected_features}."
        )
    return matrix


def _validate_target(values: Any, task: str, expected_rows: int, label: str) -> np.ndarray:
    target = np.asarray(values)
    if target.ndim != 1 or len(target) != expected_rows:
        raise ValueError(f"{label} must be one-dimensional with {expected_rows} rows.")
    if task == "regression":
        try:
            target = target.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric for regression.") from exc
        if not np.isfinite(target).all():
            raise ValueError(f"{label} contains NaN or infinity.")
    else:
        for value in target:
            if value is None:
                raise ValueError(f"{label} contains missing labels.")
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError(f"{label} contains missing/non-finite labels.")
    return target


def _validate_weights(values: Any, expected_rows: int, label: str) -> np.ndarray | None:
    if values is None:
        return None
    try:
        weights = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if weights.ndim != 1 or len(weights) != expected_rows:
        raise ValueError(f"{label} must be one-dimensional with {expected_rows} rows.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError(f"{label} must be finite, non-negative, and contain a positive value.")
    return weights


def _default_eval_metric(config: ModelConfig, class_count: int | None) -> str:
    if config.eval_metric:
        return config.eval_metric
    if config.task == "regression":
        return {"lightgbm": "l1", "catboost": "MAE", "xgboost": "mae"}.get(
            config.engine, "mae"
        )
    multiclass = bool(class_count and class_count > 2)
    return {
        "lightgbm": "multi_logloss" if multiclass else "binary_logloss",
        "catboost": "MultiClass" if multiclass else "Logloss",
        "xgboost": "mlogloss" if multiclass else "logloss",
    }.get(config.engine, "logloss")


class UnifiedModel:
    """One fitted-model contract across Ridge, LightGBM, CatBoost and XGBoost."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.estimator_: Any | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.feature_count_: int | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.best_iteration_: int | None = None
        self.best_score_: Any = None
        self.fitted_at_unix_: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.estimator_ is not None and self.feature_count_ is not None

    @property
    def classes_(self) -> np.ndarray:
        if self.config.task != "classification" or self.label_encoder_ is None:
            raise AttributeError("classes_ exists only for fitted classification models.")
        return self.label_encoder_.classes_

    @property
    def signature(self) -> str:
        payload = {
            "models_version": MODELS_VERSION,
            "config": asdict(self.config),
            "backend_version": backend_status()[self.config.engine]["version"],
            "feature_count": self.feature_count_,
            "feature_names": self.feature_names_,
            "classes": self.classes_.tolist() if self.config.task == "classification" and self.is_fitted else None,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _build_estimator(self, class_count: int | None) -> Any:
        config = self.config
        user = dict(config.params)
        if config.engine == "ridge":
            if config.task == "regression":
                defaults = {"alpha": config.ridge_alpha, "solver": "lsqr"}
                defaults.update(user)
                return Ridge(**defaults)
            defaults = {
                "C": 1.0 / config.ridge_alpha,
                "solver": "saga",
                "max_iter": config.max_iterations,
                "random_state": config.random_state,
            }
            defaults.update(user)
            return LogisticRegression(**defaults)

        module = _import_backend(config.engine)
        metric = _default_eval_metric(config, class_count)
        if config.engine == "lightgbm":
            defaults: dict[str, Any] = {
                "n_estimators": config.max_iterations,
                "learning_rate": config.learning_rate,
                "random_state": config.random_state,
                "n_jobs": config.n_jobs,
                "verbosity": -1,
                "metric": "None",
                "device_type": "gpu" if config.device == "gpu" else "cpu",
            }
            if config.task == "regression":
                if config.objective == "smape":
                    defaults["objective"] = smooth_log_smape_objective
                elif config.objective == "mape":
                    defaults["objective"] = smooth_log_mape_objective
                elif config.objective == "huber":
                    defaults["objective"] = "huber"
                    defaults["huber_alpha"] = 0.8
                elif config.objective in {"l1", "mae"}:
                    defaults["objective"] = "regression_l1"
                else:
                    defaults["objective"] = "regression"
                model_class = module.LGBMRegressor
            else:
                defaults["objective"] = "multiclass" if class_count and class_count > 2 else "binary"
                if class_count and class_count > 2:
                    defaults["num_class"] = class_count
                model_class = module.LGBMClassifier
            defaults.update(user)
            return model_class(**defaults)

        if config.engine == "catboost":
            defaults = {
                "iterations": config.max_iterations,
                "learning_rate": config.learning_rate,
                "random_seed": config.random_state,
                "thread_count": config.n_jobs,
                "task_type": "GPU" if config.device == "gpu" else "CPU",
                "verbose": False,
                "allow_writing_files": False,
                "eval_metric": metric,
            }
            if config.task == "regression" and config.objective:
                if config.objective == "huber":
                    defaults["loss_function"] = "Huber:delta=1.0"
                elif config.objective in {"l1", "mae"}:
                    defaults["loss_function"] = "MAE"
                elif config.objective == "mape":
                    defaults["loss_function"] = "MAPE"
            model_class = (
                module.CatBoostRegressor if config.task == "regression" else module.CatBoostClassifier
            )
            defaults.update(user)
            return model_class(**defaults)

        defaults = {
            "n_estimators": config.max_iterations,
            "learning_rate": config.learning_rate,
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
            "tree_method": "hist",
            "device": "cuda" if config.device == "gpu" else "cpu",
            "eval_metric": metric,
        }
        if config.early_stopping_rounds:
            defaults["early_stopping_rounds"] = config.early_stopping_rounds
        if config.task == "regression":
            if config.objective == "smape":
                defaults["objective"] = smooth_log_smape_objective
            elif config.objective == "mape":
                defaults["objective"] = smooth_log_mape_objective
            elif config.objective == "huber":
                defaults["objective"] = "reg:pseudohubererror"
            elif config.objective in {"l1", "mae"}:
                defaults["objective"] = "reg:absoluteerror"
            else:
                defaults["objective"] = "reg:squarederror"
            model_class = module.XGBRegressor
        else:
            multiclass = bool(class_count and class_count > 2)
            defaults["objective"] = "multi:softprob" if multiclass else "binary:logistic"
            if multiclass:
                defaults["num_class"] = class_count
            model_class = module.XGBClassifier
        defaults.update(user)
        return model_class(**defaults)

    def fit(
        self,
        x_train: Any,
        y_train: Any,
        *,
        x_valid: Any | None = None,
        y_valid: Any | None = None,
        sample_weight: Any | None = None,
        valid_sample_weight: Any | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "UnifiedModel":
        allow_nan = self.config.engine in TREE_ENGINES
        train_matrix = _validate_matrix(
            x_train,
            label="x_train",
            allow_nan=allow_nan,
        )
        train_target = _validate_target(
            y_train, self.config.task, train_matrix.shape[0], "y_train"
        )
        train_weight = _validate_weights(sample_weight, train_matrix.shape[0], "sample_weight")
        uses_early_stopping = (
            self.config.engine in TREE_ENGINES and self.config.early_stopping_rounds > 0
        )
        if (x_valid is None) != (y_valid is None):
            raise ValueError("x_valid and y_valid must be provided together.")
        if uses_early_stopping and x_valid is None:
            raise ValueError(
                f"{self.config.engine} early stopping requires x_valid and y_valid."
            )
        valid_matrix = None
        valid_target = None
        valid_weight = None
        if x_valid is not None:
            valid_matrix = _validate_matrix(
                x_valid,
                label="x_valid",
                expected_features=train_matrix.shape[1],
                allow_nan=allow_nan,
            )
            valid_target = _validate_target(
                y_valid, self.config.task, valid_matrix.shape[0], "y_valid"
            )
            valid_weight = _validate_weights(
                valid_sample_weight, valid_matrix.shape[0], "valid_sample_weight"
            )
        elif valid_sample_weight is not None:
            raise ValueError("valid_sample_weight requires validation data.")

        if feature_names is not None:
            names = tuple(str(name) for name in feature_names)
            if len(names) != train_matrix.shape[1] or len(set(names)) != len(names):
                raise ValueError("feature_names must be unique and match the feature count.")
            self.feature_names_ = names
        else:
            self.feature_names_ = None

        class_count: int | None = None
        if self.config.task == "classification":
            encoder = LabelEncoder().fit(train_target)
            if len(encoder.classes_) < 2:
                raise ValueError("Classification training data must contain at least two classes.")
            encoded_train = encoder.transform(train_target)
            if valid_target is not None:
                unknown = set(valid_target.tolist()) - set(encoder.classes_.tolist())
                if unknown:
                    raise ValueError(
                        f"Validation contains classes absent from training: {sorted(map(str, unknown))}"
                    )
                encoded_valid = encoder.transform(valid_target)
            else:
                encoded_valid = None
            self.label_encoder_ = encoder
            class_count = len(encoder.classes_)
            fit_train_target = encoded_train
            fit_valid_target = encoded_valid
        else:
            self.label_encoder_ = None
            fit_train_target = train_target
            fit_valid_target = valid_target

        if self.config.device == "gpu" and self.config.engine in TREE_ENGINES:
            available, reason = gpu_backend_preflight(
                self.config.engine,
                sparse_input=sparse.issparse(train_matrix),
            )
            if not available:
                raise RuntimeError(
                    "GPU preflight failed before full model training. " + reason
                )
        estimator = self._build_estimator(class_count)
        fit_kwargs: dict[str, Any] = {}
        if train_weight is not None:
            fit_kwargs["sample_weight"] = train_weight
        if self.config.engine == "lightgbm" and valid_matrix is not None:
            module = _import_backend("lightgbm")
            fit_parameters = inspect.signature(estimator.fit).parameters
            if "eval_X" in fit_parameters and "eval_y" in fit_parameters:
                fit_kwargs["eval_X"] = valid_matrix
                fit_kwargs["eval_y"] = fit_valid_target
            else:
                fit_kwargs["eval_set"] = [(valid_matrix, fit_valid_target)]
            fit_kwargs["eval_metric"] = _default_eval_metric(self.config, class_count)
            if valid_weight is not None:
                fit_kwargs["eval_sample_weight"] = [valid_weight]
            if uses_early_stopping:
                fit_kwargs["callbacks"] = [
                    module.early_stopping(
                        self.config.early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                    ),
                    module.log_evaluation(period=0),
                ]
        elif self.config.engine == "catboost" and valid_matrix is not None:
            fit_kwargs["eval_set"] = (valid_matrix, fit_valid_target)
            fit_kwargs["use_best_model"] = uses_early_stopping
            if uses_early_stopping:
                fit_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds
            if valid_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = [valid_weight]
        elif self.config.engine == "xgboost" and valid_matrix is not None:
            fit_kwargs["eval_set"] = [(valid_matrix, fit_valid_target)]
            fit_kwargs["verbose"] = False
            if valid_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = [valid_weight]

        estimator.fit(train_matrix, fit_train_target, **fit_kwargs)
        self.estimator_ = estimator
        self.feature_count_ = int(train_matrix.shape[1])
        self.best_iteration_ = self._extract_best_iteration()
        self.best_score_ = self._extract_best_score()
        self.fitted_at_unix_ = time.time()
        return self

    def _require_fitted(self) -> Any:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return self.estimator_

    def _extract_best_iteration(self) -> int | None:
        estimator = self.estimator_
        if estimator is None:
            return None
        value = getattr(estimator, "best_iteration_", None)
        if value is None:
            value = getattr(estimator, "best_iteration", None)
        if value is None and hasattr(estimator, "get_best_iteration"):
            value = estimator.get_best_iteration()
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        if numeric < 0:
            return None
        # LightGBM exposes a 1-based best_iteration_; CatBoost and XGBoost use
        # zero-based values. Public metadata is normalized to zero-based.
        if self.config.engine == "lightgbm":
            return numeric - 1 if numeric > 0 else None
        return numeric

    def _extract_best_score(self) -> Any:
        estimator = self.estimator_
        if estimator is None:
            return None
        value = getattr(estimator, "best_score_", None)
        if value is None and hasattr(estimator, "get_best_score"):
            value = estimator.get_best_score()
        return _json_ready(value)

    def _xgboost_predict_raw(self, matrix: Any) -> np.ndarray:
        """Use DMatrix inference to avoid CPU/GPU inplace-predict mismatch."""

        estimator = self._require_fitted()
        module = _import_backend("xgboost")
        dmatrix = module.DMatrix(matrix)
        predict_kwargs: dict[str, Any] = {}
        if self.best_iteration_ is not None:
            predict_kwargs["iteration_range"] = (0, self.best_iteration_ + 1)
        return np.asarray(estimator.get_booster().predict(dmatrix, **predict_kwargs))

    def predict(self, features: Any) -> np.ndarray:
        estimator = self._require_fitted()
        matrix = _validate_matrix(
            features,
            label="prediction features",
            expected_features=self.feature_count_,
            allow_nan=self.config.engine in TREE_ENGINES,
        )
        if self.config.engine == "xgboost":
            raw_backend = self._xgboost_predict_raw(matrix)
            if self.config.task == "classification":
                if raw_backend.ndim == 1:
                    raw = (raw_backend >= 0.5).astype(int)
                else:
                    raw = np.argmax(raw_backend, axis=1)
            else:
                raw = raw_backend.reshape(-1)
        else:
            raw = np.asarray(estimator.predict(matrix)).reshape(-1)
        if len(raw) != matrix.shape[0]:
            raise RuntimeError("Estimator returned an invalid prediction length.")
        if self.config.task == "regression":
            try:
                prediction = raw.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Estimator returned non-numeric regression predictions.") from exc
            if not np.isfinite(prediction).all():
                raise RuntimeError("Estimator returned NaN or infinite predictions.")
            return prediction
        assert self.label_encoder_ is not None
        try:
            encoded = np.rint(raw.astype(np.float64)).astype(int)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Classifier returned invalid encoded labels.") from exc
        if np.any(encoded < 0) or np.any(encoded >= len(self.label_encoder_.classes_)):
            raise RuntimeError("Classifier returned an out-of-range encoded label.")
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, features: Any) -> np.ndarray:
        if self.config.task != "classification":
            raise RuntimeError("predict_proba is available only for classification.")
        estimator = self._require_fitted()
        if not hasattr(estimator, "predict_proba"):
            raise RuntimeError("Underlying classifier does not expose predict_proba.")
        matrix = _validate_matrix(
            features,
            label="probability features",
            expected_features=self.feature_count_,
            allow_nan=self.config.engine in TREE_ENGINES,
        )
        if self.config.engine == "xgboost":
            probabilities = np.asarray(self._xgboost_predict_raw(matrix), dtype=np.float64)
        else:
            probabilities = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        if probabilities.shape != (matrix.shape[0], len(self.classes_)):
            raise RuntimeError(
                f"Classifier returned probability shape {probabilities.shape}; expected "
                f"({matrix.shape[0]}, {len(self.classes_)})."
            )
        row_sums = probabilities.sum(axis=1)
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < -1e-12)
            or np.any(row_sums <= 0)
        ):
            raise RuntimeError("Classifier returned invalid probabilities.")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def feature_importance(self) -> np.ndarray:
        estimator = self._require_fitted()
        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
        elif hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_, dtype=np.float64)
            importance = np.abs(coefficients).mean(axis=0) if coefficients.ndim == 2 else np.abs(coefficients)
        else:
            raise RuntimeError("Underlying estimator exposes no feature importance or coefficients.")
        importance = importance.reshape(-1)
        if len(importance) != self.feature_count_ or not np.isfinite(importance).all():
            raise RuntimeError("Estimator returned invalid feature importance values.")
        return importance

    def metadata(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "models_version": MODELS_VERSION,
            "engine": self.config.engine,
            "task": self.config.task,
            "backend_version": backend_status()[self.config.engine]["version"],
            "feature_count": self.feature_count_,
            "feature_names": list(self.feature_names_) if self.feature_names_ else None,
            "classes": self.classes_.tolist() if self.config.task == "classification" else None,
            "early_stopping_supported": self.config.engine in TREE_ENGINES,
            "early_stopping_rounds": (
                self.config.early_stopping_rounds if self.config.engine in TREE_ENGINES else 0
            ),
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "model_signature": self.signature,
            "fitted_at_unix": self.fitted_at_unix_,
        }


def create_model(config: ModelConfig) -> UnifiedModel:
    return UnifiedModel(config)


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


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".metadata.json")


def save_fold_model(
    model: UnifiedModel,
    path: str | Path,
    *,
    fold_id: int,
    run_signature: str,
    valid_positions: Sequence[int] | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically save one fitted fold model plus an authenticated sidecar."""

    model._require_fitted()
    if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
        raise ValueError("fold_id must be a non-negative integer.")
    if not isinstance(run_signature, str) or not run_signature.strip():
        raise ValueError("run_signature must be a non-empty string.")
    model_path = Path(path)
    metadata_path = _sidecar_path(model_path)
    if (model_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite fold artifact {model_path}.")
    positions = None
    if valid_positions is not None:
        positions = np.asarray(valid_positions, dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or len(set(positions.tolist())) != len(positions):
            raise ValueError("valid_positions must be unique non-negative integers.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": FOLD_CHECKPOINT_VERSION,
        "models_version": MODELS_VERSION,
        "fold_id": fold_id,
        "run_signature": run_signature,
        "model_signature": model.signature,
        "feature_count": model.feature_count_,
        "valid_positions": positions,
        "provenance": dict(provenance or {}),
        "model": model,
    }
    temporary = model_path.with_name(model_path.name + ".tmp")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "checkpoint_version": FOLD_CHECKPOINT_VERSION,
        "models_version": MODELS_VERSION,
        "fold_id": fold_id,
        "run_signature": run_signature,
        "model_signature": model.signature,
        "feature_count": model.feature_count_,
        "valid_positions": positions.tolist() if positions is not None else None,
        "provenance": dict(provenance or {}),
        "file_size": model_path.stat().st_size,
        "file_sha256": _sha256_file(model_path),
        "created_at_unix": time.time(),
    }
    _atomic_json(metadata, metadata_path)
    return model_path, metadata_path


def load_fold_model(
    path: str | Path,
    *,
    expected_fold_id: int | None = None,
    expected_run_signature: str | None = None,
    expected_feature_count: int | None = None,
    expected_valid_positions: Sequence[int] | None = None,
) -> UnifiedModel:
    """Verify sidecar/hash/provenance before loading a trusted joblib artifact."""

    model_path = Path(path)
    metadata_path = _sidecar_path(model_path)
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Fold model or metadata sidecar is missing for {model_path}.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Fold metadata is unreadable: {metadata_path}") from exc
    if metadata.get("checkpoint_version") != FOLD_CHECKPOINT_VERSION:
        raise RuntimeError("Unsupported fold checkpoint version.")
    actual_size = model_path.stat().st_size
    actual_hash = _sha256_file(model_path)
    if metadata.get("file_size") != actual_size or metadata.get("file_sha256") != actual_hash:
        raise RuntimeError("Fold model file failed size/SHA-256 authentication.")
    expectations = {
        "fold_id": expected_fold_id,
        "run_signature": expected_run_signature,
        "feature_count": expected_feature_count,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expectations.items()
        if value is not None and metadata.get(key) != value
    }
    if expected_valid_positions is not None:
        expected_positions = np.asarray(expected_valid_positions, dtype=np.int64).tolist()
        if metadata.get("valid_positions") != expected_positions:
            mismatches["valid_positions"] = (
                metadata.get("valid_positions"),
                expected_positions,
            )
    if mismatches:
        raise RuntimeError(
            "Fold checkpoint provenance mismatch: " + json.dumps(mismatches, default=str)
        )
    # joblib/pickle can execute code. Only artifacts created by this trusted
    # workspace should reach this point; the hash protects against corruption,
    # not a malicious artifact plus maliciously replaced sidecar.
    payload = joblib.load(model_path)
    if payload.get("checkpoint_version") != FOLD_CHECKPOINT_VERSION:
        raise RuntimeError("Loaded fold payload has an unsupported version.")
    for key in (
        "fold_id",
        "run_signature",
        "model_signature",
        "feature_count",
        "valid_positions",
        "provenance",
    ):
        if _json_ready(payload.get(key)) != metadata.get(key):
            raise RuntimeError(f"Fold payload/metadata mismatch for {key}.")
    model = payload.get("model")
    if not isinstance(model, UnifiedModel) or not model.is_fitted:
        raise RuntimeError("Fold payload does not contain a fitted UnifiedModel.")
    if model.signature != metadata.get("model_signature"):
        raise RuntimeError("Loaded model signature does not match authenticated metadata.")
    return model


def _mapping_signature(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_inference_bundle(
    path: str | Path,
    *,
    fold_id: int,
    run_signature: str,
    feature_count: int,
    valid_positions: Sequence[int],
    preprocessor: Any,
    model: Any,
    training_config: Mapping[str, Any],
    fold_report: Mapping[str, Any],
    class_labels: Sequence[Any] = (),
    valid_prediction: Any | None = None,
    test_prediction: Any | None = None,
    valid_probabilities: Any | None = None,
    test_probabilities: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save an authenticated, resume-capable raw-inference fold bundle."""

    if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
        raise ValueError("fold_id must be a non-negative integer.")
    if not isinstance(run_signature, str) or not run_signature.strip():
        raise ValueError("run_signature must be a non-empty string.")
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count < 1:
        raise ValueError("feature_count must be a positive integer.")
    positions = np.asarray(valid_positions, dtype=np.int64)
    if positions.ndim != 1 or positions.size == 0 or np.any(positions < 0):
        raise ValueError("valid_positions must be a non-empty 1-D non-negative sequence.")
    if len(set(positions.tolist())) != len(positions):
        raise ValueError("valid_positions must be unique.")
    if not callable(getattr(preprocessor, "transform", None)):
        raise ValueError("preprocessor must expose transform().")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("model must expose predict().")
    if not isinstance(training_config, Mapping) or not training_config:
        raise ValueError("training_config must be a non-empty mapping.")
    if not isinstance(fold_report, Mapping):
        raise ValueError("fold_report must be a mapping.")

    model_path = Path(path)
    metadata_path = _sidecar_path(model_path)
    if (model_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite inference fold artifact {model_path}.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = dict(training_config)
    config_signature = _mapping_signature(config_dict)
    payload = {
        "artifact_type": "inference_fold_bundle",
        "bundle_version": INFERENCE_BUNDLE_VERSION,
        "fold_id": fold_id,
        "run_signature": run_signature,
        "feature_count": feature_count,
        "valid_positions": positions,
        "preprocessor": preprocessor,
        "model": model,
        "training_config": config_dict,
        "training_config_signature": config_signature,
        "fold_report": dict(fold_report),
        "class_labels": list(class_labels),
        "valid_prediction": None if valid_prediction is None else np.asarray(valid_prediction),
        "test_prediction": None if test_prediction is None else np.asarray(test_prediction),
        "valid_probabilities": (
            None if valid_probabilities is None else np.asarray(valid_probabilities)
        ),
        "test_probabilities": (
            None if test_probabilities is None else np.asarray(test_probabilities)
        ),
        "provenance": dict(provenance or {}),
    }
    temporary = model_path.with_name(model_path.name + ".tmp")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "artifact_type": "inference_fold_bundle",
        "bundle_version": INFERENCE_BUNDLE_VERSION,
        "fold_id": fold_id,
        "run_signature": run_signature,
        "feature_count": feature_count,
        "valid_positions": positions.tolist(),
        "training_config_signature": config_signature,
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "preprocessor_type": f"{type(preprocessor).__module__}.{type(preprocessor).__qualname__}",
        "class_labels": _json_ready(list(class_labels)),
        "provenance": _json_ready(dict(provenance or {})),
        "file_size": model_path.stat().st_size,
        "file_sha256": _sha256_file(model_path),
        "created_at_unix": time.time(),
    }
    _atomic_json(metadata, metadata_path)
    return model_path, metadata_path


def load_inference_bundle(
    path: str | Path,
    *,
    expected_fold_id: int | None = None,
    expected_run_signature: str | None = None,
    expected_feature_count: int | None = None,
    expected_valid_positions: Sequence[int] | None = None,
) -> InferenceFoldBundle:
    """Authenticate metadata and bytes before loading a trusted inference bundle."""

    model_path = Path(path)
    metadata_path = _sidecar_path(model_path)
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Inference bundle or metadata sidecar is missing for {model_path}.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Inference metadata is unreadable: {metadata_path}") from exc
    if (
        metadata.get("artifact_type") != "inference_fold_bundle"
        or metadata.get("bundle_version") != INFERENCE_BUNDLE_VERSION
    ):
        raise RuntimeError("Unsupported or non-inference fold artifact.")
    if (
        metadata.get("file_size") != model_path.stat().st_size
        or metadata.get("file_sha256") != _sha256_file(model_path)
    ):
        raise RuntimeError("Inference bundle failed size/SHA-256 authentication.")
    expectations = {
        "fold_id": expected_fold_id,
        "run_signature": expected_run_signature,
        "feature_count": expected_feature_count,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expectations.items()
        if expected is not None and metadata.get(key) != expected
    }
    if expected_valid_positions is not None:
        expected_positions = np.asarray(expected_valid_positions, dtype=np.int64).tolist()
        if metadata.get("valid_positions") != expected_positions:
            mismatches["valid_positions"] = (metadata.get("valid_positions"), expected_positions)
    if mismatches:
        raise RuntimeError(
            "Inference fold provenance mismatch: " + json.dumps(mismatches, default=str)
        )
    # joblib/pickle can execute code. Only workspace-created artifacts whose
    # sidecar hash was authenticated above should be loaded here.
    payload = joblib.load(model_path)
    for key in (
        "artifact_type",
        "bundle_version",
        "fold_id",
        "run_signature",
        "feature_count",
        "training_config_signature",
        "class_labels",
        "provenance",
    ):
        if _json_ready(payload.get(key)) != metadata.get(key):
            raise RuntimeError(f"Inference bundle payload/metadata mismatch for {key}.")
    positions = np.asarray(payload.get("valid_positions"), dtype=np.int64)
    if positions.tolist() != metadata.get("valid_positions"):
        raise RuntimeError("Inference bundle payload/metadata mismatch for valid_positions.")
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise RuntimeError("Inference bundle contains an invalid training configuration.")
    if _mapping_signature(training_config) != metadata.get("training_config_signature"):
        raise RuntimeError("Inference bundle training configuration signature mismatch.")
    preprocessor = payload.get("preprocessor")
    model = payload.get("model")
    if not callable(getattr(preprocessor, "transform", None)):
        raise RuntimeError("Inference bundle contains no fitted preprocessor.")
    if not callable(getattr(model, "predict", None)):
        raise RuntimeError("Inference bundle contains no fitted model.")
    return InferenceFoldBundle(
        fold_id=int(payload["fold_id"]),
        run_signature=str(payload["run_signature"]),
        feature_count=int(payload["feature_count"]),
        valid_positions=positions,
        preprocessor=preprocessor,
        model=model,
        training_config=dict(training_config),
        fold_report=dict(payload.get("fold_report") or {}),
        class_labels=list(payload.get("class_labels") or []),
        valid_prediction=payload.get("valid_prediction"),
        test_prediction=payload.get("test_prediction"),
        valid_probabilities=payload.get("valid_probabilities"),
        test_probabilities=payload.get("test_probabilities"),
        provenance=dict(payload.get("provenance") or {}),
    )


class FoldModelStore:
    """Directory-scoped manager for deterministic fold artifact names."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def path_for(self, fold_id: int) -> Path:
        if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
            raise ValueError("fold_id must be a non-negative integer.")
        return self.directory / f"fold_{fold_id:03d}.joblib"

    def save(
        self,
        model: UnifiedModel,
        fold_id: int,
        run_signature: str,
        **kwargs: Any,
    ) -> tuple[Path, Path]:
        return save_fold_model(
            model,
            self.path_for(fold_id),
            fold_id=fold_id,
            run_signature=run_signature,
            **kwargs,
        )

    def load(
        self,
        fold_id: int,
        run_signature: str,
        **kwargs: Any,
    ) -> UnifiedModel:
        return load_fold_model(
            self.path_for(fold_id),
            expected_fold_id=fold_id,
            expected_run_signature=run_signature,
            **kwargs,
        )

    def save_inference_bundle(
        self,
        fold_id: int,
        run_signature: str,
        **kwargs: Any,
    ) -> tuple[Path, Path]:
        return save_inference_bundle(
            self.path_for(fold_id),
            fold_id=fold_id,
            run_signature=run_signature,
            **kwargs,
        )

    def load_inference_bundle(
        self,
        fold_id: int,
        run_signature: str | None = None,
        **kwargs: Any,
    ) -> InferenceFoldBundle:
        return load_inference_bundle(
            self.path_for(fold_id),
            expected_fold_id=fold_id,
            expected_run_signature=run_signature,
            **kwargs,
        )

    def completed_folds(self) -> list[int]:
        folds: set[int] = set()
        if not self.directory.exists():
            return []
        for path in self.directory.glob("fold_*.joblib"):
            suffix = path.stem.removeprefix("fold_")
            if suffix.isdigit() and _sidecar_path(path).is_file():
                folds.add(int(suffix))
        return sorted(folds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect unified model backend availability.")
    parser.add_argument(
        "--require",
        nargs="*",
        choices=("ridge", "lightgbm", "catboost", "xgboost"),
        default=[],
        help="Exit non-zero if any requested backend is unavailable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    status = backend_status()
    print(json.dumps(status, indent=2))
    unavailable = [name for name in args.require if not status[name]["available"]]
    if unavailable:
        print(f"Unavailable required backends: {unavailable}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
