"""Leakage-aware OOF ensembling for regression competitions.

This module learns one non-negative weight per base model from out-of-fold
(OOF) predictions. It supports:

* uniform averaging,
* non-negative least squares (NNLS), normalized onto the simplex, and
* direct SMAPE minimization with SLSQP under ``weight >= 0, sum(weight) = 1``.

The file interface consumes the artifacts emitted by :mod:`src.train`. Files
are aligned by ID, never by row position. Targets, folds, model names, and test
IDs must agree before optimization. When folds are supplied, a cross-fitted
ensemble score is also computed: weights are learned on every other fold and
evaluated on the held-out fold. That score is the honest stability diagnostic;
the all-OOF optimized score is used to obtain the final test weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, nnls

try:
    from .metrics import smape
except ImportError:  # pragma: no cover - direct script execution
    from metrics import smape


ENSEMBLE_VERSION = "1.0"
MethodName = Literal["auto", "uniform", "nnls", "slsqp"]


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for simplex-constrained OOF weight fitting."""

    method: MethodName = "auto"
    n_starts: int = 16
    random_state: int = 42
    max_iterations: int = 2_000
    tolerance: float = 1e-10
    l2_regularization: float = 0.0
    prediction_floor: float | None = 0.0
    prediction_ceiling: float | None = None
    zero_weight_threshold: float = 1e-8
    cross_fit: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"auto", "uniform", "nnls", "slsqp"}:
            raise ValueError("method must be auto, uniform, nnls, or slsqp.")
        if (
            isinstance(self.n_starts, bool)
            or not isinstance(self.n_starts, int)
            or self.n_starts < 1
        ):
            raise ValueError("n_starts must be a positive integer.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if not math.isfinite(self.l2_regularization) or self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be finite and non-negative.")
        if not math.isfinite(self.zero_weight_threshold) or not (
            0.0 <= self.zero_weight_threshold < 1.0
        ):
            raise ValueError("zero_weight_threshold must be in [0, 1).")
        for label, value in (
            ("prediction_floor", self.prediction_floor),
            ("prediction_ceiling", self.prediction_ceiling),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{label} must be finite or None.")
        if (
            self.prediction_floor is not None
            and self.prediction_ceiling is not None
            and self.prediction_floor > self.prediction_ceiling
        ):
            raise ValueError("prediction_floor cannot exceed prediction_ceiling.")


@dataclass(frozen=True)
class WeightSolution:
    """A validated set of ensemble weights and optimization diagnostics."""

    weights: np.ndarray
    method: str
    score: float
    success: bool
    message: str
    optimizer_runs: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass
class EnsembleResult:
    """Final OOF/test predictions and an auditable optimization report."""

    weights: pd.Series
    oof: pd.DataFrame
    test_predictions: pd.DataFrame | None
    report: dict[str, Any]


def _as_finite_vector(values: Any, *, label: str, expected_rows: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array.")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError(f"{label} has {len(array)} rows; expected {expected_rows}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity.")
    return array


def _as_prediction_matrix(values: Any, *, expected_rows: int | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("predictions must be numeric.") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("predictions must be a non-empty two-dimensional matrix.")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"predictions have {matrix.shape[0]} rows; expected {expected_rows}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("predictions contain NaN or infinity.")
    return matrix


def _validate_model_names(model_names: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if model_names is None:
        return tuple(f"model_{index}" for index in range(count))
    names = tuple(str(name).strip() for name in model_names)
    if len(names) != count:
        raise ValueError(f"model_names has {len(names)} entries; expected {count}.")
    if any(not name for name in names):
        raise ValueError("model_names cannot contain empty names.")
    if len(set(names)) != len(names):
        raise ValueError("model_names must be unique.")
    return names


def _clip_predictions(predictions: np.ndarray, config: EnsembleConfig) -> np.ndarray:
    lower = config.prediction_floor if config.prediction_floor is not None else -np.inf
    upper = config.prediction_ceiling if config.prediction_ceiling is not None else np.inf
    return np.clip(predictions, lower, upper)


def apply_weights(
    predictions: Any,
    weights: Any,
    *,
    prediction_floor: float | None = 0.0,
    prediction_ceiling: float | None = None,
) -> np.ndarray:
    """Blend a prediction matrix after strictly validating simplex weights."""

    matrix = _as_prediction_matrix(predictions)
    vector = _as_finite_vector(weights, label="weights", expected_rows=matrix.shape[1])
    if np.any(vector < -1e-10):
        raise ValueError("weights must be non-negative.")
    if not np.isclose(vector.sum(), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("weights must sum to one.")
    if prediction_floor is not None and not math.isfinite(prediction_floor):
        raise ValueError("prediction_floor must be finite or None.")
    if prediction_ceiling is not None and not math.isfinite(prediction_ceiling):
        raise ValueError("prediction_ceiling must be finite or None.")
    if (
        prediction_floor is not None
        and prediction_ceiling is not None
        and prediction_floor > prediction_ceiling
    ):
        raise ValueError("prediction_floor cannot exceed prediction_ceiling.")
    safe_weights = _simplex(vector)
    blended = matrix @ safe_weights
    return np.clip(
        blended,
        prediction_floor if prediction_floor is not None else -np.inf,
        prediction_ceiling if prediction_ceiling is not None else np.inf,
    )


def _score_weights(
    target: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    config: EnsembleConfig,
) -> float:
    blended = _clip_predictions(predictions @ weights, config)
    return smape(target, blended)


def _penalized_objective(
    weights: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    config: EnsembleConfig,
) -> float:
    score = _score_weights(target, predictions, weights, config)
    if config.l2_regularization:
        uniform = np.full(len(weights), 1.0 / len(weights))
        score += config.l2_regularization * float(np.sum((weights - uniform) ** 2))
    return float(score)


def _smape_gradient(
    weights: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    config: EnsembleConfig,
) -> np.ndarray:
    """Return an exact SMAPE subgradient with respect to simplex weights.

    This general form supports positive, zero, and negative targets/predictions:

    ``2 * [sign(p-y) * (|y|+|p|) - |p-y| * sign(p)] / (|y|+|p|)^2``.

    At absolute-value kinks NumPy's ``sign(0) == 0`` selects a valid central
    subgradient. Rows clipped by a prediction floor/ceiling contribute zero
    derivative outside the active interval.
    """

    raw_prediction = predictions @ weights
    blended = _clip_predictions(raw_prediction, config)
    denominator = np.abs(target) + np.abs(blended)
    difference = blended - target
    derivative = np.zeros_like(blended, dtype=np.float64)
    valid = denominator > 0.0
    numerator = 2.0 * (
        np.sign(difference) * denominator
        - np.abs(difference) * np.sign(blended)
    )
    np.divide(
        numerator,
        denominator * denominator,
        out=derivative,
        where=valid,
    )
    if config.prediction_floor is not None:
        derivative[raw_prediction <= config.prediction_floor] = 0.0
    if config.prediction_ceiling is not None:
        derivative[raw_prediction >= config.prediction_ceiling] = 0.0
    return predictions.T @ derivative / len(target)


def _penalized_gradient(
    weights: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    config: EnsembleConfig,
) -> np.ndarray:
    gradient = _smape_gradient(weights, target, predictions, config)
    if config.l2_regularization:
        uniform = np.full(len(weights), 1.0 / len(weights))
        gradient = gradient + 2.0 * config.l2_regularization * (weights - uniform)
    return np.asarray(gradient, dtype=np.float64)


def _simplex(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(vector), 1.0 / len(vector))
    return vector / total


def _nnls_weights(
    target: np.ndarray,
    predictions: np.ndarray,
    config: EnsembleConfig,
) -> tuple[np.ndarray, float]:
    """Solve least squares on the non-negative unit simplex.

    Rows are divided by ``max(abs(target), robust_epsilon)`` so low- and
    high-priced products contribute comparable relative error. SciPy NNLS
    supplies a strong non-negative starting point; a constrained quadratic
    SLSQP solve then enforces ``sum(weights) == 1`` exactly.
    """

    absolute_target = np.abs(target)
    nonzero_target = absolute_target[absolute_target > 0.0]
    reference = float(np.median(nonzero_target)) if len(nonzero_target) else 1.0
    minimum_scale = max(reference * 1e-3, np.finfo(np.float64).eps)
    row_scale = np.maximum(absolute_target, minimum_scale)
    scaled_predictions = predictions / row_scale[:, None]
    scaled_target = target / row_scale
    raw, _ = nnls(scaled_predictions, scaled_target)
    start = _simplex(raw)

    def squared_error(weights: np.ndarray) -> float:
        residual = scaled_predictions @ weights - scaled_target
        return float(np.dot(residual, residual))

    def squared_error_gradient(weights: np.ndarray) -> np.ndarray:
        residual = scaled_predictions @ weights - scaled_target
        return 2.0 * (scaled_predictions.T @ residual)

    result = minimize(
        squared_error,
        x0=start,
        jac=squared_error_gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * predictions.shape[1],
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={
            "maxiter": config.max_iterations,
            "ftol": min(config.tolerance, 1e-12),
            "disp": False,
        },
    )
    candidates = [start, _simplex(result.x)]
    candidates.extend(np.eye(predictions.shape[1]))
    weights = min(candidates, key=squared_error)
    residual_norm = float(
        np.linalg.norm((predictions @ weights - target) / row_scale)
    )
    return np.asarray(weights, dtype=np.float64), residual_norm


def _deduplicate_starts(starts: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        normalized = _simplex(start)
        key = np.round(normalized, decimals=12).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _slsqp_solution(
    target: np.ndarray,
    predictions: np.ndarray,
    config: EnsembleConfig,
    seeds: Sequence[np.ndarray],
) -> WeightSolution:
    model_count = predictions.shape[1]
    rng = np.random.default_rng(config.random_state)
    starts = list(seeds)
    while len(starts) < config.n_starts:
        starts.append(rng.dirichlet(np.ones(model_count)))
    starts = _deduplicate_starts(starts)[: config.n_starts]

    runs: list[dict[str, Any]] = []
    feasible: list[tuple[float, float, np.ndarray, str, bool]] = []
    for run_index, start in enumerate(starts):
        result = minimize(
            _penalized_objective,
            jac=_penalized_gradient,
            x0=start,
            args=(target, predictions, config),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * model_count,
            constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
            options={
                "maxiter": config.max_iterations,
                "ftol": config.tolerance,
                "disp": False,
            },
        )
        candidate = _simplex(result.x)
        score = _score_weights(target, predictions, candidate, config)
        objective = _penalized_objective(candidate, target, predictions, config)
        violation = abs(float(candidate.sum()) - 1.0) + float(
            np.maximum(-candidate, 0.0).sum()
        )
        is_feasible = bool(np.isfinite(score) and violation <= 1e-7)
        runs.append(
            {
                "run": run_index,
                "success": bool(result.success),
                "feasible": is_feasible,
                "score": float(score),
                "objective": float(objective),
                "iterations": int(getattr(result, "nit", 0)),
                "function_evaluations": int(getattr(result, "nfev", 0)),
                "gradient_evaluations": int(getattr(result, "njev", 0)),
                "message": str(result.message),
            }
        )
        if is_feasible:
            feasible.append(
                (objective, score, candidate, str(result.message), bool(result.success))
            )

    if not feasible:
        fallback = _simplex(seeds[0])
        return WeightSolution(
            fallback,
            "slsqp_fallback",
            _score_weights(target, predictions, fallback, config),
            False,
            "No SLSQP run returned a feasible finite candidate.",
            tuple(runs),
        )
    _, score, weights, message, selected_success = min(
        feasible, key=lambda item: (item[0], item[1])
    )
    return WeightSolution(
        weights,
        "slsqp",
        float(score),
        selected_success,
        message,
        tuple(runs),
    )


def optimize_weights(
    y_true: Any,
    oof_predictions: Any,
    *,
    model_names: Sequence[str] | None = None,
    config: EnsembleConfig | None = None,
) -> WeightSolution:
    """Optimize non-negative ensemble weights from genuine OOF predictions."""

    settings = config or EnsembleConfig()
    target = _as_finite_vector(y_true, label="y_true")
    predictions = _as_prediction_matrix(oof_predictions, expected_rows=len(target))
    _validate_model_names(model_names, predictions.shape[1])
    model_count = predictions.shape[1]
    uniform = np.full(model_count, 1.0 / model_count)
    if settings.method == "uniform" or model_count == 1:
        return WeightSolution(
            uniform,
            "uniform" if model_count > 1 else "single_model",
            _score_weights(target, predictions, uniform, settings),
            True,
            "No numerical optimization required.",
        )

    nnls_weight, nnls_residual = _nnls_weights(target, predictions, settings)
    if settings.method == "nnls":
        return WeightSolution(
            nnls_weight,
            "nnls",
            _score_weights(target, predictions, nnls_weight, settings),
            True,
            "Relative-scaled constrained least-squares residual norm: "
            f"{nnls_residual:.12g}.",
        )

    individual_scores = [
        _score_weights(target, predictions, np.eye(model_count)[index], settings)
        for index in range(model_count)
    ]
    inverse = 1.0 / np.maximum(np.asarray(individual_scores), 1e-12)
    one_hot = [np.eye(model_count)[index] for index in range(model_count)]
    best_single = one_hot[int(np.argmin(individual_scores))]
    # Keep the run count bounded even when dozens of base models are supplied.
    # All one-hot candidates are still compared by ``auto`` below; only the
    # strongest one is used as an SLSQP starting point.
    seeds = [nnls_weight, best_single, uniform, _simplex(inverse)]
    slsqp = _slsqp_solution(target, predictions, settings, seeds)

    if settings.method == "auto":
        candidates = [
            ("uniform", uniform),
            ("nnls", nnls_weight),
            (slsqp.method, slsqp.weights),
            *[(f"best_single_{index}", weight) for index, weight in enumerate(one_hot)],
        ]
        selected_method, selected = min(
            candidates,
            key=lambda item: _penalized_objective(item[1], target, predictions, settings),
        )
        solution = WeightSolution(
            np.asarray(selected),
            selected_method,
            _score_weights(target, predictions, selected, settings),
            slsqp.success if selected_method.startswith("slsqp") else True,
            "Selected the lowest OOF objective among uniform, NNLS, SLSQP, and single-model candidates.",
            slsqp.optimizer_runs,
        )
    else:
        solution = slsqp

    if settings.zero_weight_threshold > 0.0:
        sparse_weights = solution.weights.copy()
        sparse_weights[sparse_weights < settings.zero_weight_threshold] = 0.0
        sparse_weights = _simplex(sparse_weights)
        sparse_score = _score_weights(target, predictions, sparse_weights, settings)
        sparse_objective = _penalized_objective(
            sparse_weights, target, predictions, settings
        )
        current_objective = _penalized_objective(
            solution.weights, target, predictions, settings
        )
        if sparse_objective <= current_objective + max(settings.tolerance, 1e-12):
            solution = WeightSolution(
                sparse_weights,
                solution.method,
                sparse_score,
                solution.success,
                solution.message,
                solution.optimizer_runs,
            )
    return solution


def _validate_folds(folds: Any, rows: int) -> np.ndarray:
    values = np.asarray(folds)
    if values.ndim != 1 or len(values) != rows:
        raise ValueError(f"folds must be one-dimensional with {rows} rows.")
    missing = pd.isna(values)
    if bool(np.any(missing)):
        raise ValueError("folds contains missing values.")
    if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
        raise ValueError("folds contains infinity.")
    if len(pd.unique(values)) < 2:
        raise ValueError("Cross-fitting requires at least two distinct folds.")
    return values


def _cross_fitted_predictions(
    target: np.ndarray,
    predictions: np.ndarray,
    folds: np.ndarray,
    model_names: Sequence[str],
    config: EnsembleConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(target), np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    inner_config = replace(config, cross_fit=False)
    for fold in pd.unique(folds):
        valid = folds == fold
        train = ~valid
        if not np.any(train) or not np.any(valid):
            raise ValueError(f"Fold {fold!r} has an empty train or validation partition.")
        solution = optimize_weights(
            target[train],
            predictions[train],
            model_names=model_names,
            config=inner_config,
        )
        output[valid] = apply_weights(
            predictions[valid],
            solution.weights,
            prediction_floor=config.prediction_floor,
            prediction_ceiling=config.prediction_ceiling,
        )
        reports.append(
            {
                "fold": _json_ready(fold),
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "method": solution.method,
                "score": smape(target[valid], output[valid]),
                "weights": {
                    name: float(weight) for name, weight in zip(model_names, solution.weights)
                },
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Cross-fitting did not assign every OOF row.")
    return output, reports


def _matrix_diagnostics(predictions: np.ndarray, model_names: Sequence[str]) -> dict[str, Any]:
    duplicates: list[list[str]] = []
    for left in range(predictions.shape[1]):
        for right in range(left + 1, predictions.shape[1]):
            if np.array_equal(predictions[:, left], predictions[:, right]):
                duplicates.append([model_names[left], model_names[right]])
    centered = predictions - predictions.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    usable = singular[singular > np.finfo(float).eps * max(centered.shape)]
    condition = float(usable[0] / usable[-1]) if len(usable) > 1 else None
    return {
        "rank": int(np.linalg.matrix_rank(centered)),
        "model_count": int(predictions.shape[1]),
        "duplicate_model_pairs": duplicates,
        "condition_number_nonzero_subspace": condition,
    }


def fit_ensemble(
    y_true: Any,
    oof_predictions: Any,
    *,
    model_names: Sequence[str] | None = None,
    test_predictions: Any | None = None,
    ids: Sequence[Any] | None = None,
    test_ids: Sequence[Any] | None = None,
    folds: Any | None = None,
    target_column: str = "target",
    id_column: str = "id",
    fold_column: str = "fold",
    config: EnsembleConfig | None = None,
) -> EnsembleResult:
    """Fit final weights, produce blends, and calculate stability diagnostics."""

    started = time.perf_counter()
    settings = config or EnsembleConfig()
    target = _as_finite_vector(y_true, label="y_true")
    oof_matrix = _as_prediction_matrix(oof_predictions, expected_rows=len(target))
    names = _validate_model_names(model_names, oof_matrix.shape[1])
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("target_column must be a non-empty string.")
    if not isinstance(id_column, str) or not id_column.strip():
        raise ValueError("id_column must be a non-empty string.")
    if not isinstance(fold_column, str) or not fold_column.strip():
        raise ValueError("fold_column must be a non-empty string.")
    reserved_columns = [
        id_column,
        target_column,
        fold_column,
        "prediction",
        "cross_fitted_prediction",
    ]
    if len(set(reserved_columns)) != len(reserved_columns):
        raise ValueError(
            "id_column, target_column, fold_column, prediction, and "
            "cross_fitted_prediction names must all differ."
        )

    if ids is None:
        oof_ids = np.arange(len(target))
    else:
        oof_ids = np.asarray(ids, dtype=object)
        if oof_ids.ndim != 1 or len(oof_ids) != len(target):
            raise ValueError(f"ids must be one-dimensional with {len(target)} rows.")
        _id_index(oof_ids, label="OOF ids")

    fold_values = _validate_folds(folds, len(target)) if folds is not None else None
    solution = optimize_weights(target, oof_matrix, model_names=names, config=settings)
    final_oof = apply_weights(
        oof_matrix,
        solution.weights,
        prediction_floor=settings.prediction_floor,
        prediction_ceiling=settings.prediction_ceiling,
    )
    individual_scores = {
        name: smape(target, _clip_predictions(oof_matrix[:, index], settings))
        for index, name in enumerate(names)
    }
    uniform_prediction = _clip_predictions(oof_matrix.mean(axis=1), settings)

    cross_fitted_score = None
    fold_reports: list[dict[str, Any]] = []
    cross_fitted_oof = None
    if settings.cross_fit and fold_values is not None:
        cross_fitted_oof, fold_reports = _cross_fitted_predictions(
            target, oof_matrix, fold_values, names, settings
        )
        cross_fitted_score = smape(target, cross_fitted_oof)

    oof_frame = pd.DataFrame(
        {id_column: oof_ids, target_column: target, "prediction": final_oof}
    )
    if fold_values is not None:
        oof_frame[fold_column] = fold_values
    if cross_fitted_oof is not None:
        oof_frame["cross_fitted_prediction"] = cross_fitted_oof

    test_frame = None
    if test_predictions is not None:
        test_matrix = _as_prediction_matrix(test_predictions)
        if test_matrix.shape[1] != len(names):
            raise ValueError(
                f"test_predictions has {test_matrix.shape[1]} models; expected {len(names)}."
            )
        if test_ids is None:
            final_test_ids = np.arange(test_matrix.shape[0])
        else:
            final_test_ids = np.asarray(test_ids, dtype=object)
            if final_test_ids.ndim != 1 or len(final_test_ids) != test_matrix.shape[0]:
                raise ValueError(
                    f"test_ids must be one-dimensional with {test_matrix.shape[0]} rows."
                )
            _id_index(final_test_ids, label="test ids")
        final_test = apply_weights(
            test_matrix,
            solution.weights,
            prediction_floor=settings.prediction_floor,
            prediction_ceiling=settings.prediction_ceiling,
        )
        test_frame = pd.DataFrame({id_column: final_test_ids, "prediction": final_test})
    elif test_ids is not None:
        raise ValueError("test_ids cannot be supplied without test_predictions.")

    weights = pd.Series(solution.weights, index=names, name="weight", dtype=np.float64)
    positive = solution.weights[solution.weights > 0.0]
    entropy = -float(np.sum(positive * np.log(positive))) if len(positive) else 0.0
    report = {
        "ensemble_version": ENSEMBLE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": asdict(settings),
        "selected_method": solution.method,
        "optimization_success": bool(solution.success),
        "optimization_message": solution.message,
        "weights": {name: float(weight) for name, weight in weights.items()},
        "oof_smape": float(solution.score),
        "uniform_oof_smape": smape(target, uniform_prediction),
        "best_individual_oof_smape": float(min(individual_scores.values())),
        "individual_oof_smape": individual_scores,
        "cross_fitted_oof_smape": cross_fitted_score,
        "cross_fitted_folds": fold_reports,
        "rows": int(len(target)),
        "test_rows": int(len(test_frame)) if test_frame is not None else 0,
        "weight_sum": float(solution.weights.sum()),
        "nonzero_weights": int(np.sum(solution.weights > settings.zero_weight_threshold)),
        "weight_entropy": entropy,
        "effective_models": float(math.exp(entropy)),
        "matrix_diagnostics": _matrix_diagnostics(oof_matrix, names),
        "optimizer_runs": list(solution.optimizer_runs),
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    return EnsembleResult(weights, oof_frame, test_frame, _json_ready(report))


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


def _id_index(values: Sequence[Any], *, label: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, value in enumerate(values):
        token = _canonical_id(value)
        if token in positions:
            raise ValueError(f"{label} contains duplicate ID {value!r}.")
        positions[token] = index
    return positions


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path}; use CSV or Parquet.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_named_paths(values: Sequence[str], *, label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Each {label} must use NAME=PATH syntax; received {item!r}.")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"Each {label} must contain a non-empty name and path.")
        if name in parsed:
            raise ValueError(f"Duplicate model name {name!r} in {label} inputs.")
        parsed[name] = Path(raw_path)
    if not parsed:
        raise ValueError(f"At least one {label} is required.")
    return parsed


def _align_frame(
    frame: pd.DataFrame,
    reference_tokens: Sequence[str],
    *,
    id_column: str,
    label: str,
) -> pd.DataFrame:
    if id_column not in frame.columns:
        raise ValueError(f"{label} is missing ID column {id_column!r}.")
    positions = _id_index(frame[id_column].tolist(), label=f"{label} IDs")
    reference_set = set(reference_tokens)
    actual_set = set(positions)
    if actual_set != reference_set:
        missing = len(reference_set - actual_set)
        extra = len(actual_set - reference_set)
        raise ValueError(f"{label} ID set mismatch: missing={missing}, extra={extra}.")
    return frame.iloc[[positions[token] for token in reference_tokens]].reset_index(drop=True)


_UNEVALUATED_FOLD_TOKENS = frozenset(
    {
        "",
        "test",
        "holdout",
        "heldout",
        "held_out",
        "unassigned",
        "unused",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
    }
)


def _evaluated_fold_mask(values: pd.Series, *, label: str) -> np.ndarray:
    """Identify true OOF rows in numeric or mixed-type fold columns."""

    series = values.reset_index(drop=True)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_known = numeric.notna().to_numpy()
    if numeric_known.any() and np.isinf(numeric[numeric.notna()].to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains an infinite fold value.")
    mask = np.ones(len(series), dtype=bool)
    mask[pd.isna(series).to_numpy()] = False
    mask[numeric_known] = numeric[numeric.notna()].to_numpy(dtype=float) >= 0.0
    for index in np.flatnonzero(~numeric_known & ~pd.isna(series).to_numpy()):
        token = str(series.iloc[index]).strip().casefold().replace("-", "_")
        if token in _UNEVALUATED_FOLD_TOKENS:
            mask[index] = False
    if not np.any(mask):
        raise ValueError(f"{label} contains no evaluated OOF rows.")
    return mask


def ensemble_from_files(
    oof_files: Mapping[str, str | Path],
    *,
    test_files: Mapping[str, str | Path] | None = None,
    id_column: str,
    target_column: str,
    prediction_column: str = "prediction",
    fold_column: str = "fold",
    config: EnsembleConfig | None = None,
) -> EnsembleResult:
    """Load, ID-align, validate, and blend train.py prediction artifacts."""

    if not oof_files:
        raise ValueError("At least one OOF file is required.")
    names = tuple(oof_files)
    if any(not str(name).strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("OOF model names must be unique and non-empty.")
    paths = {name: Path(path) for name, path in oof_files.items()}
    frames = {name: _load_table(path) for name, path in paths.items()}
    first_name = names[0]
    first = frames[first_name]
    required = {id_column, target_column, prediction_column}
    missing = sorted(required - set(first.columns))
    if missing:
        raise ValueError(f"OOF file {first_name!r} is missing columns: {missing}.")

    original_first_rows = len(first)
    if fold_column in first.columns:
        eligible = _evaluated_fold_mask(
            first[fold_column], label=f"OOF {first_name!r} folds"
        )
        first = first.loc[eligible].reset_index(drop=True)
    dropped_unevaluated_rows = original_first_rows - len(first)
    _id_index(first[id_column].tolist(), label=f"OOF {first_name!r} IDs")
    reference_tokens = [_canonical_id(value) for value in first[id_column]]
    reference_target = _as_finite_vector(first[target_column], label="OOF target")
    reference_folds = first[fold_column].to_numpy() if fold_column in first.columns else None
    prediction_columns: list[np.ndarray] = []

    for name in names:
        frame = frames[name]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"OOF file {name!r} is missing columns: {missing}.")
        if reference_folds is not None and fold_column not in frame.columns:
            raise ValueError(f"OOF file {name!r} is missing fold column {fold_column!r}.")
        if reference_folds is not None:
            eligible = _evaluated_fold_mask(
                frame[fold_column], label=f"OOF {name!r} folds"
            )
            frame = frame.loc[eligible].reset_index(drop=True)
        aligned = _align_frame(
            frame, reference_tokens, id_column=id_column, label=f"OOF {name!r}"
        )
        candidate_target = _as_finite_vector(
            aligned[target_column], label=f"OOF target for {name!r}", expected_rows=len(first)
        )
        if not np.allclose(candidate_target, reference_target, rtol=1e-12, atol=1e-12):
            raise ValueError(f"OOF target mismatch for model {name!r}.")
        if reference_folds is not None:
            candidate_folds = aligned[fold_column].to_numpy()
            if not np.array_equal(candidate_folds, reference_folds):
                raise ValueError(f"OOF fold mismatch for model {name!r}.")
        prediction_columns.append(
            _as_finite_vector(
                aligned[prediction_column],
                label=f"OOF prediction for {name!r}",
                expected_rows=len(first),
            )
        )

    test_matrix = None
    final_test_ids = None
    test_paths: dict[str, Path] = {}
    if test_files is not None:
        if set(test_files) != set(names):
            raise ValueError("Test model names must exactly match OOF model names.")
        test_paths = {name: Path(test_files[name]) for name in names}
        test_frames = {name: _load_table(path) for name, path in test_paths.items()}
        test_first = test_frames[first_name]
        test_required = {id_column, prediction_column}
        missing = sorted(test_required - set(test_first.columns))
        if missing:
            raise ValueError(f"Test file {first_name!r} is missing columns: {missing}.")
        _id_index(test_first[id_column].tolist(), label=f"test {first_name!r} IDs")
        test_tokens = [_canonical_id(value) for value in test_first[id_column]]
        final_test_ids = test_first[id_column].to_numpy(dtype=object)
        test_columns: list[np.ndarray] = []
        for name in names:
            frame = test_frames[name]
            missing = sorted(test_required - set(frame.columns))
            if missing:
                raise ValueError(f"Test file {name!r} is missing columns: {missing}.")
            aligned = _align_frame(
                frame, test_tokens, id_column=id_column, label=f"test {name!r}"
            )
            test_columns.append(
                _as_finite_vector(
                    aligned[prediction_column],
                    label=f"test prediction for {name!r}",
                    expected_rows=len(test_first),
                )
            )
        test_matrix = np.column_stack(test_columns)

    result = fit_ensemble(
        reference_target,
        np.column_stack(prediction_columns),
        model_names=names,
        test_predictions=test_matrix,
        ids=first[id_column].to_numpy(dtype=object),
        test_ids=final_test_ids,
        folds=reference_folds,
        target_column=target_column,
        id_column=id_column,
        fold_column=fold_column,
        config=config,
    )
    result.report["sources"] = {
        "oof": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "test": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in test_paths.items()
        },
    }
    result.report["dropped_unevaluated_oof_rows"] = int(dropped_unevaluated_rows)
    signature_payload = {
        "config": result.report["config"],
        "sources": result.report["sources"],
        "weights": result.report["weights"],
    }
    result.report["run_signature"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


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
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def save_ensemble_result(
    result: EnsembleResult,
    output_directory: str | Path,
    *,
    target_column: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write weights, OOF/test blends, report, and submission."""

    destination = Path(output_directory)
    outputs = {
        "weights": destination / "ensemble_weights.json",
        "oof": destination / "ensemble_oof.parquet",
        "report": destination / "ensemble_report.json",
    }
    if result.test_predictions is not None:
        outputs["test_predictions"] = destination / "ensemble_test_predictions.parquet"
        outputs["submission"] = destination / "submission.csv"
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing ensemble artifacts: {[str(path) for path in existing]}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {
            "ensemble_version": ENSEMBLE_VERSION,
            "method": result.report["selected_method"],
            "weights": result.weights.to_dict(),
            "weight_sum": float(result.weights.sum()),
            "run_signature": result.report.get("run_signature"),
        },
        outputs["weights"],
    )
    _atomic_table(result.oof, outputs["oof"])
    _atomic_json(result.report, outputs["report"])
    if result.test_predictions is not None:
        _atomic_table(result.test_predictions, outputs["test_predictions"])
        submission = result.test_predictions.rename(columns={"prediction": target_column})
        _atomic_table(submission, outputs["submission"])
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit non-negative OOF ensemble weights and blend test predictions."
    )
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Base model OOF file; repeat once per model.",
    )
    parser.add_argument(
        "--test",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Matching test-prediction file; repeat once per model.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--method", choices=("auto", "uniform", "nnls", "slsqp"), default="auto")
    parser.add_argument("--n-starts", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    floor_group = parser.add_mutually_exclusive_group()
    floor_group.add_argument("--prediction-floor", type=float, default=0.0)
    floor_group.add_argument(
        "--allow-negative-predictions",
        action="store_true",
        help="Disable the default zero floor used for positive price targets.",
    )
    parser.add_argument("--prediction-ceiling", type=float)
    parser.add_argument("--zero-weight-threshold", type=float, default=1e-8)
    parser.add_argument("--no-cross-fit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    oof_files = _parse_named_paths(args.oof, label="OOF input")
    test_files = _parse_named_paths(args.test, label="test input") if args.test else None
    config = EnsembleConfig(
        method=args.method,
        n_starts=args.n_starts,
        random_state=args.random_state,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        l2_regularization=args.l2_regularization,
        prediction_floor=None if args.allow_negative_predictions else args.prediction_floor,
        prediction_ceiling=args.prediction_ceiling,
        zero_weight_threshold=args.zero_weight_threshold,
        cross_fit=not args.no_cross_fit,
    )
    result = ensemble_from_files(
        oof_files,
        test_files=test_files,
        id_column=args.id_column,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        fold_column=args.fold_column,
        config=config,
    )
    outputs = save_ensemble_result(
        result,
        args.output_dir,
        target_column=args.target_column,
        overwrite=args.overwrite,
    )
    print(json.dumps({"weights": result.weights.to_dict()}, indent=2))
    print(
        f"OOF SMAPE: {result.report['oof_smape']:.8f} | "
        f"cross-fitted: {result.report['cross_fitted_oof_smape']} | "
        f"method: {result.report['selected_method']}"
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
