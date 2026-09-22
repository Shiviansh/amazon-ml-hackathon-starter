"""Leakage-safe price calibration, domain clamping, and prediction blending.

The final scalar multiplier is learned exclusively from out-of-fold (OOF)
predictions by minimizing the repository's exact SMAPE implementation. Domain
clamps are part of the calibration objective, so the optimized multiplier is
consistent with the predictions eventually written to a submission.

Arithmetic and geometric blending are supported for wide prediction tables.
Geometric blending is performed stably in log space and refuses non-positive
inputs unless a strictly positive domain floor is explicitly configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar

try:
    from .metrics import smape
except ImportError:  # pragma: no cover - direct script execution
    from metrics import smape


POSTPROCESS_VERSION = "1.0"
BlendMethod = Literal["arithmetic", "geometric"]


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration for OOF calibration and domain-aware blending."""

    blend_method: BlendMethod = "arithmetic"
    calibrate: bool = True
    alpha_min: float = 0.5
    alpha_max: float = 2.0
    calibration_grid_size: int = 129
    local_searches: int = 3
    optimizer_tolerance: float = 1e-10
    price_floor: float | None = 0.0
    price_ceiling: float | None = None
    cross_fit: bool = True

    def __post_init__(self) -> None:
        if self.blend_method not in {"arithmetic", "geometric"}:
            raise ValueError("blend_method must be arithmetic or geometric.")
        for label, value in (
            ("alpha_min", self.alpha_min),
            ("alpha_max", self.alpha_max),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and strictly positive.")
        if self.alpha_min >= self.alpha_max:
            raise ValueError("alpha_min must be smaller than alpha_max.")
        if not (self.alpha_min <= 1.0 <= self.alpha_max):
            raise ValueError("alpha bounds must include the identity multiplier 1.0.")
        if (
            isinstance(self.calibration_grid_size, bool)
            or not isinstance(self.calibration_grid_size, int)
            or self.calibration_grid_size < 9
        ):
            raise ValueError("calibration_grid_size must be an integer of at least 9.")
        if (
            isinstance(self.local_searches, bool)
            or not isinstance(self.local_searches, int)
            or self.local_searches < 1
        ):
            raise ValueError("local_searches must be a positive integer.")
        if not math.isfinite(self.optimizer_tolerance) or self.optimizer_tolerance <= 0.0:
            raise ValueError("optimizer_tolerance must be finite and positive.")
        for label, value in (
            ("price_floor", self.price_floor),
            ("price_ceiling", self.price_ceiling),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{label} must be finite or None.")
            if value is not None and value < 0.0:
                raise ValueError(f"{label} cannot be negative for price post-processing.")
        if (
            self.price_floor is not None
            and self.price_ceiling is not None
            and self.price_floor > self.price_ceiling
        ):
            raise ValueError("price_floor cannot exceed price_ceiling.")
        if self.blend_method == "geometric" and (
            self.price_floor is not None and self.price_floor <= 0.0
        ):
            raise ValueError("Geometric blending requires a strictly positive price_floor.")


@dataclass(frozen=True)
class CalibrationSolution:
    alpha: float
    score_before: float
    score_after: float
    success: bool
    message: str
    evaluations: int


@dataclass
class PostprocessResult:
    oof: pd.DataFrame
    test_predictions: pd.DataFrame | None
    report: dict[str, Any]


def _finite_vector(values: Any, *, label: str, rows: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array.")
    if rows is not None and len(array) != rows:
        raise ValueError(f"{label} has {len(array)} rows; expected {rows}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity.")
    return array


def _prediction_matrix(values: Any, *, label: str = "predictions") -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{label} must be a non-empty one- or two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains NaN or infinity.")
    return matrix


def _simplex_weights(weights: Any | None, model_count: int) -> np.ndarray:
    if weights is None:
        return np.full(model_count, 1.0 / model_count)
    vector = _finite_vector(weights, label="blend weights", rows=model_count)
    if np.any(vector < 0.0):
        raise ValueError("Blend weights must be non-negative.")
    total = float(vector.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Blend weights must sum to one.")
    if total <= 0.0:
        raise ValueError("At least one blend weight must be positive.")
    return vector / total


def clamp_prices(predictions: Any, config: PostprocessConfig) -> np.ndarray:
    """Apply explicit catalog bounds without inventing data-derived limits."""

    values = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity.")
    lower = config.price_floor if config.price_floor is not None else -np.inf
    upper = config.price_ceiling if config.price_ceiling is not None else np.inf
    return np.clip(values, lower, upper)


def blend_predictions(
    predictions: Any,
    *,
    weights: Any | None = None,
    method: BlendMethod = "arithmetic",
    positive_floor: float | None = None,
) -> np.ndarray:
    """Blend model predictions arithmetically or geometrically in log space."""

    matrix = _prediction_matrix(predictions)
    vector = _simplex_weights(weights, matrix.shape[1])
    if method == "arithmetic":
        return matrix @ vector
    if method != "geometric":
        raise ValueError("method must be arithmetic or geometric.")
    if positive_floor is not None:
        if not math.isfinite(positive_floor) or positive_floor <= 0.0:
            raise ValueError("positive_floor must be finite and strictly positive.")
        matrix = np.maximum(matrix, positive_floor)
    elif np.any(matrix <= 0.0):
        raise ValueError(
            "Geometric blending requires strictly positive predictions or an explicit "
            "positive_floor."
        )
    return np.exp(np.log(matrix) @ vector)


def apply_postprocessing(
    predictions: Any,
    *,
    alpha: float,
    config: PostprocessConfig,
) -> np.ndarray:
    values = _finite_vector(predictions, label="predictions")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and strictly positive.")
    return clamp_prices(values * alpha, config)


def _alpha_score(
    alpha: float,
    target: np.ndarray,
    predictions: np.ndarray,
    config: PostprocessConfig,
) -> float:
    lower = config.price_floor if config.price_floor is not None else -np.inf
    upper = config.price_ceiling if config.price_ceiling is not None else np.inf
    calibrated = np.clip(predictions * alpha, lower, upper)
    denominator = np.abs(target) + np.abs(calibrated)
    errors = np.zeros_like(denominator, dtype=np.float64)
    np.divide(
        2.0 * np.abs(calibrated - target),
        denominator,
        out=errors,
        where=denominator != 0.0,
    )
    return float(errors.mean())


def _grid_alpha_scores(
    alphas: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    config: PostprocessConfig,
) -> np.ndarray:
    """Evaluate a log grid in bounded-memory row blocks."""

    totals = np.zeros(len(alphas), dtype=np.float64)
    block_rows = max(1, 2_000_000 // len(alphas))
    lower = config.price_floor if config.price_floor is not None else -np.inf
    upper = config.price_ceiling if config.price_ceiling is not None else np.inf
    for start in range(0, len(target), block_rows):
        stop = min(start + block_rows, len(target))
        y = target[start:stop, None]
        scaled = predictions[start:stop, None] * alphas[None, :]
        scaled = np.clip(scaled, lower, upper)
        denominator = np.abs(y) + np.abs(scaled)
        errors = np.zeros_like(scaled)
        np.divide(
            2.0 * np.abs(scaled - y),
            denominator,
            out=errors,
            where=denominator != 0.0,
        )
        totals += errors.sum(axis=0)
    return totals / len(target)


def optimize_smape_multiplier(
    y_true: Any,
    oof_predictions: Any,
    *,
    config: PostprocessConfig | None = None,
) -> CalibrationSolution:
    """Globally search a positive scalar multiplier on OOF SMAPE."""

    settings = config or PostprocessConfig()
    target = _finite_vector(y_true, label="y_true")
    if np.any(target < 0.0):
        raise ValueError("Price targets must be non-negative.")
    predictions = _finite_vector(
        oof_predictions, label="oof_predictions", rows=len(target)
    )
    score_before = _alpha_score(1.0, target, predictions, settings)
    if not settings.calibrate or np.all(predictions == 0.0):
        reason = (
            "Calibration disabled."
            if not settings.calibrate
            else "All predictions are zero; a scalar multiplier has no effect."
        )
        return CalibrationSolution(1.0, score_before, score_before, True, reason, 1)

    log_grid = np.linspace(
        math.log(settings.alpha_min),
        math.log(settings.alpha_max),
        settings.calibration_grid_size,
    )
    alphas = np.exp(log_grid)
    grid_scores = _grid_alpha_scores(alphas, target, predictions, settings)
    evaluations = len(alphas)
    candidates: list[tuple[float, float, str, bool]] = [
        (score_before, 1.0, "identity", True)
    ]
    for index in np.argsort(grid_scores)[: settings.local_searches]:
        candidates.append((float(grid_scores[index]), float(alphas[index]), "grid", True))
        left = max(0, int(index) - 1)
        right = min(len(log_grid) - 1, int(index) + 1)
        if left == right:
            continue
        result = minimize_scalar(
            lambda log_alpha: _alpha_score(
                math.exp(float(log_alpha)), target, predictions, settings
            ),
            bounds=(float(log_grid[left]), float(log_grid[right])),
            method="bounded",
            options={"xatol": settings.optimizer_tolerance, "maxiter": 500},
        )
        evaluations += int(getattr(result, "nfev", 0))
        alpha = math.exp(float(result.x))
        candidates.append(
            (float(result.fun), alpha, str(result.message), bool(result.success))
        )
    score_after, alpha, message, success = min(candidates, key=lambda item: item[0])
    if score_after > score_before + 1e-15:
        score_after, alpha, message, success = score_before, 1.0, "Identity fallback.", True
    return CalibrationSolution(
        float(alpha),
        float(score_before),
        float(score_after),
        bool(success),
        message,
        evaluations,
    )


def _validate_folds(values: Any, rows: int) -> np.ndarray:
    folds = np.asarray(values)
    if folds.ndim != 1 or len(folds) != rows:
        raise ValueError(f"folds must be one-dimensional with {rows} rows.")
    if pd.isna(folds).any():
        raise ValueError("folds contains missing values.")
    if len(pd.unique(folds)) < 2:
        raise ValueError("Cross-fitted calibration requires at least two folds.")
    return folds


def _cross_fitted_calibration(
    target: np.ndarray,
    predictions: np.ndarray,
    folds: np.ndarray,
    config: PostprocessConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    calibrated = np.full(len(target), np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    inner = replace(config, cross_fit=False)
    for fold in pd.unique(folds):
        valid = folds == fold
        train = ~valid
        solution = optimize_smape_multiplier(
            target[train], predictions[train], config=inner
        )
        calibrated[valid] = apply_postprocessing(
            predictions[valid], alpha=solution.alpha, config=config
        )
        reports.append(
            {
                "fold": _json_ready(fold),
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "alpha": solution.alpha,
                "valid_smape": smape(target[valid], calibrated[valid]),
            }
        )
    if not np.isfinite(calibrated).all():
        raise RuntimeError("Cross-fitted calibration did not assign every OOF row.")
    return calibrated, reports


def fit_postprocessor(
    y_true: Any,
    oof_predictions: Any,
    *,
    test_predictions: Any | None = None,
    weights: Any | None = None,
    ids: Sequence[Any] | None = None,
    test_ids: Sequence[Any] | None = None,
    folds: Any | None = None,
    id_column: str = "id",
    target_column: str = "target",
    fold_column: str = "fold",
    config: PostprocessConfig | None = None,
) -> PostprocessResult:
    """Blend, calibrate, clamp, and report OOF/test price predictions."""

    started = time.perf_counter()
    settings = config or PostprocessConfig()
    output_columns = [
        id_column,
        target_column,
        fold_column,
        "raw_prediction",
        "prediction",
        "cross_fitted_prediction",
    ]
    if any(not isinstance(column, str) or not column.strip() for column in output_columns[:3]):
        raise ValueError("id_column, target_column, and fold_column must be non-empty strings.")
    if len(set(output_columns)) != len(output_columns):
        raise ValueError(
            "ID, target, fold, raw_prediction, prediction, and "
            "cross_fitted_prediction column names must all differ."
        )
    target = _finite_vector(y_true, label="y_true")
    if np.any(target < 0.0):
        raise ValueError("Price targets must be non-negative.")
    oof_matrix = _prediction_matrix(oof_predictions, label="oof_predictions")
    if len(oof_matrix) != len(target):
        raise ValueError("OOF predictions and target row counts differ.")
    blend_weights = _simplex_weights(weights, oof_matrix.shape[1])
    positive_floor = settings.price_floor if settings.blend_method == "geometric" else None
    raw_oof = blend_predictions(
        oof_matrix,
        weights=blend_weights,
        method=settings.blend_method,
        positive_floor=positive_floor,
    )
    solution = optimize_smape_multiplier(target, raw_oof, config=settings)
    final_oof = apply_postprocessing(raw_oof, alpha=solution.alpha, config=settings)

    if ids is None:
        final_ids = np.arange(len(target))
    else:
        final_ids = np.asarray(ids, dtype=object)
        if final_ids.ndim != 1 or len(final_ids) != len(target):
            raise ValueError(f"ids must be one-dimensional with {len(target)} rows.")
        _validate_unique_ids(final_ids, label="OOF IDs")
    fold_values = _validate_folds(folds, len(target)) if folds is not None else None

    cross_prediction = None
    fold_reports: list[dict[str, Any]] = []
    if settings.cross_fit and fold_values is not None:
        cross_prediction, fold_reports = _cross_fitted_calibration(
            target, raw_oof, fold_values, settings
        )

    oof_frame = pd.DataFrame(
        {
            id_column: final_ids,
            target_column: target,
            "raw_prediction": raw_oof,
            "prediction": final_oof,
        }
    )
    if fold_values is not None:
        oof_frame[fold_column] = fold_values
    if cross_prediction is not None:
        oof_frame["cross_fitted_prediction"] = cross_prediction

    test_frame = None
    if test_predictions is not None:
        test_matrix = _prediction_matrix(test_predictions, label="test_predictions")
        if test_matrix.shape[1] != oof_matrix.shape[1]:
            raise ValueError(
                f"test_predictions has {test_matrix.shape[1]} columns; expected "
                f"{oof_matrix.shape[1]}."
            )
        raw_test = blend_predictions(
            test_matrix,
            weights=blend_weights,
            method=settings.blend_method,
            positive_floor=positive_floor,
        )
        final_test = apply_postprocessing(
            raw_test, alpha=solution.alpha, config=settings
        )
        if test_ids is None:
            final_test_ids = np.arange(len(test_matrix))
        else:
            final_test_ids = np.asarray(test_ids, dtype=object)
            if final_test_ids.ndim != 1 or len(final_test_ids) != len(test_matrix):
                raise ValueError(
                    f"test_ids must be one-dimensional with {len(test_matrix)} rows."
                )
            _validate_unique_ids(final_test_ids, label="test IDs")
        test_frame = pd.DataFrame(
            {
                id_column: final_test_ids,
                "raw_prediction": raw_test,
                "prediction": final_test,
            }
        )
    elif test_ids is not None:
        raise ValueError("test_ids cannot be supplied without test_predictions.")

    uncalibrated = apply_postprocessing(raw_oof, alpha=1.0, config=settings)
    report = {
        "postprocess_version": POSTPROCESS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": asdict(settings),
        "blend_method": settings.blend_method,
        "blend_weights": blend_weights.tolist(),
        "alpha": solution.alpha,
        "optimization_success": solution.success,
        "optimization_message": solution.message,
        "optimization_evaluations": solution.evaluations,
        "raw_oof_smape": smape(target, raw_oof),
        "uncalibrated_clamped_oof_smape": smape(target, uncalibrated),
        "calibrated_oof_smape": smape(target, final_oof),
        "cross_fitted_oof_smape": (
            smape(target, cross_prediction) if cross_prediction is not None else None
        ),
        "cross_fitted_folds": fold_reports,
        "oof_rows": int(len(target)),
        "test_rows": int(len(test_frame)) if test_frame is not None else 0,
        "oof_floor_clamped": int(
            np.sum(raw_oof * solution.alpha < settings.price_floor)
            if settings.price_floor is not None else 0
        ),
        "oof_ceiling_clamped": int(
            np.sum(raw_oof * solution.alpha > settings.price_ceiling)
            if settings.price_ceiling is not None else 0
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    return PostprocessResult(oof_frame, test_frame, _json_ready(report))


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


def _validate_unique_ids(values: Sequence[Any], *, label: str) -> list[str]:
    tokens = [_canonical_id(value) for value in values]
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{label} contains duplicate IDs.")
    return tokens


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


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input table must be CSV or Parquet.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def postprocess_from_files(
    oof_path: str | Path,
    *,
    test_path: str | Path | None = None,
    id_column: str,
    target_column: str,
    prediction_columns: Sequence[str] = ("prediction",),
    weights: Any | None = None,
    fold_column: str = "fold",
    config: PostprocessConfig | None = None,
) -> PostprocessResult:
    """Load train/ensemble artifacts and fit postprocessing strictly on OOF."""

    oof_file = Path(oof_path)
    oof = _load_table(oof_file)
    columns = tuple(str(column) for column in prediction_columns)
    if not columns or any(not column for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("prediction_columns must be unique and non-empty.")
    overlap = set(columns) & {id_column, target_column, fold_column}
    if overlap:
        raise ValueError(
            f"Prediction columns cannot include ID, target, or fold columns: {sorted(overlap)}."
        )
    required = {id_column, target_column, *columns}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF input is missing columns: {missing}.")
    original_rows = len(oof)
    if fold_column in oof.columns:
        eligible = _evaluated_fold_mask(oof[fold_column], label="OOF folds")
        oof = oof.loc[eligible].reset_index(drop=True)
    dropped = original_rows - len(oof)
    _validate_unique_ids(oof[id_column].tolist(), label="OOF IDs")

    test = None
    test_file = Path(test_path) if test_path is not None else None
    if test_file is not None:
        test = _load_table(test_file)
        test_required = {id_column, *columns}
        missing = sorted(test_required - set(test.columns))
        if missing:
            raise ValueError(f"Test input is missing columns: {missing}.")
        _validate_unique_ids(test[id_column].tolist(), label="test IDs")

    result = fit_postprocessor(
        oof[target_column],
        oof.loc[:, columns].to_numpy(),
        test_predictions=test.loc[:, columns].to_numpy() if test is not None else None,
        weights=weights,
        ids=oof[id_column].to_numpy(dtype=object),
        test_ids=test[id_column].to_numpy(dtype=object) if test is not None else None,
        folds=oof[fold_column].to_numpy() if fold_column in oof.columns else None,
        id_column=id_column,
        target_column=target_column,
        fold_column=fold_column,
        config=config,
    )
    result.report["dropped_unevaluated_oof_rows"] = int(dropped)
    result.report["prediction_columns"] = list(columns)
    result.report["sources"] = {
        "oof": {"path": str(oof_file.resolve()), "sha256": _sha256_file(oof_file)},
        "test": (
            {"path": str(test_file.resolve()), "sha256": _sha256_file(test_file)}
            if test_file is not None else None
        ),
    }
    signature = {
        "config": result.report["config"],
        "sources": result.report["sources"],
        "alpha": result.report["alpha"],
        "weights": result.report["blend_weights"],
    }
    result.report["run_signature"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
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


def save_postprocess_result(
    result: PostprocessResult,
    output_directory: str | Path,
    *,
    target_column: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_directory)
    outputs = {
        "calibration": destination / "calibration.json",
        "oof": destination / "postprocess_oof.parquet",
        "report": destination / "postprocess_report.json",
    }
    if result.test_predictions is not None:
        outputs["test_predictions"] = destination / "postprocess_test_predictions.parquet"
        outputs["submission"] = destination / "submission.csv"
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing postprocess artifacts: {[str(path) for path in existing]}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {
            "postprocess_version": POSTPROCESS_VERSION,
            "alpha": result.report["alpha"],
            "blend_method": result.report["blend_method"],
            "blend_weights": result.report["blend_weights"],
            "price_floor": result.report["config"]["price_floor"],
            "price_ceiling": result.report["config"]["price_ceiling"],
            "run_signature": result.report.get("run_signature"),
        },
        outputs["calibration"],
    )
    _atomic_table(result.oof, outputs["oof"])
    _atomic_json(result.report, outputs["report"])
    if result.test_predictions is not None:
        _atomic_table(result.test_predictions, outputs["test_predictions"])
        submission = result.test_predictions[[result.test_predictions.columns[0], "prediction"]].rename(
            columns={"prediction": target_column}
        )
        _atomic_table(submission, outputs["submission"])
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate OOF price predictions and create a domain-clamped submission."
    )
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--prediction-columns", nargs="+", default=["prediction"])
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--blend-method", choices=("arithmetic", "geometric"), default="arithmetic")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--calibration-grid-size", type=int, default=129)
    parser.add_argument("--local-searches", type=int, default=3)
    parser.add_argument("--price-floor", type=float, default=0.0)
    parser.add_argument("--price-ceiling", type=float)
    parser.add_argument("--no-cross-fit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = PostprocessConfig(
        blend_method=args.blend_method,
        calibrate=not args.no_calibration,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        calibration_grid_size=args.calibration_grid_size,
        local_searches=args.local_searches,
        price_floor=args.price_floor,
        price_ceiling=args.price_ceiling,
        cross_fit=not args.no_cross_fit,
    )
    result = postprocess_from_files(
        args.oof,
        test_path=args.test,
        id_column=args.id_column,
        target_column=args.target_column,
        prediction_columns=args.prediction_columns,
        weights=args.weights,
        fold_column=args.fold_column,
        config=config,
    )
    outputs = save_postprocess_result(
        result,
        args.output_dir,
        target_column=args.target_column,
        overwrite=args.overwrite,
    )
    print(
        f"alpha={result.report['alpha']:.10f} | "
        f"OOF SMAPE {result.report['uncalibrated_clamped_oof_smape']:.8f} -> "
        f"{result.report['calibrated_oof_smape']:.8f} | "
        f"cross-fitted={result.report['cross_fitted_oof_smape']}"
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
