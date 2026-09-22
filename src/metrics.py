"""Competition metrics with strict validation and direction metadata.

The challenge's released metric definition is always the source of truth. This
module provides safe, well-tested implementations of common regression and
classification metrics and makes any percentage scaling explicit.

All public metric functions:

* accept one-dimensional array-like inputs (lists, NumPy arrays, Series),
* reject empty, non-finite, multidimensional, or differently shaped inputs,
* return a Python ``float``, and
* document whether higher or lower values are better through METRIC_REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)

MetricFunction = Callable[..., float]
MapeZeroPolicy = Literal["epsilon", "ignore", "raise"]


def _validate_regression_inputs(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, non-empty, equally shaped 1-D float64 arrays."""
    try:
        y = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(y_pred, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Regression targets and predictions must be numeric.") from exc

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(
            "Regression metrics require 1-D inputs; "
            f"received shapes {y.shape} and {p.shape}."
        )
    if y.shape != p.shape:
        raise ValueError(
            f"Target/prediction shape mismatch: {y.shape} != {p.shape}."
        )
    if y.size == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_true contains NaN or infinity.")
    if not np.all(np.isfinite(p)):
        raise ValueError("y_pred contains NaN or infinity.")
    return y, p


def _contains_missing_or_nonfinite(values: np.ndarray) -> bool:
    """Check classification labels without requiring them to be numeric."""
    for value in values:
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return True
    return False


def _validate_classification_inputs(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-empty, equally shaped 1-D arrays of class labels."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(
            "Classification metrics require 1-D class-label inputs; "
            f"received shapes {y.shape} and {p.shape}."
        )
    if y.shape != p.shape:
        raise ValueError(
            f"Target/prediction shape mismatch: {y.shape} != {p.shape}."
        )
    if y.size == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if _contains_missing_or_nonfinite(y):
        raise ValueError("y_true contains a missing or non-finite class label.")
    if _contains_missing_or_nonfinite(p):
        raise ValueError("y_pred contains a missing or non-finite class label.")
    return y, p


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def smape(y_true: Any, y_pred: Any, *, percentage: bool = False) -> float:
    """Return symmetric mean absolute percentage error.

    ``mean(2 * abs(pred - true) / (abs(true) + abs(pred)))``

    When both values are zero, that row contributes zero error. No epsilon is
    added to valid non-zero denominators, so very small targets remain exact.
    The default result is a fraction in [0, 2]. Set ``percentage=True`` for a
    percentage in [0, 200]. Lower is better.
    """
    y, p = _validate_regression_inputs(y_true, y_pred)
    denominator = np.abs(y) + np.abs(p)
    numerator = 2.0 * np.abs(p - y)
    row_errors = np.zeros_like(denominator, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=row_errors,
        where=denominator != 0.0,
    )
    result = float(np.mean(row_errors))
    return result * 100.0 if percentage else result


def smape_percent(y_true: Any, y_pred: Any) -> float:
    """Return SMAPE expressed as a percentage in [0, 200]."""
    return smape(y_true, y_pred, percentage=True)


def custom_log_smape_objective(
    y_true: Any,
    y_pred: Any,
    *,
    soft_sign_delta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """First and second order gradients of SMAPE in log-space for tree boosting.

    Let t = ln(1 + y), y_hat = ln(1 + y_pred), and residual Delta = y_hat - t.
    SMAPE(y, y_hat) = 2 * tanh(|Delta| / 2).
    This function computes:
      g = tanh(Delta / soft_sign_delta) * sech^2(Delta / 2)
      h = sech^2(Delta / 2) + 1e-2
    Outliers (|Delta| >> 0) naturally experience up to 95% gradient suppression,
    protecting decision tree splits from extreme target distortions.
    """
    y, p = _validate_regression_inputs(y_true, y_pred)
    delta = p - y
    th = np.tanh(np.abs(delta) / 2.0)
    sech2 = np.maximum(1e-4, 1.0 - th ** 2)
    soft_sign = np.tanh(delta / max(1e-4, soft_sign_delta))
    grad = soft_sign * sech2
    hess = sech2 + 1e-2
    return grad, hess


def mape(
    y_true: Any,
    y_pred: Any,
    *,
    zero_policy: MapeZeroPolicy = "epsilon",
    eps: float = 1e-8,
    percentage: bool = False,
) -> float:
    """Return mean absolute percentage error.

    ``zero_policy`` makes the ambiguous ``y_true == 0`` behavior explicit:

    * ``"epsilon"``: divide zero targets by ``eps`` (default).
    * ``"ignore"``: exclude rows whose true value is zero.
    * ``"raise"``: reject inputs containing a zero true value.

    The default result is a fraction. Set ``percentage=True`` to multiply the
    result by 100. Lower is better. Match these options to the challenge's
    released definition before using MAPE for model selection.
    """
    y, p = _validate_regression_inputs(y_true, y_pred)
    if zero_policy not in {"epsilon", "ignore", "raise"}:
        raise ValueError(
            "zero_policy must be one of: 'epsilon', 'ignore', or 'raise'."
        )
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite number.")

    zero_mask = y == 0.0
    if zero_policy == "raise" and np.any(zero_mask):
        raise ValueError("MAPE is undefined when y_true contains zero.")
    if zero_policy == "ignore":
        keep = ~zero_mask
        if not np.any(keep):
            raise ValueError("MAPE has no rows to score after ignoring zero targets.")
        y = y[keep]
        p = p[keep]
        denominator = np.abs(y)
    else:
        denominator = np.maximum(np.abs(y), eps)

    result = float(np.mean(np.abs(y - p) / denominator))
    return result * 100.0 if percentage else result


def mape_percent(y_true: Any, y_pred: Any) -> float:
    """Return default-policy MAPE expressed as a percentage."""
    return mape(y_true, y_pred, percentage=True)


def mae(y_true: Any, y_pred: Any) -> float:
    """Return mean absolute error. Lower is better."""
    y, p = _validate_regression_inputs(y_true, y_pred)
    return float(mean_absolute_error(y, p))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Return root mean squared error. Lower is better."""
    y, p = _validate_regression_inputs(y_true, y_pred)
    return float(root_mean_squared_error(y, p))


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------

def accuracy(y_true: Any, y_pred: Any) -> float:
    """Return classification accuracy in [0, 1]. Greater is better."""
    y, p = _validate_classification_inputs(y_true, y_pred)
    return float(accuracy_score(y, p))


def f1_micro(y_true: Any, y_pred: Any) -> float:
    """Return micro-averaged F1 in [0, 1]. Greater is better."""
    y, p = _validate_classification_inputs(y_true, y_pred)
    return float(f1_score(y, p, average="micro", zero_division=0))


def f1_macro(y_true: Any, y_pred: Any) -> float:
    """Return unweighted mean F1 across observed classes. Greater is better."""
    y, p = _validate_classification_inputs(y_true, y_pred)
    return float(f1_score(y, p, average="macro", zero_division=0))


def f1_weighted(y_true: Any, y_pred: Any) -> float:
    """Return support-weighted mean F1. Greater is better."""
    y, p = _validate_classification_inputs(y_true, y_pred)
    return float(f1_score(y, p, average="weighted", zero_division=0))


# ---------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """Metadata needed by training, tuning, and early-stopping code."""

    fn: MetricFunction
    greater_is_better: bool
    description: str
    output_scale: str

    def __getitem__(self, key: str) -> Any:
        """Preserve the original ``registry[name][key]`` access style."""
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


METRIC_REGISTRY: Mapping[str, MetricSpec] = {
    "smape": MetricSpec(smape, False, "Symmetric MAPE", "fraction [0, 2]"),
    "smape_percent": MetricSpec(
        smape_percent, False, "Symmetric MAPE", "percentage [0, 200]"
    ),
    "mape": MetricSpec(mape, False, "Mean absolute percentage error", "fraction"),
    "mape_percent": MetricSpec(
        mape_percent, False, "Mean absolute percentage error", "percentage"
    ),
    "mae": MetricSpec(mae, False, "Mean absolute error", "target units"),
    "rmse": MetricSpec(rmse, False, "Root mean squared error", "target units"),
    "accuracy": MetricSpec(accuracy, True, "Classification accuracy", "fraction [0, 1]"),
    "f1_micro": MetricSpec(f1_micro, True, "Micro F1", "fraction [0, 1]"),
    "f1_macro": MetricSpec(f1_macro, True, "Macro F1", "fraction [0, 1]"),
    "f1_weighted": MetricSpec(
        f1_weighted, True, "Support-weighted F1", "fraction [0, 1]"
    ),
}


def get_metric_spec(name: str) -> MetricSpec:
    """Return the complete specification for a case-insensitive metric name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Metric name must be a non-empty string.")
    clean_name = name.strip().lower()
    try:
        return METRIC_REGISTRY[clean_name]
    except KeyError as exc:
        available = ", ".join(sorted(METRIC_REGISTRY))
        raise ValueError(
            f"Unknown metric '{name}'. Available metrics: {available}."
        ) from exc


def get_metric(name: str) -> tuple[MetricFunction, bool]:
    """Return ``(function, greater_is_better)`` for backward compatibility."""
    spec = get_metric_spec(name)
    return spec.fn, spec.greater_is_better


def evaluate_metric(name: str, y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Evaluate a registered metric, optionally passing metric-specific kwargs."""
    return get_metric_spec(name).fn(y_true, y_pred, **kwargs)


def _run_smoke_tests() -> None:
    """Dependency-free checks for quickly verifying a competition machine."""
    expected_smape = (2 / 11 + 2 / 7 + 0.0) / 3.0
    assert np.isclose(smape([10, 20, 0], [12, 15, 0]), expected_smape)
    assert np.isclose(mape([10, 20, 50], [12, 15, 50]), 0.15)
    assert np.isclose(f1_macro([0, 1, 1, 2], [0, 1, 0, 2]), 7 / 9)
    assert smape([0, 0], [0, 0]) == 0.0
    assert np.isclose(smape([1e-12], [2e-12]), 2 / 3)


if __name__ == "__main__":
    _run_smoke_tests()
    print(f"All metric smoke tests passed ({len(METRIC_REGISTRY)} metrics registered).")
