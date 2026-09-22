"""Authenticated standalone inference from saved cross-validation folds.

This module loads one fold at a time from :class:`models.FoldModelStore`,
replays that fold's fitted preprocessor on raw catalog rows, and aggregates the
predictions. It never needs the training target and never refits any component.
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
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Support both ``python -m src.predict`` and ``python src/predict.py``.
    from .models import FoldModelStore, InferenceFoldBundle
    from .train import (
        TrainingConfig,
        _restore_predictions,
        build_feature_frame,
        validate_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from models import FoldModelStore, InferenceFoldBundle
    from train import TrainingConfig, _restore_predictions, build_feature_frame, validate_config


PREDICT_VERSION = "1.0"
Aggregation = Literal["mean", "median", "geometric"]


@dataclass(frozen=True)
class PredictionConfig:
    batch_size: int = 4_096
    aggregation: Aggregation = "mean"

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be an integer.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.aggregation not in {"mean", "median", "geometric"}:
            raise ValueError("aggregation must be mean, median, or geometric.")


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    report: dict[str, Any]
    fold_predictions: pd.DataFrame | None = None


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
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
        raise ValueError("Inference IDs cannot contain missing values.")
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"number:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Inference IDs cannot contain NaN or infinity.")
        return f"number:{int(numeric)}" if numeric.is_integer() else f"number:{numeric:.17g}"
    text = str(value).strip()
    if not text:
        raise ValueError("Inference IDs cannot contain empty strings.")
    return f"text:{text}"


def _training_config(payload: Mapping[str, Any]) -> TrainingConfig:
    allowed = {item.name for item in fields(TrainingConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RuntimeError(f"Saved training config contains unknown fields: {unknown}.")
    values = dict(payload)
    for key in ("text_columns", "categorical_columns", "numeric_columns"):
        if key in values:
            values[key] = tuple(values[key])
    config = TrainingConfig(**values)
    validate_config(config)
    return config


def attach_inference_feature_table(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    *,
    id_column: str,
    prefix: str,
    fold_column: str = "fold",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Strictly align and prefix one numeric inference feature table by ID."""

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("Feature prefix must start with a letter and be alphanumeric/underscore.")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("raw must be a non-empty DataFrame.")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty DataFrame.")
    if id_column not in raw or id_column not in features:
        raise ValueError(f"Both tables must contain ID column {id_column!r}.")
    raw_tokens = [_canonical_id(value) for value in raw[id_column]]
    feature_tokens = [_canonical_id(value) for value in features[id_column]]
    if len(set(raw_tokens)) != len(raw_tokens):
        raise ValueError("Raw inference IDs must be unique.")
    if len(set(feature_tokens)) != len(feature_tokens):
        raise ValueError("Inference feature IDs must be unique.")
    if set(raw_tokens) != set(feature_tokens):
        missing = len(set(raw_tokens) - set(feature_tokens))
        extra = len(set(feature_tokens) - set(raw_tokens))
        raise ValueError(f"Inference feature ID set mismatch: missing={missing}, extra={extra}.")
    positions = {token: index for index, token in enumerate(feature_tokens)}
    aligned = features.iloc[[positions[token] for token in raw_tokens]].reset_index(drop=True)
    source_columns = [column for column in aligned if column not in {id_column, fold_column}]
    if not source_columns:
        raise ValueError("Inference feature table contains no usable columns.")
    data: dict[str, np.ndarray] = {}
    added: list[str] = []
    for column in source_columns:
        numeric = pd.to_numeric(aligned[column], errors="coerce")
        invalid = aligned[column].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Inference feature {column!r} is not finite numeric data.")
        renamed = f"{prefix}__{column}"
        if renamed in raw.columns or renamed in data:
            raise ValueError(f"Inference feature name collision: {renamed!r}.")
        data[renamed] = numeric.to_numpy()
        added.append(renamed)
    block = pd.DataFrame(data, index=raw.index)
    joined = pd.concat([raw.reset_index(drop=True), block.reset_index(drop=True)], axis=1)
    return joined, tuple(added)


def _read_bundle_metadata(store: FoldModelStore, fold_ids: Sequence[int]) -> dict[str, Any]:
    metadata: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        path = store.path_for(fold_id).with_suffix(".joblib.metadata.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable fold sidecar: {path}") from exc
        if payload.get("artifact_type") != "inference_fold_bundle":
            raise RuntimeError(f"Fold {fold_id} is not a raw-inference bundle.")
        metadata.append(payload)
    signatures = {item.get("run_signature") for item in metadata}
    config_signatures = {item.get("training_config_signature") for item in metadata}
    if len(signatures) != 1 or None in signatures:
        raise RuntimeError("Fold artifacts do not share one run signature.")
    if len(config_signatures) != 1 or None in config_signatures:
        raise RuntimeError("Fold artifacts do not share one training configuration.")
    return {
        "run_signature": next(iter(signatures)),
        "training_config_signature": next(iter(config_signatures)),
        "fold_metadata": metadata,
    }


def _validate_raw(frame: pd.DataFrame, config: TrainingConfig) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Inference input must be a non-empty DataFrame.")
    required = {
        config.id_column,
        *config.text_columns,
        *config.categorical_columns,
        *config.numeric_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Inference input is missing required feature columns: {missing}.")
    tokens = [_canonical_id(value) for value in frame[config.id_column]]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Inference IDs must be unique.")


def _aligned_probabilities(
    model: Any,
    matrix: Any,
    class_labels: Sequence[Any],
) -> np.ndarray:
    if not callable(getattr(model, "predict_proba", None)) or not hasattr(model, "classes_"):
        raise RuntimeError("Saved classifier must expose predict_proba and classes_.")
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if raw.ndim == 1 and len(class_labels) == 2:
        raw = np.column_stack([1.0 - raw, raw])
    model_classes = list(model.classes_)
    aligned = np.zeros((matrix.shape[0], len(class_labels)), dtype=np.float64)
    for target_index, label in enumerate(class_labels):
        matches = [index for index, value in enumerate(model_classes) if value == label]
        if len(matches) != 1:
            raise RuntimeError(f"Saved classifier class mismatch for {label!r}.")
        aligned[:, target_index] = raw[:, matches[0]]
    row_sums = aligned.sum(axis=1)
    if (
        not np.isfinite(aligned).all()
        or np.any(aligned < -1e-12)
        or np.any(row_sums <= 0.0)
    ):
        raise RuntimeError("Saved classifier produced invalid probabilities.")
    aligned = np.clip(aligned, 0.0, 1.0)
    return aligned / aligned.sum(axis=1, keepdims=True)


def _predict_one_fold(
    bundle: InferenceFoldBundle,
    raw: pd.DataFrame,
    training: TrainingConfig,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    started = time.perf_counter()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        feature_frame = build_feature_frame(raw.iloc[start:stop], training)
        matrix = bundle.preprocessor.transform(feature_frame)
        if matrix.ndim != 2 or matrix.shape[1] != bundle.feature_count:
            raise RuntimeError(
                f"Fold {bundle.fold_id} preprocessor produced {matrix.shape[1]} features; "
                f"expected {bundle.feature_count}."
            )
        if training.task == "regression":
            prediction = _restore_predictions(bundle.model.predict(matrix), training)
            predictions.append(np.asarray(prediction, dtype=np.float64))
        else:
            probability = _aligned_probabilities(bundle.model, matrix, bundle.class_labels)
            probabilities.append(probability)
        del feature_frame, matrix
    if training.task == "regression":
        result = np.concatenate(predictions)
        if result.shape != (len(raw),) or not np.isfinite(result).all():
            raise RuntimeError(f"Fold {bundle.fold_id} produced invalid regression predictions.")
        return result, None, time.perf_counter() - started
    probability_result = np.vstack(probabilities)
    if probability_result.shape != (len(raw), len(bundle.class_labels)):
        raise RuntimeError(f"Fold {bundle.fold_id} produced invalid probability dimensions.")
    labels = np.asarray(bundle.class_labels, dtype=object)[np.argmax(probability_result, axis=1)]
    return labels, probability_result, time.perf_counter() - started


def _aggregate_regression(values: Sequence[np.ndarray], method: Aggregation) -> np.ndarray:
    matrix = np.vstack(values).astype(np.float64)
    if method == "mean":
        result = matrix.mean(axis=0)
    elif method == "median":
        result = np.median(matrix, axis=0)
    else:
        if np.any(matrix < 0.0):
            raise ValueError("Geometric aggregation requires non-negative fold predictions.")
        result = np.exp(np.mean(np.log(np.maximum(matrix, 1e-12)), axis=0))
        result[np.all(matrix == 0.0, axis=0)] = 0.0
    if not np.isfinite(result).all():
        raise RuntimeError("Fold aggregation produced non-finite predictions.")
    return result


def predict_from_store(
    raw: pd.DataFrame,
    model_directory: str | Path,
    prediction_config: PredictionConfig | None = None,
    *,
    expected_run_signature: str | None = None,
    include_fold_predictions: bool = False,
) -> PredictionResult:
    """Generate predictions without training or target access."""

    settings = prediction_config or PredictionConfig()
    started = time.perf_counter()
    store = FoldModelStore(model_directory)
    fold_ids = store.completed_folds()
    if not fold_ids:
        raise FileNotFoundError(f"No authenticated fold bundles found in {store.directory}.")
    if fold_ids != list(range(len(fold_ids))):
        raise RuntimeError(f"Fold IDs must be contiguous from zero; found {fold_ids}.")
    discovery = _read_bundle_metadata(store, fold_ids)
    run_signature = str(discovery["run_signature"])
    if expected_run_signature is not None and run_signature != expected_run_signature:
        raise RuntimeError(
            f"Run signature mismatch: artifacts={run_signature!r}, expected={expected_run_signature!r}."
        )

    first = store.load_inference_bundle(fold_ids[0], run_signature)
    training = _training_config(first.training_config)
    if training.n_splits != len(fold_ids):
        raise RuntimeError(
            f"Expected {training.n_splits} saved folds from training config; found {len(fold_ids)}."
        )
    if training.task == "classification" and settings.aggregation != "mean":
        raise ValueError("Classification supports probability-mean aggregation only.")
    _validate_raw(raw, training)

    fold_values: list[np.ndarray] = []
    fold_probabilities: list[np.ndarray] = []
    fold_reports: list[dict[str, Any]] = []
    reference_labels = first.class_labels
    reference_config = first.training_config
    for fold_id in fold_ids:
        bundle = first if fold_id == fold_ids[0] else store.load_inference_bundle(
            fold_id, run_signature
        )
        if bundle.training_config != reference_config:
            raise RuntimeError(f"Fold {fold_id} training configuration differs from fold 0.")
        if bundle.class_labels != reference_labels:
            raise RuntimeError(f"Fold {fold_id} class labels differ from fold 0.")
        prediction, probability, seconds = _predict_one_fold(
            bundle, raw, training, settings.batch_size
        )
        fold_values.append(prediction)
        if probability is not None:
            fold_probabilities.append(probability)
        fold_reports.append({
            "fold": fold_id,
            "rows": len(raw),
            "feature_count": bundle.feature_count,
            "model_type": f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}",
            "seconds": float(seconds),
        })
        if fold_id == fold_ids[0]:
            first = None  # type: ignore[assignment]
        del bundle
        gc.collect()

    if training.task == "regression":
        final_prediction = _aggregate_regression(fold_values, settings.aggregation)
        predictions = pd.DataFrame({
            training.id_column: raw[training.id_column].to_numpy(),
            "prediction": final_prediction,
        })
        fold_frame = None
        if include_fold_predictions:
            fold_frame = pd.DataFrame({training.id_column: raw[training.id_column].to_numpy()})
            for fold_id, values in zip(fold_ids, fold_values):
                fold_frame[f"prediction_fold_{fold_id:03d}"] = values
    else:
        averaged = np.mean(np.stack(fold_probabilities, axis=0), axis=0)
        label_array = np.asarray(reference_labels, dtype=object)
        final_prediction = label_array[np.argmax(averaged, axis=1)]
        predictions = pd.DataFrame({
            training.id_column: raw[training.id_column].to_numpy(),
            "prediction": final_prediction,
        })
        for index, _label in enumerate(reference_labels):
            predictions[f"probability_{index}"] = averaged[:, index]
        fold_frame = None

    report = {
        "predict_version": PREDICT_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_signature": run_signature,
        "training_config": _json_ready(asdict(training)),
        "prediction_config": asdict(settings),
        "rows": int(len(raw)),
        "fold_count": len(fold_ids),
        "folds": fold_reports,
        "artifacts": [
            {
                "fold": int(item["fold_id"]),
                "file_sha256": item["file_sha256"],
                "file_size": int(item["file_size"]),
                "training_config_signature": item["training_config_signature"],
            }
            for item in discovery["fold_metadata"]
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return PredictionResult(predictions, _json_ready(report), fold_frame)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
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


def save_prediction_result(
    result: PredictionResult,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    destination = Path(output_directory)
    target_column = str(result.report["training_config"]["target_column"])
    outputs: dict[str, Path] = {
        "predictions": destination / "predictions.parquet",
        "submission": destination / "submission.csv",
        "report": destination / "prediction_report.json",
    }
    if result.fold_predictions is not None:
        outputs["fold_predictions"] = destination / "fold_predictions.parquet"
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prediction artifacts: {[str(p) for p in existing]}.")
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_table(result.predictions, outputs["predictions"])
    submission = result.predictions.rename(columns={"prediction": target_column})
    probability_columns = [column for column in submission if column.startswith("probability_")]
    submission = submission.drop(columns=probability_columns)
    _atomic_table(submission, outputs["submission"])
    _atomic_text(
        json.dumps(result.report, indent=2, ensure_ascii=False, allow_nan=False),
        outputs["report"],
    )
    if result.fold_predictions is not None:
        _atomic_table(result.fold_predictions, outputs["fold_predictions"])
    return outputs


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be CSV or Parquet.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run authenticated offline inference from saved CV fold bundles."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engineered-features", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--image-embeddings", type=Path)
    parser.add_argument("--expected-run-signature")
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--aggregation", choices=("mean", "median", "geometric"), default="mean")
    parser.add_argument("--save-fold-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    expected_outputs = [
        args.output_dir / "predictions.parquet",
        args.output_dir / "submission.csv",
        args.output_dir / "prediction_report.json",
    ]
    if args.save_fold_predictions:
        expected_outputs.append(args.output_dir / "fold_predictions.parquet")
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to start inference because outputs exist: {[str(path) for path in existing]}."
        )
    raw = _load_table(args.input)

    # Read only the first authenticated sidecar to discover the saved schema
    # before attaching optional auxiliary tables.
    store = FoldModelStore(args.models_dir)
    folds = store.completed_folds()
    if not folds:
        raise FileNotFoundError(f"No authenticated fold bundles found in {args.models_dir}.")
    discovery = _read_bundle_metadata(store, folds)
    first = store.load_inference_bundle(folds[0], discovery["run_signature"])
    training = _training_config(first.training_config)
    auxiliary_sources: list[dict[str, Any]] = []
    for label, path, prefix in (
        ("engineered_features", args.engineered_features, "eng"),
        ("embeddings", args.embeddings, "emb"),
        ("image_embeddings", args.image_embeddings, "img"),
    ):
        if path is None:
            continue
        raw, added = attach_inference_feature_table(
            raw,
            _load_table(path),
            id_column=training.id_column,
            prefix=prefix,
            fold_column=training.fold_column,
        )
        auxiliary_sources.append({
            "name": label,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "features": len(added),
        })
    del first
    result = predict_from_store(
        raw,
        args.models_dir,
        PredictionConfig(batch_size=args.batch_size, aggregation=args.aggregation),
        expected_run_signature=args.expected_run_signature,
        include_fold_predictions=args.save_fold_predictions,
    )
    result.report["sources"] = {
        "input": {"path": str(args.input.resolve()), "sha256": _sha256_file(args.input)},
        "models_directory": str(args.models_dir.resolve()),
        "auxiliary": auxiliary_sources,
    }
    outputs = save_prediction_result(result, args.output_dir, overwrite=args.overwrite)
    print(
        f"Predicted {len(result.predictions):,} rows with {result.report['fold_count']} folds "
        f"in {result.report['elapsed_seconds']:.2f}s."
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
