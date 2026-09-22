"""Fatal submission validation for ML competitions.

The sample submission is the schema contract. This module validates an output
file before upload, writes a machine-readable report, and exits non-zero for
fatal problems. It never edits the candidate submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

Severity = Literal["warning", "error"]
Task = Literal["auto", "regression", "classification"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    column: str | None = None
    value: Any = None
    recommendation: str | None = None


class SubmissionValidationError(ValueError):
    """Raised by ``assert_valid_submission`` for a fatal report."""


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def _add(
    issues: list[ValidationIssue],
    severity: Severity,
    code: str,
    message: str,
    **kwargs: Any,
) -> None:
    issues.append(ValidationIssue(severity, code, message, **kwargs))


def _column_tuple(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(value)
    if any(not isinstance(column, str) or not column for column in result):
        raise ValueError("Target columns must be non-empty strings.")
    if len(result) != len(set(result)):
        raise ValueError("Target columns cannot contain duplicates.")
    return result


def _strict_id_values(series: pd.Series) -> pd.Series:
    """Preserve IDs as strings while treating missing values explicitly."""
    return series.astype("string")


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    id_column: str | None = None,
    target_columns: str | Sequence[str] | None = None,
    task: Task = "auto",
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    require_id_order: bool = True,
    probabilities: bool = False,
    probability_tolerance: float = 1e-6,
    train: pd.DataFrame | None = None,
    extreme_factor: float = 10.0,
) -> dict[str, Any]:
    """Validate a submission DataFrame against the sample contract.

    Fatal problems are returned as ``severity='error'``. Distribution clues
    such as constant predictions are warnings because they may be intentional.
    """
    if not isinstance(submission, pd.DataFrame):
        raise TypeError("submission must be a pandas DataFrame.")
    if not isinstance(sample_submission, pd.DataFrame):
        raise TypeError("sample_submission must be a pandas DataFrame.")
    if sample_submission.empty:
        raise ValueError("sample_submission cannot be empty.")
    if task not in {"auto", "regression", "classification"}:
        raise ValueError("task must be 'auto', 'regression', or 'classification'.")
    if probability_tolerance <= 0 or not np.isfinite(probability_tolerance):
        raise ValueError("probability_tolerance must be positive and finite.")
    if extreme_factor <= 1 or not np.isfinite(extreme_factor):
        raise ValueError("extreme_factor must be finite and greater than 1.")
    if minimum_value is not None and maximum_value is not None:
        if minimum_value > maximum_value:
            raise ValueError("minimum_value cannot exceed maximum_value.")

    sample_columns = list(sample_submission.columns)
    resolved_id = id_column or sample_columns[0]
    resolved_targets = _column_tuple(target_columns) or tuple(
        column for column in sample_columns if column != resolved_id
    )
    if resolved_id not in sample_submission.columns:
        raise ValueError(f"ID column '{resolved_id}' is absent from the sample submission.")
    if not resolved_targets:
        raise ValueError("At least one target column is required.")
    missing_sample_targets = [
        column for column in resolved_targets if column not in sample_submission.columns
    ]
    if missing_sample_targets:
        raise ValueError(
            f"Target columns are absent from sample submission: {missing_sample_targets}."
        )
    if resolved_id in resolved_targets:
        raise ValueError("ID column cannot also be a target column.")

    if task == "auto":
        all_numeric = all(
            pd.api.types.is_numeric_dtype(sample_submission[column])
            for column in resolved_targets
        )
        resolved_task: Literal["regression", "classification"] = (
            "regression" if all_numeric and not probabilities else "classification"
        )
    else:
        resolved_task = task

    issues: list[ValidationIssue] = []
    checks: dict[str, Any] = {}
    actual_columns = list(submission.columns)
    checks["expected_columns"] = sample_columns
    checks["actual_columns"] = actual_columns

    if submission.columns.has_duplicates:
        duplicates = submission.columns[submission.columns.duplicated()].tolist()
        _add(
            issues,
            "error",
            "DUPLICATE_COLUMNS",
            f"Submission contains duplicate column names: {duplicates}.",
            recommendation="Write each required column exactly once.",
        )
    if actual_columns != sample_columns:
        if set(actual_columns) == set(sample_columns) and len(actual_columns) == len(sample_columns):
            _add(
                issues,
                "error",
                "COLUMN_ORDER_MISMATCH",
                f"Expected column order {sample_columns}, received {actual_columns}.",
                recommendation="Reorder columns to exactly match sample_submission.csv.",
            )
        else:
            missing = [column for column in sample_columns if column not in actual_columns]
            extra = [column for column in actual_columns if column not in sample_columns]
            _add(
                issues,
                "error",
                "COLUMN_SCHEMA_MISMATCH",
                f"Submission columns differ from sample; missing={missing}, extra={extra}.",
                recommendation="Use the sample submission columns exactly, without an index column.",
            )

    checks["expected_rows"] = int(len(sample_submission))
    checks["actual_rows"] = int(len(submission))
    if len(submission) != len(sample_submission):
        _add(
            issues,
            "error",
            "ROW_COUNT_MISMATCH",
            f"Expected {len(sample_submission)} rows, received {len(submission)}.",
            value=len(submission),
            recommendation="Produce exactly one prediction for every sample ID.",
        )

    if resolved_id not in submission.columns:
        _add(
            issues,
            "error",
            "ID_COLUMN_MISSING",
            f"ID column '{resolved_id}' is absent.",
            column=resolved_id,
        )
    else:
        submitted_ids = _strict_id_values(submission[resolved_id])
        expected_ids = _strict_id_values(sample_submission[resolved_id])
        missing_id_rows = int(submitted_ids.isna().sum())
        duplicate_id_rows = int(submitted_ids.duplicated(keep=False).sum())
        checks["missing_id_rows"] = missing_id_rows
        checks["duplicate_id_rows"] = duplicate_id_rows
        if missing_id_rows:
            _add(
                issues,
                "error",
                "MISSING_IDS",
                f"ID column contains {missing_id_rows} missing values.",
                column=resolved_id,
            )
        if duplicate_id_rows:
            _add(
                issues,
                "error",
                "DUPLICATE_IDS",
                f"{duplicate_id_rows} rows participate in duplicate IDs.",
                column=resolved_id,
                recommendation="There must be exactly one row per test ID.",
            )

        submitted_set = set(submitted_ids.dropna().tolist())
        expected_set = set(expected_ids.dropna().tolist())
        missing_ids = expected_set - submitted_set
        extra_ids = submitted_set - expected_set
        checks["missing_id_count"] = len(missing_ids)
        checks["extra_id_count"] = len(extra_ids)
        if missing_ids or extra_ids:
            _add(
                issues,
                "error",
                "ID_SET_MISMATCH",
                f"ID set differs: {len(missing_ids)} missing and {len(extra_ids)} unexpected IDs.",
                column=resolved_id,
                value={
                    "missing_examples": sorted(missing_ids)[:10],
                    "extra_examples": sorted(extra_ids)[:10],
                },
                recommendation="Start from the sample submission IDs and merge predictions one-to-one.",
            )
        elif require_id_order and len(submission) == len(sample_submission):
            if not submitted_ids.reset_index(drop=True).equals(
                expected_ids.reset_index(drop=True)
            ):
                _add(
                    issues,
                    "error",
                    "ID_ORDER_MISMATCH",
                    "Submission IDs contain the correct set but not the sample order.",
                    column=resolved_id,
                    recommendation="Reindex predictions to sample_submission.csv order.",
                )

    target_summaries: dict[str, Any] = {}
    numeric_targets: dict[str, pd.Series] = {}
    for column in resolved_targets:
        if column not in submission.columns:
            continue
        values = submission[column]
        missing = int(values.isna().sum())
        summary: dict[str, Any] = {
            "dtype": str(values.dtype),
            "missing": missing,
            "unique": int(values.nunique(dropna=True)),
        }
        if missing:
            _add(
                issues,
                "error",
                "MISSING_PREDICTIONS",
                f"Column '{column}' contains {missing} missing predictions.",
                column=column,
            )

        expects_numeric = (
            resolved_task == "regression"
            or probabilities
            or pd.api.types.is_numeric_dtype(sample_submission[column])
        )
        if expects_numeric:
            numeric = pd.to_numeric(values, errors="coerce")
            non_numeric = int((numeric.isna() & values.notna()).sum())
            finite_array = numeric.to_numpy(dtype=float)
            infinite = int(np.isinf(finite_array).sum())
            summary["non_numeric"] = non_numeric
            summary["infinite"] = infinite
            if non_numeric:
                _add(
                    issues,
                    "error",
                    "NON_NUMERIC_PREDICTIONS",
                    f"Column '{column}' contains {non_numeric} non-numeric predictions.",
                    column=column,
                )
            if infinite:
                _add(
                    issues,
                    "error",
                    "INFINITE_PREDICTIONS",
                    f"Column '{column}' contains {infinite} infinite predictions.",
                    column=column,
                )
            finite = numeric[np.isfinite(finite_array)]
            numeric_targets[column] = numeric
            if not finite.empty:
                summary.update({
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                    "p01": float(finite.quantile(0.01)),
                    "median": float(finite.median()),
                    "p99": float(finite.quantile(0.99)),
                })
                if minimum_value is not None:
                    below = int((finite < minimum_value).sum())
                    if below:
                        _add(
                            issues,
                            "error",
                            "PREDICTION_BELOW_MINIMUM",
                            f"Column '{column}' has {below} values below {minimum_value}.",
                            column=column,
                            value=float(finite.min()),
                        )
                if maximum_value is not None:
                    above = int((finite > maximum_value).sum())
                    if above:
                        _add(
                            issues,
                            "error",
                            "PREDICTION_ABOVE_MAXIMUM",
                            f"Column '{column}' has {above} values above {maximum_value}.",
                            column=column,
                            value=float(finite.max()),
                        )
        if (
            resolved_task == "classification"
            and not probabilities
            and train is not None
            and column in train.columns
        ):
            allowed = set(train[column].dropna().astype("string").tolist())
            predicted = values.dropna().astype("string")
            unknown = sorted(set(predicted.tolist()) - allowed)
            if unknown:
                _add(
                    issues,
                    "error",
                    "UNKNOWN_CLASS_LABELS",
                    f"Column '{column}' contains {len(unknown)} labels absent from train.",
                    column=column,
                    value=unknown[:10],
                )

        if summary["unique"] <= 1 and not missing:
            _add(
                issues,
                "warning",
                "CONSTANT_PREDICTIONS",
                f"Column '{column}' contains only one distinct prediction.",
                column=column,
                recommendation="Confirm the model did not fail or write a placeholder submission.",
            )
        if column in sample_submission.columns and len(values) == len(sample_submission):
            if values.reset_index(drop=True).equals(
                sample_submission[column].reset_index(drop=True)
            ):
                _add(
                    issues,
                    "warning",
                    "UNCHANGED_SAMPLE_VALUES",
                    f"Column '{column}' exactly matches the sample placeholder values.",
                    column=column,
                    recommendation="Ensure real predictions replaced the template values.",
                )
        target_summaries[column] = summary

    if probabilities and numeric_targets:
        probability_frame = pd.DataFrame(numeric_targets)
        outside = int(
            ((probability_frame < -probability_tolerance) | (probability_frame > 1 + probability_tolerance))
            .any(axis=1)
            .sum()
        )
        row_sums = probability_frame.sum(axis=1)
        invalid_sums = int((~np.isclose(row_sums, 1.0, atol=probability_tolerance)).sum())
        checks["probability_rows_outside_0_1"] = outside
        checks["probability_rows_not_summing_to_1"] = invalid_sums
        if outside:
            _add(
                issues,
                "error",
                "PROBABILITY_OUT_OF_RANGE",
                f"{outside} rows contain probabilities outside [0, 1].",
            )
        if len(numeric_targets) > 1 and invalid_sums:
            _add(
                issues,
                "error",
                "PROBABILITY_SUM_MISMATCH",
                f"{invalid_sums} rows do not sum to one within tolerance.",
            )

    train_summary: dict[str, Any] | None = None
    if train is not None and len(resolved_targets) == 1:
        column = resolved_targets[0]
        if column in train.columns and column in numeric_targets:
            train_numeric = pd.to_numeric(train[column], errors="coerce")
            train_finite = train_numeric[np.isfinite(train_numeric.to_numpy(dtype=float))]
            prediction_finite = numeric_targets[column][
                np.isfinite(numeric_targets[column].to_numpy(dtype=float))
            ]
            if not train_finite.empty and not prediction_finite.empty:
                train_summary = {
                    "min": float(train_finite.min()),
                    "max": float(train_finite.max()),
                    "median": float(train_finite.median()),
                }
                train_scale = max(abs(float(train_finite.min())), abs(float(train_finite.max())), 1.0)
                extreme = int((np.abs(prediction_finite) > train_scale * extreme_factor).sum())
                if extreme:
                    _add(
                        issues,
                        "warning",
                        "EXTREME_PREDICTIONS",
                        f"{extreme} predictions exceed {extreme_factor:g}x the observed train scale.",
                        column=column,
                        recommendation="Inspect inverse transforms and outlier handling.",
                    )

    issues.sort(key=lambda item: (0 if item.severity == "error" else 1, item.code, item.column or ""))
    error_count = sum(item.severity == "error" for item in issues)
    warning_count = sum(item.severity == "warning" for item in issues)
    return _json_ready({
        "validation_version": "1.0",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": error_count == 0,
        "status": "pass" if error_count == 0 else "fail",
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "rows": int(len(submission)),
            "id_column": resolved_id,
            "target_columns": list(resolved_targets),
            "task": resolved_task,
        },
        "issues": [asdict(issue) for issue in issues],
        "checks": checks,
        "target_summary": target_summaries,
        "train_target_summary": train_summary,
    })


def _inspect_raw_csv(path: Path) -> tuple[list[str] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if path.suffix.lower() != ".csv":
        _add(issues, "error", "NOT_CSV", "Submission filename must end in .csv.")
    if not path.exists() or not path.is_file():
        _add(issues, "error", "FILE_NOT_FOUND", f"Submission file does not exist: {path}.")
        return None, issues
    if path.stat().st_size == 0:
        _add(issues, "error", "EMPTY_FILE", "Submission file is empty.")
        return None, issues

    header: list[str] | None = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                _add(issues, "error", "MISSING_HEADER", "CSV has no header row.")
                return header, issues
            if len(header) != len(set(header)):
                duplicates = sorted({value for value in header if header.count(value) > 1})
                _add(
                    issues,
                    "error",
                    "DUPLICATE_COLUMNS",
                    f"Raw CSV header contains duplicate columns: {duplicates}.",
                )
            expected_width = len(header)
            bad_rows: list[int] = []
            for row_number, row in enumerate(reader, start=2):
                if len(row) != expected_width:
                    bad_rows.append(row_number)
                    if len(bad_rows) >= 10:
                        break
            if bad_rows:
                _add(
                    issues,
                    "error",
                    "MALFORMED_CSV_ROWS",
                    f"CSV rows have the wrong number of fields: {bad_rows}.",
                )
    except (UnicodeDecodeError, csv.Error, OSError) as exc:
        _add(issues, "error", "CSV_READ_ERROR", f"Cannot parse UTF-8 CSV: {exc}.")
    return header, issues


def _round_trip_check(
    submission: pd.DataFrame,
    id_column: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_trip.csv"
            submission.to_csv(path, index=False)
            reread = pd.read_csv(path, dtype={id_column: "string"})
            expected = submission.copy()
            expected[id_column] = expected[id_column].astype("string")
            pd.testing.assert_frame_equal(
                expected.reset_index(drop=True),
                reread.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
    except Exception as exc:
        _add(
            issues,
            "error",
            "CSV_ROUND_TRIP_FAILED",
            f"Submission changed after CSV write/read: {exc}.",
            recommendation="Use pandas.to_csv(index=False) and validate the written file.",
        )
    return issues


def validate_submission_file(
    submission_path: str | Path,
    sample_path: str | Path,
    *,
    id_column: str | None = None,
    target_columns: str | Sequence[str] | None = None,
    task: Task = "auto",
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    require_id_order: bool = True,
    probabilities: bool = False,
    probability_tolerance: float = 1e-6,
    train_path: str | Path | None = None,
    extreme_factor: float = 10.0,
) -> dict[str, Any]:
    """Read, structurally inspect, round-trip, and validate a submission CSV."""
    submission_file = Path(submission_path)
    sample_file = Path(sample_path)
    _, raw_issues = _inspect_raw_csv(submission_file)
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample submission does not exist: {sample_file}.")

    sample_default = pd.read_csv(sample_file)
    resolved_id = id_column or str(sample_default.columns[0])
    sample = pd.read_csv(sample_file, dtype={resolved_id: "string"})
    try:
        submission = pd.read_csv(submission_file, dtype={resolved_id: "string"})
    except Exception as exc:
        raw_issues.append(
            ValidationIssue("error", "CSV_READ_ERROR", f"Pandas could not read submission: {exc}.")
        )
        submission = pd.DataFrame(columns=sample.columns)
    train = pd.read_csv(train_path) if train_path is not None else None

    report = validate_submission(
        submission,
        sample,
        id_column=resolved_id,
        target_columns=target_columns,
        task=task,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        require_id_order=require_id_order,
        probabilities=probabilities,
        probability_tolerance=probability_tolerance,
        train=train,
        extreme_factor=extreme_factor,
    )
    round_trip_issues = _round_trip_check(submission, resolved_id)
    combined = [*raw_issues, *round_trip_issues]
    if combined:
        report["issues"].extend(asdict(issue) for issue in combined)
        report["issues"].sort(
            key=lambda item: (0 if item["severity"] == "error" else 1, item["code"])
        )
        report["summary"]["errors"] = sum(
            item["severity"] == "error" for item in report["issues"]
        )
        report["summary"]["warnings"] = sum(
            item["severity"] == "warning" for item in report["issues"]
        )
        report["valid"] = report["summary"]["errors"] == 0
        report["status"] = "pass" if report["valid"] else "fail"

    report["file"] = {
        "submission": str(submission_file.resolve()),
        "sample_submission": str(sample_file.resolve()),
        "bytes": submission_file.stat().st_size if submission_file.exists() else None,
        "sha256": hashlib.sha256(submission_file.read_bytes()).hexdigest()
        if submission_file.exists()
        else None,
        "round_trip_checked": True,
    }
    return _json_ready(report)


def assert_valid_submission(report: dict[str, Any]) -> None:
    """Raise with concise issue codes when a report contains fatal errors."""
    if not report.get("valid", False):
        codes = [
            item["code"] for item in report.get("issues", []) if item["severity"] == "error"
        ]
        raise SubmissionValidationError(
            f"Submission validation failed with {len(codes)} error(s): {codes}."
        )


def write_validation_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a competition submission CSV.")
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/sample_submission.csv"),
    )
    parser.add_argument("--id-column")
    parser.add_argument("--target-columns", nargs="+")
    parser.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    parser.add_argument("--min-value", type=float)
    parser.add_argument("--max-value", type=float)
    parser.add_argument("--allow-reordered-ids", action="store_true")
    parser.add_argument("--probabilities", action="store_true")
    parser.add_argument("--probability-tolerance", type=float, default=1e-6)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--extreme-factor", type=float, default=10.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = validate_submission_file(
        args.submission,
        args.sample,
        id_column=args.id_column,
        target_columns=args.target_columns,
        task=args.task,
        minimum_value=args.min_value,
        maximum_value=args.max_value,
        require_id_order=not args.allow_reordered_ids,
        probabilities=args.probabilities,
        probability_tolerance=args.probability_tolerance,
        train_path=args.train,
        extreme_factor=args.extreme_factor,
    )
    report_path = args.report or args.submission.with_suffix(
        args.submission.suffix + ".validation.json"
    )
    write_validation_report(report, report_path)

    summary = report["summary"]
    print(
        f"Submission validation: {report['status'].upper()} | "
        f"rows={summary['rows']:,} errors={summary['errors']} "
        f"warnings={summary['warnings']}"
    )
    for item in report["issues"]:
        column = f" [{item['column']}]" if item.get("column") else ""
        print(f"{item['severity'].upper():7} {item['code']}{column}: {item['message']}")
    print(f"Report: {report_path}")
    if not report["valid"]:
        return 2
    if args.fail_on_warning and summary["warnings"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
