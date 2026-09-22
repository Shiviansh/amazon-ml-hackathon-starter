"""Resumable end-to-end competition pipeline with content-addressed stages.

Example
-------
``python src/pipeline.py --dataset amazon --device gpu``

Every stage runs as an isolated Python process, writes a durable log, and is
cached only after all declared outputs exist and match their recorded hashes.
Changing source data, relevant code, configuration, upstream fingerprints, or
an output artifact invalidates exactly the affected stage and its descendants.
Submission validation is the mandatory terminal gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import pandas as pd
import yaml


PIPELINE_VERSION = "1.0"
DeviceName = Literal["auto", "cpu", "gpu"]
StageExecutor = Callable[["PipelineStage", Path, Path], int]


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    train: Path
    test: Path
    sample_submission: Path
    target_column: str = "PRICE"
    id_column: str = "PRODUCT_ID"
    image_url_column: str = "IMAGE_URL"
    title_column: str = "TITLE"
    description_column: str = "DESCRIPTION"
    pack_column: str = "PACK_SIZE"
    brand_column: str = "BRAND"
    category_column: str = "CATEGORY"
    text_columns: tuple[str, ...] = ("TITLE", "DESCRIPTION", "PACK_SIZE")
    categorical_columns: tuple[str, ...] = ("CATEGORY", "BRAND")
    numeric_columns: tuple[str, ...] = ()
    fold_column: str = "fold"
    n_splits: int = 5
    random_state: int = 42
    price_floor: float = 0.0


@dataclass(frozen=True)
class PipelineOptions:
    repo_root: Path
    run_directory: Path
    device: DeviceName = "gpu"
    models: tuple[str, ...] = ("xgboost", "catboost")
    use_images: bool = True
    image_backend: str = "clip"
    image_batch_size: int = 32
    download_workers: int = 16
    feature_workers: int = 1
    ocr_workers: int = 4
    max_image_failure_rate: float = 0.20
    local_files_only: bool = False
    fail_on_validation_warning: bool = False
    model_params: Mapping[str, Path] = field(default_factory=dict)
    objective: str = "auto"
    extract_domain_features: bool = False
    text_embedding_model: str = "BAAI/bge-large-en-v1.5"

    def __post_init__(self) -> None:
        allowed_models = {"linear", "lightgbm", "xgboost", "catboost"}
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError(f"Unsupported device: {self.device!r}.")
        if not self.models:
            raise ValueError("At least one model is required.")
        if len(self.models) != len(set(self.models)):
            raise ValueError("Model names must be unique.")
        unknown = sorted(set(self.models) - allowed_models)
        if unknown:
            raise ValueError(f"Unsupported models: {unknown}.")
        if self.image_backend not in {"clip", "dinov2"}:
            raise ValueError(f"Unsupported image backend: {self.image_backend!r}.")
        for name, value in {
            "image_batch_size": self.image_batch_size,
            "download_workers": self.download_workers,
            "feature_workers": self.feature_workers,
            "ocr_workers": self.ocr_workers,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
        if not math.isfinite(self.max_image_failure_rate) or not (
            0.0 <= self.max_image_failure_rate <= 1.0
        ):
            raise ValueError("max_image_failure_rate must be finite and between 0 and 1.")
        unknown_params = sorted(set(self.model_params) - set(self.models))
        if unknown_params:
            raise ValueError(
                f"Parameter files were supplied for unselected models: {unknown_params}."
            )


@dataclass(frozen=True)
class PipelineStage:
    name: str
    description: str
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    code_files: tuple[Path, ...]


@dataclass
class StageRun:
    name: str
    status: str
    fingerprint: str
    seconds: float
    log_path: str
    reason: str | None = None


@dataclass
class PipelineResult:
    stages: list[StageRun]
    report_path: Path
    final_submission: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def _path_digest(path: Path) -> dict[str, Any]:
    """Fingerprint a file or a deterministic directory tree."""

    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return {
            "kind": "file",
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    if not path.is_dir():
        raise ValueError(f"Unsupported artifact type: {path}")
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(_sha256_file(child).encode("ascii"))
        file_count += 1
        total_size += size
    return {
        "kind": "directory",
        "files": file_count,
        "size": total_size,
        "sha256": digest.hexdigest(),
    }


def _path_records(paths: Iterable[Path]) -> dict[str, Any]:
    return {str(path.resolve()): _path_digest(path) for path in paths}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class PipelineLock:
    """Process-level guard preventing two writers from sharing one run directory."""

    def __init__(self, path: Path):
        self.path = path
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            details = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Pipeline run directory is locked by another process: {self.path}\n{details}"
            ) from exc
        payload = json.dumps({
            "pid": os.getpid(),
            "token": self.token,
            "created_at_utc": _utc_now(),
        }).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        self.acquired = True
        return self

    def __exit__(self, *_: Any) -> None:
        if not self.acquired:
            return
        current = _read_json(self.path)
        if current and current.get("token") == self.token:
            self.path.unlink(missing_ok=True)
        self.acquired = False


def _subprocess_executor(stage: PipelineStage, repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(list(stage.command)) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            list(stage.command),
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[{stage.name}] {line}", end="")
                log.write(line)
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


class PipelineRunner:
    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        repo_root: Path,
        run_directory: Path,
        executor: StageExecutor = _subprocess_executor,
    ):
        self.stages = {stage.name: stage for stage in stages}
        if len(self.stages) != len(stages):
            raise ValueError("Pipeline stage names must be unique.")
        self.repo_root = repo_root
        self.run_directory = run_directory
        self.state_directory = run_directory / ".pipeline" / "stages"
        self.log_directory = run_directory / "logs"
        self.executor = executor
        self.order = self._topological_order()
        self.last_runs: list[StageRun] = []

    def _topological_order(self) -> list[str]:
        missing = {
            dependency
            for stage in self.stages.values()
            for dependency in stage.dependencies
            if dependency not in self.stages
        }
        if missing:
            raise ValueError(f"Pipeline dependencies reference missing stages: {sorted(missing)}.")
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Pipeline dependency cycle detected at {name!r}.")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.stages[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for name in self.stages:
            visit(name)
        return ordered

    def manifest_path(self, stage_name: str) -> Path:
        return self.state_directory / f"{stage_name}.json"

    def _fingerprint(self, stage: PipelineStage) -> tuple[str, dict[str, Any]]:
        missing_inputs = [str(path) for path in stage.inputs if not path.exists()]
        if missing_inputs:
            raise FileNotFoundError(
                f"Stage {stage.name!r} is missing inputs: {missing_inputs}."
            )
        dependency_fingerprints: dict[str, str] = {}
        for dependency in stage.dependencies:
            manifest = _read_json(self.manifest_path(dependency))
            if not manifest or manifest.get("status") != "complete":
                raise RuntimeError(
                    f"Dependency {dependency!r} has no complete stage manifest."
                )
            dependency_fingerprints[dependency] = str(manifest["fingerprint"])
        details = {
            "pipeline_version": PIPELINE_VERSION,
            "stage": stage.name,
            "command": list(stage.command),
            "inputs": _path_records(stage.inputs),
            "code": _path_records(stage.code_files),
            "dependencies": dependency_fingerprints,
            "python": platform.python_version(),
        }
        fingerprint = hashlib.sha256(
            json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return fingerprint, details

    def _cache_hit(self, stage: PipelineStage, fingerprint: str) -> tuple[bool, str]:
        manifest = _read_json(self.manifest_path(stage.name))
        if not manifest:
            return False, "no prior manifest"
        if manifest.get("status") != "complete":
            return False, f"prior status={manifest.get('status')}"
        if manifest.get("fingerprint") != fingerprint:
            return False, "fingerprint changed"
        if any(not path.exists() for path in stage.outputs):
            return False, "declared output missing"
        try:
            current = _path_records(stage.outputs)
        except (OSError, ValueError):
            return False, "output cannot be fingerprinted"
        if current != manifest.get("outputs"):
            return False, "output content changed"
        return True, "content hash match"

    def _dry_run_state(
        self,
        stage: PipelineStage,
        projected_dependencies: Mapping[str, str],
    ) -> tuple[str, bool, str]:
        """Return a read-only projected fingerprint and cache status.

        A normal fingerprint is exact only when every dependency remains at its
        currently materialized fingerprint. Otherwise generated inputs may be
        stale or absent, so the projected fingerprint deliberately records that
        uncertainty and forces the affected descendant into the plan.
        """

        dependencies_current = True
        for dependency in stage.dependencies:
            manifest = _read_json(self.manifest_path(dependency))
            if (
                not manifest
                or manifest.get("status") != "complete"
                or manifest.get("fingerprint") != projected_dependencies[dependency]
            ):
                dependencies_current = False
                break
        inputs_exist = all(path.exists() for path in stage.inputs)
        if dependencies_current and inputs_exist:
            fingerprint, _ = self._fingerprint(stage)
            hit, reason = self._cache_hit(stage, fingerprint)
            return fingerprint, hit, reason

        input_state = {
            str(path.resolve()): (
                _path_digest(path) if path.exists() else {"kind": "missing-generated-input"}
            )
            for path in stage.inputs
        }
        details = {
            "pipeline_version": PIPELINE_VERSION,
            "stage": stage.name,
            "command": list(stage.command),
            "inputs": input_state,
            "code": _path_records(stage.code_files),
            "dependencies": {
                dependency: projected_dependencies[dependency]
                for dependency in stage.dependencies
            },
            "python": platform.python_version(),
            "projection": True,
        }
        fingerprint = hashlib.sha256(
            json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        reason = (
            "upstream stage would change"
            if not dependencies_current
            else "generated input does not exist yet"
        )
        return fingerprint, False, reason

    def run(
        self,
        *,
        force_stages: Sequence[str] = (),
        no_cache: bool = False,
        dry_run: bool = False,
        until_stage: str | None = None,
    ) -> list[StageRun]:
        unknown_force = sorted(set(force_stages) - set(self.stages))
        if unknown_force:
            raise ValueError(f"Unknown forced stages: {unknown_force}.")
        if until_stage is not None and until_stage not in self.stages:
            raise ValueError(f"Unknown --until-stage {until_stage!r}.")
        selected = self.order
        if until_stage is not None:
            selected = selected[: selected.index(until_stage) + 1]
        results: list[StageRun] = []
        self.last_runs = results
        forced = set(force_stages)
        if dry_run:
            projected: dict[str, str] = {}
            for name in selected:
                stage = self.stages[name]
                fingerprint, hit, reason = self._dry_run_state(stage, projected)
                should_run = no_cache or name in forced or not hit
                status = "planned" if should_run else "cached"
                if name in forced:
                    reason = "forced"
                elif no_cache:
                    reason = "cache disabled"
                verb = "PLAN" if should_run else "SKIP"
                print(f"[pipeline] {verb} {name}: {reason}")
                log_path = self.log_directory / f"{name}.log"
                results.append(
                    StageRun(name, status, fingerprint, 0.0, str(log_path), reason)
                )
                projected[name] = fingerprint
            return results

        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        for name in selected:
            stage = self.stages[name]
            fingerprint, details = self._fingerprint(stage)
            hit, reason = self._cache_hit(stage, fingerprint)
            should_run = no_cache or name in forced or not hit
            log_path = self.log_directory / f"{name}.log"
            if not should_run:
                print(f"[pipeline] SKIP {name}: {reason}")
                results.append(StageRun(name, "cached", fingerprint, 0.0, str(log_path), reason))
                continue
            print(f"[pipeline] RUN  {name}: {stage.description}")
            started = time.perf_counter()
            started_at_utc = _utc_now()
            _atomic_json({
                "status": "running",
                "stage": name,
                "fingerprint": fingerprint,
                "started_at_utc": started_at_utc,
                "details": details,
                "command": list(stage.command),
                "log": str(log_path),
            }, self.manifest_path(name))
            try:
                return_code = self.executor(stage, self.repo_root, log_path)
                if return_code != 0:
                    raise RuntimeError(
                        f"Stage {name!r} failed with exit code {return_code}. Log: {log_path}"
                    )
                missing = [str(path) for path in stage.outputs if not path.exists()]
                if missing:
                    raise RuntimeError(
                        f"Stage {name!r} exited successfully but outputs are missing: {missing}."
                    )
                output_records = _path_records(stage.outputs)
                seconds = time.perf_counter() - started
                _atomic_json({
                    "status": "complete",
                    "stage": name,
                    "fingerprint": fingerprint,
                    "started_at_utc": started_at_utc,
                    "elapsed_seconds": seconds,
                    "details": details,
                    "outputs": output_records,
                    "command": list(stage.command),
                    "log": str(log_path),
                }, self.manifest_path(name))
                results.append(StageRun(name, "complete", fingerprint, seconds, str(log_path)))
            except BaseException as exc:
                seconds = time.perf_counter() - started
                _atomic_json({
                    "status": "failed",
                    "stage": name,
                    "fingerprint": fingerprint,
                    "elapsed_seconds": seconds,
                    "failed_at_utc": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": list(stage.command),
                    "log": str(log_path),
                }, self.manifest_path(name))
                results.append(
                    StageRun(
                        name,
                        "failed",
                        fingerprint,
                        seconds,
                        str(log_path),
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
        return results


def _python_command(repo_root: Path, script: str, *arguments: Any) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        str((repo_root / "src" / script).resolve()),
        *(str(argument) for argument in arguments),
    )


def _load_profile(
    dataset: str,
    repo_root: Path,
    *,
    train_override: Path | None = None,
    test_override: Path | None = None,
    sample_override: Path | None = None,
) -> DatasetProfile:
    aliases = {"amazon", "amazon_india", "amazon-ml"}
    if dataset not in aliases:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available built-in profile: amazon. "
            "Use --train/--test/--sample only with --dataset amazon."
        )
    config_path = repo_root / "configs" / "amazon_india.yaml"
    values: dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Dataset config must be a YAML mapping: {config_path}")
        values = loaded
    raw = repo_root / "data" / "raw"
    return DatasetProfile(
        name="amazon",
        train=(train_override or raw / "train.csv").resolve(),
        test=(test_override or raw / "test.csv").resolve(),
        sample_submission=(sample_override or raw / "sample_submission.csv").resolve(),
        target_column=str(values.get("target_column", "PRICE")),
        id_column=str(values.get("id_column", "PRODUCT_ID")),
        text_columns=tuple(values.get("text_columns", ["TITLE", "DESCRIPTION", "PACK_SIZE"])),
        categorical_columns=tuple(values.get("categorical_columns", ["CATEGORY", "BRAND"])),
        numeric_columns=tuple(values.get("numeric_columns", [])),
        fold_column=str(values.get("fold_column", "fold")),
        n_splits=int(values.get("n_splits", 5)),
        random_state=int(values.get("random_state", 42)),
        price_floor=float(values.get("prediction_floor", 0.0)),
    )


def _validate_profile(profile: DatasetProfile, use_images: bool) -> None:
    missing_paths = [
        str(path)
        for path in (profile.train, profile.test, profile.sample_submission)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Dataset profile is missing files: {missing_paths}.")
    def read_head(path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path).head(5)
        return pd.read_csv(path, nrows=5)

    train_head = read_head(profile.train)
    test_head = read_head(profile.test)
    required_train = {
        profile.id_column,
        profile.target_column,
        *profile.text_columns,
        *profile.categorical_columns,
        *profile.numeric_columns,
    }
    required_test = required_train - {profile.target_column}
    if use_images:
        required_train.add(profile.image_url_column)
        required_test.add(profile.image_url_column)
    missing_train = sorted(required_train - set(train_head.columns))
    missing_test = sorted(required_test - set(test_head.columns))
    if missing_train or missing_test:
        raise ValueError(
            f"Dataset schema mismatch: train_missing={missing_train}, test_missing={missing_test}."
        )


def _parse_model_params(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--model-params entries must use MODEL=PATH.")
        model, raw_path = item.split("=", 1)
        if model not in {"linear", "lightgbm", "xgboost", "catboost"}:
            raise ValueError(f"Unknown model in --model-params: {model!r}.")
        if model in output:
            raise ValueError(f"Duplicate --model-params entry for {model!r}.")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output[model] = path
    return output


def build_pipeline_stages(
    profile: DatasetProfile,
    options: PipelineOptions,
) -> list[PipelineStage]:
    root = options.repo_root
    run = options.run_directory
    processed = run / "processed"
    reports = run / "reports"
    folds = processed / "folds.parquet"
    stages: list[PipelineStage] = []
    requirements = root / "requirements.txt"

    stages.append(PipelineStage(
        "folds",
        "Create deterministic leakage-safe validation folds",
        (),
        _python_command(
            root, "splits.py",
            "--input", profile.train,
            "--output", folds,
            "--target-column", profile.target_column,
            "--id-column", profile.id_column,
            "--task", "regression",
            "--strategy", "stratified",
            "--n-splits", profile.n_splits,
            "--random-state", profile.random_state,
            "--fold-column", profile.fold_column,
        ),
        (profile.train,),
        (folds, folds.with_suffix(".parquet.metadata.json")),
        (root / "src" / "splits.py", requirements),
    ))

    if options.use_images:
        for split, source in (("train", profile.train), ("test", profile.test)):
            image_root = processed / "images" / split
            stages.append(PipelineStage(
                f"download_{split}_images",
                f"Download, validate, deduplicate, and cache {split} images",
                (),
                _python_command(
                    root, "download_images.py",
                    "--input", source,
                    "--output-dir", image_root,
                    "--id-column", profile.id_column,
                    "--url-column", profile.image_url_column,
                    "--workers", options.download_workers,
                    "--max-failure-rate", options.max_image_failure_rate,
                    "--cache-verification", "sha256",
                    "--no-progress",
                ),
                (source,),
                (image_root / "image_manifest.parquet", image_root / "download_report.json"),
                (root / "src" / "download_images.py", requirements),
            ))

        ocr_dir = processed / "ocr"
        ocr_command: list[Any] = [
            "--train", profile.train,
            "--test", profile.test,
            "--train-manifest", processed / "images" / "train" / "image_manifest.parquet",
            "--test-manifest", processed / "images" / "test" / "image_manifest.parquet",
            "--train-image-root", processed / "images" / "train",
            "--test-image-root", processed / "images" / "test",
            "--output-train", ocr_dir / "ocr_train.parquet",
            "--output-test", ocr_dir / "ocr_test.parquet",
            "--backfilled-train", ocr_dir / "catalog_train_backfilled.parquet",
            "--backfilled-test", ocr_dir / "catalog_test_backfilled.parquet",
            "--diagnostics-dir", reports / "ocr",
            "--cache", ocr_dir / "ocr_cache.sqlite3",
            "--report", reports / "ocr" / "ocr_run.json",
            "--id-column", profile.id_column,
            "--title-column", profile.title_column,
            "--description-column", profile.description_column,
            "--pack-column", profile.pack_column,
            "--workers", options.ocr_workers,
            "--max-failure-rate", 1.0,
        ]
        if options.device == "gpu":
            ocr_command.append("--gpu")
        stages.append(PipelineStage(
            "ocr",
            "Extract packaging text and backfill physical quantities",
            ("download_train_images", "download_test_images"),
            _python_command(root, "ocr.py", *ocr_command),
            (
                profile.train,
                profile.test,
                processed / "images" / "train" / "image_manifest.parquet",
                processed / "images" / "test" / "image_manifest.parquet",
            ),
            (
                ocr_dir / "ocr_train.parquet",
                ocr_dir / "ocr_test.parquet",
                ocr_dir / "catalog_train_backfilled.parquet",
                ocr_dir / "catalog_test_backfilled.parquet",
                reports / "ocr" / "ocr_run.json",
            ),
            (root / "src" / "ocr.py", root / "src" / "features.py", requirements),
        ))

        image_dir = processed / "image_embeddings"
        image_command: list[Any] = [
            "--train", profile.train,
            "--test", profile.test,
            "--train-manifest", processed / "images" / "train" / "image_manifest.parquet",
            "--test-manifest", processed / "images" / "test" / "image_manifest.parquet",
            "--train-image-root", processed / "images" / "train",
            "--test-image-root", processed / "images" / "test",
            "--output-dir", image_dir,
            "--report", reports / "image_embeddings_run.json",
            "--id-column", profile.id_column,
            "--backend", options.image_backend,
            "--device", "cuda" if options.device == "gpu" else options.device,
            "--batch-size", options.image_batch_size,
            "--verify-hashes",
            "--failure-policy", "zero",
        ]
        if options.local_files_only:
            image_command.append("--local-files-only")
        stages.append(PipelineStage(
            "image_embeddings",
            "Encode product images with CLIP/DINOv2",
            ("download_train_images", "download_test_images"),
            _python_command(root, "image_embeddings.py", *image_command),
            (
                profile.train,
                profile.test,
                processed / "images" / "train" / "image_manifest.parquet",
                processed / "images" / "test" / "image_manifest.parquet",
            ),
            (
                image_dir / "image_embeddings_train.parquet",
                image_dir / "image_embeddings_test.parquet",
                reports / "image_embeddings_run.json",
            ),
            (root / "src" / "image_embeddings.py", root / "src" / "models.py", requirements),
        ))
        catalog_train = ocr_dir / "catalog_train_backfilled.parquet"
        catalog_test = ocr_dir / "catalog_test_backfilled.parquet"
        feature_dependencies = ("folds", "ocr")
    else:
        catalog_train = profile.train
        catalog_test = profile.test
        feature_dependencies = ("folds",)

    feature_dir = processed / "features"
    stages.append(PipelineStage(
        "features",
        "Extract numeric, pack, unit, text-statistic, and OOF target-encoded features",
        feature_dependencies,
        _python_command(
            root, "features.py",
            "--train", catalog_train,
            "--test", catalog_test,
            "--folds", folds,
            "--output-train", feature_dir / "features_train.parquet",
            "--output-test", feature_dir / "features_test.parquet",
            "--target-column", profile.target_column,
            "--id-column", profile.id_column,
            "--title-column", profile.title_column,
            "--desc-column", profile.description_column,
            "--pack-column", profile.pack_column,
            "--brand-column", profile.brand_column,
            "--category-column", profile.category_column,
            "--fold-column", profile.fold_column,
            "--n-jobs", options.feature_workers,
        ),
        (catalog_train, catalog_test, folds),
        (
            feature_dir / "features_train.parquet",
            feature_dir / "features_test.parquet",
            feature_dir / "features_manifest.json",
        ),
        (root / "src" / "features.py", requirements),
    ))

    train_dependencies = ["features"]
    if options.use_images:
        train_dependencies.append("image_embeddings")
    train_stage_names: list[str] = []
    for model in options.models:
        name = f"train_{model}"
        train_stage_names.append(name)
        output_dir = run / "models" / model
        train_device = "cpu" if model == "linear" else options.device
        command: list[Any] = [
            "--train", catalog_train,
            "--test", catalog_test,
            "--folds", folds,
            "--output-dir", output_dir,
            "--target-column", profile.target_column,
            "--id-column", profile.id_column,
            "--text-columns", *profile.text_columns,
            "--categorical-columns", *profile.categorical_columns,
        ]
        if profile.numeric_columns:
            command.extend(["--numeric-columns", *profile.numeric_columns])
        command.extend([
            "--engineered-features",
            feature_dir / "features_train.parquet",
            feature_dir / "features_test.parquet",
            "--model", model,
            "--device", train_device,
            "--target-transform", "log1p",
            "--fold-column", profile.fold_column,
            "--n-splits", profile.n_splits,
            "--random-state", profile.random_state,
            "--prediction-floor", profile.price_floor,
            "--objective", options.objective,
            "--save-models",
            "--overwrite",
        ])
        if options.extract_domain_features:
            command.append("--extract-domain-features")
        if options.use_images:
            command.extend([
                "--image-embeddings",
                processed / "image_embeddings" / "image_embeddings_train.parquet",
                processed / "image_embeddings" / "image_embeddings_test.parquet",
            ])
        if model in options.model_params:
            command.extend(["--model-params-json", options.model_params[model]])
        inputs = [
            catalog_train,
            catalog_test,
            folds,
            feature_dir / "features_train.parquet",
            feature_dir / "features_test.parquet",
        ]
        if options.use_images:
            inputs.extend([
                processed / "image_embeddings" / "image_embeddings_train.parquet",
                processed / "image_embeddings" / "image_embeddings_test.parquet",
            ])
        if model in options.model_params:
            inputs.append(options.model_params[model])
        stages.append(PipelineStage(
            name,
            f"Train {model} with fold-local preprocessing and GPU acceleration where supported",
            tuple(train_dependencies),
            _python_command(root, "train.py", *command),
            tuple(inputs),
            (
                output_dir / "oof.parquet",
                output_dir / "test_predictions.parquet",
                output_dir / "submission.csv",
                output_dir / "training_report.json",
                output_dir / "models",
            ),
            (
                root / "src" / "train.py",
                root / "src" / "models.py",
                root / "src" / "metrics.py",
                root / "src" / "splits.py",
                requirements,
            ),
        ))

    ensemble_dir = run / "ensemble"
    ensemble_command: list[Any] = []
    ensemble_inputs: list[Path] = []
    for model in options.models:
        model_dir = run / "models" / model
        ensemble_command.extend(["--oof", f"{model}={model_dir / 'oof.parquet'}"])
        ensemble_command.extend([
            "--test", f"{model}={model_dir / 'test_predictions.parquet'}"
        ])
        ensemble_inputs.extend([model_dir / "oof.parquet", model_dir / "test_predictions.parquet"])
    ensemble_command.extend([
        "--output-dir", ensemble_dir,
        "--id-column", profile.id_column,
        "--target-column", profile.target_column,
        "--fold-column", profile.fold_column,
        "--method", "auto",
        "--random-state", profile.random_state,
        "--prediction-floor", profile.price_floor,
        "--overwrite",
    ])
    stages.append(PipelineStage(
        "ensemble",
        "Optimize non-negative OOF ensemble weights",
        tuple(train_stage_names),
        _python_command(root, "ensemble.py", *ensemble_command),
        tuple(ensemble_inputs),
        (
            ensemble_dir / "ensemble_weights.json",
            ensemble_dir / "ensemble_oof.parquet",
            ensemble_dir / "ensemble_test_predictions.parquet",
            ensemble_dir / "submission.csv",
            ensemble_dir / "ensemble_report.json",
        ),
        (root / "src" / "ensemble.py", root / "src" / "metrics.py", requirements),
    ))

    postprocess_dir = run / "postprocess"
    stages.append(PipelineStage(
        "postprocess",
        "Calibrate OOF SMAPE and apply domain-safe price clamping",
        ("ensemble",),
        _python_command(
            root, "postprocess.py",
            "--oof", ensemble_dir / "ensemble_oof.parquet",
            "--test", ensemble_dir / "ensemble_test_predictions.parquet",
            "--output-dir", postprocess_dir,
            "--id-column", profile.id_column,
            "--target-column", profile.target_column,
            "--fold-column", profile.fold_column,
            "--price-floor", profile.price_floor,
            "--overwrite",
        ),
        (
            ensemble_dir / "ensemble_oof.parquet",
            ensemble_dir / "ensemble_test_predictions.parquet",
        ),
        (
            postprocess_dir / "calibration.json",
            postprocess_dir / "postprocess_oof.parquet",
            postprocess_dir / "postprocess_test_predictions.parquet",
            postprocess_dir / "submission.csv",
            postprocess_dir / "postprocess_report.json",
        ),
        (root / "src" / "postprocess.py", root / "src" / "metrics.py", requirements),
    ))

    validation_report = run / "final" / "submission.validation.json"
    validate_command: list[Any] = [
        "--submission", postprocess_dir / "submission.csv",
        "--sample", profile.sample_submission,
        "--id-column", profile.id_column,
        "--target-columns", profile.target_column,
        "--task", "regression",
        "--min-value", profile.price_floor,
        "--train", catalog_train,
        "--report", validation_report,
    ]
    if options.fail_on_validation_warning:
        validate_command.append("--fail-on-warning")
    stages.append(PipelineStage(
        "validate_submission",
        "Mandatory final schema, ID-order, finiteness, range, and sanity gate",
        ("postprocess",),
        _python_command(root, "validate_submit.py", *validate_command),
        (postprocess_dir / "submission.csv", profile.sample_submission, catalog_train),
        (validation_report,),
        (root / "src" / "validate_submit.py", requirements),
    ))
    return stages


def _verify_validation_gate(report_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    if not report:
        raise RuntimeError(f"Final validation report is missing or unreadable: {report_path}")
    if report.get("valid") is not True or report.get("status") not in {"pass", "valid"}:
        raise RuntimeError(
            f"Final submission validation did not pass: status={report.get('status')!r}."
        )
    return report


def run_pipeline(
    profile: DatasetProfile,
    options: PipelineOptions,
    *,
    force_stages: Sequence[str] = (),
    no_cache: bool = False,
    dry_run: bool = False,
    until_stage: str | None = None,
    executor: StageExecutor = _subprocess_executor,
) -> PipelineResult:
    _validate_profile(profile, options.use_images)
    stages = build_pipeline_stages(profile, options)
    runner = PipelineRunner(
        stages,
        repo_root=options.repo_root,
        run_directory=options.run_directory,
        executor=executor,
    )
    report_path = options.run_directory / "pipeline_report.json"
    final_submission = options.run_directory / "postprocess" / "submission.csv"
    started = time.perf_counter()
    options.run_directory.mkdir(parents=True, exist_ok=True)
    stage_runs: list[StageRun] = []
    with PipelineLock(options.run_directory / ".pipeline" / "pipeline.lock"):
        try:
            stage_runs = runner.run(
                force_stages=force_stages,
                no_cache=no_cache,
                dry_run=dry_run,
                until_stage=until_stage,
            )
            terminal_reached = until_stage is None or until_stage == "validate_submission"
            validation: dict[str, Any] | None = None
            if terminal_reached and not dry_run:
                validation = _verify_validation_gate(
                    options.run_directory / "final" / "submission.validation.json"
                )
            report = {
                "pipeline_version": PIPELINE_VERSION,
                "status": "planned" if dry_run else "complete",
                "created_at_utc": _utc_now(),
                "dataset": _json_ready(asdict(profile)),
                "options": _json_ready(asdict(options)),
                "stages": [_json_ready(asdict(item)) for item in stage_runs],
                "final_submission": str(final_submission) if terminal_reached else None,
                "validation": validation,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_json(report, report_path)
        except BaseException as exc:
            stage_runs = runner.last_runs
            failure_report = {
                "pipeline_version": PIPELINE_VERSION,
                "status": "failed",
                "created_at_utc": _utc_now(),
                "dataset": _json_ready(asdict(profile)),
                "options": _json_ready(asdict(options)),
                "stages": [_json_ready(asdict(item)) for item in stage_runs],
                "error": f"{type(exc).__name__}: {exc}",
                "final_submission": None,
                "validation": None,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_json(failure_report, report_path)
            raise
    return PipelineResult(stage_runs, report_path, final_submission)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete cached Amazon ML competition pipeline."
    )
    parser.add_argument("--dataset", default="amazon")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("linear", "lightgbm", "xgboost", "catboost"),
        help="Default: xgboost+catboost on GPU; linear+lightgbm otherwise.",
    )
    parser.add_argument(
        "--model-params",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Best-parameter JSON produced by tune.py; repeat per model.",
    )
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--image-backend", choices=("clip", "dinov2"), default="clip")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--ocr-workers", type=int, default=4)
    parser.add_argument("--max-image-failure-rate", type=float, default=0.20)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fail-on-validation-warning", action="store_true")
    parser.add_argument(
        "--objective",
        default="auto",
        choices=("auto", "smape", "mape", "huber", "l1", "l2", "regression", "mae"),
        help="Loss objective for tree learners ('auto' selects based on metric).",
    )
    parser.add_argument(
        "--extract-domain-features",
        action="store_true",
        help="Extract deterministic pack, unit, and dimension features from text.",
    )
    parser.add_argument(
        "--text-embedding-model",
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for dense text representations.",
    )
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--until-stage")
    parser.add_argument("--list-stages", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    profile = _load_profile(
        args.dataset,
        repo_root,
        train_override=args.train,
        test_override=args.test,
        sample_override=args.sample,
    )
    models = tuple(args.models or (
        ("xgboost", "catboost") if args.device == "gpu" else ("linear", "lightgbm")
    ))
    if len(models) != len(set(models)):
        raise ValueError("--models cannot contain duplicates.")
    run_directory = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (repo_root / "runs" / profile.name).resolve()
    )
    options = PipelineOptions(
        repo_root=repo_root,
        run_directory=run_directory,
        device=args.device,
        models=models,
        use_images=not args.skip_images,
        image_backend=args.image_backend,
        image_batch_size=args.image_batch_size,
        download_workers=args.download_workers,
        feature_workers=args.feature_workers,
        ocr_workers=args.ocr_workers,
        max_image_failure_rate=args.max_image_failure_rate,
        local_files_only=args.local_files_only,
        fail_on_validation_warning=args.fail_on_validation_warning,
        model_params=_parse_model_params(args.model_params),
        objective=args.objective,
        extract_domain_features=args.extract_domain_features,
        text_embedding_model=args.text_embedding_model,
    )
    stages = build_pipeline_stages(profile, options)
    if args.list_stages:
        for index, stage in enumerate(stages, start=1):
            dependencies = ", ".join(stage.dependencies) or "none"
            print(f"{index:02d}. {stage.name:24} deps=[{dependencies}]  {stage.description}")
        return 0
    result = run_pipeline(
        profile,
        options,
        force_stages=args.force_stage,
        no_cache=args.no_cache,
        dry_run=args.dry_run,
        until_stage=args.until_stage,
    )
    print(f"Pipeline report: {result.report_path}")
    if not args.dry_run and (args.until_stage is None or args.until_stage == "validate_submission"):
        print(f"VALIDATED submission: {result.final_submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
