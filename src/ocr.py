"""Resilient OCR harvesting for e-commerce packaging images.

The module deliberately separates three concerns:

* OCR engines return text together with confidence and bounding boxes.
* packaging filtering rejects promotions and nutrition-panel measurements.
* physical quantities are parsed by :mod:`features`, so OCR and catalog text
  obey exactly the same unit conversion and multipack rules.

The command line interface preserves one output row per catalog row, in the
same order, and writes the competition-facing six-column parquet atomically.
An additional diagnostics parquet records failures, bounding boxes, cache use,
and engine details without polluting model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

try:  # Works both as ``python -m src.ocr`` and ``python src/ocr.py``.
    from .features import extract_row_physical_specs
except ImportError:  # pragma: no cover - exercised by the CLI, not imports.
    from features import extract_row_physical_specs


OCR_VERSION = "1.1"
OUTPUT_COLUMNS = [
    "PRODUCT_ID",
    "ocr_text",
    "ocr_weight_g",
    "ocr_volume_ml",
    "ocr_pack_count",
    "has_ocr",
]
SUCCESS_STATUSES = frozenset({"downloaded", "cached"})


@dataclass(frozen=True)
class OCRDetection:
    """One OCR token or line, with pixel coordinates in its input variant."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    variant: str = "original"


@dataclass(frozen=True)
class OCRConfig:
    """Runtime and filtering controls shared by both OCR engines."""

    engine: str = "auto"
    languages: tuple[str, ...] = ("eng",)
    workers: int = max(1, min(4, os.cpu_count() or 1))
    min_confidence: float = 0.35
    early_exit_confidence: float = 0.60
    adaptive_ocr: bool = True
    timeout_seconds: float = 45.0
    psm_modes: tuple[int, ...] = (6, 11)
    gpu: bool = False
    easyocr_download_enabled: bool = False
    easyocr_model_dir: str | None = None
    tesseract_cmd: str = "tesseract"
    min_ocr_side: int = 900
    max_upscale: float = 4.0
    include_all_images: bool = False
    fail_on_missing_images: bool = False
    max_failure_rate: float = 1.0

    def __post_init__(self) -> None:
        engine = self.engine.lower()
        if engine not in {"auto", "tesseract", "easyocr"}:
            raise ValueError("engine must be one of: auto, tesseract, easyocr")
        if not self.languages or any(not str(x).strip() for x in self.languages):
            raise ValueError("At least one non-empty OCR language is required.")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= self.early_exit_confidence <= 1.0:
            raise ValueError("early_exit_confidence must be between 0 and 1")
        if self.early_exit_confidence < self.min_confidence:
            raise ValueError("early_exit_confidence must be >= min_confidence")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.min_ocr_side < 1 or self.max_upscale < 1.0:
            raise ValueError("min_ocr_side must be positive and max_upscale must be >= 1")
        if not 0.0 <= self.max_failure_rate <= 1.0:
            raise ValueError("max_failure_rate must be between 0 and 1")
        if not self.psm_modes or any(mode < 3 or mode > 13 for mode in self.psm_modes):
            raise ValueError("psm_modes must contain Tesseract page modes from 3 through 13")


class OCREngine(Protocol):
    """Minimal interface used by the pipeline and by test doubles."""

    @property
    def signature(self) -> str: ...

    def recognize(self, image: Image.Image, variant: str) -> list[OCRDetection]: ...


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_id(value: Any) -> str:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if value is None or (isinstance(missing, (bool, np.bool_)) and bool(missing)):
        raise ValueError("PRODUCT_ID values must be non-missing.")
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("PRODUCT_ID values must be finite.")
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("PRODUCT_ID values must be non-empty.")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TesseractEngine:
    """Tesseract CLI backend that needs no ``pytesseract`` dependency."""

    def __init__(self, config: OCRConfig):
        executable = shutil.which(config.tesseract_cmd)
        if executable is None:
            raise RuntimeError(
                f"Tesseract executable {config.tesseract_cmd!r} was not found on PATH. "
                "Install Tesseract, pass --tesseract-cmd, or use --engine easyocr."
            )
        self._executable = executable
        tesseract_languages = ["eng" if code.lower() == "en" else code for code in config.languages]
        self._languages = "+".join(tesseract_languages)
        self._psm_modes = config.psm_modes
        self._timeout = config.timeout_seconds
        self._min_confidence = config.min_confidence
        self._early_exit_confidence = config.early_exit_confidence
        self._adaptive = config.adaptive_ocr
        self.max_parallelism = config.workers
        completed = subprocess.run(
            [self._executable, "--version"],
            capture_output=True,
            text=True,
            timeout=min(10.0, self._timeout),
            check=False,
        )
        first_line = (completed.stdout or completed.stderr).splitlines()
        version = first_line[0].strip() if first_line else "unknown"
        self._signature = (
            f"tesseract|{version}|lang={self._languages}|psm="
            + ",".join(map(str, self._psm_modes))
        )

    @property
    def signature(self) -> str:
        return self._signature

    def recognize(self, image: Image.Image, variant: str) -> list[OCRDetection]:
        payload = io.BytesIO()
        image.save(payload, format="PNG", optimize=False)
        detections: list[OCRDetection] = []
        for psm in self._psm_modes:
            command = [
                self._executable,
                "stdin",
                "stdout",
                "-l",
                self._languages,
                "--oem",
                "3",
                "--psm",
                str(psm),
                "tsv",
            ]
            try:
                completed = subprocess.run(
                    command,
                    input=payload.getvalue(),
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Tesseract exceeded {self._timeout:g}s on {variant} (PSM {psm})."
                ) from exc
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Tesseract failed on {variant} (PSM {psm}): {detail}")
            text = completed.stdout.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            line_groups: dict[tuple[str, str, str, str], list[tuple[str, float, tuple[int, int, int, int]]]] = {}
            for row in reader:
                token = (row.get("text") or "").strip()
                if not token:
                    continue
                try:
                    confidence = float(row.get("conf", "-1")) / 100.0
                    left = int(row["left"])
                    top = int(row["top"])
                    width = int(row["width"])
                    height = int(row["height"])
                except (KeyError, TypeError, ValueError):
                    continue
                if confidence < 0:
                    continue
                line_key = tuple(
                    str(row.get(column, "0"))
                    for column in ("page_num", "block_num", "par_num", "line_num")
                )
                line_groups.setdefault(line_key, []).append(
                    (
                        token,
                        max(0.0, min(1.0, confidence)),
                        (left, top, left + width, top + height),
                    )
                )
            for words in line_groups.values():
                # TSV rows are already in reading order. Reconstructing lines is
                # essential because the number and its unit are often separate tokens.
                line_text = " ".join(word[0] for word in words)
                confidence = float(np.mean([word[1] for word in words]))
                bbox = (
                    min(word[2][0] for word in words),
                    min(word[2][1] for word in words),
                    max(word[2][2] for word in words),
                    max(word[2][3] for word in words),
                )
                detections.append(
                    OCRDetection(line_text, confidence, bbox, f"{variant}:psm{psm}")
                )
            # PSM 6 is usually enough for a centered package front. Sparse-text
            # PSMs are fallbacks, not mandatory duplicate work.
            if self._adaptive and _has_reliable_physical_entity(
                detections, self._min_confidence, self._early_exit_confidence
            ):
                break
        return detections


class EasyOCREngine:
    """Lazy optional EasyOCR backend. One reader is guarded for thread safety."""

    def __init__(self, config: OCRConfig):
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Run `pip install easyocr`, or use "
                "the default Tesseract backend."
            ) from exc
        model_dir = config.easyocr_model_dir
        kwargs: dict[str, Any] = {
            "gpu": config.gpu,
            "download_enabled": config.easyocr_download_enabled,
            "verbose": False,
        }
        if model_dir:
            kwargs["model_storage_directory"] = model_dir
        easy_languages = ["en" if code.lower() == "eng" else code for code in config.languages]
        self._easyocr = easyocr
        self._languages = easy_languages
        self._reader_kwargs = kwargs
        self._thread_local = threading.local()
        self._initialization_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        # Construct one reader immediately so missing models/configuration fail
        # before a long run starts. CPU workers lazily claim/create one reader
        # each; GPU deliberately stays single-reader to avoid VRAM duplication.
        self._seed_reader = easyocr.Reader(easy_languages, **kwargs)
        self._seed_claimed = False
        self._gpu = config.gpu
        self.max_parallelism = 1 if config.gpu else config.workers
        self._signature = (
            f"easyocr|lang={'+'.join(easy_languages)}|gpu={int(config.gpu)}|ocr={OCR_VERSION}"
        )

    @property
    def signature(self) -> str:
        return self._signature

    def _cpu_reader(self) -> Any:
        reader = getattr(self._thread_local, "reader", None)
        if reader is not None:
            return reader
        with self._initialization_lock:
            if not self._seed_claimed:
                reader = self._seed_reader
                self._seed_claimed = True
            else:
                reader = self._easyocr.Reader(self._languages, **self._reader_kwargs)
        self._thread_local.reader = reader
        return reader

    def recognize(self, image: Image.Image, variant: str) -> list[OCRDetection]:
        array = np.asarray(image.convert("RGB"))
        if self._gpu:
            with self._inference_lock:
                results = self._seed_reader.readtext(array, detail=1, paragraph=False)
        else:
            results = self._cpu_reader().readtext(array, detail=1, paragraph=False)
        detections: list[OCRDetection] = []
        for polygon, text, confidence in results:
            try:
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                detections.append(
                    OCRDetection(str(text).strip(), float(confidence), bbox, variant)
                )
            except (TypeError, ValueError, IndexError):
                continue
        return detections


def create_ocr_engine(config: OCRConfig) -> OCREngine:
    """Create the requested engine; ``auto`` prefers the lightweight CLI."""

    if config.engine.lower() == "tesseract":
        return TesseractEngine(config)
    if config.engine.lower() == "easyocr":
        return EasyOCREngine(config)
    try:
        return TesseractEngine(config)
    except RuntimeError as tesseract_error:
        try:
            return EasyOCREngine(config)
        except RuntimeError as easyocr_error:
            raise RuntimeError(
                "No usable OCR engine was found. "
                f"Tesseract: {tesseract_error} EasyOCR: {easyocr_error}"
            ) from easyocr_error


def prepare_image_variants(image: Image.Image, config: OCRConfig) -> list[tuple[str, Image.Image]]:
    """Build conservative OCR views without altering the cached source image."""

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if min(rgb.size) <= 0:
        raise ValueError("Image has an invalid zero-sized dimension.")
    scale = min(config.max_upscale, max(1.0, config.min_ocr_side / min(rgb.size)))
    if scale > 1.01:
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
    gray = ImageOps.autocontrast(ImageOps.grayscale(rgb), cutoff=1)
    sharp = ImageEnhance.Contrast(gray.filter(ImageFilter.SHARPEN)).enhance(1.45)
    # A fixed midpoint is intentionally a secondary view: the grayscale view
    # remains available for packages with colored or low-contrast backgrounds.
    threshold = int(np.asarray(sharp.resize((64, 64))).mean())
    binary = sharp.point(lambda px: 255 if px >= threshold else 0, mode="1").convert("L")
    return [("rgb", rgb), ("gray", sharp), ("binary", binary)]


_PROMOTION_RE = re.compile(
    r"(?i)\b(?:100\s*%\s*satisfaction|best\s*seller|money\s*back|free\s*shipping|"
    r"limited\s*(?:time\s*)?offer|special\s*offer|premium\s+quality|buy\s+now|"
    r"save\s*\d+\s*%|\d+\s*%\s*off)\b"
)
_BUY_GET_RE = re.compile(
    r"(?i)\bbuy\s+(\d+)\s*(?:&|and)?\s*get\s+(\d+)\s+free\b"
)
_NUTRITION_RE = re.compile(
    r"(?i)\b(?:nutrition|nutritional|serving(?:s|\s+size)?|calories?|energy|protein|"
    r"carbohydrate|sugars?|sodium|cholesterol|daily\s+value|ingredients?|per\s+100\s*[gm]l?)\b"
)
_NET_CONTEXT_RE = re.compile(
    r"(?i)\b(?:net\s*(?:wt|weight|vol(?:ume)?)|net\s+contents?|contents?|quantity|qty)\b"
)
_UNIT_RE = re.compile(
    r"(?i)(?<![a-z])\d+(?:[.,]\d+)?\s*(?:kg|kgs|kilograms?|g|gm|gms|grams?|"
    r"mg|lb|lbs|oz|ounces?|l|ltr|lit(?:er|re)s?|ml|millilit(?:er|re)s?|cl)(?![a-z])"
)
_EXTENDED_UNIT_RE = re.compile(
    r"(?i)(?<![a-z])\d+(?:[.,]\d+)?\s*[- ]?\s*(?:mm|cm|m|met(?:er|re)s?|"
    r"inch(?:es)?|in|ft|feet|\"|kilowatts?|kw|watts?|w|kv|volts?|v|mah|ah|wh|"
    r"mb|gb|tb)(?![a-z])"
)
_PACK_RE = re.compile(
    r"(?i)\b(?:pack|set|box|case|bundle|combo)\s*(?:of\s*)?\d+\b|"
    r"\b\d+\s*(?:pack|packs|count|ct|pcs|pieces)\b|"
    r"\b\d+\s*[x×*]\s*\d+(?:[.,]\d+)?\s*(?:kg|g|gm|mg|l|ml|cl|oz|lb)\b"
)


def normalize_ocr_text(text: str) -> str:
    """Normalize Unicode and common unit-spacing errors without inventing text."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("×", "x").replace("✕", "x").replace("•", " ")
    value = re.sub(r"(?i)\b(m|k|c)\s+(l|g)\b", r"\1\2", value)
    value = re.sub(r"(?i)(\d)\s*([.,])\s*(\d)", r"\1\2\3", value)
    # Correct O/o only inside a number immediately followed by a physical unit.
    def repair_number(match: re.Match[str]) -> str:
        number = match.group(1).replace("O", "0").replace("o", "0")
        return number + " " + match.group(2)

    value = re.sub(
        r"(?i)\b([0-9Oo]+(?:[.,][0-9Oo]+)?)\s*(kg|g|gm|mg|ml|cl|l|oz|lb)\b",
        repair_number,
        value,
    )
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip(" |;,_-")


def filter_packaging_text(lines: Iterable[str]) -> tuple[str, list[str]]:
    """Keep likely front-of-pack quantity evidence and reject common traps.

    Returns ``(clean_text, rejected_lines)``. Text-bearing non-quantity lines
    are preserved only when they sit next to useful packaging evidence; this
    avoids feeding entire ingredient/nutrition panels to the quantity parser.
    """

    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = normalize_ocr_text(raw)
        if not line:
            continue
        # Convert a promotion that contains real pack information before
        # removing marketing phrases. "Buy 1 get 1 free" is a pack of two.
        line = _BUY_GET_RE.sub(
            lambda match: f"pack of {int(match.group(1)) + int(match.group(2))}", line
        )
        cleaned_line = normalize_ocr_text(_PROMOTION_RE.sub(" ", line))
        key = re.sub(r"\W+", "", cleaned_line).casefold()
        if not key:
            rejected.append(line)
            continue
        if key in seen:
            continue
        seen.add(key)
        has_strong_context = bool(_NET_CONTEXT_RE.search(cleaned_line) or _PACK_RE.search(cleaned_line))
        # Nutrition/ingredient measurements are unsafe unless the same OCR
        # line explicitly identifies net contents or a package count.
        if _NUTRITION_RE.search(cleaned_line) and not has_strong_context:
            rejected.append(line)
            continue
        if (
            _UNIT_RE.search(cleaned_line)
            or _EXTENDED_UNIT_RE.search(cleaned_line)
            or _PACK_RE.search(cleaned_line)
            or _NET_CONTEXT_RE.search(cleaned_line)
        ):
            accepted.append(cleaned_line)
        else:
            rejected.append(line)
    return " | ".join(accepted), rejected


def harvest_ocr_entities(text: str) -> dict[str, float]:
    """Parse all supported OCR entities through the catalog feature pipeline."""

    specs = extract_row_physical_specs(text, None, None)
    weight = float(specs["total_weight_g"]) if int(specs["has_weight"]) else math.nan
    volume = float(specs["total_volume_ml"]) if int(specs["has_volume"]) else math.nan
    # The feature parser's neutral default is one. OCR must not claim a pack of
    # one unless a multiplier/count was explicitly visible.
    pack = float(specs["pack_count"]) if int(specs["is_multipack"]) else math.nan
    entities = {
        "ocr_weight_g": weight,
        "ocr_volume_ml": volume,
        "ocr_pack_count": pack,
        "ocr_dim_length_cm": float(specs["dim_length_cm"]),
        "ocr_dim_width_cm": float(specs["dim_width_cm"]),
        "ocr_dim_height_cm": float(specs["dim_height_cm"]),
        "ocr_power_watts": float(specs["power_watts"]),
        "ocr_voltage_volts": float(specs["voltage_volts"]),
        "ocr_battery_mah": float(specs["battery_mah"]),
        "ocr_storage_gb": float(specs["storage_gb"]),
    }
    return entities


def harvest_ocr_quantities(text: str) -> dict[str, float]:
    """Return the stable three-quantity subset used by public OCR outputs."""

    entities = harvest_ocr_entities(text)
    return {
        key: entities[key]
        for key in ("ocr_weight_g", "ocr_volume_ml", "ocr_pack_count")
    }


def _has_reliable_physical_entity(
    detections: Iterable[OCRDetection],
    min_confidence: float,
    early_exit_confidence: float,
) -> bool:
    """Return true when current OCR is good enough to skip fallback passes."""

    merged = merge_detections(detections, min_confidence)
    clean_text, _ = filter_packaging_text(d.text for d in merged)
    if not clean_text:
        return False
    entities = harvest_ocr_entities(clean_text)
    if not any(not math.isnan(value) for value in entities.values()):
        return False
    accepted_keys = {
        re.sub(r"\W+", "", normalize_ocr_text(part)).casefold()
        for part in clean_text.split("|")
        if part.strip()
    }
    confidences = [
        detection.confidence
        for detection in merged
        if re.sub(r"\W+", "", normalize_ocr_text(detection.text)).casefold()
        in accepted_keys
    ]
    return bool(confidences and float(np.mean(confidences)) >= early_exit_confidence)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def merge_detections(
    detections: Iterable[OCRDetection], min_confidence: float
) -> list[OCRDetection]:
    """Confidence-filter and deduplicate repeated OCR across image variants."""

    ordered = sorted(
        (d for d in detections if d.text.strip() and d.confidence >= min_confidence),
        key=lambda d: (-d.confidence, d.bbox[1], d.bbox[0]),
    )
    kept: list[OCRDetection] = []
    for candidate in ordered:
        normalized = re.sub(r"\W+", "", normalize_ocr_text(candidate.text)).casefold()
        duplicate = False
        for existing in kept:
            other = re.sub(r"\W+", "", normalize_ocr_text(existing.text)).casefold()
            # Coordinates from differently scaled variants are not always
            # comparable, so exact normalized text is itself sufficient.
            if normalized == other and normalized:
                duplicate = True
                break
            if candidate.variant == existing.variant and _bbox_iou(candidate.bbox, existing.bbox) > 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda d: (d.bbox[1], d.bbox[0], -d.confidence))


def _union_bbox(detections: Sequence[OCRDetection]) -> list[int] | None:
    if not detections:
        return None
    return [
        min(d.bbox[0] for d in detections),
        min(d.bbox[1] for d in detections),
        max(d.bbox[2] for d in detections),
        max(d.bbox[3] for d in detections),
    ]


def _resolve_inside(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes image root: {relative!r}") from exc
    return candidate


def build_image_index(
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    image_root: Path,
    id_column: str = "PRODUCT_ID",
    include_all_images: bool = False,
) -> dict[str, list[tuple[Path, str | None]]]:
    """Validate and align downloader manifest entries to unique catalog IDs."""

    if id_column not in catalog.columns:
        raise ValueError(f"Catalog is missing ID column {id_column!r}.")
    if id_column not in manifest.columns:
        raise ValueError(f"Image manifest is missing ID column {id_column!r}.")
    required = {"status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Image manifest is missing columns: {missing}")
    path_column = "by_id_path" if "by_id_path" in manifest.columns else "relative_path"
    if path_column not in manifest.columns:
        raise ValueError("Image manifest must contain by_id_path or relative_path.")

    catalog_keys = [_canonical_id(value) for value in catalog[id_column]]
    duplicates = pd.Series(catalog_keys).duplicated(keep=False)
    if duplicates.any():
        sample = sorted(set(pd.Series(catalog_keys)[duplicates].tolist()))[:5]
        raise ValueError(f"Catalog IDs must be unique; duplicates include {sample}.")
    allowed = set(catalog_keys)
    work = manifest.copy()
    work["__key"] = [_canonical_id(value) for value in work[id_column]]
    work = work[work["__key"].isin(allowed) & work["status"].isin(SUCCESS_STATUSES)]
    if "image_index" not in work.columns:
        work["image_index"] = 0
    work["image_index"] = pd.to_numeric(work["image_index"], errors="coerce").fillna(0).astype(int)
    work = work.sort_values(["__key", "image_index"], kind="stable")
    ambiguous = work.duplicated(["__key", "image_index"], keep=False)
    if ambiguous.any():
        sample = work.loc[ambiguous, ["__key", "image_index"]].head(5).values.tolist()
        raise ValueError(
            "Image manifest has multiple successful paths for the same product/image index; "
            f"examples: {sample}"
        )
    if not include_all_images:
        work = work[work["image_index"] == 0].drop_duplicates("__key", keep="first")

    index: dict[str, list[tuple[Path, str | None]]] = {key: [] for key in catalog_keys}
    for _, row in work.iterrows():
        relative = row.get(path_column)
        if not isinstance(relative, str) or not relative.strip():
            continue
        path = _resolve_inside(image_root, relative)
        stated_hash = row.get("content_sha256")
        content_hash = stated_hash if isinstance(stated_hash, str) and len(stated_hash) == 64 else None
        index[row["__key"]].append((path, content_hash))
    return index


class OCRCache:
    """Small SQLite checkpoint store; safe across interrupted reruns."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS ocr_cache ("
            "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload FROM ocr_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO ocr_cache(cache_key, payload, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False, allow_nan=False), time.time()),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "OCRCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _cache_key(
    product_key: str,
    paths: Sequence[tuple[Path, str | None]],
    engine_signature: str,
    config: OCRConfig,
) -> str:
    image_parts: list[str] = []
    for path, stated_hash in paths:
        if not path.is_file():
            image_parts.append(f"missing:{path.name}")
        else:
            image_parts.append(stated_hash or _sha256_file(path))
    relevant = {
        "version": OCR_VERSION,
        "min_confidence": config.min_confidence,
        "early_exit_confidence": config.early_exit_confidence,
        "adaptive_ocr": config.adaptive_ocr,
        "min_ocr_side": config.min_ocr_side,
        "max_upscale": config.max_upscale,
        "engine": engine_signature,
    }
    raw = json.dumps([product_key, image_parts, relevant], sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _recognize_product(
    product_key: str,
    paths: Sequence[tuple[Path, str | None]],
    engine: OCREngine,
    config: OCRConfig,
) -> dict[str, Any]:
    started = time.monotonic()
    all_detections: list[OCRDetection] = []
    errors: list[str] = []
    images_processed = 0
    variant_passes = 0
    adaptive_exit = False
    for path, _ in paths:
        reliable_entity_found = False
        if not path.is_file():
            errors.append(f"missing image: {path}")
            continue
        try:
            with Image.open(path) as source:
                variants = prepare_image_variants(source, config)
            for name, variant in variants:
                variant_passes += 1
                all_detections.extend(engine.recognize(variant, name))
                if config.adaptive_ocr and _has_reliable_physical_entity(
                    all_detections, config.min_confidence, config.early_exit_confidence
                ):
                    reliable_entity_found = True
                    adaptive_exit = True
                    break
            images_processed += 1
            if reliable_entity_found:
                break
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    detections = merge_detections(all_detections, config.min_confidence)
    clean_text, rejected = filter_packaging_text(d.text for d in detections)
    entities = harvest_ocr_entities(clean_text)
    useful_tokens = {
        re.sub(r"\W+", "", normalize_ocr_text(part)).casefold()
        for part in clean_text.split("|")
        if part.strip()
    }
    useful_detections = [
        d
        for d in detections
        if re.sub(r"\W+", "", normalize_ocr_text(d.text)).casefold() in useful_tokens
    ]
    payload: dict[str, Any] = {
        "product_key": product_key,
        "ocr_text": clean_text,
        **entities,
        "has_ocr": int(bool(clean_text)),
        "status": "ok" if clean_text else ("no_text" if images_processed else "image_error"),
        "error": " | ".join(errors)[:4000],
        "images_requested": len(paths),
        "images_processed": images_processed,
        "variant_passes": variant_passes,
        "adaptive_exit": adaptive_exit,
        "detections_kept": len(detections),
        "detections_rejected": len(rejected),
        "mean_confidence": (
            float(np.mean([d.confidence for d in detections])) if detections else None
        ),
        "packaging_bbox": _union_bbox(useful_detections),
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }
    return payload


def run_ocr_catalog(
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    image_root: Path,
    engine: OCREngine,
    config: OCRConfig,
    cache_path: Path,
    id_column: str = "PRODUCT_ID",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """OCR a catalog with deterministic alignment, caching, and diagnostics."""

    image_index = build_image_index(
        catalog, manifest, image_root, id_column, config.include_all_images
    )
    keys = [_canonical_id(value) for value in catalog[id_column]]
    completed: dict[str, dict[str, Any]] = {}
    pending: dict[Any, tuple[str, str]] = {}
    cache_hits = 0

    with OCRCache(cache_path) as cache:
        effective_workers = max(
            1, min(config.workers, int(getattr(engine, "max_parallelism", config.workers)))
        )
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="ocr") as pool:
            for key in keys:
                paths = image_index[key]
                if not paths:
                    completed[key] = {
                        "product_key": key,
                        "ocr_text": "",
                        "ocr_weight_g": None,
                        "ocr_volume_ml": None,
                        "ocr_pack_count": None,
                        "has_ocr": 0,
                        "status": "missing_image",
                        "error": "No successful image-manifest entry for this product.",
                        "images_requested": 0,
                        "images_processed": 0,
                        "variant_passes": 0,
                        "adaptive_exit": False,
                        "detections_kept": 0,
                        "detections_rejected": 0,
                        "mean_confidence": None,
                        "packaging_bbox": None,
                        "elapsed_seconds": 0.0,
                        "cache_hit": False,
                    }
                    continue
                key_hash = _cache_key(key, paths, engine.signature, config)
                cached = cache.get(key_hash)
                if cached is not None:
                    cached["cache_hit"] = True
                    completed[key] = cached
                    cache_hits += 1
                    continue
                future = pool.submit(_recognize_product, key, paths, engine, config)
                pending[future] = (key, key_hash)

            for future in as_completed(pending):
                key, key_hash = pending[future]
                try:
                    result = future.result()
                except Exception as exc:  # Keep one damaged item from killing a long run.
                    result = {
                        "product_key": key,
                        "ocr_text": "",
                        "ocr_weight_g": None,
                        "ocr_volume_ml": None,
                        "ocr_pack_count": None,
                        "has_ocr": 0,
                        "status": "worker_error",
                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                        "images_requested": len(image_index[key]),
                        "images_processed": 0,
                        "variant_passes": 0,
                        "adaptive_exit": False,
                        "detections_kept": 0,
                        "detections_rejected": 0,
                        "mean_confidence": None,
                        "packaging_bbox": None,
                        "elapsed_seconds": 0.0,
                    }
                result["cache_hit"] = False
                completed[key] = result
                # JSON has no NaN; cache null and restore pandas NaN later.
                cacheable = {
                    name: (None if isinstance(value, float) and math.isnan(value) else value)
                    for name, value in result.items()
                }
                cache.put(key_hash, cacheable)

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for original_id, key in zip(catalog[id_column].tolist(), keys):
        result = completed[key]
        rows.append(
            {
                "PRODUCT_ID": original_id,
                "ocr_text": result.get("ocr_text") or "",
                "ocr_weight_g": result.get("ocr_weight_g"),
                "ocr_volume_ml": result.get("ocr_volume_ml"),
                "ocr_pack_count": result.get("ocr_pack_count"),
                "has_ocr": result.get("has_ocr", 0),
            }
        )
        diagnostics.append({"PRODUCT_ID": original_id, **result, "engine": engine.signature})

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in ("ocr_weight_g", "ocr_volume_ml", "ocr_pack_count"):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    output["ocr_text"] = output["ocr_text"].fillna("").astype("string")
    output["has_ocr"] = output["has_ocr"].fillna(0).astype("int8")
    diagnostics_frame = pd.DataFrame(diagnostics)
    failure_mask = (
        diagnostics_frame["status"].isin(["missing_image", "image_error", "worker_error"])
        if "status" in diagnostics_frame
        else pd.Series([], dtype=bool)
    )
    failure_rate = float(failure_mask.mean()) if len(diagnostics_frame) else 0.0
    if (
        config.fail_on_missing_images
        and "status" in diagnostics_frame
        and (diagnostics_frame["status"] == "missing_image").any()
    ):
        raise RuntimeError("At least one catalog row has no successful downloaded image.")
    if failure_rate > config.max_failure_rate:
        raise RuntimeError(
            f"OCR failure rate {failure_rate:.1%} exceeds maximum {config.max_failure_rate:.1%}."
        )
    summary = {
        "rows": len(output),
        "has_ocr_rows": int(output["has_ocr"].sum()),
        "quantity_rows": int(
            output[["ocr_weight_g", "ocr_volume_ml", "ocr_pack_count"]].notna().any(axis=1).sum()
        ),
        "extended_entity_rows": int(
            diagnostics_frame[
                [
                    column
                    for column in (
                        "ocr_dim_length_cm",
                        "ocr_power_watts",
                        "ocr_voltage_volts",
                        "ocr_battery_mah",
                        "ocr_storage_gb",
                    )
                    if column in diagnostics_frame
                ]
            ].notna().any(axis=1).sum()
        )
        if len(diagnostics_frame)
        else 0,
        "cache_hits": cache_hits,
        "workers_used": effective_workers,
        "failure_rate": failure_rate,
        "status_counts": {
            str(k): int(v)
            for k, v in (
                diagnostics_frame["status"].value_counts().items()
                if "status" in diagnostics_frame
                else []
            )
        },
        "engine": engine.signature,
        "ocr_version": OCR_VERSION,
    }
    return output, diagnostics_frame, summary


def backfill_catalog_quantities(
    catalog: pd.DataFrame,
    ocr: pd.DataFrame,
    id_column: str = "PRODUCT_ID",
    title_column: str = "TITLE",
    pack_column: str = "PACK_SIZE",
    description_column: str = "DESCRIPTION",
) -> pd.DataFrame:
    """Resolve physical specs as Title > OCR > Pack Size > Description.

    This prevents serving sizes buried in seller descriptions from overriding
    a high-confidence net quantity printed on the package. The input catalog is
    never modified in place and its row order is preserved.
    """

    if id_column not in catalog or "PRODUCT_ID" not in ocr:
        raise ValueError("Both catalog and OCR data must contain PRODUCT_ID.")
    catalog_keys = [_canonical_id(v) for v in catalog[id_column]]
    ocr_keys = [_canonical_id(v) for v in ocr["PRODUCT_ID"]]
    if pd.Series(catalog_keys).duplicated().any() or pd.Series(ocr_keys).duplicated().any():
        raise ValueError("PRODUCT_ID must be unique in catalog and OCR data.")
    ocr_map = ocr.copy()
    ocr_map["__key"] = ocr_keys
    ocr_map = ocr_map.set_index("__key")
    missing_ocr = sorted(set(catalog_keys) - set(ocr_keys))
    if missing_ocr:
        raise ValueError(f"OCR output is missing {len(missing_ocr)} catalog IDs.")

    result = catalog.copy()
    metric_definitions = {
        "weight_g": ("total_weight_g", "has_weight"),
        "volume_ml": ("total_volume_ml", "has_volume"),
        "pack_count": ("pack_count", "is_multipack"),
        "dim_length_cm": ("dim_length_cm", "has_dimensions"),
        "dim_width_cm": ("dim_width_cm", "has_dimensions"),
        "dim_height_cm": ("dim_height_cm", "has_dimensions"),
        "power_watts": ("power_watts", None),
        "voltage_volts": ("voltage_volts", None),
        "battery_mah": ("battery_mah", None),
        "storage_gb": ("storage_gb", None),
    }
    effective: dict[str, list[float]] = {name: [] for name in metric_definitions}
    sources: dict[str, list[str]] = {name: [] for name in metric_definitions}

    def finite_float(value: Any) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return math.nan
        return converted if math.isfinite(converted) else math.nan

    def spec_value(specs: dict[str, Any], value_key: str, flag_key: str | None) -> float:
        if flag_key is not None and not int(specs[flag_key]):
            return math.nan
        return finite_float(specs[value_key])

    for position, key in enumerate(catalog_keys):
        row = catalog.iloc[position]
        ocr_row = ocr_map.loc[key]
        ocr_text_value = ocr_row.get("ocr_text")
        try:
            ocr_text_missing = bool(pd.isna(ocr_text_value))
        except (TypeError, ValueError):
            ocr_text_missing = False
        ocr_text = "" if ocr_text_value is None or ocr_text_missing else str(ocr_text_value)
        source_specs = [
            ("title", extract_row_physical_specs(row.get(title_column), None, None)),
            ("ocr", extract_row_physical_specs(ocr_text, None, None)),
            ("pack_size", extract_row_physical_specs(None, row.get(pack_column), None)),
            ("description", extract_row_physical_specs(None, None, row.get(description_column))),
        ]

        # Prefer already materialized public OCR quantities when available.
        ocr_overrides = {
            "weight_g": finite_float(ocr_row.get("ocr_weight_g")),
            "volume_ml": finite_float(ocr_row.get("ocr_volume_ml")),
            "pack_count": finite_float(ocr_row.get("ocr_pack_count")),
        }
        for metric_name, (value_key, flag_key) in metric_definitions.items():
            selected = math.nan
            selected_source = "missing"
            for source_name, specs in source_specs:
                if source_name == "ocr" and metric_name in ocr_overrides:
                    candidate = ocr_overrides[metric_name]
                else:
                    candidate = spec_value(specs, value_key, flag_key)
                if not math.isnan(candidate):
                    selected = candidate
                    selected_source = source_name
                    break
            effective[metric_name].append(selected)
            sources[metric_name].append(selected_source)

    for metric_name in metric_definitions:
        result[f"effective_{metric_name}"] = np.asarray(effective[metric_name], dtype=np.float32)
    result["weight_source"] = sources["weight_g"]
    result["volume_source"] = sources["volume_ml"]
    result["pack_source"] = sources["pack_count"]
    result["dimension_source"] = sources["dim_length_cm"]
    result["power_source"] = sources["power_watts"]
    result["voltage_source"] = sources["voltage_volts"]
    result["battery_source"] = sources["battery_mah"]
    result["storage_source"] = sources["storage_gb"]
    ocr_text_values: list[str] = []
    for key in catalog_keys:
        value = ocr_map.loc[key].get("ocr_text")
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        ocr_text_values.append("" if value is None or is_missing else str(value))
    result["ocr_text"] = ocr_text_values
    return result


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}; use CSV or Parquet.")


def _run_split(
    name: str,
    catalog_path: Path,
    manifest_path: Path,
    image_root: Path,
    output_path: Path,
    diagnostics_path: Path,
    cache_path: Path,
    engine: OCREngine,
    config: OCRConfig,
    id_column: str,
    backfilled_path: Path | None,
    title_column: str,
    pack_column: str,
    description_column: str,
) -> dict[str, Any]:
    catalog = _read_table(catalog_path)
    manifest = _read_table(manifest_path)
    output, diagnostics, summary = run_ocr_catalog(
        catalog, manifest, image_root, engine, config, cache_path, id_column
    )
    _atomic_parquet(output, output_path)
    _atomic_parquet(diagnostics, diagnostics_path)
    if backfilled_path is not None:
        backfilled = backfill_catalog_quantities(
            catalog,
            output,
            id_column=id_column,
            title_column=title_column,
            pack_column=pack_column,
            description_column=description_column,
        )
        _atomic_parquet(backfilled, backfilled_path)
    print(
        f"{name}: {len(output):,} rows, {int(output['has_ocr'].sum()):,} with OCR, "
        f"{summary['quantity_rows']:,} with recovered quantities -> {output_path}"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest packaging text and physical quantities from downloaded product images."
    )
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/processed/images/train/image_manifest.parquet"),
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("data/processed/images/test/image_manifest.parquet"),
    )
    parser.add_argument("--train-image-root", type=Path, default=Path("data/processed/images/train"))
    parser.add_argument("--test-image-root", type=Path, default=Path("data/processed/images/test"))
    parser.add_argument("--output-train", type=Path, default=Path("data/processed/ocr_train.parquet"))
    parser.add_argument("--output-test", type=Path, default=Path("data/processed/ocr_test.parquet"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("reports/ocr"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/ocr_cache.sqlite3"))
    parser.add_argument("--report", type=Path, default=Path("reports/ocr/ocr_run.json"))
    parser.add_argument("--backfilled-train", type=Path)
    parser.add_argument("--backfilled-test", type=Path)
    parser.add_argument("--id-column", default="PRODUCT_ID")
    parser.add_argument("--title-column", default="TITLE")
    parser.add_argument("--pack-column", default="PACK_SIZE")
    parser.add_argument("--description-column", default="DESCRIPTION")
    parser.add_argument("--engine", choices=("auto", "tesseract", "easyocr"), default="auto")
    parser.add_argument("--languages", default="eng", help="Comma-separated language codes.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--early-exit-confidence", type=float, default=0.60)
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run every preprocessing/PSM fallback even after a reliable entity is found.",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--psm", default="6,11", help="Comma-separated Tesseract page modes.")
    parser.add_argument("--tesseract-cmd", default="tesseract")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR when available.")
    parser.add_argument("--easyocr-download", action="store_true")
    parser.add_argument("--easyocr-model-dir")
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--fail-on-missing-images", action="store_true")
    parser.add_argument("--max-failure-rate", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = OCRConfig(
        engine=args.engine,
        languages=tuple(part.strip() for part in args.languages.split(",") if part.strip()),
        workers=args.workers,
        min_confidence=args.min_confidence,
        early_exit_confidence=args.early_exit_confidence,
        adaptive_ocr=not args.exhaustive,
        timeout_seconds=args.timeout,
        psm_modes=tuple(int(part.strip()) for part in args.psm.split(",") if part.strip()),
        gpu=args.gpu,
        easyocr_download_enabled=args.easyocr_download,
        easyocr_model_dir=args.easyocr_model_dir,
        tesseract_cmd=args.tesseract_cmd,
        include_all_images=args.all_images,
        fail_on_missing_images=args.fail_on_missing_images,
        max_failure_rate=args.max_failure_rate,
    )
    engine = create_ocr_engine(config)
    started = time.time()
    summaries: dict[str, Any] = {}
    summaries["train"] = _run_split(
        "train",
        args.train,
        args.train_manifest,
        args.train_image_root,
        args.output_train,
        args.diagnostics_dir / "ocr_train_diagnostics.parquet",
        args.cache,
        engine,
        config,
        args.id_column,
        args.backfilled_train,
        args.title_column,
        args.pack_column,
        args.description_column,
    )
    summaries["test"] = _run_split(
        "test",
        args.test,
        args.test_manifest,
        args.test_image_root,
        args.output_test,
        args.diagnostics_dir / "ocr_test_diagnostics.parquet",
        args.cache,
        engine,
        config,
        args.id_column,
        args.backfilled_test,
        args.title_column,
        args.pack_column,
        args.description_column,
    )
    report = {
        "ocr_version": OCR_VERSION,
        "created_at_unix": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "config": asdict(config),
        "splits": summaries,
    }
    _atomic_json(report, args.report)
    print(f"Run report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
