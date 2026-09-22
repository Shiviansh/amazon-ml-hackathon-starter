"""Dense visual embeddings for product catalogs using CLIP or DINOv2.

The module consumes the immutable manifest produced by ``download_images.py``
and emits exactly one normalized vector per catalog product. It is designed for
long competition runs: image identity is content-hashed, chunks are committed
atomically, interrupted jobs resume only when provenance matches, and missing
or corrupt images are represented by a zero vector plus explicit flags.

No synthetic fallback is provided. If CLIP/DINOv2 cannot be loaded, execution
fails rather than silently generating features with different semantics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

try:
    from .metrics import smape
except ImportError:  # pragma: no cover - direct script execution.
    from metrics import smape


IMAGE_EMBEDDINGS_VERSION = "1.1"
CHECKPOINT_VERSION = "1.0"
SUCCESS_STATUSES = frozenset({"downloaded", "cached"})
SHA256_RE_LENGTH = 64
DEFAULT_MODELS = {
    "clip": "openai/clip-vit-base-patch32",
    "dinov2": "facebook/dinov2-base",
}
DEFAULT_REVISIONS = {
    "clip": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "dinov2": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}


@dataclass(frozen=True)
class ImageEmbeddingConfig:
    backend: str = "clip"
    model_name: str | None = None
    revision: str | None = None
    device: str = "auto"
    precision: str = "auto"
    batch_size: int = 32
    chunk_size: int = 512
    max_images_per_product: int = 1
    pooling: str = "weighted_mean"
    secondary_image_decay: float = 0.5
    output_dtype: str = "float16"
    decode_workers: int = max(1, min(4, os.cpu_count() or 1))
    max_image_pixels: int = 100_000_000
    verify_hashes: bool = False
    failure_policy: str = "zero"
    local_files_only: bool = False
    keep_checkpoints: bool = False
    empty_cuda_cache_each_chunk: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"clip", "dinov2"}:
            raise ValueError("backend must be 'clip' or 'dinov2'.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps.")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("precision must be auto, float32, float16, or bfloat16.")
        if self.batch_size < 1 or self.chunk_size < 1:
            raise ValueError("batch_size and chunk_size must be positive.")
        if self.max_images_per_product < 0:
            raise ValueError("max_images_per_product must be >= 0 (0 means all).")
        if self.pooling not in {"weighted_mean", "mean", "max"}:
            raise ValueError("pooling must be weighted_mean, mean, or max.")
        if not 0.0 < self.secondary_image_decay <= 1.0:
            raise ValueError("secondary_image_decay must be in (0, 1].")
        if self.output_dtype not in {"float16", "float32"}:
            raise ValueError("output_dtype must be float16 or float32.")
        if self.decode_workers < 1 or self.max_image_pixels < 1:
            raise ValueError("decode_workers and max_image_pixels must be positive.")
        if self.failure_policy not in {"zero", "error"}:
            raise ValueError("failure_policy must be zero or error.")

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or DEFAULT_MODELS[self.backend]

    @property
    def resolved_revision(self) -> str | None:
        if self.revision is not None:
            return self.revision
        if self.model_name is None or self.model_name == DEFAULT_MODELS[self.backend]:
            return DEFAULT_REVISIONS[self.backend]
        return None


@dataclass(frozen=True)
class ProductImage:
    path: Path
    content_sha256: str
    image_index: int


class ImageEncoder(Protocol):
    dimension: int
    signature: str
    description: str

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _canonical_id(value: Any) -> str:
    """Type-aware identity token; int 1 is intentionally not string '1'."""

    if _is_missing(value):
        raise ValueError("IDs must be non-missing.")
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("IDs must be finite.")
        return f"float:{numeric.hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _canonical_ids(values: Sequence[Any], label: str) -> list[str]:
    tokens = [_canonical_id(value) for value in values]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate IDs detected in {label}.")
    return tokens


def _resolve_inside(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes image root: {relative!r}") from exc
    return candidate


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_RE_LENGTH:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def build_product_image_index(
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    image_root: Path,
    config: ImageEmbeddingConfig,
    id_column: str = "PRODUCT_ID",
) -> tuple[list[str], dict[str, list[ProductImage]]]:
    """Strictly align successful manifest images to catalog IDs.

    The returned lists follow catalog order. Extra successful manifest IDs,
    duplicate image indices, hash mismatches, and path traversal fail closed.
    Products with no successful manifest entry remain present with an empty list.
    """

    if id_column not in catalog:
        raise ValueError(f"Catalog is missing ID column {id_column!r}.")
    if id_column not in manifest:
        raise ValueError(f"Image manifest is missing ID column {id_column!r}.")
    if "status" not in manifest:
        raise ValueError("Image manifest is missing status column.")
    # The content-addressed object path deduplicates repeated/hard-linked aliases
    # and is the fastest source for optional integrity verification.
    path_column = "relative_path" if "relative_path" in manifest else "by_id_path"
    if path_column not in manifest:
        raise ValueError("Image manifest must contain by_id_path or relative_path.")

    catalog_tokens = _canonical_ids(catalog[id_column].tolist(), "catalog")
    allowed = set(catalog_tokens)
    work = manifest.loc[manifest["status"].isin(SUCCESS_STATUSES)].copy()
    work["__token"] = [_canonical_id(value) for value in work[id_column]]
    extra = sorted(set(work["__token"]) - allowed)
    if extra:
        raise ValueError(
            f"Image manifest contains {len(extra)} successful IDs outside the catalog boundary."
        )
    if "image_index" not in work:
        work["image_index"] = 0
    numeric_indices = pd.to_numeric(work["image_index"], errors="coerce")
    if numeric_indices.isna().any() or (numeric_indices < 0).any():
        raise ValueError("Manifest image_index values must be non-negative integers.")
    if not np.equal(numeric_indices, np.floor(numeric_indices)).all():
        raise ValueError("Manifest image_index values must be integers.")
    work["image_index"] = numeric_indices.astype(int)
    ambiguous = work.duplicated(["__token", "image_index"], keep=False)
    if ambiguous.any():
        sample = work.loc[ambiguous, [id_column, "image_index"]].head(5).values.tolist()
        raise ValueError(f"Duplicate successful product/image indices in manifest: {sample}")
    work = work.sort_values(["__token", "image_index"], kind="stable")
    if config.max_images_per_product:
        work = work.groupby("__token", sort=False).head(config.max_images_per_product)

    if "content_sha256" not in work:
        work["content_sha256"] = None
    records: list[tuple[str, int, Path, Any]] = []
    selected = work[["__token", "image_index", path_column, "content_sha256"]]
    for token, image_number, relative, stated_hash in selected.itertuples(index=False, name=None):
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("A successful manifest row has no usable image path.")
        path = _resolve_inside(image_root, relative)
        records.append((token, int(image_number), path, stated_hash))

    paths_to_hash = sorted(
        {
            path
            for _, _, path, stated_hash in records
            if path.is_file() and (config.verify_hashes or not _valid_sha256(stated_hash))
        },
        key=str,
    )
    computed_hashes: dict[Path, str] = {}
    if paths_to_hash:
        with ThreadPoolExecutor(
            max_workers=config.decode_workers, thread_name_prefix="image-hash"
        ) as pool:
            computed_hashes = dict(zip(paths_to_hash, pool.map(_sha256_file, paths_to_hash)))

    index: dict[str, list[ProductImage]] = {token: [] for token in catalog_tokens}
    for token, image_number, path, stated_hash in records:
        if not path.is_file():
            content_hash = f"missing:{path.name}"
        elif path in computed_hashes:
            content_hash = computed_hashes[path]
            if (
                config.verify_hashes
                and _valid_sha256(stated_hash)
                and content_hash != str(stated_hash).lower()
            ):
                raise RuntimeError(f"Image SHA-256 mismatch for {path}.")
        else:
            content_hash = str(stated_hash).lower()
        index[token].append(ProductImage(path, content_hash, image_number))
    return catalog_tokens, index


def _dataset_fingerprint(
    raw_ids: Sequence[Any], tokens: Sequence[str], image_index: dict[str, list[ProductImage]]
) -> str:
    digest = hashlib.sha256()
    for raw_id, token in zip(raw_ids, tokens):
        raw = _canonical_id(raw_id).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        for image in image_index[token]:
            record = f"{image.image_index}:{image.content_sha256}".encode("utf-8")
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
    return digest.hexdigest()


def _hash_tokens(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        payload = token.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
    return requested


class TransformersImageEncoder:
    """Hugging Face CLIP/DINOv2 inference adapter with normalized outputs."""

    def __init__(self, config: ImageEmbeddingConfig):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection
        except ImportError as exc:
            raise RuntimeError(
                "Image embeddings require torch and transformers. Install requirements.txt first."
            ) from exc

        self._torch = torch
        self.device = _resolve_device(torch, config.device)
        if config.precision == "auto":
            self.precision = "float16" if self.device in {"cuda", "mps"} else "float32"
        else:
            self.precision = config.precision
        if self.device == "cpu" and self.precision == "float16":
            raise ValueError("float16 inference on CPU is unsupported; use float32 or bfloat16.")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._dtype = dtype_map[self.precision]
        model_name = config.resolved_model_name
        load_kwargs: dict[str, Any] = {
            "revision": config.resolved_revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        model_class = CLIPVisionModelWithProjection if config.backend == "clip" else AutoModel
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, **load_kwargs)
            self.model = model_class.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:
            mode = "local cache only" if config.local_files_only else "local cache/network"
            raise RuntimeError(
                f"Could not load {config.backend} image model {model_name!r} at revision "
                f"{config.resolved_revision or 'main'!r} using {mode}. Ensure the model is "
                "cached or network access is available, and verify the revision."
            ) from exc
        self.model.eval()
        self.model.to(self.device)
        # CPU bfloat16 should use autocast with FP32 master weights; eagerly
        # converting all CPU parameters can route unsupported operations poorly.
        if self.precision != "float32" and not (
            self.device == "cpu" and self.precision == "bfloat16"
        ):
            self.model.to(dtype=self._dtype)

        model_config = self.model.config
        if config.backend == "clip":
            dimension = getattr(model_config, "projection_dim", None)
        else:
            dimension = getattr(model_config, "hidden_size", None)
        if dimension is None or int(dimension) < 1:
            raise RuntimeError("Could not determine image embedding dimension from model config.")
        self.dimension = int(dimension)
        resolved_revision = (
            getattr(model_config, "_commit_hash", None)
            or config.resolved_revision
            or "unresolved"
        )
        signature_payload = {
            "version": IMAGE_EMBEDDINGS_VERSION,
            "backend": config.backend,
            "model_name": model_name,
            "resolved_revision": resolved_revision,
            "processor": f"{type(self.processor).__module__}.{type(self.processor).__qualname__}",
            "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "dimension": self.dimension,
            "precision": self.precision,
        }
        self.signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.description = (
            f"{config.backend}:{model_name}@{resolved_revision} "
            f"[{self.device}/{self.precision}, {self.dimension}d]"
        )
        self.resolved_revision = str(resolved_revision)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = self.processor(images=list(images), return_tensors="pt")
        moved_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            value = value.to(device=self.device, non_blocking=self.device == "cuda")
            if (
                value.is_floating_point()
                and self.precision != "float32"
                and not (self.device == "cpu" and self.precision == "bfloat16")
            ):
                value = value.to(dtype=self._dtype)
            moved_inputs[key] = value
        inputs = moved_inputs
        autocast: Any = nullcontext()
        if self.device == "cuda" and self.precision in {"float16", "bfloat16"}:
            autocast = self._torch.autocast(device_type="cuda", dtype=self._dtype)
        elif self.device == "cpu" and self.precision == "bfloat16":
            autocast = self._torch.autocast(device_type="cpu", dtype=self._dtype)
        with self._torch.inference_mode(), autocast:
            outputs = self.model(**inputs)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                vectors = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vectors = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                vectors = outputs.last_hidden_state[:, 0]
            else:
                raise RuntimeError("Model output contains no supported image representation.")
            vectors = self._torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
        array = vectors.detach().cpu().numpy().astype(np.float32, copy=False)
        if array.shape != (len(images), self.dimension) or not np.isfinite(array).all():
            raise RuntimeError(f"Image model returned invalid embedding shape/data: {array.shape}.")
        return array

    def release_memory(self, empty_cuda_cache: bool = False) -> None:
        """Release temporary tensors; optionally return cached CUDA blocks."""

        if self.device == "cuda" and empty_cuda_cache:
            self._torch.cuda.empty_cache()


def create_image_encoder(config: ImageEmbeddingConfig) -> TransformersImageEncoder:
    return TransformersImageEncoder(config)


def _decode_image(path: Path, max_pixels: int) -> tuple[Image.Image | None, str | None]:
    if not path.is_file():
        return None, f"missing image: {path}"
    try:
        with Image.open(path) as source:
            if source.width * source.height > max_pixels:
                raise ValueError(
                    f"image has {source.width * source.height:,} pixels; limit is {max_pixels:,}"
                )
            source.load()
            # convert("RGB") owns a detached buffer; a second .copy() only
            # doubles peak decode memory without improving lifetime safety.
            image = ImageOps.exif_transpose(source).convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        return None, f"{path.name}: {type(exc).__name__}: {exc}"


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Image encoder/pooling produced a zero or non-finite vector.")
    return np.asarray(vector / norm, dtype=np.float32)


def _encode_chunk(
    raw_ids: Sequence[Any],
    tokens: Sequence[str],
    image_index: dict[str, list[ProductImage]],
    encoder: ImageEncoder,
    config: ImageEmbeddingConfig,
    id_column: str,
) -> pd.DataFrame:
    per_product: list[list[tuple[int, np.ndarray]]] = [[] for _ in raw_ids]
    errors: list[list[str]] = [[] for _ in raw_ids]
    references: list[tuple[int, ProductImage]] = []
    manifest_counts = np.zeros(len(raw_ids), dtype=np.int32)
    for product_position, token in enumerate(tokens):
        manifest_counts[product_position] = len(image_index[token])
        if not image_index[token]:
            errors[product_position].append("no successful manifest image")
        references.extend((product_position, item) for item in image_index[token])

    def submit_decode_batch(
        pool: ThreadPoolExecutor, batch: Sequence[tuple[int, ProductImage]]
    ) -> list[tuple[tuple[int, ProductImage], Future[tuple[Image.Image | None, str | None]]]]:
        return [
            (
                item,
                pool.submit(_decode_image, item[1].path, config.max_image_pixels),
            )
            for item in batch
        ]

    def close_pending(
        pending_batch: Sequence[
            tuple[tuple[int, ProductImage], Future[tuple[Image.Image | None, str | None]]]
        ],
    ) -> None:
        for _, future in pending_batch:
            try:
                image, _ = future.result()
                if image is not None:
                    image.close()
            except Exception:
                pass

    batch_starts = list(range(0, len(references), config.batch_size))
    with ThreadPoolExecutor(
        max_workers=config.decode_workers, thread_name_prefix="image-decode"
    ) as pool:
        pending = (
            submit_decode_batch(pool, references[batch_starts[0]:batch_starts[0] + config.batch_size])
            if batch_starts
            else []
        )
        for batch_number, start in enumerate(batch_starts):
            current = pending
            batch_refs = [item for item, _ in current]
            decoded = [future.result() for _, future in current]
            # Double buffering: while the current batch runs on the accelerator,
            # decoder workers prepare the next batch.
            if batch_number + 1 < len(batch_starts):
                next_start = batch_starts[batch_number + 1]
                pending = submit_decode_batch(
                    pool, references[next_start:next_start + config.batch_size]
                )
            else:
                pending = []
            valid_images: list[Image.Image] = []
            valid_targets: list[tuple[int, int]] = []
            for (product_position, product_image), (image, error) in zip(batch_refs, decoded):
                if error is not None or image is None:
                    errors[product_position].append(error or "unknown image decode failure")
                else:
                    valid_images.append(image)
                    valid_targets.append((product_position, product_image.image_index))
            if not valid_images:
                continue
            try:
                try:
                    embeddings = np.asarray(encoder.encode(valid_images), dtype=np.float32)
                except Exception:
                    close_pending(pending)
                    pending = []
                    raise
            finally:
                for image in valid_images:
                    image.close()
            if embeddings.shape != (len(valid_images), encoder.dimension):
                raise RuntimeError(
                    f"Encoder returned {embeddings.shape}; expected "
                    f"({len(valid_images)}, {encoder.dimension})."
                )
            if not np.isfinite(embeddings).all():
                raise RuntimeError("Encoder returned non-finite image embeddings.")
            for (position, image_number), vector in zip(valid_targets, embeddings):
                per_product[position].append((image_number, _normalize_vector(vector)))

    embedding_rows = np.zeros((len(raw_ids), encoder.dimension), dtype=np.float32)
    has_embedding = np.zeros(len(raw_ids), dtype=np.int8)
    image_counts = np.zeros(len(raw_ids), dtype=np.int32)
    failure_counts = np.zeros(len(raw_ids), dtype=np.int32)
    for position, indexed_vectors in enumerate(per_product):
        image_counts[position] = len(indexed_vectors)
        failure_counts[position] = len(errors[position]) if manifest_counts[position] else 0
        if indexed_vectors:
            stack = np.vstack([vector for _, vector in indexed_vectors])
            if config.pooling == "max":
                pooled = stack.max(axis=0)
            elif config.pooling == "weighted_mean":
                weights = np.asarray(
                    [config.secondary_image_decay ** image_number for image_number, _ in indexed_vectors],
                    dtype=np.float32,
                )
                pooled = np.average(stack, axis=0, weights=weights)
            else:
                pooled = stack.mean(axis=0)
            embedding_rows[position] = _normalize_vector(pooled)
            has_embedding[position] = 1
        elif config.failure_policy == "error":
            detail = " | ".join(errors[position]) or "no successful manifest image"
            raise RuntimeError(f"No usable image for product {raw_ids[position]!r}: {detail}")

    dtype = np.float16 if config.output_dtype == "float16" else np.float32
    embedding_rows = embedding_rows.astype(dtype)
    columns = [f"img_emb_{index:04d}" for index in range(encoder.dimension)]
    frame = pd.DataFrame(embedding_rows, columns=columns)
    frame.insert(0, "image_failure_count", failure_counts)
    frame.insert(0, "image_count", image_counts)
    frame.insert(0, "manifest_image_count", manifest_counts)
    frame.insert(0, "has_image_embedding", has_embedding)
    frame.insert(0, id_column, list(raw_ids))
    frame["_image_errors"] = [" | ".join(items)[:8000] for items in errors]
    return frame


def encode_image_catalog_resumable(
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    image_root: Path,
    encoder: ImageEncoder,
    output_path: Path,
    config: ImageEmbeddingConfig,
    id_column: str = "PRODUCT_ID",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encode a catalog with atomic, provenance-checked chunk checkpoints."""

    if catalog.empty:
        raise ValueError("Catalog must be non-empty.")
    if encoder.dimension < 1:
        raise ValueError("Encoder dimension must be positive.")
    raw_ids = catalog[id_column].tolist() if id_column in catalog else []
    tokens, image_index = build_product_image_index(
        catalog, manifest, image_root, config, id_column
    )
    dataset_hash = _dataset_fingerprint(raw_ids, tokens, image_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / f".checkpoints_{output_path.stem}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest_path = checkpoint_dir / "manifest.json"
    number_of_chunks = math.ceil(len(catalog) / config.chunk_size)
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "image_embeddings_version": IMAGE_EMBEDDINGS_VERSION,
        "dataset_hash": dataset_hash,
        "row_count": len(catalog),
        "id_column": id_column,
        "model_signature": encoder.signature,
        "embedding_dimension": encoder.dimension,
        "batch_size": config.batch_size,
        "chunk_size": config.chunk_size,
        "max_images_per_product": config.max_images_per_product,
        "pooling": config.pooling,
        "secondary_image_decay": config.secondary_image_decay,
        "output_dtype": config.output_dtype,
        "failure_policy": config.failure_policy,
    }

    existing_chunks = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    if checkpoint_manifest_path.exists():
        try:
            recorded = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Image embedding checkpoint manifest is unreadable.") from exc
        mismatches = {
            key: (recorded.get(key), value)
            for key, value in expected.items()
            if recorded.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Checkpoint provenance mismatch; refusing stale image embeddings: "
                + json.dumps(mismatches, default=str)
            )
        if not isinstance(recorded.get("chunks"), dict):
            raise RuntimeError("Checkpoint manifest has invalid chunk metadata.")
    elif existing_chunks:
        raise RuntimeError("Checkpoint chunks exist without a manifest; refusing unsafe resume.")
    else:
        recorded = {**expected, "chunks": {}}
        _atomic_json(recorded, checkpoint_manifest_path)

    expected_names = {f"chunk_{index:05d}.parquet" for index in range(number_of_chunks)}
    unexpected = [path.name for path in existing_chunks if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(f"Unexpected image embedding checkpoint chunks: {unexpected}")

    chunk_paths: list[Path] = []
    resumed_chunks = 0
    for chunk_index in range(number_of_chunks):
        start = chunk_index * config.chunk_size
        end = min(start + config.chunk_size, len(catalog))
        chunk_path = checkpoint_dir / f"chunk_{chunk_index:05d}.parquet"
        chunk_paths.append(chunk_path)
        chunk_token_hash = _hash_tokens(tokens[start:end])
        metadata = recorded["chunks"].get(str(chunk_index))
        if chunk_path.exists() and metadata is not None:
            actual = {
                "start": start,
                "end": end,
                "row_count": end - start,
                "id_hash": chunk_token_hash,
                "file_size": chunk_path.stat().st_size,
                "file_sha256": _sha256_file(chunk_path),
            }
            differences = {
                key: (metadata.get(key), value)
                for key, value in actual.items()
                if metadata.get(key) != value
            }
            if differences:
                raise RuntimeError(
                    f"Checkpoint metadata mismatch in {chunk_path}: "
                    + json.dumps(differences, default=str)
                )
            resumed_chunks += 1
            print(f"Resuming image embeddings: chunk {chunk_index + 1}/{number_of_chunks} cached.")
            continue
        if chunk_path.exists() and metadata is None:
            # The file may have been atomically renamed immediately before a
            # crash but never committed to the manifest. Recompute safely.
            chunk_path.unlink()

        chunk_frame = _encode_chunk(
            raw_ids[start:end],
            tokens[start:end],
            image_index,
            encoder,
            config,
            id_column,
        )
        _atomic_parquet(chunk_frame, chunk_path)
        recorded["chunks"][str(chunk_index)] = {
            "start": start,
            "end": end,
            "row_count": end - start,
            "id_hash": chunk_token_hash,
            "file_size": chunk_path.stat().st_size,
            "file_sha256": _sha256_file(chunk_path),
        }
        _atomic_json(recorded, checkpoint_manifest_path)
        print(f"Encoded image chunk {chunk_index + 1}/{number_of_chunks} ({start}-{end}).")
        gc.collect()
        release_memory = getattr(encoder, "release_memory", None)
        if callable(release_memory):
            release_memory(config.empty_cuda_cache_each_chunk)

    chunks = [pd.read_parquet(path) for path in chunk_paths]
    assembled = pd.concat(chunks, ignore_index=True)
    assembled_tokens = _canonical_ids(assembled[id_column].tolist(), "assembled image embeddings")
    if assembled_tokens != tokens:
        raise RuntimeError("Assembled image embeddings are not in catalog order.")
    embedding_columns = [column for column in assembled if column.startswith("img_emb_")]
    if len(embedding_columns) != encoder.dimension:
        raise RuntimeError("Assembled image embedding dimension is invalid.")
    values = assembled[embedding_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Assembled image embeddings contain non-finite values.")
    present = assembled["has_image_embedding"].astype(bool).to_numpy()
    if present.any():
        norms = np.linalg.norm(values[present], axis=1)
        # float16 storage creates small norm error.
        if not np.allclose(norms, 1.0, atol=2e-3):
            raise RuntimeError("Present image embeddings are not unit-normalized.")
    if (~present).any() and np.any(values[~present] != 0):
        raise RuntimeError("Missing-image embeddings must be exact zero vectors.")

    diagnostics = assembled[
        [
            id_column,
            "has_image_embedding",
            "manifest_image_count",
            "image_count",
            "image_failure_count",
            "_image_errors",
        ]
    ].rename(columns={"_image_errors": "image_errors"})
    output = assembled.drop(columns=["_image_errors"])
    _atomic_parquet(output, output_path)
    summary = {
        "rows": len(output),
        "embedding_dimension": encoder.dimension,
        "embedded_products": int(output["has_image_embedding"].sum()),
        "missing_products": int((output["has_image_embedding"] == 0).sum()),
        "decoded_images": int(output["image_count"].sum()),
        "failed_images": int(output["image_failure_count"].sum()),
        "resumed_chunks": resumed_chunks,
        "dataset_hash": dataset_hash,
        "model_signature": encoder.signature,
    }

    if not config.keep_checkpoints:
        for path in chunk_paths:
            path.unlink(missing_ok=True)
        checkpoint_manifest_path.unlink(missing_ok=True)
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass
    return output, diagnostics, summary


def verify_and_align_image_embeddings(
    embeddings: pd.DataFrame,
    target: pd.DataFrame,
    id_column: str = "PRODUCT_ID",
) -> pd.DataFrame:
    """Return embeddings in target order after an exact typed-ID set check."""

    if id_column not in embeddings or id_column not in target:
        raise ValueError(f"Both frames must contain {id_column!r}.")
    embedding_tokens = _canonical_ids(embeddings[id_column].tolist(), "image embeddings")
    target_tokens = _canonical_ids(target[id_column].tolist(), "target data")
    if set(embedding_tokens) != set(target_tokens):
        missing = len(set(target_tokens) - set(embedding_tokens))
        extra = len(set(embedding_tokens) - set(target_tokens))
        raise ValueError(f"Image embedding ID mismatch: missing={missing}, extra={extra}.")
    positions = {token: position for position, token in enumerate(embedding_tokens)}
    aligned = embeddings.iloc[[positions[token] for token in target_tokens]].reset_index(drop=True)
    if _canonical_ids(aligned[id_column].tolist(), "aligned image embeddings") != target_tokens:
        raise RuntimeError("Image embedding alignment invariant failed.")
    return aligned


def project_image_embeddings(
    train_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame | None,
    n_components: int,
    id_column: str = "PRODUCT_ID",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None, PCA, dict[str, Any]]:
    """Fit PCA on train only and transform optional test embeddings.

    Presence/count flags are retained beside the compact visual components.
    Test data never contributes to the fitted projection.
    """

    if n_components < 1:
        raise ValueError("n_components must be positive.")
    if id_column not in train_embeddings:
        raise ValueError(f"Training embeddings are missing {id_column!r}.")
    _canonical_ids(train_embeddings[id_column].tolist(), "PCA training embeddings")
    embedding_columns = [c for c in train_embeddings if c.startswith("img_emb_")]
    if not embedding_columns:
        raise ValueError("Training frame contains no img_emb_* columns.")
    maximum = min(len(train_embeddings), len(embedding_columns))
    if n_components > maximum:
        raise ValueError(
            f"n_components={n_components} exceeds min(rows, features)={maximum}."
        )
    train_values = train_embeddings[embedding_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(train_values).all():
        raise ValueError("Training image embeddings contain non-finite values.")
    if test_embeddings is not None:
        if id_column not in test_embeddings:
            raise ValueError(f"Test embeddings are missing {id_column!r}.")
        _canonical_ids(test_embeddings[id_column].tolist(), "PCA test embeddings")
        test_columns = [c for c in test_embeddings if c.startswith("img_emb_")]
        if test_columns != embedding_columns:
            raise ValueError("Train/test raw image embedding schemas do not match exactly.")
        test_values = test_embeddings[embedding_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(test_values).all():
            raise ValueError("Test image embeddings contain non-finite values.")
    else:
        test_values = None

    projection = PCA(
        n_components=n_components,
        svd_solver="randomized" if n_components < maximum else "full",
        random_state=random_state,
    )
    train_compact = projection.fit_transform(train_values).astype(np.float32)
    test_compact = (
        projection.transform(test_values).astype(np.float32)
        if test_values is not None
        else None
    )
    component_columns = [f"img_svd_{index:03d}" for index in range(n_components)]
    metadata_columns = [
        column
        for column in (
            "has_image_embedding",
            "manifest_image_count",
            "image_count",
            "image_failure_count",
        )
        if column in train_embeddings
    ]
    train_output = train_embeddings[[id_column, *metadata_columns]].reset_index(drop=True).copy()
    train_output[component_columns] = train_compact
    test_output: pd.DataFrame | None = None
    if test_embeddings is not None and test_compact is not None:
        missing_metadata = sorted(set(metadata_columns) - set(test_embeddings.columns))
        if missing_metadata:
            raise ValueError(f"Test embeddings are missing metadata columns: {missing_metadata}")
        test_output = test_embeddings[[id_column, *metadata_columns]].reset_index(drop=True).copy()
        test_output[component_columns] = test_compact
    report = {
        "components": n_components,
        "input_dimension": len(embedding_columns),
        "explained_variance_ratio_sum": float(projection.explained_variance_ratio_.sum()),
        "fit_rows": len(train_embeddings),
        "random_state": random_state,
    }
    return train_output, test_output, projection, report


def train_image_embedding_head(
    train_embeddings: pd.DataFrame,
    folds: pd.DataFrame,
    test_embeddings: pd.DataFrame | None = None,
    *,
    target_column: str = "PRICE",
    id_column: str = "PRODUCT_ID",
    fold_column: str = "fold",
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Train a fold-safe log1p Ridge head and produce OOF/test predictions."""

    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")
    required = {id_column, target_column, fold_column}
    missing = sorted(required - set(folds.columns))
    if missing:
        raise ValueError(f"Folds table is missing columns: {missing}")
    if folds.empty:
        raise ValueError("Folds table must be non-empty.")
    aligned = verify_and_align_image_embeddings(train_embeddings, folds, id_column)
    vector_columns = [c for c in aligned if c.startswith("img_svd_")]
    if not vector_columns:
        vector_columns = [c for c in aligned if c.startswith("img_emb_")]
    metadata_columns = [
        column
        for column in (
            "has_image_embedding",
            "manifest_image_count",
            "image_count",
            "image_failure_count",
        )
        if column in aligned
    ]
    feature_columns = [*metadata_columns, *vector_columns]
    if not vector_columns:
        raise ValueError("Image embeddings contain no img_emb_* or img_svd_* columns.")
    X = aligned[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Image head features contain non-finite values.")
    target = pd.to_numeric(folds[target_column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Image Ridge head requires finite, non-negative targets.")
    fold_values = pd.to_numeric(folds[fold_column], errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(fold_values).all()
        or np.any(fold_values < 0)
        or not np.allclose(fold_values, np.floor(fold_values))
    ):
        raise ValueError("Fold assignments must be non-negative integers.")
    fold_values = fold_values.astype(np.int64)
    unique_folds = sorted(np.unique(fold_values).tolist())
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required for OOF evaluation.")

    target_log = np.log1p(target)
    oof = np.full(len(folds), np.nan, dtype=np.float64)
    models: list[Ridge] = []
    fold_scores: list[float] = []
    for fold_id in unique_folds:
        train_mask = fold_values != fold_id
        valid_mask = fold_values == fold_id
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"Fold {fold_id} has an empty train or validation partition.")
        model = Ridge(alpha=alpha)
        model.fit(X[train_mask], target_log[train_mask])
        prediction = np.maximum(np.expm1(model.predict(X[valid_mask])), 0.0)
        oof[valid_mask] = prediction
        fold_scores.append(float(smape(target[valid_mask], prediction)))
        models.append(model)
    if not np.isfinite(oof).all():
        raise RuntimeError("Not every training row received an OOF image prediction.")

    oof_frame = pd.DataFrame(
        {
            id_column: folds[id_column].reset_index(drop=True),
            "actual": target,
            fold_column: fold_values,
            "prediction": oof,
        }
    )
    test_frame: pd.DataFrame | None = None
    if test_embeddings is not None:
        _canonical_ids(test_embeddings[id_column].tolist(), "image head test embeddings")
        test_feature_columns = [
            column
            for column in test_embeddings
            if column in metadata_columns
            or column.startswith("img_svd_")
            or column.startswith("img_emb_")
        ]
        if test_feature_columns != feature_columns:
            raise ValueError("Train/test image head feature schemas do not match exactly.")
        X_test = test_embeddings[feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(X_test).all():
            raise ValueError("Test image head features contain non-finite values.")
        prediction_log = np.mean([model.predict(X_test) for model in models], axis=0)
        test_frame = pd.DataFrame(
            {
                id_column: test_embeddings[id_column].reset_index(drop=True),
                "prediction": np.maximum(np.expm1(prediction_log), 0.0),
            }
        )
    return {
        "oof_smape": float(smape(target, oof)),
        "fold_smapes": fold_scores,
        "mean_fold_smape": float(np.mean(fold_scores)),
        "std_fold_smape": float(np.std(fold_scores)),
        "feature_count": len(feature_columns),
        "oof_df": oof_frame,
        "test_pred_df": test_frame,
    }


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


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
    encoder: ImageEncoder,
    config: ImageEmbeddingConfig,
    id_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    catalog = _read_table(catalog_path)
    manifest = _read_table(manifest_path)
    output, diagnostics, summary = encode_image_catalog_resumable(
        catalog, manifest, image_root, encoder, output_path, config, id_column
    )
    _atomic_parquet(diagnostics, diagnostics_path)
    print(
        f"{name}: {summary['embedded_products']:,}/{len(output):,} products embedded, "
        f"{encoder.dimension} dimensions -> {output_path}"
    )
    return output, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create resumable CLIP or DINOv2 product-image embeddings."
    )
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--test",
        type=Path,
        help="Optional test catalog. Omit for train-only CV/backbone experiments.",
    )
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/image_embeddings")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/image_embeddings_run.json"))
    parser.add_argument("--id-column", default="PRODUCT_ID")
    parser.add_argument("--backend", choices=("clip", "dinov2"), default="clip")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--max-images-per-product", type=int, default=1, help="0 means use every image."
    )
    parser.add_argument(
        "--pooling", choices=("weighted_mean", "mean", "max"), default="weighted_mean"
    )
    parser.add_argument("--secondary-image-decay", type=float, default=0.5)
    parser.add_argument("--output-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--decode-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-image-pixels", type=int, default=100_000_000)
    hash_group = parser.add_mutually_exclusive_group()
    hash_group.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Concurrently re-hash image files instead of trusting downloader manifest hashes.",
    )
    hash_group.add_argument(
        "--no-verify-hashes",
        action="store_false",
        dest="verify_hashes",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(verify_hashes=False)
    parser.add_argument("--failure-policy", choices=("zero", "error"), default="zero")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument(
        "--empty-cuda-cache",
        action="store_true",
        help="Release CUDA allocator cache after each chunk (use only when memory-constrained).",
    )
    parser.add_argument("--svd-components", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--eval-head", action="store_true")
    parser.add_argument("--target-column", default="PRICE")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ImageEmbeddingConfig(
        backend=args.backend,
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_images_per_product=args.max_images_per_product,
        pooling=args.pooling,
        secondary_image_decay=args.secondary_image_decay,
        output_dtype=args.output_dtype,
        decode_workers=args.decode_workers,
        max_image_pixels=args.max_image_pixels,
        verify_hashes=args.verify_hashes,
        failure_policy=args.failure_policy,
        local_files_only=args.local_files_only,
        keep_checkpoints=args.keep_checkpoints,
        empty_cuda_cache_each_chunk=args.empty_cuda_cache,
    )
    if args.svd_components < 0:
        raise ValueError("--svd-components must be >= 0.")
    if args.eval_head and args.folds is None:
        raise ValueError("--eval-head requires --folds.")
    required_paths = [args.train, args.train_manifest]
    if args.test is not None:
        required_paths.extend([args.test, args.test_manifest])
    if args.folds is not None:
        required_paths.append(args.folds)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required input files do not exist: {missing_paths}")
    started = time.time()
    encoder = create_image_encoder(config)
    print(f"Loaded image encoder: {encoder.description}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output = args.output_dir / "image_embeddings_train.parquet"
    test_output = args.output_dir / "image_embeddings_test.parquet"
    train_embeddings, train_summary = _run_split(
        "train",
        args.train,
        args.train_manifest,
        args.train_image_root,
        train_output,
        args.output_dir / "image_embeddings_train_diagnostics.parquet",
        encoder,
        config,
        args.id_column,
    )
    test_embeddings: pd.DataFrame | None = None
    test_summary: dict[str, Any] | None = None
    if args.test is not None:
        test_embeddings, test_summary = _run_split(
            "test",
            args.test,
            args.test_manifest,
            args.test_image_root,
            test_output,
            args.output_dir / "image_embeddings_test_diagnostics.parquet",
            encoder,
            config,
            args.id_column,
        )

    head_train = train_embeddings
    head_test = test_embeddings
    projection_report: dict[str, Any] | None = None
    if args.svd_components:
        compact_train, compact_test, projection, projection_report = project_image_embeddings(
            train_embeddings,
            test_embeddings,
            args.svd_components,
            id_column=args.id_column,
            random_state=args.random_state,
        )
        compact_train_path = args.output_dir / "image_svd_train.parquet"
        _atomic_parquet(compact_train, compact_train_path)
        if compact_test is not None:
            _atomic_parquet(compact_test, args.output_dir / "image_svd_test.parquet")
        _atomic_joblib(projection, args.output_dir / "image_svd_pca.joblib")
        head_train, head_test = compact_train, compact_test

    head_report: dict[str, Any] | None = None
    if args.eval_head:
        folds = _read_table(args.folds)
        head_results = train_image_embedding_head(
            head_train,
            folds,
            head_test,
            target_column=args.target_column,
            id_column=args.id_column,
            fold_column=args.fold_column,
            alpha=args.ridge_alpha,
        )
        _atomic_parquet(
            head_results["oof_df"], args.output_dir / "image_ridge_oof.parquet"
        )
        if head_results["test_pred_df"] is not None:
            _atomic_parquet(
                head_results["test_pred_df"],
                args.output_dir / "image_ridge_test.parquet",
            )
        head_report = {
            key: value
            for key, value in head_results.items()
            if key not in {"oof_df", "test_pred_df"}
        }
        print(
            f"Image Ridge OOF SMAPE: {head_report['oof_smape']:.6f} "
            f"({head_report['feature_count']} features)"
        )
    report = {
        "image_embeddings_version": IMAGE_EMBEDDINGS_VERSION,
        "created_at_unix": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
        "encoder": encoder.description,
        "model_signature": encoder.signature,
        "config": asdict(config),
        "train": train_summary,
        "test": test_summary,
        "projection": projection_report,
        "ridge_head": head_report,
        "artifacts": {
            "train": str(train_output),
            "test": str(test_output) if args.test is not None else None,
        },
    }
    _atomic_json(report, args.report)
    print(f"Run report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
