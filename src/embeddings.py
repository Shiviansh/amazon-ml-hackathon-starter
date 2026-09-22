"""Defensive dense text embedding utilities for competitive ML.

Features:
- Explicit dual-backend architecture: SentenceTransformers or an opt-in
  TF-IDF + TruncatedSVD fallback.
- Strict immutable ID-to-row alignment checks (zero row-order assumptions).
- Resumable chunked encoding with checkpoint recovery.
- Compact float16 precision caching (halves memory and disk footprint).
- Smart catalog text formatting (prioritizing TITLE, BRAND, and CATEGORY over DESC).
- Built-in fold-safe Ridge head training with local SMAPE evaluation.
- Scikit-learn compatible ``TextEmbeddingTransformer``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection

try:
    from .metrics import smape
except ImportError:
    from metrics import smape


EMBEDDINGS_VERSION = "2.1"
CHECKPOINT_VERSION = "3.0"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
FAST_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = None
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _resolve_model_revision(model_name: str, revision: str | None) -> str | None:
    if revision is not None:
        return revision
    if model_name == FAST_MINILM_MODEL:
        return MINILM_REVISION
    return None


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


def _is_missing_id(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _canonical_id(value: Any) -> str:
    """Canonical ID token that preserves type and rejects missing values."""
    if _is_missing_id(value):
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


def _canonical_ids(values: Sequence[Any], *, label: str) -> list[str]:
    tokens = [_canonical_id(value) for value in values]
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"Duplicate IDs detected in {label}.")
    return tokens


def _hash_records(ids: Sequence[Any], texts: Sequence[str] | None = None) -> str:
    digest = hashlib.sha256()
    for idx, value in enumerate(ids):
        token = _canonical_id(value).encode("utf-8")
        digest.update(len(token).to_bytes(8, "big"))
        digest.update(token)
        if texts is not None:
            text = str(texts[idx]).encode("utf-8")
            digest.update(len(text).to_bytes(8, "big"))
            digest.update(text)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_signature(model: Any) -> str:
    payload: dict[str, Any] = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    for name in (
        "model_name_or_path", "n_components", "random_state", "strategy",
        "large_corpus_threshold", "hash_features", "mode_",
    ):
        if hasattr(model, name):
            payload[name] = getattr(model, name)
    if hasattr(model, "get_sentence_embedding_dimension"):
        try:
            payload["dimension"] = int(model.get_sentence_embedding_dimension())
        except Exception:
            pass
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    )
    if isinstance(model, OfflineDenseEmbeddingBackend) and model.pipeline is not None:
        if model.mode_ == "tfidf-svd":
            tfidf = model.pipeline.named_steps["tfidf"]
            svd = model.pipeline.named_steps["svd"]
            digest.update(
                json.dumps(
                    {key: int(value) for key, value in tfidf.vocabulary_.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(np.asarray(tfidf.idf_, dtype=np.float64).tobytes(order="C"))
            digest.update(np.asarray(svd.components_, dtype=np.float64).tobytes(order="C"))
        elif model.mode_ == "hash-rp":
            hashing = model.pipeline.named_steps["hashing"]
            projection = model.pipeline.named_steps["projection"]
            digest.update(
                json.dumps(hashing.get_params(deep=False), sort_keys=True, default=str).encode("utf-8")
            )
            components = projection.components_.tocsr()
            digest.update(np.asarray(components.data).tobytes(order="C"))
            digest.update(np.asarray(components.indices).tobytes(order="C"))
            digest.update(np.asarray(components.indptr).tobytes(order="C"))
    return digest.hexdigest()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    if not np.isscalar(value):
        raise TypeError(f"Expected scalar text, got {type(value).__name__}.")
    return str(value).strip()


# ---------------------------------------------------------------------------
# Smart Catalog Text Formatting
# ---------------------------------------------------------------------------

def format_catalog_text(
    title: Any,
    brand: Any = None,
    category: Any = None,
    desc: Any = None,
    max_desc_chars: int = 256,
) -> str:
    """Format catalog fields into a structured, high-signal text sequence.

    Guarantees title, brand, and category appear first so they are never truncated
    by tokenizers when descriptions are long.
    """
    if not isinstance(max_desc_chars, int) or max_desc_chars < 0:
        raise ValueError("max_desc_chars must be a non-negative integer.")
    parts: list[str] = []

    t_str = _clean_text(title)
    if t_str:
        parts.append(f"Title: {t_str}")

    b_str = _clean_text(brand)
    if b_str:
        parts.append(f"Brand: {b_str}")

    c_str = _clean_text(category)
    if c_str:
        parts.append(f"Category: {c_str}")

    d_str = _clean_text(desc)
    if d_str:
        truncated_desc = d_str[:max_desc_chars]
        parts.append(f"Description: {truncated_desc}")

    return " | ".join(parts) if parts else "Unknown Product"


def format_dataframe_texts(
    df: pd.DataFrame,
    title_column: str = "TITLE",
    brand_column: str | None = "BRAND",
    category_column: str | None = "CATEGORY",
    desc_column: str | None = "DESCRIPTION",
    max_desc_chars: int = 256,
) -> list[str]:
    """Vectorized formatting of catalog DataFrame rows."""
    titles = df[title_column].values if title_column in df.columns else [None] * len(df)
    brands = df[brand_column].values if brand_column and brand_column in df.columns else [None] * len(df)
    cats = df[category_column].values if category_column and category_column in df.columns else [None] * len(df)
    descs = df[desc_column].values if desc_column and desc_column in df.columns else [None] * len(df)

    formatted: list[str] = []
    for t, b, c, d in zip(titles, brands, cats, descs):
        formatted.append(format_catalog_text(t, b, c, d, max_desc_chars=max_desc_chars))
    return formatted


# ---------------------------------------------------------------------------
# Offline Fallback Backend: SVD-Dense Projection
# ---------------------------------------------------------------------------

class OfflineDenseEmbeddingBackend:
    """Deterministic dense fallback with bounded-memory large-corpus mode.

    Small corpora use TF-IDF + TruncatedSVD for quality. Large corpora switch
    to stateless feature hashing + sparse random projection, avoiding a fitted
    vocabulary and a full-corpus TF-IDF matrix during ``fit``.
    """

    def __init__(
        self,
        n_components: int = 384,
        random_state: int = 42,
        *,
        strategy: str = "auto",
        large_corpus_threshold: int = 100_000,
        hash_features: int = 2**18,
    ):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if strategy not in {"auto", "svd", "hash"}:
            raise ValueError("strategy must be one of: auto, svd, hash.")
        if not isinstance(large_corpus_threshold, int) or large_corpus_threshold < 1:
            raise ValueError("large_corpus_threshold must be a positive integer.")
        if not isinstance(hash_features, int) or hash_features < 2:
            raise ValueError("hash_features must be an integer of at least 2.")
        self.n_components = n_components
        self.random_state = random_state
        self.strategy = strategy
        self.large_corpus_threshold = large_corpus_threshold
        self.hash_features = hash_features
        self.pipeline: Pipeline | None = None
        self.mode_: str | None = None

    def fit(self, texts: Sequence[str]) -> OfflineDenseEmbeddingBackend:
        if len(texts) == 0:
            raise ValueError("Cannot fit embeddings on an empty corpus.")
        use_hashing = self.strategy == "hash" or (
            self.strategy == "auto" and len(texts) >= self.large_corpus_threshold
        )
        if use_hashing:
            hashing = HashingVectorizer(
                n_features=self.hash_features,
                alternate_sign=True,
                norm="l2",
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b",
            )
            projection = SparseRandomProjection(
                n_components=self.n_components,
                dense_output=True,
                random_state=self.random_state,
            )
            # Fitting creates the deterministic sparse projection matrix from
            # feature width only; it never scans/materializes the corpus.
            projection.fit(hashing.transform(["__projection_seed__"]))
            self.pipeline = Pipeline([
                ("hashing", hashing),
                ("projection", projection),
            ])
            self.mode_ = "hash-rp"
            return self

        safe_texts = [str(text).strip() or "__empty__" for text in texts]
        min_df = 1 if len(texts) < 50 else 2
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            min_df=min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        X_tfidf = tfidf.fit_transform(safe_texts)
        if X_tfidf.shape[1] < 2:
            # TruncatedSVD requires at least two features. Add a deterministic
            # corpus sentinel only for degenerate one-token corpora.
            safe_texts = [f"{text} __fallback_dimension__" for text in safe_texts]
            X_tfidf = tfidf.fit_transform(safe_texts)
        n_features = X_tfidf.shape[1]
        n_samples = X_tfidf.shape[0]

        actual_components = min(self.n_components, max(1, min(n_samples, n_features)))
        if actual_components < 1:
            actual_components = 1

        svd = TruncatedSVD(n_components=actual_components, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            svd.fit(X_tfidf)

        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("svd", svd),
        ])
        self.mode_ = "tfidf-svd"
        return self

    def encode(self, texts: Sequence[str], batch_size: int = 512) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("OfflineDenseEmbeddingBackend must be fitted before encode().")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        safe_texts = [str(text).strip() or "__empty__" for text in texts]
        dense = self.pipeline.transform(safe_texts)
        # Pad to n_components if necessary
        if dense.shape[1] < self.n_components:
            pad_width = self.n_components - dense.shape[1]
            dense = np.pad(dense, ((0, 0), (0, pad_width)), mode="constant")
        # Ensure zero-norm vectors (e.g. empty texts) become stable unit vectors
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        zero_mask = (norms == 0).ravel()
        if np.any(zero_mask):
            dense[zero_mask, 0] = 1.0
        # L2-normalize vectors
        normed = normalize(dense, norm="l2", axis=1)
        if not np.isfinite(normed).all():
            raise RuntimeError("Offline embedding backend produced non-finite values.")
        return normed.astype(np.float16)


# ---------------------------------------------------------------------------
# Resilient Model Loader
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    allow_fallback: bool = False,
    fallback_components: int = 384,
    fallback_strategy: str = "auto",
    fallback_large_corpus_threshold: int = 100_000,
    fallback_hash_features: int = 2**18,
) -> tuple[Any, str]:
    """Load a reproducibly pinned default model or an explicit dense fallback."""
    load_error: Exception | None = None
    resolved_revision = _resolve_model_revision(model_name, revision)
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            kwargs: dict[str, Any] = {
                "device": device,
                "local_files_only": local_files_only,
            }
            if resolved_revision is not None:
                kwargs["revision"] = resolved_revision
            model = SentenceTransformer(model_name, **kwargs)
            revision_label = resolved_revision or "un-pinned"
            return model, f"sentence-transformers ({model_name}@{revision_label}) [{device}]"
        except Exception as err:
            load_error = err
            if (
                model_name in {DEFAULT_MODEL_NAME, "BAAI/bge-large-en-v1.5", "BAAI/bge-m3"}
                and allow_fallback
            ):
                try:
                    kwargs_fallback: dict[str, Any] = {
                        "device": device,
                        "local_files_only": local_files_only,
                        "revision": MINILM_REVISION,
                    }
                    model = SentenceTransformer(FAST_MINILM_MODEL, **kwargs_fallback)
                    warnings.warn(
                        f"Failed to load primary embedding model '{model_name}' ({err}). "
                        f"Cascaded to fast fallback '{FAST_MINILM_MODEL}'."
                    )
                    return (
                        model,
                        f"sentence-transformers fallback ({FAST_MINILM_MODEL}@{MINILM_REVISION}) [{device}]",
                    )
                except Exception:
                    pass
    except ImportError as err:
        load_error = err

    if not allow_fallback:
        raise RuntimeError(
            f"Unable to load embedding model '{model_name}'. "
        "Pass allow_fallback=True only if the offline dense backend is an acceptable model change."
        ) from load_error
    warnings.warn(
        f"Failed to load transformer '{model_name}' ({load_error}). "
        "Using explicitly authorized offline dense fallback."
    )
    return (
        OfflineDenseEmbeddingBackend(
            n_components=fallback_components,
            strategy=fallback_strategy,
            large_corpus_threshold=fallback_large_corpus_threshold,
            hash_features=fallback_hash_features,
        ),
        f"offline-{fallback_strategy}-fallback-{fallback_components}d [cpu]",
    )


# ---------------------------------------------------------------------------
# Resumable Chunked Batch Encoding
# ---------------------------------------------------------------------------

def encode_texts_resumable(
    texts: list[str],
    ids: Sequence[Any],
    model: Any,
    output_path: Path,
    batch_size: int = 64,
    chunk_size: int = 2500,
    show_progress: bool = True,
    model_signature: str | None = None,
) -> pd.DataFrame:
    """Encode atomic chunks and resume only when provenance matches exactly."""
    if len(texts) == 0:
        raise ValueError("texts must be non-empty.")
    if len(texts) != len(ids):
        raise ValueError(f"texts length {len(texts)} does not match ids length {len(ids)}.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if not hasattr(model, "encode"):
        raise ValueError("Model has no .encode() method.")
    if isinstance(model, OfflineDenseEmbeddingBackend) and model.pipeline is None:
        raise RuntimeError("Fit the offline embedding backend on training text before encoding chunks.")

    ids_list = list(ids)
    canonical_ids = _canonical_ids(ids_list, label="embedding input")
    safe_texts = [str(text) for text in texts]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / f".checkpoints_{output_path.stem}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = checkpoint_dir / "manifest.json"

    n_samples = len(safe_texts)
    n_chunks = math.ceil(n_samples / chunk_size)
    signature = model_signature or _model_signature(model)
    expected_manifest: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "input_hash": _hash_records(ids_list, safe_texts),
        "id_hash": _hash_records(ids_list),
        "model_signature": signature,
        "n_samples": n_samples,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "dtype": "float16",
    }

    existing_chunks = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    embedding_dim: int | None = None
    recorded: dict[str, Any]
    if checkpoint_manifest.exists():
        try:
            recorded = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise RuntimeError(f"Checkpoint manifest is unreadable: {checkpoint_manifest}") from err
        mismatches = {
            key: (recorded.get(key), value)
            for key, value in expected_manifest.items()
            if recorded.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Checkpoint provenance mismatch; refusing to reuse stale embeddings: "
                + json.dumps(mismatches, default=str)
            )
        if recorded.get("embedding_dim") is not None:
            embedding_dim = int(recorded["embedding_dim"])
        if not isinstance(recorded.get("chunks"), dict):
            raise RuntimeError("Checkpoint manifest has invalid or missing chunk metadata.")
    elif existing_chunks:
        raise RuntimeError(
            f"Checkpoint chunks exist without a manifest in {checkpoint_dir}; refusing unsafe resume."
        )
    else:
        recorded = {**expected_manifest, "embedding_dim": None, "chunks": {}}
        _atomic_json(recorded, checkpoint_manifest)

    expected_chunk_names = {f"chunk_{index:05d}.parquet" for index in range(n_chunks)}
    unexpected_chunks = [path for path in existing_chunks if path.name not in expected_chunk_names]
    if unexpected_chunks:
        raise RuntimeError(
            "Unexpected checkpoint chunks found; refusing ambiguous resume: "
            + str([str(path) for path in unexpected_chunks])
        )

    chunk_files: list[Path] = []

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_file = checkpoint_dir / f"chunk_{chunk_idx:05d}.parquet"
        chunk_files.append(chunk_file)

        expected_chunk_id_hash = _hash_records(ids_list[start_idx:end_idx])
        if chunk_file.exists():
            chunk_meta = recorded["chunks"].get(str(chunk_idx))
            # A chunk written immediately before a process crash may exist
            # without committed metadata. It is recomputed safely. Committed
            # chunks are checked using size + SHA-256, without deserializing.
            if chunk_meta is not None:
                expected_meta = {
                    "start": start_idx,
                    "end": end_idx,
                    "row_count": end_idx - start_idx,
                    "id_hash": expected_chunk_id_hash,
                    "embedding_dim": embedding_dim,
                }
                mismatched_meta = {
                    key: (chunk_meta.get(key), value)
                    for key, value in expected_meta.items()
                    if chunk_meta.get(key) != value
                }
                actual_size = chunk_file.stat().st_size
                if chunk_meta.get("file_size") != actual_size:
                    mismatched_meta["file_size"] = (chunk_meta.get("file_size"), actual_size)
                actual_hash = _sha256_file(chunk_file)
                if chunk_meta.get("file_sha256") != actual_hash:
                    mismatched_meta["file_sha256"] = (chunk_meta.get("file_sha256"), actual_hash)
                if mismatched_meta:
                    raise RuntimeError(
                        f"Checkpoint metadata mismatch in {chunk_file}: "
                        + json.dumps(mismatched_meta, default=str)
                    )
                if show_progress:
                    print(
                        f"Resuming: chunk {chunk_idx + 1}/{n_chunks} already cached "
                        f"({start_idx}-{end_idx})."
                    )
                continue

        chunk_texts = safe_texts[start_idx:end_idx]
        chunk_ids = ids_list[start_idx:end_idx]

        # Encode via model backend
        if isinstance(model, OfflineDenseEmbeddingBackend):
            chunk_embs = model.encode(chunk_texts, batch_size=batch_size)
        else:
            chunk_embs = model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        chunk_embs = np.asarray(chunk_embs)
        if chunk_embs.ndim != 2 or chunk_embs.shape[0] != len(chunk_texts) or chunk_embs.shape[1] < 1:
            raise RuntimeError(
                f"Model returned invalid embedding shape {chunk_embs.shape} for {len(chunk_texts)} rows."
            )
        if not np.issubdtype(chunk_embs.dtype, np.number) or not np.isfinite(chunk_embs).all():
            raise RuntimeError("Model returned non-numeric or non-finite embeddings.")
        chunk_embs = chunk_embs.astype(np.float16)
        if embedding_dim is None:
            embedding_dim = int(chunk_embs.shape[1])
            recorded["embedding_dim"] = embedding_dim
            _atomic_json(recorded, checkpoint_manifest)
        elif chunk_embs.shape[1] != embedding_dim:
            raise RuntimeError(
                f"Embedding dimension changed from {embedding_dim} to {chunk_embs.shape[1]}."
            )

        # Construct DataFrame with ID and float16 embedding columns
        emb_cols = [f"emb_{i:03d}" for i in range(chunk_embs.shape[1])]
        chunk_df = pd.DataFrame(chunk_embs, columns=emb_cols)
        chunk_df.insert(0, "id", chunk_ids)

        # Atomic chunk write
        _atomic_parquet(chunk_df, chunk_file)
        recorded["chunks"][str(chunk_idx)] = {
            "start": start_idx,
            "end": end_idx,
            "row_count": end_idx - start_idx,
            "id_hash": expected_chunk_id_hash,
            "embedding_dim": embedding_dim,
            "file_size": chunk_file.stat().st_size,
            "file_sha256": _sha256_file(chunk_file),
        }
        _atomic_json(recorded, checkpoint_manifest)

        if show_progress:
            print(f"Encoded chunk {chunk_idx + 1}/{n_chunks} ({start_idx}-{end_idx} rows).")

        gc.collect()

    # Concatenate all verified chunks in deterministic order
    assembled: list[pd.DataFrame] = []
    for cf in chunk_files:
        if not cf.exists():
            raise RuntimeError(f"Missing expected chunk file: {cf}")
        assembled.append(pd.read_parquet(cf))

    full_df = pd.concat(assembled, ignore_index=True)
    if len(full_df) != n_samples:
        raise RuntimeError(f"Assembled {len(full_df)} rows; expected {n_samples}.")
    assembled_ids = _canonical_ids(full_df["id"].tolist(), label="assembled embeddings")
    if assembled_ids != canonical_ids:
        raise RuntimeError("Assembled embedding IDs are not in the exact requested order.")
    emb_cols = [col for col in full_df.columns if col.startswith("emb_")]
    if len(emb_cols) != embedding_dim:
        raise RuntimeError("Assembled embedding dimension does not match checkpoint manifest.")
    if not np.isfinite(full_df[emb_cols].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Assembled embeddings contain non-finite values.")

    _atomic_parquet(full_df, output_path)

    # Clean up checkpoints on successful completion
    for cf in chunk_files:
        try:
            cf.unlink()
        except OSError:
            pass
    try:
        checkpoint_manifest.unlink()
    except OSError:
        pass
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass

    return full_df


# ---------------------------------------------------------------------------
# Immutable ID Alignment & Verification
# ---------------------------------------------------------------------------

def verify_and_align_embeddings(
    embeddings_df: pd.DataFrame,
    target_df: pd.DataFrame,
    id_column: str = "PRODUCT_ID",
    emb_id_column: str = "id",
) -> pd.DataFrame:
    """Verify and enforce strict ID alignment between embeddings and target rows.

    Raises ``ValueError`` on count mismatch, missing IDs, or duplicate IDs.
    """
    if emb_id_column not in embeddings_df.columns:
        raise ValueError(f"Embeddings frame missing ID column '{emb_id_column}'.")
    if id_column not in target_df.columns:
        raise ValueError(f"Target frame missing ID column '{id_column}'.")

    if emb_id_column != id_column and id_column in embeddings_df.columns:
        raise ValueError(
            f"Embeddings frame contains both '{emb_id_column}' and '{id_column}', creating an ID collision."
        )
    emb_ids = _canonical_ids(embeddings_df[emb_id_column].tolist(), label="embeddings frame")
    target_ids = _canonical_ids(target_df[id_column].tolist(), label="target frame")

    if len(emb_ids) != len(target_ids):
        raise ValueError(
            f"Embedding row count ({len(emb_ids)}) does not match target count ({len(target_ids)})."
        )

    emb_set = set(emb_ids)
    target_set = set(target_ids)
    if emb_set != target_set:
        missing = list(target_set - emb_set)[:5]
        extra = list(emb_set - target_set)[:5]
        raise ValueError(
            f"ID mismatch between embeddings and target. Missing: {missing}, Extra: {extra}"
        )

    id_to_idx = {pid: i for i, pid in enumerate(emb_ids)}
    reorder_indices = [id_to_idx[pid] for pid in target_ids]
    aligned_df = embeddings_df.iloc[reorder_indices].reset_index(drop=True)

    # Rename embedding ID column to match target id_column
    if emb_id_column != id_column:
        aligned_df = aligned_df.rename(columns={emb_id_column: id_column})

    emb_cols = [column for column in aligned_df.columns if column.startswith("emb_")]
    if not emb_cols:
        raise ValueError("Embeddings frame has no emb_* feature columns.")
    values = aligned_df[emb_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain non-finite values.")

    return aligned_df


# ---------------------------------------------------------------------------
# Downstream Embedding Head Evaluation (Ridge on log1p)
# ---------------------------------------------------------------------------

def train_embedding_head(
    train_embeddings: pd.DataFrame,
    folds_df: pd.DataFrame,
    target_column: str = "PRICE",
    id_column: str = "PRODUCT_ID",
    fold_column: str = "fold",
    alpha: float = 10.0,
    test_embeddings: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train a fold-safe Ridge regression head directly on text embeddings.

    Returns:
        dict containing 'oof_smape', 'fold_smapes', 'oof_predictions', 'test_predictions'
    """
    required = {id_column, fold_column, target_column}
    missing = required - set(folds_df.columns)
    if missing:
        raise ValueError(f"folds_df is missing required columns: {sorted(missing)}")
    if folds_df.empty:
        raise ValueError("folds_df must be non-empty.")
    aligned_embeddings = verify_and_align_embeddings(
        train_embeddings, folds_df, id_column=id_column, emb_id_column=id_column
    )
    emb_cols = [c for c in aligned_embeddings.columns if c.startswith("emb_")]
    X = aligned_embeddings[emb_cols].to_numpy(dtype=np.float32)
    y_series = pd.to_numeric(folds_df[target_column], errors="coerce")
    if y_series.isna().any() or not np.isfinite(y_series.to_numpy(dtype=np.float64)).all():
        raise ValueError("Training targets must be numeric, finite, and non-missing.")
    if (y_series < 0).any():
        raise ValueError("log1p Ridge head requires non-negative targets.")
    y = y_series.to_numpy(dtype=np.float64)
    y_log = np.log1p(y)
    fold_series = pd.to_numeric(folds_df[fold_column], errors="coerce")
    if fold_series.isna().any() or (fold_series < 0).any() or not np.allclose(fold_series, np.floor(fold_series)):
        raise ValueError("Fold assignments must be non-negative integers.")
    folds = fold_series.to_numpy(dtype=np.int64)

    unique_folds = sorted(np.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required.")
    oof_preds = np.full(len(folds_df), np.nan, dtype=np.float64)
    fold_scores: list[float] = []

    models: list[Ridge] = []

    for fold_id in unique_folds:
        train_mask = (folds != fold_id)
        val_mask = (folds == fold_id)

        X_train, y_train = X[train_mask], y_log[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X_train, y_train)
        models.append(reg)

        val_pred_log = reg.predict(X_val)
        val_pred = np.maximum(np.expm1(val_pred_log), 0.0)
        oof_preds[val_mask] = val_pred

        fold_score = float(smape(y_val, val_pred))
        fold_scores.append(fold_score)

    if not np.isfinite(oof_preds).all():
        raise RuntimeError("Some rows did not receive an OOF prediction.")
    overall_smape = float(smape(y, oof_preds))

    oof_df = pd.DataFrame({
        id_column: folds_df[id_column].reset_index(drop=True),
        "actual": y,
        "fold": folds,
        "prediction": oof_preds,
    })

    test_pred_df: pd.DataFrame | None = None
    if test_embeddings is not None:
        _canonical_ids(test_embeddings[id_column].tolist(), label="test embeddings")
        missing_test_cols = set(emb_cols) - set(test_embeddings.columns)
        extra_test_cols = {
            col for col in test_embeddings.columns if col.startswith("emb_")
        } - set(emb_cols)
        if missing_test_cols or extra_test_cols:
            raise ValueError(
                f"Test embedding schema mismatch. Missing={sorted(missing_test_cols)}, "
                f"extra={sorted(extra_test_cols)}"
            )
        X_test = test_embeddings[emb_cols].values.astype(np.float32)
        if not np.isfinite(X_test).all():
            raise ValueError("Test embeddings contain non-finite values.")
        test_preds_log = np.mean([m.predict(X_test) for m in models], axis=0)
        test_preds = np.maximum(np.expm1(test_preds_log), 0.0)
        test_pred_df = pd.DataFrame({
            id_column: test_embeddings[id_column],
            "prediction": test_preds,
        })

    return {
        "oof_smape": overall_smape,
        "fold_smapes": fold_scores,
        "mean_fold_smape": float(np.mean(fold_scores)),
        "std_fold_smape": float(np.std(fold_scores)),
        "oof_df": oof_df,
        "test_pred_df": test_pred_df,
    }


# ---------------------------------------------------------------------------
# Scikit-Learn Transformer Interface
# ---------------------------------------------------------------------------

@dataclass
class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for dense text embeddings."""

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 64
    max_desc_chars: int = 256
    device: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    allow_fallback: bool = False
    fallback_components: int = 384
    fallback_strategy: str = "auto"
    fallback_large_corpus_threshold: int = 100_000
    fallback_hash_features: int = 2**18
    title_column: str = "TITLE"
    brand_column: str | None = "BRAND"
    category_column: str | None = "CATEGORY"
    desc_column: str | None = "DESCRIPTION"
    model_: Any = field(default=None, init=False)
    backend_info_: str = field(default="", init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    output_dimension_: int = field(default=0, init=False)

    def _texts(self, X: pd.DataFrame | Sequence[str]) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return format_dataframe_texts(
                X,
                title_column=self.title_column,
                brand_column=self.brand_column,
                category_column=self.category_column,
                desc_column=self.desc_column,
                max_desc_chars=self.max_desc_chars,
            )
        return [str(value) for value in X]

    def fit(self, X: pd.DataFrame | Sequence[str], y: Any = None) -> TextEmbeddingTransformer:
        self.model_, self.backend_info_ = load_embedding_model(
            model_name=self.model_name,
            device=self.device,
            revision=self.revision,
            local_files_only=self.local_files_only,
            allow_fallback=self.allow_fallback,
            fallback_components=self.fallback_components,
            fallback_strategy=self.fallback_strategy,
            fallback_large_corpus_threshold=self.fallback_large_corpus_threshold,
            fallback_hash_features=self.fallback_hash_features,
        )
        texts = self._texts(X)
        if not texts:
            raise ValueError("Cannot fit TextEmbeddingTransformer on empty input.")
        # Fit offline SVD if fallback is active
        if isinstance(self.model_, OfflineDenseEmbeddingBackend):
            self.model_.fit(texts)
            self.output_dimension_ = self.model_.n_components
        elif hasattr(self.model_, "get_sentence_embedding_dimension"):
            self.output_dimension_ = int(self.model_.get_sentence_embedding_dimension())
        else:
            probe = np.asarray(
                self.model_.encode(
                    texts[:1],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            )
            if probe.ndim != 2 or probe.shape[0] != 1 or probe.shape[1] < 1:
                raise RuntimeError(f"Cannot infer embedding dimension from shape {probe.shape}.")
            self.output_dimension_ = int(probe.shape[1])
        self.feature_names_ = [f"emb_{i:03d}" for i in range(self.output_dimension_)]
        return self

    def transform(self, X: pd.DataFrame | Sequence[str]) -> np.ndarray:
        if self.model_ is None or self.output_dimension_ < 1:
            raise NotFittedError("TextEmbeddingTransformer must be fitted before transform().")
        texts = self._texts(X)

        if isinstance(self.model_, OfflineDenseEmbeddingBackend):
            result = self.model_.encode(texts, batch_size=self.batch_size)
        else:
            result = self.model_.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        result = np.asarray(result)
        if result.shape != (len(texts), self.output_dimension_):
            raise RuntimeError(
                f"Embedding output shape {result.shape} does not match expected "
                f"({len(texts)}, {self.output_dimension_})."
            )
        if not np.isfinite(result).all():
            raise RuntimeError("Embedding model returned non-finite values.")
        return result.astype(np.float16)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self.model_ is None or self.output_dimension_ < 1:
            raise NotFittedError("TextEmbeddingTransformer is not fitted.")
        return np.asarray(self.feature_names_, dtype=object)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache dense text embeddings with immutable ID alignment.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", help="Pinned Hugging Face model revision/commit.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Explicitly allow the offline dense backend if the transformer cannot load.",
    )
    parser.add_argument("--fallback-components", type=int, default=384)
    parser.add_argument(
        "--fallback-strategy", choices=["auto", "svd", "hash"], default="auto",
        help="Offline mode: quality-oriented SVD, bounded-memory hashing, or automatic selection.",
    )
    parser.add_argument("--fallback-large-corpus-threshold", type=int, default=100_000)
    parser.add_argument("--fallback-hash-features", type=int, default=2**18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--max-desc-chars", type=int, default=256)
    parser.add_argument("--id-column", default="PRODUCT_ID")
    parser.add_argument("--title-column", default="TITLE")
    parser.add_argument("--brand-column", default="BRAND")
    parser.add_argument("--category-column", default="CATEGORY")
    parser.add_argument("--desc-column", default="DESCRIPTION")
    parser.add_argument("--target-column", default="PRICE")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--eval-head", action="store_true", help="Train a Ridge head on embeddings and output CV SMAPE.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start_time = time.time()

    print(f"Loading train dataset: {args.train}")
    train_df = pd.read_csv(args.train)
    print(f"Loading test dataset: {args.test}")
    test_df = pd.read_csv(args.test)
    for name, frame in (("train", train_df), ("test", test_df)):
        if frame.empty:
            raise ValueError(f"{name} data is empty.")
        if args.id_column not in frame.columns:
            raise ValueError(f"{name} is missing ID column '{args.id_column}'.")
        _canonical_ids(frame[args.id_column].tolist(), label=f"{name} data")
    if args.target_column not in train_df.columns:
        raise ValueError(f"train is missing target column '{args.target_column}'.")

    # Format texts
    print("Formatting catalog text sequences...")
    train_texts = format_dataframe_texts(
        train_df,
        title_column=args.title_column,
        brand_column=args.brand_column,
        category_column=args.category_column,
        desc_column=args.desc_column,
        max_desc_chars=args.max_desc_chars,
    )
    test_texts = format_dataframe_texts(
        test_df,
        title_column=args.title_column,
        brand_column=args.brand_column,
        category_column=args.category_column,
        desc_column=args.desc_column,
        max_desc_chars=args.max_desc_chars,
    )

    # Load model backend
    print(f"Loading embedding model: {args.model_name}")
    model, backend_info = load_embedding_model(
        model_name=args.model_name,
        device=args.device,
        revision=args.revision,
        local_files_only=args.local_files_only,
        allow_fallback=args.allow_fallback,
        fallback_components=args.fallback_components,
        fallback_strategy=args.fallback_strategy,
        fallback_large_corpus_threshold=args.fallback_large_corpus_threshold,
        fallback_hash_features=args.fallback_hash_features,
    )
    print(f"Active embedding backend: {backend_info}")

    # Fit fallback if offline
    if isinstance(model, OfflineDenseEmbeddingBackend):
        print("Fitting offline dense projection on train texts...")
        model.fit(train_texts)
        print(f"Offline fallback mode: {model.mode_}")
    resolved_revision = _resolve_model_revision(args.model_name, args.revision)
    run_model_signature = hashlib.sha256(
        json.dumps(
            {
                "model_name": args.model_name,
                "revision": resolved_revision,
                "backend_info": backend_info,
                "implementation": _model_signature(model),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    # Encode train
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "embeddings_train.parquet"
    test_out = args.output_dir / "embeddings_test.parquet"

    print(f"\nEncoding train texts ({len(train_texts)} rows) in float16...")
    train_embs = encode_texts_resumable(
        texts=train_texts,
        ids=train_df[args.id_column].values,
        model=model,
        output_path=train_out,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        model_signature=run_model_signature,
    )

    print(f"\nEncoding test texts ({len(test_texts)} rows) in float16...")
    test_embs = encode_texts_resumable(
        texts=test_texts,
        ids=test_df[args.id_column].values,
        model=model,
        output_path=test_out,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        model_signature=run_model_signature,
    )

    # Verify ID alignment
    print("\nVerifying immutable ID alignment...")
    train_aligned = verify_and_align_embeddings(
        train_embs, train_df, id_column=args.id_column, emb_id_column="id"
    )
    test_aligned = verify_and_align_embeddings(
        test_embs, test_df, id_column=args.id_column, emb_id_column="id"
    )

    _atomic_parquet(train_aligned, train_out)
    _atomic_parquet(test_aligned, test_out)

    elapsed = time.time() - start_time
    dim_count = train_aligned.shape[1] - 1  # minus id column

    manifest = {
        "embeddings_version": EMBEDDINGS_VERSION,
        "model_name": args.model_name,
        "model_revision": resolved_revision,
        "backend_info": backend_info,
        "model_signature": run_model_signature,
        "local_files_only": args.local_files_only,
        "fallback_authorized": args.allow_fallback,
        "fallback_strategy": args.fallback_strategy,
        "fallback_large_corpus_threshold": args.fallback_large_corpus_threshold,
        "fallback_hash_features": args.fallback_hash_features,
        "fallback_mode": model.mode_ if isinstance(model, OfflineDenseEmbeddingBackend) else None,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_desc_chars": args.max_desc_chars,
        "train_rows": len(train_aligned),
        "test_rows": len(test_aligned),
        "embedding_dim": dim_count,
        "dtype": "float16",
        "elapsed_seconds": round(elapsed, 2),
        "train_file": str(train_out),
        "test_file": str(test_out),
        "train_source_sha256": _sha256_file(args.train),
        "test_source_sha256": _sha256_file(args.test),
        "train_id_sha256": _hash_records(train_aligned[args.id_column].tolist()),
        "test_id_sha256": _hash_records(test_aligned[args.id_column].tolist()),
        "train_text_sha256": _hash_records(train_aligned[args.id_column].tolist(), train_texts),
        "test_text_sha256": _hash_records(test_aligned[args.id_column].tolist(), test_texts),
    }

    head_results: dict[str, Any] | None = None
    if args.eval_head and args.folds and args.folds.exists():
        print("\n--- Training Fold-Safe Ridge Head on Embeddings ---")
        folds_df = pd.read_parquet(args.folds) if args.folds.suffix.lower() == ".parquet" else pd.read_csv(args.folds)
        required_fold_columns = {args.id_column, args.fold_column}
        missing_fold_columns = required_fold_columns - set(folds_df.columns)
        if missing_fold_columns:
            raise ValueError(f"Folds file is missing columns: {sorted(missing_fold_columns)}")
        _canonical_ids(folds_df[args.id_column].tolist(), label="folds data")
        if args.target_column not in folds_df.columns and args.target_column in train_df.columns:
            target_lookup = train_df[[args.id_column, args.target_column]]
            original_ids = folds_df[args.id_column].copy()
            folds_df = folds_df.merge(
                target_lookup,
                on=args.id_column,
                how="left",
                validate="one_to_one",
                sort=False,
            )
            if not folds_df[args.id_column].reset_index(drop=True).equals(original_ids.reset_index(drop=True)):
                raise RuntimeError("Target merge changed fold row order.")
            if folds_df[args.target_column].isna().any():
                raise ValueError("Some fold IDs have no matching training target.")

        head_results = train_embedding_head(
            train_embeddings=train_aligned,
            folds_df=folds_df,
            target_column=args.target_column,
            id_column=args.id_column,
            fold_column=args.fold_column,
            test_embeddings=test_aligned,
        )

        manifest["head_evaluation"] = {
            "oof_smape": head_results["oof_smape"],
            "mean_fold_smape": head_results["mean_fold_smape"],
            "std_fold_smape": head_results["std_fold_smape"],
            "fold_smapes": head_results["fold_smapes"],
        }
        print(f"Embedding Ridge Head OOF SMAPE: {head_results['oof_smape']:.6f} (+/- {head_results['std_fold_smape']:.6f})")

        # Save OOF and test predictions
        _atomic_parquet(head_results["oof_df"], args.output_dir / "embedding_ridge_oof.parquet")
        if head_results["test_pred_df"] is not None:
            _atomic_parquet(head_results["test_pred_df"], args.output_dir / "embedding_ridge_test.parquet")

    manifest_path = args.output_dir / "embeddings_manifest.json"
    _atomic_json(manifest, manifest_path)

    print("\n--- Text Embeddings Complete ---")
    print(f"Train embeddings: {train_aligned.shape} | Test embeddings: {test_aligned.shape}")
    print(f"Saved manifest: {manifest_path} (Elapsed: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
