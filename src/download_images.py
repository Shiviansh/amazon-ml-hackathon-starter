"""Resilient, concurrent image acquisition with a content-addressed cache.

The downloader is designed for multimodal competition pipelines where a broken
URL, interrupted process, duplicate image, or HTML error page must never become
a silently corrupted training example.

Highlights
----------
* immutable ID/URL validation and deterministic output ordering;
* bounded thread-pool concurrency with retry/backoff and Retry-After support;
* redirect and private-network protection (SSRF guard);
* streamed downloads with byte limits, MIME checks, Pillow decode verification,
  pixel/dimension limits, and atomic writes;
* SHA-256 content-addressed object storage and duplicate-image deduplication;
* resumable URL cache with configurable size/SHA verification;
* Parquet manifest and strict JSON run report for downstream image encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import random
import re
import shutil
import socket
import sqlite3
import tempfile
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import certifi
import urllib3
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.util import Timeout


DOWNLOADER_VERSION = "1.0"
CACHE_VERSION = "2.0"
CacheVerification = Literal["none", "size", "sha256"]

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}
FORMAT_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DownloadError(RuntimeError):
    """Permanent image download/validation failure."""


class RetryableDownloadError(DownloadError):
    """Failure that may succeed on a later attempt."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for :func:`download_images`."""

    output_dir: Path
    id_column: str = "PRODUCT_ID"
    url_column: str = "IMAGE_URL"
    url_separator: str | None = "|"
    max_images_per_id: int = 1
    workers: int = 16
    retries: int = 4
    connect_timeout: float = 8.0
    read_timeout: float = 30.0
    backoff_base: float = 0.75
    max_retry_after: float = 30.0
    max_redirects: int = 5
    max_bytes: int = 25 * 1024 * 1024
    min_width: int = 16
    min_height: int = 16
    max_pixels: int = 50_000_000
    max_image_size: int = 384
    output_format: Literal["webp", "jpeg"] = "webp"
    image_quality: int = 85
    user_agent: str = "amazon-ml-image-fetcher/1.0"
    allow_private_hosts: bool = False
    dns_cache_ttl: float = 300.0
    cache_verification: CacheVerification = "size"
    refresh: bool = False
    show_progress: bool = True


@dataclass
class FetchResult:
    """One unique URL fetch result, later expanded back to input rows."""

    source_url: str
    url_sha256: str
    status: str
    cache_hit: bool = False
    http_status: int | None = None
    mime_type: str | None = None
    image_format: str | None = None
    source_format: str | None = None
    extension: str | None = None
    width: int | None = None
    height: int | None = None
    source_width: int | None = None
    source_height: int | None = None
    source_byte_count: int | None = None
    source_sha256: str | None = None
    byte_count: int | None = None
    content_sha256: str | None = None
    relative_path: str | None = None
    attempts: int = 0
    redirect_count: int = 0
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


_thread_local = threading.local()
_object_commit_lock = threading.Lock()
_dns_cache_lock = threading.Lock()
_dns_cache: dict[tuple[str, int, bool], tuple[float, tuple[str, ...]]] = {}
_dns_inflight: dict[tuple[str, int, bool], threading.Event] = {}


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
    if _is_missing(value):
        raise ValueError("Image manifest IDs must be non-missing.")
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Image manifest IDs must be finite.")
        return f"float:{numeric.hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _validate_config(config: DownloadConfig) -> None:
    if not config.id_column or not config.url_column:
        raise ValueError("id_column and url_column must be non-empty.")
    if config.url_separator == "":
        raise ValueError("url_separator must be non-empty or None.")
    if not isinstance(config.max_images_per_id, int) or config.max_images_per_id < 1:
        raise ValueError("max_images_per_id must be a positive integer.")
    if not isinstance(config.workers, int) or not 1 <= config.workers <= 128:
        raise ValueError("workers must be an integer between 1 and 128.")
    if not isinstance(config.retries, int) or not 0 <= config.retries <= 20:
        raise ValueError("retries must be an integer between 0 and 20.")
    for name, value in (
        ("connect_timeout", config.connect_timeout),
        ("read_timeout", config.read_timeout),
        ("backoff_base", config.backoff_base),
        ("max_retry_after", config.max_retry_after),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if config.connect_timeout == 0 or config.read_timeout == 0:
        raise ValueError("connect_timeout and read_timeout must be positive.")
    if not isinstance(config.max_redirects, int) or not 0 <= config.max_redirects <= 20:
        raise ValueError("max_redirects must be between 0 and 20.")
    if not isinstance(config.max_bytes, int) or config.max_bytes < 1024:
        raise ValueError("max_bytes must be an integer of at least 1024.")
    if config.min_width < 1 or config.min_height < 1 or config.max_pixels < 1:
        raise ValueError("Image dimension limits must be positive.")
    if not isinstance(config.max_image_size, int) or config.max_image_size < 32:
        raise ValueError("max_image_size must be an integer of at least 32.")
    if config.output_format not in {"webp", "jpeg"}:
        raise ValueError("output_format must be webp or jpeg.")
    if not isinstance(config.image_quality, int) or not 1 <= config.image_quality <= 100:
        raise ValueError("image_quality must be between 1 and 100.")
    if not math.isfinite(config.dns_cache_ttl) or config.dns_cache_ttl < 0:
        raise ValueError("dns_cache_ttl must be finite and non-negative.")
    if config.cache_verification not in {"none", "size", "sha256"}:
        raise ValueError("cache_verification must be none, size, or sha256.")


def _normalize_url(value: Any) -> str:
    if _is_missing(value):
        raise ValueError("missing URL")
    if not np.isscalar(value):
        raise ValueError("URL must be a scalar value")
    raw = str(value).strip()
    if not raw:
        raise ValueError("empty URL")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("only http:// and https:// image URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs containing credentials are not allowed")
    if not parts.hostname:
        raise ValueError("URL has no hostname")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc
    hostname = parts.hostname.lower().rstrip(".")
    if ":" in hostname and not hostname.startswith("["):
        rendered_host = f"[{hostname}]"
    else:
        rendered_host = hostname
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = rendered_host if port is None or default_port else f"{rendered_host}:{port}"
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


class _PinnedHTTPConnection(HTTPConnection):
    """urllib3 connection whose socket uses an already-validated IP address."""

    def __init__(self, *args: Any, pinned_ip: str, **kwargs: Any):
        self._pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def _new_conn(self) -> socket.socket:
        original = self._dns_host
        self._dns_host = self._pinned_ip
        try:
            return super()._new_conn()
        finally:
            self._dns_host = original


class _PinnedHTTPSConnection(HTTPSConnection):
    def __init__(self, *args: Any, pinned_ip: str, **kwargs: Any):
        self._pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def _new_conn(self) -> socket.socket:
        original = self._dns_host
        self._dns_host = self._pinned_ip
        try:
            return super()._new_conn()
        finally:
            self._dns_host = original


class _PinnedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PinnedHTTPConnection


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PinnedHTTPSConnection


class _PinnedResponse:
    def __init__(self, response: urllib3.HTTPResponse):
        self._response = response
        self.status_code = int(response.status)
        self.headers = response.headers

    def iter_content(self, chunk_size: int) -> Any:
        yield from self._response.stream(chunk_size, decode_content=True)

    def close(self) -> None:
        self._response.close()
        self._response.release_conn()


def _resolve_addresses(
    hostname: str,
    port: int,
    *,
    allow_private_hosts: bool,
    ttl: float,
) -> tuple[str, ...]:
    """Resolve once, validate every result, and cache to avoid DNS floods."""
    key = (hostname, port, allow_private_hosts)
    while True:
        now = time.monotonic()
        with _dns_cache_lock:
            cached = _dns_cache.get(key)
            if cached is not None and cached[0] >= now:
                return cached[1]
            waiter = _dns_inflight.get(key)
            if waiter is None:
                waiter = threading.Event()
                _dns_inflight[key] = waiter
                resolver = True
            else:
                resolver = False
        if resolver:
            break
        # Wake when the owner publishes a result or failure. The timeout is a
        # dead-owner safeguard; the loop then elects another resolver.
        if not waiter.wait(timeout=30.0):
            with _dns_cache_lock:
                if _dns_inflight.get(key) is waiter:
                    _dns_inflight.pop(key, None)
                    waiter.set()
    try:
        resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        with _dns_cache_lock:
            _dns_inflight.pop(key, None)
            waiter.set()
        raise RetryableDownloadError(f"DNS resolution failed for {hostname}: {exc}") from exc
    addresses = tuple(dict.fromkeys(item[4][0] for item in resolved))
    if not addresses:
        with _dns_cache_lock:
            _dns_inflight.pop(key, None)
            waiter.set()
        raise RetryableDownloadError(f"DNS resolution returned no addresses for {hostname}")
    if not allow_private_hosts:
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                with _dns_cache_lock:
                    _dns_inflight.pop(key, None)
                    waiter.set()
                raise DownloadError(f"Refusing non-public network target: {hostname} -> {ip}")
    with _dns_cache_lock:
        _dns_cache[key] = (now + ttl, addresses)
        _dns_inflight.pop(key, None)
        waiter.set()
    return addresses


def _get_pinned_pool(
    scheme: str,
    hostname: str,
    port: int,
    pinned_ip: str,
) -> HTTPConnectionPool:
    pools = getattr(_thread_local, "pinned_pools", None)
    if pools is None:
        pools = {}
        _thread_local.pinned_pools = pools
    key = (scheme, hostname, port, pinned_ip)
    pool = pools.get(key)
    if pool is None:
        if scheme == "https":
            pool = _PinnedHTTPSConnectionPool(
                hostname,
                port,
                maxsize=1,
                block=True,
                retries=False,
                cert_reqs="CERT_REQUIRED",
                ca_certs=certifi.where(),
                assert_hostname=hostname,
                server_hostname=hostname,
                pinned_ip=pinned_ip,
            )
        else:
            pool = _PinnedHTTPConnectionPool(
                hostname,
                port,
                maxsize=1,
                block=True,
                retries=False,
                pinned_ip=pinned_ip,
            )
        pools[key] = pool
    return pool


def _retry_after_seconds(value: str | None, *, maximum: float) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = target.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds):
        return None
    return min(max(0.0, seconds), maximum)


def _request_with_redirects(
    url: str,
    config: DownloadConfig,
    *,
    address_offset: int = 0,
) -> tuple[_PinnedResponse, str, int]:
    current = url
    for redirect_count in range(config.max_redirects + 1):
        parts = urlsplit(current)
        hostname = parts.hostname
        if hostname is None:
            raise DownloadError("URL has no hostname")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = _resolve_addresses(
            hostname,
            port,
            allow_private_hosts=config.allow_private_hosts,
            ttl=config.dns_cache_ttl,
        )
        pinned_ip = addresses[(address_offset + redirect_count) % len(addresses)]
        pool = _get_pinned_pool(parts.scheme, hostname, port, pinned_ip)
        request_target = urlunsplit(("", "", parts.path or "/", parts.query, ""))
        host_header = hostname
        if ":" in hostname:
            host_header = f"[{hostname}]"
        if port not in {80, 443}:
            host_header = f"{host_header}:{port}"
        try:
            raw_response = pool.urlopen(
                "GET",
                request_target,
                headers={
                    "Host": host_header,
                    "User-Agent": config.user_agent,
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.5",
                },
                redirect=False,
                retries=False,
                preload_content=False,
                timeout=Timeout(connect=config.connect_timeout, read=config.read_timeout),
            )
        except urllib3.exceptions.HTTPError as exc:
            raise RetryableDownloadError(f"Connection failed for {hostname}: {exc}") from exc
        response = _PinnedResponse(raw_response)
        if response.status_code not in REDIRECT_STATUS_CODES:
            return response, current, redirect_count
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise DownloadError(f"HTTP {response.status_code} redirect had no Location header")
        if redirect_count >= config.max_redirects:
            raise DownloadError(f"Exceeded {config.max_redirects} redirects")
        try:
            current = _normalize_url(urljoin(current, location))
        except ValueError as exc:
            raise DownloadError(f"Invalid redirect target: {exc}") from exc
    raise DownloadError("Redirect handling reached an impossible state")


def _process_image_file(
    path: Path,
    config: DownloadConfig,
) -> tuple[str, int, int, str, str, str, int, int]:
    """Decode once, orient, downscale, and atomically normalize an image."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    normalized_path = path.with_name(path.name + ".normalized")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                source_format = (image.format or "").upper()
                source_width, source_height = image.size
                if source_format not in FORMAT_EXTENSIONS:
                    raise DownloadError(
                        f"Unsupported raster image format: {source_format or 'unknown'}"
                    )
                if source_width < config.min_width or source_height < config.min_height:
                    raise DownloadError(
                        f"Image is too small ({source_width}x{source_height}); minimum is "
                        f"{config.min_width}x{config.min_height}"
                    )
                if source_width * source_height > config.max_pixels:
                    raise DownloadError(
                        f"Image has {source_width * source_height:,} pixels; "
                        f"maximum is {config.max_pixels:,}"
                    )
                if getattr(image, "is_animated", False):
                    image.seek(0)
                image.load()  # One full decode pass also rejects truncated/corrupt payloads.
                normalized = ImageOps.exif_transpose(image)
                normalized.thumbnail(
                    (config.max_image_size, config.max_image_size),
                    Image.Resampling.LANCZOS,
                )
                if config.output_format == "webp":
                    output_format, extension, mime_type = "WEBP", ".webp", "image/webp"
                    if normalized.mode not in {"RGB", "RGBA"}:
                        normalized = normalized.convert("RGBA" if "transparency" in normalized.info else "RGB")
                    normalized.save(
                        normalized_path,
                        format="WEBP",
                        quality=config.image_quality,
                        method=4,
                    )
                else:
                    output_format, extension, mime_type = "JPEG", ".jpg", "image/jpeg"
                    if normalized.mode in {"RGBA", "LA"}:
                        background = Image.new("RGB", normalized.size, "white")
                        alpha = normalized.getchannel("A")
                        background.paste(normalized.convert("RGB"), mask=alpha)
                        normalized = background
                    elif normalized.mode != "RGB":
                        normalized = normalized.convert("RGB")
                    normalized.save(
                        normalized_path,
                        format="JPEG",
                        quality=config.image_quality,
                        optimize=True,
                        progressive=True,
                    )
                width, height = normalized.size
        with normalized_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(normalized_path, path)
    except DownloadError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise DownloadError(f"Invalid or corrupt image: {exc}") from exc
    finally:
        if normalized_path.exists():
            try:
                normalized_path.unlink()
            except OSError:
                pass
    return (
        source_format,
        int(source_width),
        int(source_height),
        output_format,
        extension,
        mime_type,
        int(width),
        int(height),
    )


def _object_path(output_dir: Path, content_sha256: str, extension: str) -> Path:
    return output_dir / "objects" / content_sha256[:2] / f"{content_sha256}{extension}"


def _safe_cached_path(output_dir: Path, relative_path: str) -> Path:
    root = output_dir.resolve()
    candidate = (output_dir / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError(f"Cache index path escapes output directory: {relative_path}")
    return candidate


def _build_alias_stems(ids: Sequence[Any]) -> list[str]:
    """Build readable, path-safe, collision-free product aliases."""
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    bases: list[str] = []
    tokens = [_canonical_id(value) for value in ids]
    for value, token in zip(ids, tokens):
        rendered = str(value).strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", rendered).strip(" ._") or "item"
        if len(safe) > 120 or safe.upper() in reserved:
            safe = f"{safe[:100]}-{hashlib.sha256(token.encode()).hexdigest()[:12]}"
        bases.append(safe)
    counts: dict[str, int] = {}
    for base in bases:
        counts[base.lower()] = counts.get(base.lower(), 0) + 1
    return [
        base if counts[base.lower()] == 1 else (
            f"{base}-{hashlib.sha256(token.encode()).hexdigest()[:12]}"
        )
        for base, token in zip(bases, tokens)
    ]


def _materialize_by_id_alias(source: Path, destination: Path) -> None:
    """Atomically create a hardlink alias, falling back to a file copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _clear_product_aliases(output_dir: Path, stems: Sequence[str]) -> None:
    """Remove stale generated aliases for products included in the current run."""
    alias_dir = output_dir / "by_id"
    if not alias_dir.exists():
        return
    for stem in stems:
        for candidate in alias_dir.glob(f"{stem}.*"):
            if candidate.is_file():
                candidate.unlink()
        for candidate in alias_dir.glob(f"{stem}__??.*"):
            if candidate.is_file():
                candidate.unlink()


def _cache_hit_result(
    url: str,
    url_sha256: str,
    entry: dict[str, Any],
    config: DownloadConfig,
) -> FetchResult | None:
    if entry.get("normalization_signature") != _normalization_signature(config):
        return None
    relative_path = entry.get("relative_path")
    content_sha256 = entry.get("content_sha256")
    byte_count = entry.get("byte_count")
    if not isinstance(relative_path, str) or not isinstance(content_sha256, str):
        return None
    if not SHA256_PATTERN.fullmatch(content_sha256):
        return None
    path = _safe_cached_path(config.output_dir, relative_path)
    if not path.is_file():
        return None
    if config.cache_verification in {"size", "sha256"}:
        if not isinstance(byte_count, int) or path.stat().st_size != byte_count:
            return None
    if config.cache_verification == "sha256" and _sha256_file(path) != content_sha256:
        return None
    return FetchResult(
        source_url=url,
        url_sha256=url_sha256,
        status="cached",
        cache_hit=True,
        http_status=entry.get("http_status"),
        mime_type=entry.get("mime_type"),
        image_format=entry.get("image_format"),
        source_format=entry.get("source_format"),
        extension=entry.get("extension"),
        width=entry.get("width"),
        height=entry.get("height"),
        source_width=entry.get("source_width"),
        source_height=entry.get("source_height"),
        source_byte_count=entry.get("source_byte_count"),
        source_sha256=entry.get("source_sha256"),
        byte_count=byte_count,
        content_sha256=content_sha256,
        relative_path=relative_path,
        attempts=0,
        redirect_count=int(entry.get("redirect_count", 0)),
        etag=entry.get("etag"),
        last_modified=entry.get("last_modified"),
        elapsed_seconds=0.0,
    )


def _download_one(url: str, url_sha256: str, config: DownloadConfig) -> FetchResult:
    started = time.perf_counter()
    last_error = "unknown download failure"
    last_status: int | None = None
    total_attempts = config.retries + 1

    for attempt in range(1, total_attempts + 1):
        temporary_path: Path | None = None
        response: _PinnedResponse | None = None
        try:
            response, final_url, redirects = _request_with_redirects(
                url, config, address_offset=attempt - 1
            )
            last_status = response.status_code
            if response.status_code in RETRYABLE_STATUS_CODES:
                retry_after = _retry_after_seconds(
                    response.headers.get("Retry-After"), maximum=config.max_retry_after
                )
                raise RetryableDownloadError(
                    f"HTTP {response.status_code} from {final_url}",
                    retry_after=retry_after,
                )
            if not 200 <= response.status_code < 300:
                raise DownloadError(f"HTTP {response.status_code} from {final_url}")

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                raise DownloadError(f"Response Content-Type is not an image: {content_type}")
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise DownloadError("Response has an invalid Content-Length") from exc
                if declared_size < 0 or declared_size > config.max_bytes:
                    raise DownloadError(
                        f"Declared image size {declared_size:,} exceeds limit {config.max_bytes:,}"
                    )

            temp_dir = config.output_dir / ".tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(prefix="image_", suffix=".part", dir=temp_dir)
            temporary_path = Path(raw_path)
            digest = hashlib.sha256()
            downloaded = 0
            with os.fdopen(descriptor, "wb") as handle:
                for block in response.iter_content(chunk_size=128 * 1024):
                    if not block:
                        continue
                    downloaded += len(block)
                    if downloaded > config.max_bytes:
                        raise DownloadError(
                            f"Downloaded image exceeded limit {config.max_bytes:,} bytes"
                        )
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if downloaded == 0:
                raise DownloadError("Response body was empty")

            source_sha256 = digest.hexdigest()
            (
                source_format,
                source_width,
                source_height,
                image_format,
                extension,
                detected_mime,
                width,
                height,
            ) = _process_image_file(temporary_path, config)
            source_mime = FORMAT_MIME_TYPES[source_format]
            if content_type.startswith("image/") and content_type not in {
                source_mime,
                "image/jpg" if source_mime == "image/jpeg" else source_mime,
                "image/x-png" if source_mime == "image/png" else source_mime,
            }:
                raise DownloadError(
                    f"Content-Type {content_type} disagrees with decoded {source_mime}"
                )

            normalized_bytes = temporary_path.stat().st_size
            content_sha256 = _sha256_file(temporary_path)
            destination = _object_path(config.output_dir, content_sha256, extension)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _object_commit_lock:
                if destination.exists():
                    # A content-addressed path is valid only when its bytes hash
                    # to its filename. Repair corruption atomically; do not
                    # trust equal file sizes.
                    if _sha256_file(destination) == content_sha256:
                        temporary_path.unlink()
                    else:
                        os.replace(temporary_path, destination)
                else:
                    os.replace(temporary_path, destination)
                temporary_path = None

            return FetchResult(
                source_url=url,
                url_sha256=url_sha256,
                status="downloaded",
                cache_hit=False,
                http_status=response.status_code,
                mime_type=detected_mime,
                image_format=image_format,
                source_format=source_format,
                extension=extension,
                width=width,
                height=height,
                source_width=source_width,
                source_height=source_height,
                source_byte_count=downloaded,
                source_sha256=source_sha256,
                byte_count=normalized_bytes,
                content_sha256=content_sha256,
                relative_path=destination.relative_to(config.output_dir).as_posix(),
                attempts=attempt,
                redirect_count=redirects,
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                elapsed_seconds=time.perf_counter() - started,
            )
        except RetryableDownloadError as exc:
            last_error = str(exc)
            retry_after = exc.retry_after
        except (urllib3.exceptions.HTTPError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            retry_after = None
        except DownloadError as exc:
            last_error = str(exc)
            break
        finally:
            if response is not None:
                response.close()
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        if attempt < total_attempts:
            delay = retry_after if retry_after is not None else (
                config.backoff_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
            )
            time.sleep(min(delay, config.max_retry_after))

    return FetchResult(
        source_url=url,
        url_sha256=url_sha256,
        status="failed",
        cache_hit=False,
        http_status=last_status,
        attempts=min(total_attempts, attempt),
        error=last_error[:2_000],
        elapsed_seconds=time.perf_counter() - started,
    )


def _open_cache_database(path: Path) -> sqlite3.Connection:
    """Open a scalable transactional URL cache with one JSON payload per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                url_sha256 TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'cache_version'"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES('cache_version', ?)",
                (CACHE_VERSION,),
            )
            connection.commit()
        elif row[0] != CACHE_VERSION:
            raise RuntimeError(
                f"Unsupported cache database version {row[0]!r}; expected {CACHE_VERSION!r}."
            )
        return connection
    except Exception:
        connection.close()
        raise


def _load_cache_entries(path: Path, url_hashes: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not url_hashes:
        return {}
    connection = _open_cache_database(path)
    try:
        output: dict[str, dict[str, Any]] = {}
        # SQLite commonly limits bound variables to 999; stay comfortably below.
        for start in range(0, len(url_hashes), 500):
            batch = list(url_hashes[start:start + 500])
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"SELECT url_sha256, payload_json FROM cache_entries "
                f"WHERE url_sha256 IN ({placeholders})",
                batch,
            ).fetchall()
            for url_sha256, payload_json in rows:
                try:
                    payload = json.loads(payload_json)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Corrupt cache payload for URL hash {url_sha256}."
                    ) from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Invalid cache payload for URL hash {url_sha256}.")
                output[url_sha256] = payload
        return output
    finally:
        connection.close()


def _upsert_cache_entries(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    if not entries:
        return
    connection = _open_cache_database(path)
    try:
        updated = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                key,
                json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                updated,
            )
            for key, value in entries.items()
        ]
        with connection:
            connection.executemany(
                """
                INSERT INTO cache_entries(url_sha256, payload_json, updated_at_utc)
                VALUES(?, ?, ?)
                ON CONFLICT(url_sha256) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                rows,
            )
    finally:
        connection.close()


def _cache_entry(result: FetchResult, config: DownloadConfig) -> dict[str, Any]:
    entry = {
        key: value
        for key, value in asdict(result).items()
        if key not in {"source_url", "url_sha256", "status", "cache_hit", "error", "elapsed_seconds"}
    }
    entry["normalization_signature"] = _normalization_signature(config)
    return entry


def _normalization_signature(config: DownloadConfig) -> str:
    payload = {
        "downloader_version": DOWNLOADER_VERSION,
        "pillow_version": Image.__version__,
        "max_image_size": config.max_image_size,
        "output_format": config.output_format,
        "image_quality": config.image_quality,
        "min_width": config.min_width,
        "min_height": config.min_height,
        "max_pixels": config.max_pixels,
        "max_bytes": config.max_bytes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def download_images(
    frame: pd.DataFrame,
    config: DownloadConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and cache all images, returning row manifest and run summary.

    Network failures become explicit manifest rows rather than exceptions. Input
    schema/ID/cache corruption remains fail-closed because continuing would make
    row-to-image alignment untrustworthy.
    """
    _validate_config(config)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")
    missing_columns = {config.id_column, config.url_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Input is missing columns: {sorted(missing_columns)}")
    canonical_ids = [_canonical_id(value) for value in frame[config.id_column]]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError(f"ID column '{config.id_column}' must be unique.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache_database_path = config.output_dir / "cache.sqlite3"

    row_references: list[dict[str, Any]] = []
    unique_urls: dict[str, tuple[str, str]] = {}
    truncated_url_count = 0
    alias_stems = _build_alias_stems(frame[config.id_column].tolist())
    for position, (raw_id, value) in enumerate(
        zip(frame[config.id_column], frame[config.url_column])
    ):
        if _is_missing(value):
            parts: list[Any] = [value]
        else:
            raw_value = str(value).strip()
            if config.url_separator is None:
                parts = [raw_value]
            else:
                parts = [part.strip() for part in raw_value.split(config.url_separator) if part.strip()]
                if not parts:
                    parts = [""]
        if len(parts) > config.max_images_per_id:
            truncated_url_count += len(parts) - config.max_images_per_id
            parts = parts[:config.max_images_per_id]
        for image_index, part in enumerate(parts):
            try:
                normalized = _normalize_url(part)
                url_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                error = None
                unique_urls.setdefault(url_sha256, (normalized, url_sha256))
            except ValueError as exc:
                normalized = None
                url_sha256 = None
                error = str(exc)
            row_references.append({
                "input_position": position,
                "raw_id": raw_id,
                "alias_stem": alias_stems[position],
                "image_index": image_index,
                "input_url": "" if _is_missing(part) else str(part),
                "normalized_url": normalized,
                "url_sha256": url_sha256,
                "invalid_error": error,
            })

    cache_entries = _load_cache_entries(cache_database_path, list(unique_urls))

    results: dict[str, FetchResult] = {}
    pending: list[tuple[str, str]] = []
    if not config.refresh:
        for url_sha256, (url, _) in unique_urls.items():
            cached = _cache_hit_result(
                url, url_sha256, cache_entries.get(url_sha256, {}), config
            )
            if cached is not None:
                results[url_sha256] = cached
            else:
                pending.append((url, url_sha256))
    else:
        pending = list(unique_urls.values())

    if pending:
        with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="image-fetch") as pool:
            futures: dict[Future[FetchResult], str] = {
                pool.submit(_download_one, url, url_sha256, config): url_sha256
                for url, url_sha256 in pending
            }
            completed: Any = as_completed(futures)
            if config.show_progress:
                try:
                    from tqdm import tqdm

                    completed = tqdm(completed, total=len(futures), desc="Downloading images", unit="image")
                except ImportError:
                    pass
            for future in completed:
                url_sha256 = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # Defensive: preserve manifest completeness.
                    url = unique_urls[url_sha256][0]
                    result = FetchResult(
                        source_url=url,
                        url_sha256=url_sha256,
                        status="failed",
                        error=f"Unexpected worker error: {type(exc).__name__}: {exc}"[:2_000],
                    )
                results[url_sha256] = result

    successful_cache_updates: dict[str, dict[str, Any]] = {}
    for url_sha256, result in results.items():
        if result.status in {"downloaded", "cached"}:
            successful_cache_updates[url_sha256] = _cache_entry(result, config)
    _upsert_cache_entries(cache_database_path, successful_cache_updates)

    _clear_product_aliases(config.output_dir, alias_stems)
    manifest_rows: list[dict[str, Any]] = []
    for reference in row_references:
        url = reference["normalized_url"]
        url_sha256 = reference["url_sha256"]
        invalid_error = reference["invalid_error"]
        if invalid_error is not None or url_sha256 is None:
            result = FetchResult(
                source_url="" if url is None else url,
                url_sha256="" if url_sha256 is None else url_sha256,
                status="invalid_url",
                error=invalid_error,
            )
        else:
            result = results[url_sha256]
        by_id_path: str | None = None
        if result.status in {"downloaded", "cached"} and result.relative_path and result.extension:
            source_path = _safe_cached_path(config.output_dir, result.relative_path)
            image_suffix = "" if reference["image_index"] == 0 else f"__{reference['image_index']:02d}"
            alias_path = config.output_dir / "by_id" / (
                f"{reference['alias_stem']}{image_suffix}{result.extension}"
            )
            _materialize_by_id_alias(source_path, alias_path)
            by_id_path = alias_path.relative_to(config.output_dir).as_posix()
        row = {
            config.id_column: reference["raw_id"],
            "input_position": reference["input_position"],
            "image_index": reference["image_index"],
            "input_url": reference["input_url"],
            "by_id_path": by_id_path,
            **asdict(result),
        }
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    success_mask = manifest["status"].isin(["downloaded", "cached"])
    failures = int((~success_mask).sum())
    product_success = success_mask.groupby(manifest["input_position"]).any()
    successful_products = int(product_success.sum())
    failed_products = int(len(frame) - successful_products)
    summary = {
        "downloader_version": DOWNLOADER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(frame)),
        "image_references": int(len(manifest)),
        "unique_valid_urls": int(len(unique_urls)),
        "duplicate_url_rows": int(
            sum(1 for value in row_references if value["url_sha256"]) - len(unique_urls)
        ),
        "truncated_url_count": int(truncated_url_count),
        "successful_products": successful_products,
        "failed_products": failed_products,
        "successful_rows": int(success_mask.sum()),
        "downloaded_rows": int((manifest["status"] == "downloaded").sum()),
        "cached_rows": int((manifest["status"] == "cached").sum()),
        "failed_rows": failures,
        "failure_rate": float(failed_products / len(frame)),
        "unique_content_objects": int(manifest.loc[success_mask, "content_sha256"].nunique()),
        "total_referenced_bytes": int(
            pd.to_numeric(manifest.loc[success_mask, "byte_count"], errors="coerce").fillna(0).sum()
        ),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "status_counts": {
            str(key): int(value) for key, value in manifest["status"].value_counts().items()
        },
    }
    return manifest, summary


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be CSV or Parquet.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrently fetch, validate, deduplicate, and cache product images."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-column", default="PRODUCT_ID")
    parser.add_argument("--url-column", default="IMAGE_URL")
    parser.add_argument(
        "--url-separator",
        default="|",
        help="Separator for multiple URLs in one cell; default is '|'.",
    )
    parser.add_argument(
        "--max-images-per-id",
        type=int,
        default=1,
        help="Fetch at most this many URLs per product, in source order.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--backoff-base", type=float, default=0.75)
    parser.add_argument("--max-retry-after", type=float, default=30.0)
    parser.add_argument("--max-redirects", type=int, default=5)
    parser.add_argument("--max-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--min-height", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=50_000_000)
    parser.add_argument(
        "--max-image-size", type=int, default=384,
        help="Downscale so the longest side is at most this many pixels.",
    )
    parser.add_argument("--output-format", choices=["webp", "jpeg"], default="webp")
    parser.add_argument("--image-quality", type=int, default=85)
    parser.add_argument("--user-agent", default="amazon-ml-image-fetcher/1.0")
    parser.add_argument(
        "--allow-private-hosts",
        action="store_true",
        help="Permit localhost/private IPs. Intended only for trusted test servers.",
    )
    parser.add_argument("--dns-cache-ttl", type=float, default=300.0)
    parser.add_argument(
        "--cache-verification", choices=["none", "size", "sha256"], default="size"
    )
    parser.add_argument("--refresh", action="store_true", help="Redownload even valid cache entries.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.05,
        help="Exit non-zero when failed/invalid rows exceed this fraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not math.isfinite(args.max_failure_rate) or not 0.0 <= args.max_failure_rate <= 1.0:
        raise ValueError("--max-failure-rate must be between 0 and 1.")
    frame = _load_table(args.input)
    config = DownloadConfig(
        output_dir=args.output_dir,
        id_column=args.id_column,
        url_column=args.url_column,
        url_separator=args.url_separator,
        max_images_per_id=args.max_images_per_id,
        workers=args.workers,
        retries=args.retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        backoff_base=args.backoff_base,
        max_retry_after=args.max_retry_after,
        max_redirects=args.max_redirects,
        max_bytes=args.max_bytes,
        min_width=args.min_width,
        min_height=args.min_height,
        max_pixels=args.max_pixels,
        max_image_size=args.max_image_size,
        output_format=args.output_format,
        image_quality=args.image_quality,
        user_agent=args.user_agent,
        allow_private_hosts=args.allow_private_hosts,
        dns_cache_ttl=args.dns_cache_ttl,
        cache_verification=args.cache_verification,
        refresh=args.refresh,
        show_progress=not args.no_progress,
    )
    started = time.perf_counter()
    manifest, summary = download_images(frame, config)
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    summary["input_path"] = str(args.input)
    summary["input_sha256"] = _sha256_file(args.input)

    manifest_path = args.output_dir / "image_manifest.parquet"
    failures_path = args.output_dir / "image_failures.csv"
    report_path = args.output_dir / "download_report.json"
    _atomic_parquet(manifest, manifest_path)
    failed = manifest.loc[~manifest["status"].isin(["downloaded", "cached"])]
    if not failed.empty:
        temporary = failures_path.with_name(failures_path.name + ".tmp")
        failed.to_csv(temporary, index=False)
        os.replace(temporary, failures_path)
    elif failures_path.exists():
        failures_path.unlink()
    _atomic_json(summary, report_path)

    print(
        f"Images complete: {summary['successful_products']:,}/{summary['input_rows']:,} products covered | "
        f"downloaded={summary['downloaded_rows']:,} cached={summary['cached_rows']:,} "
        f"failed={summary['failed_rows']:,}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    if summary["failure_rate"] > args.max_failure_rate:
        print(
            f"Failure rate {summary['failure_rate']:.2%} exceeds allowed "
            f"{args.max_failure_rate:.2%}.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
