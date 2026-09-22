"""Tests for resilient concurrent image fetching and cache recovery."""

import hashlib
import io
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from PIL import Image

from src.download_images import (
    DownloadConfig,
    _dns_cache,
    _dns_cache_lock,
    download_images,
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 18), color=(20, 120, 220)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1200, 800), color=(190, 80, 30)).save(
        buffer, format="JPEG", quality=95
    )
    return buffer.getvalue()


class _ImageHandler(BaseHTTPRequestHandler):
    png = _png_bytes()
    jpeg = _jpeg_bytes()
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def log_message(self, format, *args):  # noqa: A003 - stdlib signature
        return

    def _count(self) -> int:
        with self.lock:
            self.counts[self.path] = self.counts.get(self.path, 0) + 1
            return self.counts[self.path]

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        count = self._count()
        if self.path == "/retry.png" and count == 1:
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        if self.path == "/redirect.png":
            self.send_response(302)
            self.send_header("Location", "/image.png")
            self.end_headers()
            return
        if self.path == "/html":
            payload = b"<html>not an image</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in {"/image.png", "/retry.png"}:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.png)))
            self.send_header("ETag", '"test-image"')
            self.end_headers()
            self.wfile.write(self.png)
            return
        if self.path == "/large.jpg":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(self.jpeg)))
            self.end_headers()
            self.wfile.write(self.jpeg)
            return
        self.send_response(404)
        self.end_headers()


class DownloadImagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ImageHandler.counts = {}
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _config(self, root: Path, **overrides) -> DownloadConfig:
        values = {
            "output_dir": root,
            "id_column": "id",
            "url_column": "url",
            "workers": 4,
            "retries": 2,
            "backoff_base": 0.0,
            "max_retry_after": 0.0,
            "min_width": 1,
            "min_height": 1,
            "allow_private_hosts": True,
            "show_progress": False,
        }
        values.update(overrides)
        return DownloadConfig(**values)

    def test_concurrent_retry_redirect_deduplication_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_url = f"{self.base_url}/image.png"
            frame = pd.DataFrame({
                "id": [1, 2, 3, 4],
                "url": [
                    image_url,
                    image_url,
                    f"{self.base_url}/retry.png",
                    f"{self.base_url}/redirect.png",
                ],
            })
            before = dict(_ImageHandler.counts)
            manifest, summary = download_images(frame, self._config(root))
            self.assertTrue((manifest["status"] == "downloaded").all())
            self.assertEqual(summary["successful_rows"], 4)
            self.assertEqual(summary["duplicate_url_rows"], 1)
            self.assertEqual(summary["unique_content_objects"], 1)
            self.assertEqual(manifest["content_sha256"].nunique(), 1)
            relative = manifest.loc[0, "relative_path"]
            self.assertTrue((root / relative).is_file())
            self.assertTrue((root / manifest.loc[0, "by_id_path"]).is_file())
            self.assertTrue((root / "cache.sqlite3").is_file())
            self.assertFalse((root / "cache_index.json").exists())
            self.assertEqual(_ImageHandler.counts.get("/retry.png", 0) - before.get("/retry.png", 0), 2)

            counts_after_first = dict(_ImageHandler.counts)
            resumed, resumed_summary = download_images(frame, self._config(root))
            self.assertTrue((resumed["status"] == "cached").all())
            self.assertEqual(resumed_summary["cached_rows"], 4)
            self.assertEqual(_ImageHandler.counts, counts_after_first)

    def test_invalid_url_and_non_image_are_explicit_failures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame = pd.DataFrame({
                "id": [1, 2, 3],
                "url": [None, "file:///etc/passwd", f"{self.base_url}/html"],
            })
            manifest, summary = download_images(frame, self._config(Path(tmp_dir), retries=0))
            self.assertEqual(manifest["status"].tolist(), ["invalid_url", "invalid_url", "failed"])
            self.assertIn("not an image", manifest.loc[2, "error"])
            self.assertEqual(summary["failed_rows"], 3)

    def test_pipe_delimited_urls_default_to_primary_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            value = f"{self.base_url}/image.png|{self.base_url}/html"
            frame = pd.DataFrame({"id": [1], "url": [value]})
            manifest, summary = download_images(frame, self._config(Path(tmp_dir)))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest.loc[0, "image_index"], 0)
            self.assertEqual(manifest.loc[0, "status"], "downloaded")
            self.assertEqual(summary["truncated_url_count"], 1)

            multi_root = Path(tmp_dir) / "multi"
            multi, multi_summary = download_images(
                frame, self._config(multi_root, max_images_per_id=2, retries=0)
            )
            self.assertEqual(len(multi), 2)
            self.assertEqual(multi["status"].tolist(), ["downloaded", "failed"])
            self.assertEqual(multi_summary["successful_products"], 1)

    def test_private_network_targets_are_blocked_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame = pd.DataFrame({"id": [1], "url": [f"{self.base_url}/image.png"]})
            config = self._config(Path(tmp_dir), allow_private_hosts=False, retries=0)
            manifest, _ = download_images(frame, config)
            self.assertEqual(manifest.loc[0, "status"], "failed")
            self.assertIn("non-public network target", manifest.loc[0, "error"])

    def test_sha256_cache_verification_repairs_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            url = f"{self.base_url}/image.png"
            frame = pd.DataFrame({"id": [1], "url": [url]})
            config = self._config(root, cache_verification="sha256")
            first, _ = download_images(frame, config)
            object_path = root / first.loc[0, "relative_path"]
            expected_hash = first.loc[0, "content_sha256"]
            original_size = object_path.stat().st_size
            object_path.write_bytes(b"x" * original_size)
            self.assertNotEqual(hashlib.sha256(object_path.read_bytes()).hexdigest(), expected_hash)

            repaired, _ = download_images(frame, config)
            self.assertEqual(repaired.loc[0, "status"], "downloaded")
            self.assertEqual(hashlib.sha256(object_path.read_bytes()).hexdigest(), expected_hash)

    def test_large_image_is_single_pass_downscaled_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame = pd.DataFrame({"id": ["product-1"], "url": [f"{self.base_url}/large.jpg"]})
            manifest, _ = download_images(frame, self._config(root, max_image_size=384))
            row = manifest.iloc[0]
            self.assertEqual(row["source_format"], "JPEG")
            self.assertEqual((row["source_width"], row["source_height"]), (1200, 800))
            self.assertEqual(row["image_format"], "WEBP")
            self.assertEqual((row["width"], row["height"]), (384, 256))
            self.assertEqual(row["by_id_path"], "by_id/product-1.webp")
            self.assertLess(row["byte_count"], row["source_byte_count"])
            with Image.open(root / row["relative_path"]) as image:
                self.assertEqual(image.size, (384, 256))
                self.assertEqual(image.format, "WEBP")

            changed, _ = download_images(
                frame, self._config(root, max_image_size=256, image_quality=75)
            )
            self.assertEqual(changed.loc[0, "status"], "downloaded")
            self.assertEqual((changed.loc[0, "width"], changed.loc[0, "height"]), (256, 171))

    def test_hostname_is_resolved_once_then_connection_uses_pinned_ip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            local_base = f"http://localhost:{self.server.server_port}"
            frame = pd.DataFrame({
                "id": [1, 2],
                "url": [f"{local_base}/image.png", f"{local_base}/large.jpg"],
            })
            with _dns_cache_lock:
                _dns_cache.clear()
            original = __import__("socket").getaddrinfo
            hostname_calls = 0

            def counted_getaddrinfo(host, *args, **kwargs):
                nonlocal hostname_calls
                if host == "localhost":
                    hostname_calls += 1
                return original(host, *args, **kwargs)

            with patch("src.download_images.socket.getaddrinfo", side_effect=counted_getaddrinfo):
                manifest, _ = download_images(
                    frame,
                    self._config(root, retries=2, dns_cache_ttl=300.0),
                )
            self.assertTrue((manifest["status"] == "downloaded").all())
            self.assertEqual(hostname_calls, 1)

    def test_duplicate_and_missing_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._config(Path(tmp_dir))
            duplicate = pd.DataFrame({"id": [1, 1], "url": ["https://a.test/a", "https://a.test/b"]})
            with self.assertRaisesRegex(ValueError, "must be unique"):
                download_images(duplicate, config)
            missing = pd.DataFrame({"id": [None], "url": ["https://a.test/a"]})
            with self.assertRaisesRegex(ValueError, "non-missing"):
                download_images(missing, config)


if __name__ == "__main__":
    unittest.main()
