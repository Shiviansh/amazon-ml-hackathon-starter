import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from src.image_embeddings import (
    ImageEmbeddingConfig,
    build_arg_parser,
    build_product_image_index,
    encode_image_catalog_resumable,
    project_image_embeddings,
    train_image_embedding_head,
    verify_and_align_image_embeddings,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeEncoder:
    dimension = 4
    signature = "fake-image-encoder-v1"
    description = "fake 4d encoder"

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def encode(self, images):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("simulated encoder interruption")
        rows = []
        for image in images:
            rgb = np.asarray(image.resize((1, 1)), dtype=np.float32)[0, 0] / 255.0
            rows.append([rgb[0], rgb[1], rgb[2], 1.0])
        return np.asarray(rows, dtype=np.float32)


class BadEncoder(FakeEncoder):
    def encode(self, images):
        return np.full((len(images), self.dimension), np.nan, dtype=np.float32)


class ImageIndexTests(unittest.TestCase):
    def test_default_models_are_revision_pinned_but_custom_models_are_not(self):
        clip = ImageEmbeddingConfig(backend="clip")
        dino = ImageEmbeddingConfig(backend="dinov2")
        custom = ImageEmbeddingConfig(backend="clip", model_name="org/custom-model")
        self.assertEqual(len(clip.resolved_revision), 40)
        self.assertEqual(len(dino.resolved_revision), 40)
        self.assertIsNone(custom.resolved_revision)

    def test_strict_alignment_limits_images_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, color in (("a.webp", "red"), ("b.webp", "blue"), ("a2.webp", "green")):
                Image.new("RGB", (20, 20), color).save(root / name, "WEBP")
            catalog = pd.DataFrame({"PRODUCT_ID": [2, 1, 3]})
            manifest = pd.DataFrame(
                {
                    "PRODUCT_ID": [1, 2, 1],
                    "status": ["downloaded", "cached", "downloaded"],
                    "image_index": [0, 0, 1],
                    "by_id_path": ["a.webp", "b.webp", "a2.webp"],
                    "content_sha256": [sha256(root / "a.webp"), sha256(root / "b.webp"), sha256(root / "a2.webp")],
                }
            )
            tokens, index = build_product_image_index(
                catalog, manifest, root, ImageEmbeddingConfig(max_images_per_product=1)
            )
            self.assertEqual(len(tokens), 3)
            self.assertEqual(index[tokens[1]][0].path, (root / "a.webp").resolve())
            self.assertEqual(len(index[tokens[1]]), 1)
            self.assertEqual(index[tokens[2]], [])

            bad = manifest.copy()
            bad.loc[0, "content_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                build_product_image_index(
                    catalog, bad, root, ImageEmbeddingConfig(verify_hashes=True)
                )

    def test_manifest_boundary_duplicates_and_traversal_fail_closed(self):
        catalog = pd.DataFrame({"PRODUCT_ID": [1]})
        base = {
            "status": ["downloaded"],
            "image_index": [0],
            "content_sha256": [None],
        }
        with self.assertRaisesRegex(ValueError, "outside the catalog"):
            build_product_image_index(
                catalog,
                pd.DataFrame({"PRODUCT_ID": [2], "by_id_path": ["x"], **base}),
                Path("."),
                ImageEmbeddingConfig(verify_hashes=False),
            )
        duplicate = pd.DataFrame(
            {
                "PRODUCT_ID": [1, 1],
                "status": ["downloaded", "cached"],
                "image_index": [0, 0],
                "by_id_path": ["a", "b"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Duplicate successful"):
            build_product_image_index(
                catalog, duplicate, Path("."), ImageEmbeddingConfig(verify_hashes=False)
            )
        traversal = pd.DataFrame(
            {
                "PRODUCT_ID": [1],
                "status": ["downloaded"],
                "image_index": [0],
                "by_id_path": ["../escape.webp"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "escapes image root"):
                build_product_image_index(
                    catalog, traversal, Path(tmp), ImageEmbeddingConfig(verify_hashes=False)
                )


class ImageEmbeddingPipelineTests(unittest.TestCase):
    def _fixture(self, root):
        image_dir = root / "by_id"
        image_dir.mkdir(parents=True)
        paths = []
        for name, color in (("p1.webp", "red"), ("p2.webp", "blue")):
            path = image_dir / name
            Image.new("RGB", (32, 24), color).save(path, "WEBP")
            paths.append(path)
        catalog = pd.DataFrame({"PRODUCT_ID": ["p2", "p1", "missing"]})
        manifest = pd.DataFrame(
            {
                "PRODUCT_ID": ["p1", "p2", "missing"],
                "status": ["downloaded", "downloaded", "failed"],
                "image_index": [0, 0, 0],
                "by_id_path": ["by_id/p1.webp", "by_id/p2.webp", None],
                "content_sha256": [sha256(paths[0]), sha256(paths[1]), None],
            }
        )
        return catalog, manifest

    def test_order_normalization_missing_flags_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            encoder = FakeEncoder()
            output, diagnostics, summary = encode_image_catalog_resumable(
                catalog,
                manifest,
                root,
                encoder,
                root / "embeddings.parquet",
                ImageEmbeddingConfig(batch_size=2, chunk_size=2),
            )
            self.assertEqual(output["PRODUCT_ID"].tolist(), catalog["PRODUCT_ID"].tolist())
            embedding_columns = [c for c in output if c.startswith("img_emb_")]
            self.assertEqual(len(embedding_columns), 4)
            vectors = output[embedding_columns].to_numpy(dtype=np.float32)
            self.assertTrue(np.allclose(np.linalg.norm(vectors[:2], axis=1), 1.0, atol=2e-3))
            self.assertTrue(np.array_equal(vectors[2], np.zeros(4)))
            self.assertEqual(output["has_image_embedding"].tolist(), [1, 1, 0])
            self.assertEqual(summary["embedded_products"], 2)
            self.assertEqual(diagnostics.loc[2, "image_errors"], "no successful manifest image")
            round_trip = pd.read_parquet(root / "embeddings.parquet")
            self.assertEqual(round_trip.columns.tolist(), output.columns.tolist())

    def test_corrupt_image_is_zero_or_error_according_to_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.webp"
            bad.write_bytes(b"not an image")
            catalog = pd.DataFrame({"PRODUCT_ID": [1]})
            manifest = pd.DataFrame(
                {
                    "PRODUCT_ID": [1],
                    "status": ["downloaded"],
                    "image_index": [0],
                    "by_id_path": ["bad.webp"],
                    "content_sha256": [sha256(bad)],
                }
            )
            output, diagnostics, _ = encode_image_catalog_resumable(
                catalog,
                manifest,
                root,
                FakeEncoder(),
                root / "zero.parquet",
                ImageEmbeddingConfig(),
            )
            self.assertEqual(int(output.loc[0, "has_image_embedding"]), 0)
            self.assertEqual(int(output.loc[0, "image_failure_count"]), 1)
            self.assertIn("UnidentifiedImageError", diagnostics.loc[0, "image_errors"])
            with self.assertRaisesRegex(RuntimeError, "No usable image"):
                encode_image_catalog_resumable(
                    catalog,
                    manifest,
                    root,
                    FakeEncoder(),
                    root / "error.parquet",
                    ImageEmbeddingConfig(failure_policy="error"),
                )

    def test_multi_image_mean_pool_is_deterministic_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            red, blue = root / "red.webp", root / "blue.webp"
            Image.new("RGB", (20, 20), "red").save(red, "WEBP")
            Image.new("RGB", (20, 20), "blue").save(blue, "WEBP")
            catalog = pd.DataFrame({"PRODUCT_ID": [1]})
            manifest = pd.DataFrame(
                {
                    "PRODUCT_ID": [1, 1],
                    "status": ["downloaded", "downloaded"],
                    "image_index": [1, 0],
                    "by_id_path": ["blue.webp", "red.webp"],
                    "content_sha256": [sha256(blue), sha256(red)],
                }
            )
            output, _, _ = encode_image_catalog_resumable(
                catalog,
                manifest,
                root,
                FakeEncoder(),
                root / "multi.parquet",
                ImageEmbeddingConfig(max_images_per_product=0, pooling="mean"),
            )
            self.assertEqual(int(output.loc[0, "image_count"]), 2)
            vector = output.filter(like="img_emb_").iloc[0].to_numpy(dtype=np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=3)
            self.assertGreater(vector[0], 0)
            self.assertGreater(vector[2], 0)
            weighted, _, _ = encode_image_catalog_resumable(
                catalog,
                manifest,
                root,
                FakeEncoder(),
                root / "weighted.parquet",
                ImageEmbeddingConfig(
                    max_images_per_product=0,
                    pooling="weighted_mean",
                    secondary_image_decay=0.25,
                ),
            )
            weighted_vector = weighted.filter(like="img_emb_").iloc[0].to_numpy(dtype=np.float32)
            self.assertGreater(weighted_vector[0], weighted_vector[2])

    def test_interrupted_run_resumes_verified_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            output_path = root / "resume.parquet"
            config = ImageEmbeddingConfig(batch_size=1, chunk_size=1)
            with self.assertRaisesRegex(RuntimeError, "simulated encoder interruption"):
                encode_image_catalog_resumable(
                    catalog, manifest, root, FakeEncoder(fail_on_call=2), output_path, config
                )
            checkpoint = root / ".checkpoints_resume" / "chunk_00000.parquet"
            self.assertTrue(checkpoint.is_file())
            resumed_encoder = FakeEncoder()
            output, _, summary = encode_image_catalog_resumable(
                catalog, manifest, root, resumed_encoder, output_path, config
            )
            self.assertEqual(summary["resumed_chunks"], 1)
            # p1/p2 each need inference; one was cached and the missing row needs none.
            self.assertEqual(resumed_encoder.calls, 1)
            self.assertEqual(len(output), 3)
            self.assertFalse((root / ".checkpoints_resume").exists())

    def test_resume_rejects_changed_pooling_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            output_path = root / "stale.parquet"
            with self.assertRaisesRegex(RuntimeError, "simulated encoder interruption"):
                encode_image_catalog_resumable(
                    catalog,
                    manifest,
                    root,
                    FakeEncoder(fail_on_call=2),
                    output_path,
                    ImageEmbeddingConfig(batch_size=1, chunk_size=1, pooling="mean"),
                )
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                encode_image_catalog_resumable(
                    catalog,
                    manifest,
                    root,
                    FakeEncoder(),
                    output_path,
                    ImageEmbeddingConfig(batch_size=1, chunk_size=1, pooling="max"),
                )

    def test_bad_encoder_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                encode_image_catalog_resumable(
                    catalog,
                    manifest,
                    root,
                    BadEncoder(),
                    root / "bad.parquet",
                    ImageEmbeddingConfig(),
                )

    def test_next_decode_batch_is_prefetched_during_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            # Keep only the two products that have images.
            catalog = catalog.iloc[:2].reset_index(drop=True)
            manifest = manifest.iloc[:2].reset_index(drop=True)
            second_decode_started = threading.Event()
            lock = threading.Lock()
            decode_calls = 0

            def fake_decode(_path, _max_pixels):
                nonlocal decode_calls
                with lock:
                    decode_calls += 1
                    if decode_calls == 2:
                        second_decode_started.set()
                return Image.new("RGB", (8, 8), "red"), None

            class PrefetchAssertingEncoder(FakeEncoder):
                def encode(self, images):
                    if self.calls == 0 and not second_decode_started.wait(timeout=2):
                        raise AssertionError("next decode was not prefetched during inference")
                    return super().encode(images)

            with patch("src.image_embeddings._decode_image", side_effect=fake_decode):
                output, _, _ = encode_image_catalog_resumable(
                    catalog,
                    manifest,
                    root,
                    PrefetchAssertingEncoder(),
                    root / "prefetch.parquet",
                    ImageEmbeddingConfig(batch_size=1, chunk_size=2, decode_workers=1),
                )
            self.assertEqual(output["has_image_embedding"].tolist(), [1, 1])


class AlignmentTests(unittest.TestCase):
    def test_alignment_is_typed_and_target_ordered(self):
        embeddings = pd.DataFrame(
            {
                "PRODUCT_ID": [2, 1],
                "has_image_embedding": [1, 1],
                "img_emb_0000": [0.2, 0.1],
            }
        )
        target = pd.DataFrame({"PRODUCT_ID": [1, 2]})
        aligned = verify_and_align_image_embeddings(embeddings, target)
        self.assertEqual(aligned["PRODUCT_ID"].tolist(), [1, 2])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_and_align_image_embeddings(
                embeddings, pd.DataFrame({"PRODUCT_ID": ["1", "2"]})
            )


class ProjectionAndHeadTests(unittest.TestCase):
    def _embedding_frames(self):
        rng = np.random.default_rng(42)
        train_vectors = rng.normal(size=(12, 6)).astype(np.float32)
        train_vectors /= np.linalg.norm(train_vectors, axis=1, keepdims=True)
        test_vectors = rng.normal(loc=100.0, size=(3, 6)).astype(np.float32)
        test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)
        columns = [f"img_emb_{i:04d}" for i in range(6)]
        train = pd.DataFrame(train_vectors, columns=columns)
        train.insert(0, "image_count", 1)
        train.insert(0, "manifest_image_count", 1)
        train.insert(0, "has_image_embedding", 1)
        train.insert(0, "PRODUCT_ID", np.arange(12))
        test = pd.DataFrame(test_vectors, columns=columns)
        test.insert(0, "image_count", 1)
        test.insert(0, "manifest_image_count", 1)
        test.insert(0, "has_image_embedding", 1)
        test.insert(0, "PRODUCT_ID", np.arange(100, 103))
        return train, test, train_vectors

    def test_pca_is_fit_on_train_only_and_preserves_flags(self):
        train, test, train_vectors = self._embedding_frames()
        compact_train, compact_test, projection, report = project_image_embeddings(
            train, test, n_components=3
        )
        np.testing.assert_allclose(projection.mean_, train_vectors.mean(axis=0), atol=1e-7)
        self.assertEqual(len([c for c in compact_train if c.startswith("img_svd_")]), 3)
        self.assertIn("has_image_embedding", compact_train)
        self.assertEqual(compact_test["PRODUCT_ID"].tolist(), [100, 101, 102])
        self.assertEqual(report["fit_rows"], 12)

    def test_fold_safe_ridge_head_generates_oof_and_test_predictions(self):
        train, test, _ = self._embedding_frames()
        # Shuffle embeddings so the head must align by typed ID, not row position.
        train = train.sample(frac=1.0, random_state=7).reset_index(drop=True)
        folds = pd.DataFrame(
            {
                "PRODUCT_ID": np.arange(12),
                "PRICE": np.linspace(10.0, 120.0, 12),
                "fold": np.tile([0, 1, 2], 4),
            }
        )
        result = train_image_embedding_head(
            train, folds, test, target_column="PRICE", alpha=2.0
        )
        self.assertEqual(len(result["oof_df"]), 12)
        self.assertEqual(len(result["test_pred_df"]), 3)
        self.assertTrue(np.isfinite(result["oof_df"]["prediction"]).all())
        self.assertTrue(np.isfinite(result["oof_smape"]))
        self.assertEqual(len(result["fold_smapes"]), 3)

    def test_cli_defaults_to_train_only_and_manifest_hash_trust(self):
        args = build_arg_parser().parse_args([])
        self.assertIsNone(args.test)
        self.assertFalse(args.verify_hashes)


if __name__ == "__main__":
    unittest.main()
