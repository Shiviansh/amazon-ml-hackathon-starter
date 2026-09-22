import math
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from src.ocr import (
    OCRConfig,
    OCRDetection,
    EasyOCREngine,
    backfill_catalog_quantities,
    build_image_index,
    filter_packaging_text,
    harvest_ocr_entities,
    harvest_ocr_quantities,
    merge_detections,
    normalize_ocr_text,
    run_ocr_catalog,
)


class FakeEngine:
    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = 0

    @property
    def signature(self):
        return "fake-ocr-v1"

    def recognize(self, image, variant):
        self.calls += 1
        return [
            OCRDetection(item.text, item.confidence, item.bbox, variant)
            for item in self.detections
        ]


class PackagingTextTests(unittest.TestCase):
    def test_noise_filter_and_compound_weight(self):
        clean, rejected = filter_packaging_text(
            [
                "BEST SELLER",
                "100% satisfaction guaranteed",
                "Nutrition per 100 g",
                "NET WT 4 x 100 g",
            ]
        )
        self.assertEqual(clean, "NET WT 4 x 100 g")
        self.assertEqual(len(rejected), 3)
        quantities = harvest_ocr_quantities(clean)
        self.assertAlmostEqual(quantities["ocr_weight_g"], 400.0)
        self.assertAlmostEqual(quantities["ocr_pack_count"], 4.0)
        self.assertTrue(math.isnan(quantities["ocr_volume_ml"]))

    def test_volume_without_explicit_pack_does_not_invent_count(self):
        clean, _ = filter_packaging_text(["Net volume 750 ml"])
        quantities = harvest_ocr_quantities(clean)
        self.assertAlmostEqual(quantities["ocr_volume_ml"], 750.0)
        self.assertTrue(math.isnan(quantities["ocr_pack_count"]))

    def test_normalization_repairs_unit_spacing_and_numeric_o(self):
        self.assertEqual(normalize_ocr_text(" 5O0 m l "), "500 ml")

    def test_detection_filter_and_deduplication(self):
        detections = [
            OCRDetection("NET WT 500 g", 0.95, (0, 0, 100, 20), "gray"),
            OCRDetection("net wt 500 g", 0.70, (0, 0, 100, 20), "binary"),
            OCRDetection("noise", 0.10, (0, 30, 50, 50), "gray"),
        ]
        merged = merge_detections(detections, 0.35)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "NET WT 500 g")

    def test_promotions_are_stripped_without_losing_valid_entities(self):
        clean, _ = filter_packaging_text(
            ["Special Offer: Pack of 3, Net Wt 500g", "Buy 1 Get 1 Free (250 ml each)"]
        )
        self.assertNotIn("Special Offer", clean)
        self.assertIn("Pack of 3", clean)
        self.assertIn("500 g", clean)
        self.assertIn("pack of 2", clean)
        entities = harvest_ocr_entities(clean)
        self.assertFalse(math.isnan(entities["ocr_weight_g"]))
        self.assertFalse(math.isnan(entities["ocr_volume_ml"]))

    def test_dimensions_and_technical_specs_are_retained(self):
        clean, _ = filter_packaging_text(
            ["Dimensions: 30 x 20 x 10 cm", "65W Fast Charger", "Memory 256 GB"]
        )
        entities = harvest_ocr_entities(clean)
        self.assertEqual(entities["ocr_dim_length_cm"], 30.0)
        self.assertEqual(entities["ocr_dim_width_cm"], 20.0)
        self.assertEqual(entities["ocr_dim_height_cm"], 10.0)
        self.assertEqual(entities["ocr_power_watts"], 65.0)
        self.assertEqual(entities["ocr_storage_gb"], 256.0)


class AlignmentTests(unittest.TestCase):
    def test_manifest_alignment_is_id_based_and_primary_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one.webp", "two.webp", "extra.webp"):
                Image.new("RGB", (20, 20), "white").save(root / name, "WEBP")
            catalog = pd.DataFrame({"PRODUCT_ID": [2, 1]})
            manifest = pd.DataFrame(
                {
                    "PRODUCT_ID": [1, 2, 1],
                    "status": ["downloaded", "cached", "downloaded"],
                    "image_index": [0, 0, 1],
                    "by_id_path": ["one.webp", "two.webp", "extra.webp"],
                }
            )
            index = build_image_index(catalog, manifest, root)
            self.assertEqual(index["2"][0][0], (root / "two.webp").resolve())
            self.assertEqual(index["1"][0][0], (root / "one.webp").resolve())
            self.assertEqual(len(index["1"]), 1)

    def test_manifest_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = pd.DataFrame({"PRODUCT_ID": [1]})
            manifest = pd.DataFrame(
                {
                    "PRODUCT_ID": [1],
                    "status": ["downloaded"],
                    "by_id_path": ["../escape.webp"],
                }
            )
            with self.assertRaisesRegex(ValueError, "escapes image root"):
                build_image_index(catalog, manifest, Path(tmp))

    def test_duplicate_catalog_ids_fail_closed(self):
        catalog = pd.DataFrame({"PRODUCT_ID": [1, 1]})
        manifest = pd.DataFrame(
            {"PRODUCT_ID": [1], "status": ["downloaded"], "by_id_path": ["x.webp"]}
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_image_index(catalog, manifest, Path("."))

    def test_missing_catalog_id_and_ambiguous_manifest_fail_closed(self):
        missing_catalog = pd.DataFrame({"PRODUCT_ID": [pd.NA]})
        manifest = pd.DataFrame(
            {"PRODUCT_ID": [1], "status": ["downloaded"], "by_id_path": ["x.webp"]}
        )
        with self.assertRaisesRegex(ValueError, "non-missing"):
            build_image_index(missing_catalog, manifest, Path("."))

        catalog = pd.DataFrame({"PRODUCT_ID": [1]})
        ambiguous = pd.DataFrame(
            {
                "PRODUCT_ID": [1, 1],
                "status": ["downloaded", "cached"],
                "image_index": [0, 0],
                "by_id_path": ["x.webp", "y.webp"],
            }
        )
        with self.assertRaisesRegex(ValueError, "multiple successful paths"):
            build_image_index(catalog, ambiguous, Path("."))


class PipelineTests(unittest.TestCase):
    def _fixture(self, root):
        image_path = root / "by_id" / "p1.webp"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (300, 180), "white").save(image_path, "WEBP")
        catalog = pd.DataFrame(
            {
                "PRODUCT_ID": ["p1", "p2"],
                "TITLE": ["Coffee", "Tea"],
                "PACK_SIZE": [None, None],
                "DESCRIPTION": [None, None],
            }
        )
        manifest = pd.DataFrame(
            {
                "PRODUCT_ID": ["p1", "p2"],
                "status": ["downloaded", "failed"],
                "image_index": [0, 0],
                "by_id_path": ["by_id/p1.webp", None],
            }
        )
        return catalog, manifest

    def test_exact_schema_missing_images_and_cache_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            engine = FakeEngine(
                [OCRDetection("NET WT 2 x 250 g", 0.98, (10, 20, 180, 60))]
            )
            config = OCRConfig(workers=2, min_ocr_side=300)
            cache = root / "cache.sqlite3"
            first, diagnostics, summary = run_ocr_catalog(
                catalog, manifest, root, engine, config, cache
            )
            self.assertEqual(first.columns.tolist(), [
                "PRODUCT_ID", "ocr_text", "ocr_weight_g", "ocr_volume_ml",
                "ocr_pack_count", "has_ocr",
            ])
            self.assertEqual(first["PRODUCT_ID"].tolist(), ["p1", "p2"])
            self.assertEqual(float(first.loc[0, "ocr_weight_g"]), 500.0)
            self.assertEqual(float(first.loc[0, "ocr_pack_count"]), 2.0)
            self.assertEqual(int(first.loc[1, "has_ocr"]), 0)
            self.assertEqual(diagnostics.loc[1, "status"], "missing_image")
            self.assertEqual(summary["has_ocr_rows"], 1)
            # Both public and diagnostic schemas must survive Arrow serialization.
            first.to_parquet(root / "ocr.parquet", index=False)
            diagnostics.to_parquet(root / "diagnostics.parquet", index=False)
            self.assertEqual(pd.read_parquet(root / "ocr.parquet").columns.tolist(), first.columns.tolist())
            # A reliable primary pass short-circuits two expensive fallbacks.
            self.assertEqual(engine.calls, 1)
            second, _, second_summary = run_ocr_catalog(
                catalog, manifest, root, engine, config, cache
            )
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(second_summary["cache_hits"], 1)

    def test_adaptive_pipeline_falls_back_when_primary_has_no_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, manifest = self._fixture(root)
            engine = FakeEngine([OCRDetection("COFFEE BRAND", 0.99, (0, 0, 100, 20))])
            output, _, _ = run_ocr_catalog(
                catalog,
                manifest,
                root,
                engine,
                OCRConfig(min_ocr_side=300),
                root / "fallback-cache.db",
            )
            self.assertEqual(engine.calls, 3)
            self.assertEqual(int(output.loc[0, "has_ocr"]), 0)

    def test_corrupt_image_becomes_explicit_failure(self):
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
                }
            )
            engine = FakeEngine([])
            output, diagnostic, _ = run_ocr_catalog(
                catalog, manifest, root, engine, OCRConfig(min_ocr_side=20), root / "cache.db"
            )
            self.assertEqual(int(output.loc[0, "has_ocr"]), 0)
            self.assertEqual(diagnostic.loc[0, "status"], "image_error")
            self.assertIn("UnidentifiedImageError", diagnostic.loc[0, "error"])

    def test_title_then_ocr_then_pack_then_description_hierarchy(self):
        catalog = pd.DataFrame(
            {
                "PRODUCT_ID": [1, 2, 3],
                "TITLE": ["Rice 1 kg", "Juice", "Tea"],
                "PACK_SIZE": [None, "100 g", "200 g"],
                "DESCRIPTION": ["contains 2 g salt", "contains 2 g sweetener", "contains 3 g spice"],
            }
        )
        ocr = pd.DataFrame(
            {
                "PRODUCT_ID": [2, 1, 3],
                "ocr_text": ["Net Wt 500 g", "Net 500 g", ""],
                "ocr_weight_g": [500.0, 500.0, np.nan],
                "ocr_volume_ml": [np.nan, np.nan, np.nan],
                "ocr_pack_count": [np.nan, np.nan, np.nan],
                "has_ocr": [1, 1, 0],
            }
        )
        result = backfill_catalog_quantities(catalog, ocr)
        self.assertEqual(float(result.loc[0, "effective_weight_g"]), 1000.0)
        self.assertEqual(result.loc[0, "weight_source"], "title")
        self.assertEqual(float(result.loc[1, "effective_weight_g"]), 500.0)
        self.assertEqual(result.loc[1, "weight_source"], "ocr")
        self.assertEqual(float(result.loc[2, "effective_weight_g"]), 200.0)
        self.assertEqual(result.loc[2, "weight_source"], "pack_size")
        self.assertEqual(catalog.columns.tolist(), [
            "PRODUCT_ID", "TITLE", "PACK_SIZE", "DESCRIPTION"
        ])


class EasyOCRConcurrencyTests(unittest.TestCase):
    def test_cpu_uses_one_reader_per_active_worker_and_gpu_stays_single(self):
        barrier = threading.Barrier(2)

        class Reader:
            instances = 0

            def __init__(self, *_args, **_kwargs):
                type(self).instances += 1

            def readtext(self, *_args, **_kwargs):
                barrier.wait(timeout=2)
                return []

        fake_module = types.SimpleNamespace(Reader=Reader)
        with patch.dict(sys.modules, {"easyocr": fake_module}):
            engine = EasyOCREngine(OCRConfig(engine="easyocr", workers=2))
            image = Image.new("RGB", (20, 20), "white")
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(engine.recognize, image, f"v{i}") for i in range(2)]
                for future in futures:
                    self.assertEqual(future.result(), [])
            self.assertEqual(engine.max_parallelism, 2)
            self.assertEqual(Reader.instances, 2)

        class GPUReader:
            def __init__(self, *_args, **_kwargs):
                pass

            def readtext(self, *_args, **_kwargs):
                return []

        with patch.dict(sys.modules, {"easyocr": types.SimpleNamespace(Reader=GPUReader)}):
            gpu_engine = EasyOCREngine(OCRConfig(engine="easyocr", workers=4, gpu=True))
            self.assertEqual(gpu_engine.max_parallelism, 1)


if __name__ == "__main__":
    unittest.main()
