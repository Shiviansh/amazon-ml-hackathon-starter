"""Tests for adapting the 2025-style competition CSV schema."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.prepare_challenge_data import adapt_files


class PrepareChallengeDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.train_path = self.root / "raw_train.csv"
        self.test_path = self.root / "raw_test.csv"
        self.sample_path = self.root / "raw_sample.csv"
        self.output_dir = self.root / "canonical"
        self.train = pd.DataFrame({
            "sample_id": ["001", "002"],
            "catalog_content": ["Widget, 2 pack", None],
            "image_link": ["https://example/1.jpg", None],
            "price": [12.5, 0.0],
        })
        self.test = pd.DataFrame({
            "sample_id": ["010", "011"],
            "catalog_content": ["Test item A", "Test item B"],
            "image_link": ["https://example/10.jpg", ""],
        })
        self.sample = pd.DataFrame({"sample_id": ["010", "011"], "price": [None, None]})
        self.write_inputs()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_inputs(self):
        self.train.to_csv(self.train_path, index=False)
        self.test.to_csv(self.test_path, index=False)
        self.sample.to_csv(self.sample_path, index=False)

    def run_adapter(self, *, overwrite=False):
        return adapt_files(
            self.train_path,
            self.test_path,
            self.sample_path,
            self.output_dir,
            overwrite=overwrite,
        )

    def test_valid_files_convert_to_pipeline_schema_and_preserve_ids(self):
        train_path, test_path, sample_path = self.run_adapter()
        train = pd.read_csv(
            train_path, dtype={"PRODUCT_ID": "string"}, keep_default_na=False
        )
        test = pd.read_csv(
            test_path, dtype={"PRODUCT_ID": "string"}, keep_default_na=False
        )
        sample = pd.read_csv(
            sample_path, dtype={"PRODUCT_ID": "string"}, keep_default_na=False
        )

        self.assertEqual(train.columns.tolist(), [
            "PRODUCT_ID", "TITLE", "DESCRIPTION", "PACK_SIZE", "BRAND",
            "CATEGORY", "IMAGE_URL", "PRICE",
        ])
        self.assertEqual(test.columns.tolist(), train.columns[:-1].tolist())
        self.assertEqual(train["PRODUCT_ID"].tolist(), ["001", "002"])
        self.assertEqual(train["TITLE"].iloc[0], "Widget, 2 pack")
        self.assertEqual(train["TITLE"].iloc[1], "")
        self.assertTrue(train[["DESCRIPTION", "PACK_SIZE", "BRAND", "CATEGORY"]].eq("").all().all())
        self.assertEqual(train["PRICE"].tolist(), [12.5, 0.0])
        self.assertEqual(test["PRODUCT_ID"].tolist(), ["010", "011"])
        self.assertEqual(sample.columns.tolist(), ["PRODUCT_ID", "PRICE"])
        self.assertEqual(sample["PRODUCT_ID"].tolist(), test["PRODUCT_ID"].tolist())

    def test_rejects_test_sample_id_order_mismatch(self):
        self.sample = self.sample.iloc[::-1].reset_index(drop=True)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "same order"):
            self.run_adapter()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_duplicate_ids_in_each_input(self):
        original = {name: getattr(self, name).copy() for name in ("train", "test", "sample")}
        for frame_name in ("train", "test", "sample"):
            with self.subTest(frame=frame_name):
                for name, frame in original.items():
                    setattr(self, name, frame.copy())
                frame = getattr(self, frame_name)
                frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                    self.run_adapter()

    def test_rejects_blank_ids(self):
        self.train.loc[0, "sample_id"] = " "
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "blank sample_id"):
            self.run_adapter()

    def test_rejects_missing_required_columns(self):
        self.train = self.train.drop(columns="catalog_content")
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "catalog_content"):
            self.run_adapter()

    def test_rejects_negative_price(self):
        self.train.loc[0, "price"] = -0.01
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.run_adapter()

    def test_rejects_non_numeric_or_infinite_price(self):
        for bad_price in ("not-a-price", "inf"):
            with self.subTest(price=bad_price):
                self.train["price"] = self.train["price"].astype(object)
                self.train.loc[0, "price"] = bad_price
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, "finite, non-negative"):
                    self.run_adapter()

    def test_refuses_to_overwrite_without_explicit_flag(self):
        paths = self.run_adapter()
        before = [path.read_bytes() for path in paths]
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.run_adapter()
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_overwrite_requires_explicit_flag_and_replaces_outputs(self):
        paths = self.run_adapter()
        self.train.loc[0, "price"] = 99.0
        self.write_inputs()
        self.run_adapter(overwrite=True)
        updated = pd.read_csv(paths[0])
        self.assertEqual(updated["PRICE"].iloc[0], 99.0)


if __name__ == "__main__":
    unittest.main()
