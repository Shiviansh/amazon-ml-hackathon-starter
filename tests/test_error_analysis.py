"""Tests for OOF residual profiling and failure diagnostics."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.error_analysis import (
    ErrorAnalysisConfig,
    analyze_errors,
    analyze_from_files,
    main,
    save_error_analysis,
)
from src.metrics import smape


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        count = 120
        ids = np.arange(1000, 1000 + count)
        target = np.geomspace(12.0, 2400.0, count)
        prediction = target * (1.0 + 0.08 * np.sin(np.arange(count)))
        prediction[5] = target[5] * 5.0
        prediction[90] = target[90] * 0.05
        self.oof = pd.DataFrame(
            {
                "PRODUCT_ID": ids,
                "PRICE": target,
                "fold": np.arange(count) % 4,
                "prediction": prediction,
            }
        )
        brands = np.array(["common"] * 70 + ["medium"] * 30 + [f"rare_{i // 2}" for i in range(20)])
        categories = np.where(np.arange(count) % 3 == 0, "grocery", "electronics")
        titles = [f"Product {index}" for index in range(count)]
        titles[5] = "Imported bundle pack of 12 price ₹999"
        titles[90] = "Protein powder 2 x 500 g"
        descriptions = ["basic catalog description"] * count
        descriptions[0] = ""
        self.metadata = pd.DataFrame(
            {
                "PRODUCT_ID": ids,
                "CATEGORY": categories,
                "BRAND": brands,
                "TITLE": titles,
                "DESCRIPTION": descriptions,
            }
        ).sample(frac=1, random_state=11).reset_index(drop=True)
        self.config = ErrorAnalysisConfig(
            worst_n=12,
            min_group_size=5,
            brand_medium_max=50,
        )

    def analyze(self, **kwargs):
        return analyze_errors(
            self.oof,
            metadata=self.metadata,
            id_column="PRODUCT_ID",
            target_column="PRICE",
            category_columns=["CATEGORY"],
            brand_column="BRAND",
            text_columns=["TITLE", "DESCRIPTION"],
            config=self.config,
            **kwargs,
        )

    def test_row_metrics_and_overall_score_are_exact(self):
        result = self.analyze()
        expected = smape(self.oof["PRICE"], self.oof["prediction"])
        self.assertAlmostEqual(result.report["overall"]["smape"], expected)
        self.assertAlmostEqual(result.rows["signed_error"].iloc[0], self.oof["prediction"].iloc[0] - self.oof["PRICE"].iloc[0])
        self.assertTrue((result.rows["absolute_error"] >= 0).all())
        self.assertTrue(result.rows["row_smape"].between(0.0, 2.0).all())

    def test_metadata_is_aligned_by_id_not_row_order(self):
        result = self.analyze()
        expected = self.metadata.set_index("PRODUCT_ID").loc[result.rows["PRODUCT_ID"], "CATEGORY"].to_numpy()
        np.testing.assert_array_equal(result.rows["CATEGORY"], expected)

    def test_worst_errors_and_packaging_signals_are_actionable(self):
        result = self.analyze()
        worst_ids = result.worst_errors["PRODUCT_ID"].tolist()
        self.assertIn(1005, worst_ids[:3])
        self.assertIn(1090, worst_ids[:3])
        row = result.rows.set_index("PRODUCT_ID").loc[1005]
        self.assertTrue(row["has_multipack_signal"])
        self.assertTrue(row["has_currency_signal"])
        other = result.rows.set_index("PRODUCT_ID").loc[1090]
        self.assertTrue(other["has_unit_signal"])

    def test_all_requested_segment_tables_exist(self):
        result = self.analyze()
        expected = {
            "price_tier",
            "text_length",
            "fold",
            "brand",
            "brand_frequency",
            "category__CATEGORY",
            "failure_modes",
        }
        self.assertEqual(set(result.segment_tables), expected)
        for name in expected - {"failure_modes"}:
            table = result.segment_tables[name]
            self.assertEqual(int(table["count"].sum()), len(self.oof))
            self.assertIn("share_total_absolute_error", table.columns)

    def test_brand_frequency_tiers_and_price_tiers(self):
        result = self.analyze()
        self.assertTrue(
            {"frequent", "medium", "rare"}.issubset(
                set(result.rows["brand_frequency_tier"])
            )
        )
        self.assertEqual(set(result.rows["price_tier"]), {"cheap", "mid", "luxury"})
        self.assertEqual(result.report["price_tiers"]["source"], "target_quantiles")

    def test_fixed_price_tiers(self):
        config = ErrorAnalysisConfig(
            worst_n=5,
            min_group_size=1,
            price_tier_cuts=(100.0, 1000.0),
        )
        result = analyze_errors(
            self.oof,
            id_column="PRODUCT_ID",
            target_column="PRICE",
            config=config,
        )
        self.assertEqual(result.report["price_tiers"]["source"], "fixed")
        self.assertEqual(set(result.rows["price_tier"]), {"cheap", "mid", "luxury"})

    def test_rank_by_absolute_error_changes_primary_sort(self):
        config = ErrorAnalysisConfig(worst_n=10, rank_by="absolute_error", min_group_size=1)
        result = analyze_errors(
            self.oof,
            id_column="PRODUCT_ID",
            target_column="PRICE",
            config=config,
        )
        values = result.worst_errors["absolute_error"].to_numpy()
        self.assertTrue(np.all(values[:-1] >= values[1:]))

    def test_invalid_inputs_fail_closed(self):
        duplicate = self.metadata.copy()
        duplicate.loc[1, "PRODUCT_ID"] = duplicate.loc[0, "PRODUCT_ID"]
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            analyze_errors(
                self.oof,
                metadata=duplicate,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                category_columns=["CATEGORY"],
            )
        incomplete = self.metadata.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "does not cover"):
            analyze_errors(
                self.oof,
                metadata=incomplete,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                category_columns=["CATEGORY"],
            )
        broken = self.oof.copy()
        broken.loc[0, "prediction"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            analyze_errors(
                broken,
                id_column="PRODUCT_ID",
                target_column="PRICE",
            )
        with self.assertRaisesRegex(ValueError, "distinct names"):
            analyze_errors(
                self.oof,
                id_column="PRODUCT_ID",
                target_column="prediction",
            )
        with self.assertRaisesRegex(ValueError, "multiple analysis roles"):
            analyze_errors(
                self.oof,
                metadata=self.metadata,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                category_columns=["BRAND"],
                brand_column="BRAND",
            )

    def test_temporal_and_mixed_unevaluated_rows_are_removed(self):
        extra = pd.DataFrame(
            {
                "PRODUCT_ID": [9000, 9001, 9002],
                "PRICE": [20.0, 30.0, 40.0],
                "fold": ["-1", "test", None],
                "prediction": [np.nan, np.nan, np.nan],
            }
        )
        oof = self.oof.copy()
        oof["fold"] = oof["fold"].astype(str)
        combined = pd.concat([extra, oof], ignore_index=True)
        result = analyze_errors(
            combined,
            id_column="PRODUCT_ID",
            target_column="PRICE",
            config=ErrorAnalysisConfig(min_group_size=1),
        )
        self.assertEqual(len(result.rows), len(self.oof))
        self.assertEqual(result.report["dropped_unevaluated_oof_rows"], 3)

    def test_degenerate_price_tiers_are_reported(self):
        frame = pd.DataFrame(
            {"id": np.arange(10), "target": 10.0, "prediction": 11.0}
        )
        result = analyze_errors(
            frame,
            id_column="id",
            target_column="target",
            config=ErrorAnalysisConfig(min_group_size=1),
        )
        self.assertTrue(result.report["price_tiers"]["degenerate"])
        self.assertEqual(set(result.rows["price_tier"]), {"mid"})

    def test_html_escapes_catalog_text(self):
        metadata = self.metadata.copy()
        product_id = self.oof.loc[5, "PRODUCT_ID"]
        metadata.loc[metadata["PRODUCT_ID"] == product_id, "TITLE"] = "<script>alert(1)</script> pack of 5"
        result = analyze_errors(
            self.oof,
            metadata=metadata,
            id_column="PRODUCT_ID",
            target_column="PRICE",
            text_columns=["TITLE"],
            config=self.config,
        )
        self.assertNotIn("<script>alert(1)</script>", result.html_report)
        self.assertIn("&lt;script&gt;", result.html_report)


class FileAndCliTests(unittest.TestCase):
    @staticmethod
    def make_files(root: Path) -> tuple[Path, Path]:
        count = 50
        ids = np.arange(count)
        target = np.linspace(12.0, 500.0, count)
        oof = pd.DataFrame(
            {
                "PRODUCT_ID": ids,
                "PRICE": target,
                "fold": ids % 5,
                "prediction": target * np.where(ids % 2, 0.8, 1.15),
            }
        )
        metadata = pd.DataFrame(
            {
                "PRODUCT_ID": ids[::-1],
                "CATEGORY": np.where(ids[::-1] % 2, "food", "home"),
                "BRAND": np.where(ids[::-1] % 3, "brand_a", "brand_b"),
                "TITLE": [f"Pack of {i % 5 + 1} item {i}" for i in ids[::-1]],
            }
        )
        oof_path = root / "oof.parquet"
        metadata_path = root / "metadata.csv"
        oof.to_parquet(oof_path, index=False)
        metadata.to_csv(metadata_path, index=False)
        return oof_path, metadata_path

    def test_file_pipeline_hashes_and_saves_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, metadata = self.make_files(root)
            result = analyze_from_files(
                oof,
                metadata_path=metadata,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                category_columns=["CATEGORY"],
                brand_column="BRAND",
                text_columns=["TITLE"],
                config=ErrorAnalysisConfig(worst_n=10, min_group_size=2),
            )
            self.assertEqual(len(result.report["sources"]["oof"]["sha256"]), 64)
            self.assertEqual(len(result.report["run_signature"]), 64)
            output = root / "analysis"
            paths = save_error_analysis(result, output)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertIn("segment__price_tier", paths)
            self.assertIn("Worst 10 prediction misses", paths["html"].read_text(encoding="utf-8"))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_error_analysis(result, output)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, metadata = self.make_files(root)
            output = root / "cli"
            arguments = [
                "--oof", str(oof),
                "--metadata", str(metadata),
                "--output-dir", str(output),
                "--id-column", "PRODUCT_ID",
                "--target-column", "PRICE",
                "--category-columns", "CATEGORY",
                "--brand-column", "BRAND",
                "--text-columns", "TITLE",
                "--worst-n", "8",
                "--min-group-size", "2",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertTrue((output / "error_analysis_report.html").is_file())
            worst = pd.read_csv(output / "worst_errors.csv")
            self.assertEqual(len(worst), 8)


if __name__ == "__main__":
    unittest.main()
