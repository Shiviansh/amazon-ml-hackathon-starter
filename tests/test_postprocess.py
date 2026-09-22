"""Tests for leakage-safe price calibration and domain post-processing."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.postprocess import (
    PostprocessConfig,
    apply_postprocessing,
    blend_predictions,
    fit_postprocessor,
    main,
    optimize_smape_multiplier,
    postprocess_from_files,
    save_postprocess_result,
)


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_smape_multiplier(self):
        prediction = np.linspace(10.0, 500.0, 300)
        target = prediction * 1.25
        solution = optimize_smape_multiplier(
            target,
            prediction,
            config=PostprocessConfig(price_floor=None),
        )
        self.assertAlmostEqual(solution.alpha, 1.25, places=6)
        self.assertLess(solution.score_after, 1e-8)
        self.assertLessEqual(solution.score_after, solution.score_before)

    def test_calibration_objective_includes_domain_clamp(self):
        target = np.array([12.0, 20.0, 40.0, 100.0])
        prediction = np.array([-10.0, 10.0, 20.0, 50.0])
        config = PostprocessConfig(price_floor=12.0, price_ceiling=90.0)
        solution = optimize_smape_multiplier(target, prediction, config=config)
        output = apply_postprocessing(prediction, alpha=solution.alpha, config=config)
        self.assertTrue(np.all(output >= 12.0))
        self.assertTrue(np.all(output <= 90.0))
        self.assertLessEqual(solution.score_after, solution.score_before + 1e-15)

    def test_disabled_and_all_zero_calibration_use_identity(self):
        target = [10.0, 20.0, 30.0]
        disabled = optimize_smape_multiplier(
            target,
            [9.0, 18.0, 27.0],
            config=PostprocessConfig(calibrate=False),
        )
        zeros = optimize_smape_multiplier(target, [0.0, 0.0, 0.0])
        self.assertEqual(disabled.alpha, 1.0)
        self.assertEqual(zeros.alpha, 1.0)

    def test_invalid_configuration_and_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "include the identity"):
            PostprocessConfig(alpha_min=1.1, alpha_max=2.0)
        with self.assertRaisesRegex(ValueError, "price_floor"):
            PostprocessConfig(blend_method="geometric", price_floor=0.0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            PostprocessConfig(price_floor=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            optimize_smape_multiplier([-1.0, 2.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            optimize_smape_multiplier([1.0, 2.0], [1.0, np.nan])


class BlendingTests(unittest.TestCase):
    def test_arithmetic_and_geometric_blends(self):
        matrix = np.array([[4.0, 16.0], [9.0, 25.0]])
        arithmetic = blend_predictions(matrix, weights=[0.5, 0.5])
        geometric = blend_predictions(
            matrix, weights=[0.5, 0.5], method="geometric"
        )
        np.testing.assert_allclose(arithmetic, [10.0, 17.0])
        np.testing.assert_allclose(geometric, [8.0, 15.0])

    def test_geometric_rejects_nonpositive_without_explicit_floor(self):
        matrix = np.array([[0.0, 10.0], [-2.0, 8.0]])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            blend_predictions(matrix, method="geometric")
        result = blend_predictions(
            matrix,
            method="geometric",
            positive_floor=2.0,
        )
        np.testing.assert_allclose(result, [np.sqrt(20.0), 4.0])

    def test_weight_contract_is_strict(self):
        matrix = np.ones((3, 2))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            blend_predictions(matrix, weights=[0.2, 0.2])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            blend_predictions(matrix, weights=[-0.1, 1.1])
        with self.assertRaisesRegex(ValueError, "expected 2"):
            blend_predictions(matrix, weights=[1.0])


class PostprocessorFitTests(unittest.TestCase):
    def test_cross_fitted_calibration_and_test_application(self):
        rng = np.random.default_rng(44)
        raw = rng.uniform(10.0, 200.0, 120)
        target = raw * 1.12
        test = np.array([20.0, 40.0, 80.0])
        result = fit_postprocessor(
            target,
            raw,
            test_predictions=test,
            ids=np.arange(120),
            test_ids=["a", "b", "c"],
            folds=np.arange(120) % 4,
            id_column="PRODUCT_ID",
            target_column="PRICE",
            config=PostprocessConfig(price_floor=12.0),
        )
        self.assertAlmostEqual(result.report["alpha"], 1.12, places=5)
        self.assertLess(result.report["calibrated_oof_smape"], 1e-7)
        self.assertLess(result.report["cross_fitted_oof_smape"], 1e-7)
        self.assertEqual(len(result.report["cross_fitted_folds"]), 4)
        np.testing.assert_allclose(
            result.test_predictions["prediction"], test * 1.12, rtol=1e-6
        )

    def test_geometric_wide_prediction_pipeline(self):
        target = np.array([8.0, 15.0, 24.0])
        matrix = np.array([[4.0, 16.0], [9.0, 25.0], [16.0, 36.0]])
        result = fit_postprocessor(
            target,
            matrix,
            weights=[0.5, 0.5],
            config=PostprocessConfig(
                blend_method="geometric",
                calibrate=False,
                price_floor=1.0,
                cross_fit=False,
            ),
        )
        np.testing.assert_allclose(result.oof["prediction"], target)

    def test_duplicate_ids_and_single_fold_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_postprocessor(
                [10.0, 20.0], [9.0, 18.0], ids=[1, 1]
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_postprocessor(
                [10.0, 20.0], [9.0, 18.0], folds=[0, 0]
            )
        with self.assertRaisesRegex(ValueError, "column names must all differ"):
            fit_postprocessor(
                [10.0, 20.0],
                [9.0, 18.0],
                id_column="prediction",
            )


class FileAndCliTests(unittest.TestCase):
    @staticmethod
    def make_files(root: Path) -> tuple[Path, Path]:
        ids = np.arange(20)
        raw = np.linspace(10.0, 100.0, 20)
        oof = pd.DataFrame(
            {
                "PRODUCT_ID": ids,
                "PRICE": raw * 1.1,
                "fold": (ids % 4).astype(str),
                "model_a": raw,
                "model_b": raw * 1.02,
            }
        )
        excluded = pd.DataFrame(
            {
                "PRODUCT_ID": [999, 1000],
                "PRICE": [20.0, 30.0],
                "fold": ["test", None],
                "model_a": [np.nan, np.nan],
                "model_b": [np.nan, np.nan],
            }
        )
        oof = pd.concat([excluded, oof], ignore_index=True)
        test = pd.DataFrame(
            {
                "PRODUCT_ID": [102, 101],
                "model_a": [40.0, 20.0],
                "model_b": [44.0, 22.0],
            }
        )
        oof_path = root / "oof.parquet"
        test_path = root / "test.parquet"
        oof.to_parquet(oof_path, index=False)
        test.to_parquet(test_path, index=False)
        return oof_path, test_path

    def test_file_pipeline_filters_rows_hashes_sources_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof_path, test_path = self.make_files(root)
            result = postprocess_from_files(
                oof_path,
                test_path=test_path,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                prediction_columns=["model_a", "model_b"],
                weights=[0.75, 0.25],
                config=PostprocessConfig(price_floor=12.0),
            )
            self.assertEqual(len(result.oof), 20)
            self.assertEqual(result.report["dropped_unevaluated_oof_rows"], 2)
            self.assertEqual(len(result.report["sources"]["oof"]["sha256"]), 64)
            self.assertEqual(result.test_predictions["PRODUCT_ID"].tolist(), [102, 101])
            output = root / "postprocessed"
            paths = save_postprocess_result(
                result, output, target_column="PRICE"
            )
            self.assertEqual(
                set(paths),
                {"calibration", "oof", "report", "test_predictions", "submission"},
            )
            submission = pd.read_csv(paths["submission"])
            self.assertEqual(list(submission), ["PRODUCT_ID", "PRICE"])
            calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
            self.assertGreater(calibration["alpha"], 0.0)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_postprocess_result(result, output, target_column="PRICE")

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof_path, test_path = self.make_files(root)
            output = root / "cli"
            arguments = [
                "--oof", str(oof_path),
                "--test", str(test_path),
                "--output-dir", str(output),
                "--id-column", "PRODUCT_ID",
                "--target-column", "PRICE",
                "--prediction-columns", "model_a", "model_b",
                "--weights", "0.75", "0.25",
                "--price-floor", "12",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertTrue((output / "submission.csv").is_file())
            self.assertTrue((output / "postprocess_report.json").is_file())

    def test_missing_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oof.csv"
            pd.DataFrame({"id": [1], "target": [2.0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                postprocess_from_files(
                    path,
                    id_column="id",
                    target_column="target",
                )

    def test_target_column_cannot_be_used_as_a_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oof.csv"
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "target": [10.0, 20.0],
                    "fold": [0, 1],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "cannot include"):
                postprocess_from_files(
                    path,
                    id_column="id",
                    target_column="target",
                    prediction_columns=["target"],
                )


if __name__ == "__main__":
    unittest.main()
