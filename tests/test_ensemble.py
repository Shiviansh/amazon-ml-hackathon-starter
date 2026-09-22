"""Tests for robust OOF ensemble optimization and artifact handling."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.ensemble import (
    EnsembleConfig,
    _smape_gradient,
    apply_weights,
    ensemble_from_files,
    fit_ensemble,
    main,
    optimize_weights,
    save_ensemble_result,
)
from src.metrics import smape


class WeightOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.target = np.linspace(10.0, 100.0, 80)
        self.predictions = np.column_stack(
            [self.target * 0.7, self.target * 1.1, self.target + 8.0]
        )

    def test_slsqp_recovers_exact_simplex_blend(self):
        solution = optimize_weights(
            self.target,
            self.predictions[:, :2],
            model_names=["low", "high"],
            config=EnsembleConfig(
                method="slsqp", n_starts=8, prediction_floor=None
            ),
        )
        self.assertAlmostEqual(solution.weights.sum(), 1.0, places=10)
        self.assertTrue(np.all(solution.weights >= 0.0))
        self.assertLess(solution.score, 1e-8)
        np.testing.assert_allclose(solution.weights, [0.25, 0.75], atol=1e-5)
        self.assertTrue(
            any(run["gradient_evaluations"] > 0 for run in solution.optimizer_runs)
        )

    def test_analytic_smape_gradient_matches_central_difference(self):
        target = np.array([3.0, 7.0, -4.0, 0.0, 12.0])
        predictions = np.array(
            [
                [2.0, 5.0, 4.0],
                [9.0, 5.0, 8.0],
                [-2.0, -7.0, 1.0],
                [1.0, -2.0, 3.0],
                [15.0, 8.0, 10.0],
            ]
        )
        weights = np.array([0.2, 0.35, 0.45])
        config = EnsembleConfig(prediction_floor=None, cross_fit=False)
        analytic = _smape_gradient(weights, target, predictions, config)
        numerical = np.empty_like(weights)
        step = 1e-6
        for index in range(len(weights)):
            delta = np.zeros_like(weights)
            delta[index] = step
            numerical[index] = (
                smape(target, predictions @ (weights + delta))
                - smape(target, predictions @ (weights - delta))
            ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-6)

    def test_clipped_rows_have_zero_smape_gradient(self):
        config = EnsembleConfig(prediction_floor=0.0, cross_fit=False)
        gradient = _smape_gradient(
            np.array([0.5, 0.5]),
            np.array([5.0, 10.0]),
            np.array([[-2.0, -3.0], [-4.0, -1.0]]),
            config,
        )
        np.testing.assert_array_equal(gradient, [0.0, 0.0])

    def test_auto_is_no_worse_than_best_candidate(self):
        config = EnsembleConfig(method="auto", n_starts=6, prediction_floor=None)
        solution = optimize_weights(self.target, self.predictions, config=config)
        individual = [
            optimize_weights(
                self.target,
                self.predictions[:, [index]],
                config=EnsembleConfig(method="uniform", prediction_floor=None),
            ).score
            for index in range(self.predictions.shape[1])
        ]
        self.assertLessEqual(solution.score, min(individual) + 1e-10)
        self.assertAlmostEqual(solution.weights.sum(), 1.0, places=10)

    def test_nnls_returns_normalized_nonnegative_weights(self):
        solution = optimize_weights(
            self.target,
            self.predictions[:, :2],
            config=EnsembleConfig(method="nnls", prediction_floor=None),
        )
        self.assertEqual(solution.method, "nnls")
        self.assertTrue(np.all(solution.weights >= 0.0))
        self.assertAlmostEqual(float(solution.weights.sum()), 1.0, places=12)
        self.assertLess(solution.score, 1e-7)
        np.testing.assert_allclose(solution.weights, [0.25, 0.75], atol=1e-5)

    def test_nnls_is_relative_scale_aware_for_broad_prices(self):
        target = np.concatenate([np.full(100, 10.0), [10_000.0]])
        low_price_model = np.concatenate([np.full(100, 10.0), [5_000.0]])
        high_price_model = np.concatenate([np.full(100, 20.0), [10_000.0]])
        solution = optimize_weights(
            target,
            np.column_stack([low_price_model, high_price_model]),
            model_names=["low_price_specialist", "high_price_specialist"],
            config=EnsembleConfig(
                method="nnls", prediction_floor=None, cross_fit=False
            ),
        )
        self.assertGreater(solution.weights[0], 0.99)
        self.assertIn("Relative-scaled", solution.message)

    def test_uniform_and_single_model_paths(self):
        uniform = optimize_weights(
            self.target,
            self.predictions,
            config=EnsembleConfig(method="uniform", prediction_floor=None),
        )
        np.testing.assert_allclose(uniform.weights, np.full(3, 1 / 3))
        single = optimize_weights(
            self.target,
            self.predictions[:, [0]],
            config=EnsembleConfig(method="slsqp", prediction_floor=None),
        )
        np.testing.assert_array_equal(single.weights, [1.0])
        self.assertEqual(single.method, "single_model")

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            optimize_weights([1.0, np.nan], [[1.0], [2.0]])
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            optimize_weights([1.0, 2.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "unique"):
            optimize_weights(
                self.target,
                self.predictions[:, :2],
                model_names=["same", "same"],
            )
        with self.assertRaises(ValueError):
            EnsembleConfig(prediction_floor=2.0, prediction_ceiling=1.0)
        with self.assertRaises(ValueError):
            EnsembleConfig(n_starts=0)
        with self.assertRaises(ValueError):
            EnsembleConfig(n_starts=2.5)
        with self.assertRaises(ValueError):
            EnsembleConfig(max_iterations=10.5)

    def test_apply_weights_enforces_simplex_and_clipping(self):
        blended = apply_weights(
            [[-5.0, 5.0], [10.0, 20.0]], [0.5, 0.5], prediction_floor=0.0
        )
        np.testing.assert_allclose(blended, [0.0, 15.0])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            apply_weights([[1.0, 2.0]], [0.2, 0.2])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_weights([[1.0, 2.0]], [-0.1, 1.1])


class EnsembleFitTests(unittest.TestCase):
    def test_cross_fitted_diagnostics_and_test_blend(self):
        rng = np.random.default_rng(4)
        target = rng.uniform(5.0, 80.0, 90)
        predictions = np.column_stack(
            [target + rng.normal(0, 4, 90), target * 1.08, target * 0.9]
        )
        test = predictions[:7] + 1.0
        result = fit_ensemble(
            target,
            predictions,
            model_names=["a", "b", "c"],
            test_predictions=test,
            ids=[f"p{index}" for index in range(90)],
            test_ids=[f"t{index}" for index in range(7)],
            folds=np.arange(90) % 3,
            target_column="PRICE",
            id_column="PRODUCT_ID",
            config=EnsembleConfig(method="auto", n_starts=5),
        )
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=10)
        self.assertEqual(len(result.test_predictions), 7)
        self.assertTrue(np.isfinite(result.oof["cross_fitted_prediction"]).all())
        self.assertIsNotNone(result.report["cross_fitted_oof_smape"])
        self.assertEqual(len(result.report["cross_fitted_folds"]), 3)
        self.assertIn("matrix_diagnostics", result.report)

    def test_duplicate_prediction_diagnostic(self):
        target = np.arange(1.0, 11.0)
        predictions = np.column_stack([target, target])
        result = fit_ensemble(
            target,
            predictions,
            model_names=["copy_a", "copy_b"],
            config=EnsembleConfig(method="uniform", cross_fit=False),
        )
        self.assertEqual(
            result.report["matrix_diagnostics"]["duplicate_model_pairs"],
            [["copy_a", "copy_b"]],
        )

    def test_cross_fit_requires_multiple_folds(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_ensemble(
                [1.0, 2.0],
                [[1.0], [2.0]],
                folds=[0, 0],
                config=EnsembleConfig(cross_fit=True),
            )


class FileInterfaceTests(unittest.TestCase):
    @staticmethod
    def write_runs(root: Path):
        ids = np.arange(100, 112)
        target = np.linspace(10.0, 30.0, len(ids))
        fold = np.arange(len(ids)) % 3
        paths = {}
        test_paths = {}
        for index, name in enumerate(("ridge", "lgbm")):
            oof = pd.DataFrame(
                {
                    "PRODUCT_ID": ids,
                    "PRICE": target,
                    "fold": fold,
                    "prediction": target * (0.9 + 0.2 * index),
                }
            )
            test = pd.DataFrame(
                {
                    "PRODUCT_ID": [9, 8, 7],
                    "prediction": [20.0, 30.0, 40.0] * np.asarray(0.9 + 0.2 * index),
                }
            )
            if index:
                oof = oof.sample(frac=1, random_state=8).reset_index(drop=True)
                test = test.sample(frac=1, random_state=9).reset_index(drop=True)
            oof_path = root / f"{name}_oof.parquet"
            test_path = root / f"{name}_test.parquet"
            oof.to_parquet(oof_path, index=False)
            test.to_parquet(test_path, index=False)
            paths[name] = oof_path
            test_paths[name] = test_path
        return paths, test_paths

    def test_files_align_by_id_and_save_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, test = self.write_runs(root)
            result = ensemble_from_files(
                oof,
                test_files=test,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                config=EnsembleConfig(method="slsqp", n_starts=5),
            )
            self.assertLess(result.report["oof_smape"], 1e-8)
            self.assertEqual(result.test_predictions["PRODUCT_ID"].tolist(), [9, 8, 7])
            self.assertEqual(len(result.report["sources"]["oof"]["ridge"]["sha256"]), 64)
            output = root / "ensemble"
            written = save_ensemble_result(
                result, output, target_column="PRICE"
            )
            self.assertEqual(
                set(written),
                {"weights", "oof", "report", "test_predictions", "submission"},
            )
            report = json.loads(written["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            submission = pd.read_csv(written["submission"])
            self.assertEqual(list(submission), ["PRODUCT_ID", "PRICE"])
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_ensemble_result(result, output, target_column="PRICE")

    def test_target_fold_and_id_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, _ = self.write_runs(root)
            broken = pd.read_parquet(oof["lgbm"])
            broken.loc[0, "PRICE"] += 10.0
            broken.to_parquet(oof["lgbm"], index=False)
            with self.assertRaisesRegex(ValueError, "target mismatch"):
                ensemble_from_files(
                    oof, id_column="PRODUCT_ID", target_column="PRICE"
                )

            oof, _ = self.write_runs(root)
            broken = pd.read_parquet(oof["lgbm"])
            broken.loc[0, "fold"] = 99
            broken.to_parquet(oof["lgbm"], index=False)
            with self.assertRaisesRegex(ValueError, "fold mismatch"):
                ensemble_from_files(
                    oof, id_column="PRODUCT_ID", target_column="PRICE"
                )

            oof, _ = self.write_runs(root)
            broken = pd.read_parquet(oof["lgbm"])
            broken.loc[0, "PRODUCT_ID"] = 999
            broken.to_parquet(oof["lgbm"], index=False)
            with self.assertRaisesRegex(ValueError, "ID set mismatch"):
                ensemble_from_files(
                    oof, id_column="PRODUCT_ID", target_column="PRICE"
                )

    def test_test_model_set_must_match_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, test = self.write_runs(root)
            del test["lgbm"]
            with self.assertRaisesRegex(ValueError, "exactly match"):
                ensemble_from_files(
                    oof,
                    test_files=test,
                    id_column="PRODUCT_ID",
                    target_column="PRICE",
                )

    def test_cli_writes_submission_and_custom_fold_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, test = self.write_runs(root)
            for path in oof.values():
                frame = pd.read_parquet(path).rename(columns={"fold": "cv_fold"})
                frame.to_parquet(path, index=False)
            output = root / "cli-output"
            arguments = [
                "--oof", f"ridge={oof['ridge']}",
                "--oof", f"lgbm={oof['lgbm']}",
                "--test", f"ridge={test['ridge']}",
                "--test", f"lgbm={test['lgbm']}",
                "--output-dir", str(output),
                "--id-column", "PRODUCT_ID",
                "--target-column", "PRICE",
                "--fold-column", "cv_fold",
                "--method", "nnls",
                "--allow-negative-predictions",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            saved_oof = pd.read_parquet(output / "ensemble_oof.parquet")
            self.assertIn("cv_fold", saved_oof.columns)
            self.assertTrue((output / "submission.csv").is_file())

    def test_negative_temporal_fold_rows_are_ignored_consistently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, _ = self.write_runs(root)
            for path in oof.values():
                frame = pd.read_parquet(path)
                extra = pd.DataFrame(
                    {
                        "PRODUCT_ID": [999],
                        "PRICE": [12.0],
                        "fold": [-1],
                        "prediction": [np.nan],
                    }
                )
                pd.concat([extra, frame], ignore_index=True).to_parquet(path, index=False)
            result = ensemble_from_files(
                oof,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                config=EnsembleConfig(method="uniform"),
            )
            self.assertEqual(len(result.oof), 12)
            self.assertNotIn(999, result.oof["PRODUCT_ID"].tolist())

    def test_mixed_test_and_missing_fold_markers_are_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof, _ = self.write_runs(root)
            for path in oof.values():
                frame = pd.read_parquet(path)
                excluded = pd.DataFrame(
                    {
                        "PRODUCT_ID": [900, 901],
                        "PRICE": [12.0, 14.0],
                        "fold": ["test", None],
                        "prediction": [np.nan, np.nan],
                    }
                )
                frame["fold"] = frame["fold"].astype(str)
                pd.concat([excluded, frame], ignore_index=True).to_parquet(path, index=False)
            result = ensemble_from_files(
                oof,
                id_column="PRODUCT_ID",
                target_column="PRICE",
                config=EnsembleConfig(method="uniform"),
            )
            self.assertEqual(len(result.oof), 12)
            self.assertEqual(result.report["dropped_unevaluated_oof_rows"], 2)
            self.assertNotIn(900, result.oof["PRODUCT_ID"].tolist())


if __name__ == "__main__":
    unittest.main()
