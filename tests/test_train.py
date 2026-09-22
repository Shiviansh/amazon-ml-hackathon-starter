"""Tests for the fold-aware training runner."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.train import (
    TrainingConfig,
    _make_lgbm_eval_metric,
    attach_auxiliary_feature_pair,
    attach_folds,
    build_feature_frame,
    build_preprocessor,
    main as train_main,
    resolve_training_device,
    save_training_result,
    train_cross_validated,
    validate_config,
)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        count = 60
        self.train = pd.DataFrame({
            "id": np.arange(count),
            "title": [f"product category {i % 6} feature {i % 4}" for i in range(count)],
            "brand": np.tile(["a", "b", None], 20),
            "weight": np.linspace(1.0, 4.0, count),
            "price": 10.0 + np.arange(count) % 10 + np.tile([0.0, 4.0], 30),
        })
        self.test = pd.DataFrame({
            "id": np.arange(100, 112),
            "title": [f"product category {i % 6} feature {i % 4}" for i in range(12)],
            "brand": np.tile(["a", "b", None], 4),
            "weight": np.linspace(1.2, 3.8, 12),
        })
        self.folds = pd.DataFrame({
            "id": np.arange(count),
            "fold": np.arange(count) % 3,
        })
        self.config = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            categorical_columns=("brand",),
            numeric_columns=("weight",),
            n_splits=3,
            min_df=1,
            word_max_features=500,
            char_max_features=500,
            linear_strength=1.0,
        )

    def test_feature_frame_does_not_mutate_input(self):
        original = self.train.copy(deep=True)
        features = build_feature_frame(self.train, self.config)
        pd.testing.assert_frame_equal(self.train, original)
        self.assertIn("__combined_text", features.columns)
        self.assertFalse(features["brand"].isna().any())

    def test_external_folds_join_by_id_not_row_order(self):
        shuffled = self.folds.sample(frac=1, random_state=4)
        folded = attach_folds(self.train, shuffled, self.config)
        expected = self.folds.set_index("id")["fold"]
        actual = folded.set_index("id")["fold"]
        pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())

    def test_auxiliary_features_join_by_id_and_prefix_columns(self):
        train_aux = pd.DataFrame({
            "id": self.train["id"].iloc[::-1].to_numpy(),
            "feature": np.arange(len(self.train), dtype=float),
        })
        test_aux = pd.DataFrame({
            "id": self.test["id"].iloc[::-1].to_numpy(),
            "feature": np.arange(len(self.test), dtype=float),
        })
        joined_train, joined_test, columns = attach_auxiliary_feature_pair(
            self.train,
            self.test,
            train_aux,
            test_aux,
            self.config,
            prefix="eng",
            label="engineered",
        )
        self.assertEqual(columns, ("eng__feature",))
        expected_train = train_aux.set_index("id").loc[self.train["id"], "feature"].to_numpy()
        expected_test = test_aux.set_index("id").loc[self.test["id"], "feature"].to_numpy()
        np.testing.assert_array_equal(joined_train["eng__feature"], expected_train)
        np.testing.assert_array_equal(joined_test["eng__feature"], expected_test)

    def test_auxiliary_id_mismatch_is_rejected(self):
        train_aux = pd.DataFrame({"id": self.train["id"], "feature": 1.0})
        train_aux.loc[0, "id"] = 999_999
        test_aux = pd.DataFrame({"id": self.test["id"], "feature": 1.0})
        with self.assertRaisesRegex(ValueError, "ID set mismatch"):
            attach_auxiliary_feature_pair(
                self.train, self.test, train_aux, test_aux, self.config,
                prefix="eng", label="engineered",
            )

    def test_target_encoded_auxiliary_requires_matching_folds(self):
        train_aux = pd.DataFrame({
            "id": self.train["id"],
            "fold": self.folds["fold"],
            "te_brand": np.linspace(1.0, 2.0, len(self.train)),
        })
        test_aux = pd.DataFrame({
            "id": self.test["id"],
            "te_brand": np.linspace(1.0, 2.0, len(self.test)),
        })
        with self.assertRaisesRegex(ValueError, "no verifiable external folds"):
            attach_auxiliary_feature_pair(
                self.train, self.test, train_aux, test_aux, self.config,
                prefix="eng", label="engineered",
            )

        _, _, columns = attach_auxiliary_feature_pair(
            self.train, self.test, train_aux, test_aux, self.config,
            prefix="eng", label="engineered", folds=self.folds,
        )
        self.assertEqual(columns, ("eng__te_brand",))

    def test_joined_auxiliary_columns_train_end_to_end(self):
        train_aux = pd.DataFrame({
            "id": self.train["id"],
            "signal": np.sin(np.arange(len(self.train))),
        })
        test_aux = pd.DataFrame({
            "id": self.test["id"],
            "signal": np.sin(np.arange(len(self.test))),
        })
        train, test, added = attach_auxiliary_feature_pair(
            self.train, self.test, train_aux, test_aux, self.config,
            prefix="eng", label="engineered",
        )
        numeric_only = replace(
            self.config,
            text_columns=(),
            categorical_columns=(),
            numeric_columns=("weight", *added),
        )
        result = train_cross_validated(train, test, numeric_only, folds=self.folds)
        self.assertEqual(result.report["feature_count_min"], 2)
        self.assertTrue(np.isfinite(result.test_predictions["prediction"]).all())

    def test_incomplete_fold_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            attach_folds(self.train, self.folds.iloc[:-1], self.config)

    def test_temporal_training_requires_precomputed_folds(self):
        config = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            temporal=True,
            n_splits=3,
        )
        with self.assertRaisesRegex(ValueError, "precomputed temporal"):
            attach_folds(self.train, None, config)

    def test_feature_type_overlap_is_rejected(self):
        invalid = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            categorical_columns=("title",),
        )
        with self.assertRaisesRegex(ValueError, "both text and categorical"):
            validate_config(invalid)

    def test_metric_task_mismatch_is_rejected(self):
        invalid = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            task="classification",
            metric="smape",
            target_transform="none",
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_config(invalid)

    def test_backend_model_parameters_are_validated(self):
        valid = replace(
            self.config,
            model="xgboost",
            model_params={"max_depth": 7, "subsample": 0.8},
        )
        validate_config(valid)
        invalid = replace(valid, model_params={"n_estimators": 50})
        with self.assertRaisesRegex(ValueError, "controlled settings"):
            validate_config(invalid)

    def test_tuning_artifact_is_applied_by_training_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.parquet"
            test_path = root / "test.parquet"
            folds_path = root / "folds.csv"
            artifact_path = root / "best_params_xgboost.json"
            output = root / "output"
            self.train.to_parquet(train_path, index=False)
            self.test.to_parquet(test_path, index=False)
            self.folds.to_csv(folds_path, index=False)
            artifact_path.write_text(json.dumps({
                "engine": "xgboost",
                "learning_rate": 0.07,
                "max_iterations": 7,
                "early_stopping_rounds": 2,
                "model_params": {
                    "max_depth": 4,
                    "subsample": 0.8,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.01,
                    "reg_lambda": 1.5,
                    "min_child_weight": 1.0,
                },
            }), encoding="utf-8")
            self.assertEqual(train_main([
                "--train", str(train_path),
                "--test", str(test_path),
                "--folds", str(folds_path),
                "--output-dir", str(output),
                "--target-column", "price",
                "--id-column", "id",
                "--numeric-columns", "weight",
                "--model", "xgboost",
                "--device", "cpu",
                "--n-splits", "3",
                "--model-params-json", str(artifact_path),
            ]), 0)
            report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
            config = report["config"]
            self.assertEqual(config["lgbm_learning_rate"], 0.07)
            self.assertEqual(config["lgbm_estimators"], 7)
            self.assertEqual(config["model_params"]["max_depth"], 4)

    def test_regression_training_is_complete_and_deterministic(self):
        first = train_cross_validated(
            self.train,
            self.test,
            self.config,
            folds=self.folds,
        )
        second = train_cross_validated(
            self.train,
            self.test,
            self.config,
            folds=self.folds,
        )
        self.assertFalse(first.oof["prediction"].isna().any())
        self.assertTrue(np.isfinite(first.test_predictions["prediction"]).all())
        np.testing.assert_allclose(
            first.test_predictions["prediction"],
            second.test_predictions["prediction"],
        )
        self.assertEqual(len(first.report["folds"]), 3)
        self.assertGreater(first.report["feature_count_min"], 0)

    def test_classification_training_saves_aligned_probabilities(self):
        train = self.train.copy()
        train["price"] = np.tile(["low", "high"], 30)
        config = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            categorical_columns=("brand",),
            task="classification",
            metric="accuracy",
            target_transform="none",
            n_splits=3,
            min_df=1,
            word_max_features=200,
            char_max_features=200,
            linear_strength=1.0,
        )
        result = train_cross_validated(train, self.test, config, folds=self.folds)
        self.assertTrue(set(result.test_predictions["prediction"]) <= {"low", "high"})
        self.assertGreaterEqual(result.report["overall_score"], 0.0)
        probability_columns = [
            item["column"] for item in result.report["class_probability_columns"]
        ]
        self.assertEqual(len(probability_columns), 2)
        np.testing.assert_allclose(
            result.test_predictions[probability_columns].sum(axis=1),
            1.0,
        )
        self.assertFalse(result.oof[probability_columns].isna().any().any())

    def test_lightgbm_regression_path(self):
        config = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            categorical_columns=("brand",),
            numeric_columns=("weight",),
            model="lightgbm",
            n_splits=3,
            min_df=1,
            word_max_features=100,
            char_max_features=100,
            lgbm_estimators=20,
            lgbm_early_stopping_rounds=5,
        )
        result = train_cross_validated(
            self.train,
            self.test,
            config,
            folds=self.folds,
        )
        self.assertTrue(np.isfinite(result.test_predictions["prediction"]).all())
        self.assertEqual(len(result.report["folds"]), 3)
        self.assertTrue(all(
            item["early_stopping_metric"] == "smape"
            for item in result.report["folds"]
        ))

    def test_device_resolution_never_silently_downgrades_explicit_gpu(self):
        gpu_config = replace(self.config, model="xgboost", device="gpu")
        with patch("src.train.gpu_backend_preflight", return_value=(False, "no CUDA build")):
            with self.assertRaisesRegex(RuntimeError, "explicitly requested"):
                resolve_training_device(gpu_config)
        auto_config = replace(gpu_config, device="auto")
        with patch("src.train.gpu_backend_preflight", return_value=(False, "no CUDA build")):
            device, report = resolve_training_device(auto_config)
        self.assertEqual(device, "cpu")
        self.assertEqual(report["resolved"], "cpu")
        self.assertIn("fallback", report["reason"].lower())

    def test_xgboost_gpu_training_path_when_available(self):
        config = replace(
            self.config,
            text_columns=(),
            categorical_columns=(),
            numeric_columns=("weight",),
            model="xgboost",
            device="gpu",
            target_transform="none",
            lgbm_estimators=8,
            lgbm_early_stopping_rounds=3,
        )
        available, reason = resolve_training_device(config)
        if available != "gpu":
            self.skipTest(reason)
        result = train_cross_validated(
            self.train,
            self.test,
            config,
            folds=self.folds,
        )
        self.assertEqual(result.report["device"]["resolved"], "gpu")
        self.assertTrue(all(item["device"] == "gpu" for item in result.report["folds"]))
        self.assertTrue(np.isfinite(result.test_predictions["prediction"]).all())

    def test_lightgbm_text_features_are_capped(self):
        config = TrainingConfig(
            target_column="price",
            id_column="id",
            text_columns=("title",),
            model="lightgbm",
            word_max_features=120_000,
            char_max_features=120_000,
            lightgbm_text_max_features_per_block=123,
        )
        preprocessor = build_preprocessor(config)
        transformers = {name: transformer for name, transformer, _ in preprocessor.transformers}
        self.assertEqual(transformers["word_tfidf"].max_features, 123)
        self.assertEqual(transformers["char_tfidf"].max_features, 123)

    def test_custom_lightgbm_metric_uses_original_target_scale(self):
        metric = _make_lgbm_eval_metric(self.config)
        self.assertIsNotNone(metric)
        y = np.array([10.0, 20.0, 30.0])
        name, score, greater = metric(np.log1p(y), np.log1p(y))
        self.assertEqual(name, "smape")
        self.assertEqual(score, 0.0)
        self.assertFalse(greater)

    def test_resume_reuses_validated_fold_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            first = train_cross_validated(
                self.train,
                self.test,
                self.config,
                folds=self.folds,
                model_directory=directory,
            )
            second = train_cross_validated(
                self.train,
                self.test,
                self.config,
                folds=self.folds,
                model_directory=directory,
                resume=True,
            )
            self.assertEqual(second.report["resumed_folds"], 3)
            np.testing.assert_allclose(
                first.test_predictions["prediction"],
                second.test_predictions["prediction"],
            )

    def test_resume_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            train_cross_validated(
                self.train,
                self.test,
                self.config,
                folds=self.folds,
                model_directory=directory,
            )
            changed = self.train.copy()
            changed.loc[0, "title"] = "changed after checkpoint"
            with self.assertRaisesRegex(ValueError, "configuration/data mismatch"):
                train_cross_validated(
                    changed,
                    self.test,
                    self.config,
                    folds=self.folds,
                    model_directory=directory,
                    resume=True,
                )

    def test_models_are_not_retained_by_default(self):
        result = train_cross_validated(
            self.train,
            self.test,
            self.config,
            folds=self.folds,
        )
        self.assertEqual(result.models, [])

    def test_log_transform_rejects_invalid_targets(self):
        train = self.train.copy()
        train.loc[0, "price"] = -2.0
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            train_cross_validated(train, self.test, self.config, folds=self.folds)

    def test_artifacts_are_written_and_protected(self):
        result = train_cross_validated(
            self.train,
            self.test,
            self.config,
            folds=self.folds,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_training_result(result, directory, self.config)
            self.assertTrue(all(path.exists() for path in paths.values()))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            submission = pd.read_csv(paths["submission"])
            self.assertEqual(list(submission.columns), ["id", "price"])
            with self.assertRaises(FileExistsError):
                save_training_result(result, directory, self.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
