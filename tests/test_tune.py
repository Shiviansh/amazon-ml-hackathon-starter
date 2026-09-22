"""Tests for fold-safe Optuna hyperparameter tuning."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import TrainingConfig
from src.tune import (
    TuningConfig,
    _to_backend_parameters,
    attach_tuning_feature_table,
    main,
    save_tuning_result,
    tune_hyperparameters,
)


class TuneValidationTests(unittest.TestCase):
    def setUp(self):
        count = 60
        self.train = pd.DataFrame({
            "id": np.arange(count),
            "x1": np.linspace(-2.0, 2.0, count),
            "x2": np.sin(np.arange(count) / 4.0),
            "price": 30.0 + 4.0 * np.linspace(-2.0, 2.0, count) + np.arange(count) % 3,
        })
        self.folds = pd.DataFrame({"id": np.arange(count), "fold": np.arange(count) % 3})
        self.config = TrainingConfig(
            target_column="price",
            id_column="id",
            numeric_columns=("x1", "x2"),
            model="xgboost",
            target_transform="log1p",
            n_splits=3,
        )

    def test_tuning_config_rejects_parallel_single_gpu_trials(self):
        with self.assertRaisesRegex(ValueError, "VRAM contention"):
            TuningConfig(device="gpu", n_jobs_trials=2)
        with self.assertRaisesRegex(ValueError, "without duplicates"):
            TuningConfig(engines=("xgboost", "xgboost"))

    def test_backend_parameter_mapping_is_explicit(self):
        search = {
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "reg_lambda": 2.5,
            "min_child_weight": 6.2,
            "random_strength": 0.4,
            "colsample_bytree": 0.7,
        }
        gpu = _to_backend_parameters("catboost", search, "gpu")
        self.assertEqual(gpu["depth"], 7)
        self.assertEqual(gpu["min_data_in_leaf"], 6)
        self.assertNotIn("rsm", gpu)
        cpu = _to_backend_parameters("catboost", search, "cpu")
        self.assertEqual(cpu["rsm"], 0.7)

    def test_auxiliary_features_align_by_id_and_verify_target_encoding_folds(self):
        ids = self.train["id"].iloc[::-1].to_numpy()
        fold_by_id = self.folds.set_index("id")["fold"]
        features = pd.DataFrame({
            "id": ids,
            "fold": fold_by_id.loc[ids].to_numpy(),
            "signal": np.arange(len(ids), dtype=float),
            "te_brand": np.linspace(1.0, 2.0, len(ids)),
        })
        joined, added = attach_tuning_feature_table(
            self.train,
            features,
            self.config,
            prefix="eng",
            expected_folds=self.folds.sample(frac=1, random_state=4),
        )
        self.assertEqual(added, ("eng__signal", "eng__te_brand"))
        expected = features.set_index("id").loc[self.train["id"], "signal"].to_numpy()
        np.testing.assert_array_equal(joined["eng__signal"], expected)
        broken = features.copy()
        broken.loc[0, "fold"] = (int(broken.loc[0, "fold"]) + 1) % 3
        with self.assertRaisesRegex(ValueError, "do not match"):
            attach_tuning_feature_table(
                self.train,
                broken,
                self.config,
                prefix="eng",
                expected_folds=self.folds,
            )

    def test_both_engines_tune_and_artifacts_are_train_ready(self):
        tuning = TuningConfig(
            engines=("xgboost", "catboost"),
            n_trials=1,
            device="cpu",
            max_iterations=12,
            early_stopping_rounds=3,
            startup_trials=1,
            pruner_warmup_folds=1,
            sampler_seed=9,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = tune_hyperparameters(
                self.train,
                self.config,
                tuning,
                folds=self.folds,
                cache_directory=root / "cache",
            )
            self.assertEqual(set(result.best_parameters), {"xgboost", "catboost"})
            self.assertEqual(len(result.trials), 2)
            self.assertTrue(np.isfinite(result.trials["value"]).all())
            for engine, artifact in result.best_parameters.items():
                self.assertEqual(artifact["engine"], engine)
                self.assertIn("learning_rate", artifact)
                self.assertIsInstance(artifact["model_params"], dict)
                self.assertEqual(len(artifact["fold_scores"]), 3)
            outputs = save_tuning_result(result, root / "output")
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            payload = json.loads(outputs["best__xgboost"].read_text(encoding="utf-8"))
            self.assertIn("max_depth", payload["model_params"])
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_tuning_result(result, root / "output")

    def test_cli_uses_persistent_sqlite_and_preflights_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.parquet"
            fold_path = root / "folds.csv"
            output = root / "tuning"
            self.train.iloc[:40].to_parquet(train_path, index=False)
            pd.DataFrame({"id": np.arange(40), "fold": np.arange(40) % 2}).to_csv(
                fold_path, index=False
            )
            arguments = [
                "--train", str(train_path),
                "--folds", str(fold_path),
                "--output-dir", str(output),
                "--target-column", "price",
                "--id-column", "id",
                "--numeric-columns", "x1", "x2",
                "--n-splits", "2",
                "--engines", "xgboost",
                "--device", "cpu",
                "--trials", "1",
                "--max-iterations", "6",
                "--early-stopping-rounds", "2",
                "--startup-trials", "1",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertTrue((output / "optuna_studies.db").is_file())
            self.assertTrue((output / "best_params_xgboost.json").is_file())
            with self.assertRaisesRegex(FileExistsError, "long tuning run"):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
