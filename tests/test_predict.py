"""Tests for authenticated standalone fold inference."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.predict import (
    PredictionConfig,
    attach_inference_feature_table,
    main,
    predict_from_store,
    save_prediction_result,
)
from src.train import TrainingConfig, train_cross_validated


class OfflinePredictionTests(unittest.TestCase):
    def setUp(self):
        count = 40
        self.train = pd.DataFrame({
            "id": np.arange(count),
            "x": np.linspace(0.0, 4.0, count),
            "z": np.sin(np.arange(count) / 3.0),
            "price": 15.0 + 3.0 * np.linspace(0.0, 4.0, count),
        })
        self.test = pd.DataFrame({
            "id": np.arange(100, 108),
            "x": np.linspace(0.25, 3.75, 8),
            "z": np.sin(np.arange(8) / 3.0),
        })
        self.folds = pd.DataFrame({"id": np.arange(count), "fold": np.arange(count) % 2})
        self.config = TrainingConfig(
            target_column="price",
            id_column="id",
            numeric_columns=("x", "z"),
            model="linear",
            target_transform="log1p",
            n_splits=2,
        )

    def train_models(self, directory: Path, **kwargs):
        return train_cross_validated(
            self.train,
            self.test,
            kwargs.pop("config", self.config),
            folds=self.folds,
            model_directory=directory,
            **kwargs,
        )

    def test_offline_predictions_match_training_side_effect_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained = self.train_models(root / "models")
            self.assertTrue((root / "models" / "fold_000.joblib").is_file())
            self.assertTrue((root / "models" / "fold_000.joblib.metadata.json").is_file())
            inferred = predict_from_store(
                self.test,
                root / "models",
                PredictionConfig(batch_size=3),
                include_fold_predictions=True,
            )
            np.testing.assert_allclose(
                inferred.predictions["prediction"],
                trained.test_predictions["prediction"],
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertEqual(inferred.report["fold_count"], 2)
            self.assertEqual(len(inferred.fold_predictions.columns), 3)

    def test_can_predict_a_new_partition_with_different_ids_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.train_models(root / "models")
            new = pd.DataFrame({
                "id": [500, 501, 502],
                "x": [0.1, 2.0, 3.9],
                "z": [0.0, 0.2, -0.4],
            })
            result = predict_from_store(new, root / "models")
            self.assertEqual(result.predictions["id"].tolist(), [500, 501, 502])
            self.assertTrue(np.isfinite(result.predictions["prediction"]).all())

    def test_authentication_failure_prevents_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.train_models(root / "models")
            sidecar = root / "models" / "fold_000.joblib.metadata.json"
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["file_sha256"] = "0" * 64
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "authentication"):
                predict_from_store(self.test, root / "models")

    def test_classification_averages_aligned_probabilities(self):
        train = self.train.copy()
        train["price"] = np.where((np.arange(len(train)) // 2) % 2, "high", "low")
        config = TrainingConfig(
            target_column="price",
            id_column="id",
            numeric_columns=("x", "z"),
            task="classification",
            metric="accuracy",
            model="linear",
            target_transform="none",
            n_splits=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained = train_cross_validated(
                train,
                self.test,
                config,
                folds=self.folds,
                model_directory=root / "models",
            )
            inferred = predict_from_store(self.test, root / "models")
            self.assertEqual(
                inferred.predictions["prediction"].tolist(),
                trained.test_predictions["prediction"].tolist(),
            )
            probability_columns = [
                column for column in inferred.predictions if column.startswith("probability_")
            ]
            np.testing.assert_allclose(
                inferred.predictions[probability_columns].sum(axis=1), 1.0
            )
            with self.assertRaisesRegex(ValueError, "probability-mean"):
                predict_from_store(
                    self.test,
                    root / "models",
                    PredictionConfig(aggregation="median"),
                )

    def test_auxiliary_table_is_aligned_by_id(self):
        raw = self.test[["id", "x", "z"]].copy()
        features = pd.DataFrame({
            "id": raw["id"].iloc[::-1].to_numpy(),
            "signal": np.arange(len(raw), dtype=float),
        })
        joined, columns = attach_inference_feature_table(
            raw, features, id_column="id", prefix="eng"
        )
        expected = features.set_index("id").loc[raw["id"], "signal"].to_numpy()
        np.testing.assert_array_equal(joined["eng__signal"], expected)
        self.assertEqual(columns, ("eng__signal",))

    def test_cli_and_output_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.train_models(root / "models")
            input_path = root / "new.csv"
            output = root / "output"
            self.test.to_csv(input_path, index=False)
            arguments = [
                "--input", str(input_path),
                "--models-dir", str(root / "models"),
                "--output-dir", str(output),
                "--batch-size", "3",
                "--save-fold-predictions",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertTrue((output / "predictions.parquet").is_file())
            self.assertTrue((output / "submission.csv").is_file())
            self.assertTrue((output / "prediction_report.json").is_file())
            self.assertTrue((output / "fold_predictions.parquet").is_file())
            with self.assertRaisesRegex(FileExistsError, "outputs exist"):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
