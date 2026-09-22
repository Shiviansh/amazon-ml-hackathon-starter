import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.models import (
    FoldModelStore,
    ModelConfig,
    UnifiedModel,
    backend_status,
    create_model,
    load_fold_model,
    save_fold_model,
)


class ConfigAndValidationTests(unittest.TestCase):
    def test_invalid_configurations_fail_closed(self):
        with self.assertRaises(ValueError):
            ModelConfig(engine="unknown")
        with self.assertRaisesRegex(ValueError, "only device='cpu'"):
            ModelConfig(engine="ridge", device="gpu")
        with self.assertRaisesRegex(ValueError, "controlled settings"):
            ModelConfig(engine="lightgbm", params={"random_state": 99})
        for key, value in (
            ("n_estimators", 50),
            ("iterations", 50),
            ("max_iter", 50),
            ("objective", "regression"),
            ("learning_rate", 0.2),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "controlled settings"):
                    ModelConfig(engine="lightgbm", params={key: value})
        with self.assertRaises(ValueError):
            ModelConfig(early_stopping_rounds=-1)

    def test_backend_status_is_complete(self):
        status = backend_status()
        self.assertEqual(set(status), {"ridge", "lightgbm", "catboost", "xgboost"})
        self.assertTrue(status["ridge"]["available"])

    def test_tree_early_stopping_requires_validation(self):
        model = create_model(
            ModelConfig(engine="lightgbm", early_stopping_rounds=5, max_iterations=10)
        )
        with self.assertRaisesRegex(ValueError, "requires x_valid"):
            model.fit(np.ones((5, 2)), np.arange(5.0))

    def test_matrix_and_weight_validation(self):
        model = create_model(ModelConfig())
        with self.assertRaisesRegex(ValueError, "NaN"):
            model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model.fit(np.ones((2, 2)), [1.0, 2.0], sample_weight=[1.0, -1.0])

    def test_tree_models_allow_nan_but_all_models_reject_infinity(self):
        features = np.arange(240, dtype=np.float32).reshape(80, 3)
        target = np.linspace(1.0, 9.0, 80)
        features[4, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            create_model(ModelConfig(engine="ridge")).fit(features, target)
        if backend_status()["lightgbm"]["available"]:
            tree = create_model(
                ModelConfig(
                    engine="lightgbm",
                    max_iterations=10,
                    early_stopping_rounds=2,
                )
            )
            tree.fit(
                features[:60], target[:60],
                x_valid=features[60:], y_valid=target[60:],
            )
            self.assertTrue(np.isfinite(tree.predict(features[60:])).all())
            infinite = features[60:].copy()
            infinite[0, 0] = np.inf
            with self.assertRaisesRegex(ValueError, "infinity"):
                tree.predict(infinite)

    def test_explicit_gpu_failure_is_detected_before_full_fit(self):
        model = create_model(
            ModelConfig(
                engine="xgboost",
                device="gpu",
                max_iterations=2,
                early_stopping_rounds=0,
            )
        )
        with patch(
            "src.models.gpu_backend_preflight",
            return_value=(False, "simulated unsupported sparse path"),
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                model.fit(np.ones((20, 3)), np.linspace(1.0, 2.0, 20))


class RidgeInterfaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(80, 8)).astype(np.float32)
        self.y = 3.0 * self.X[:, 0] - 1.5 * self.X[:, 1] + 0.1

    def test_regression_dense_sparse_prediction_and_importance(self):
        model = create_model(ModelConfig(engine="ridge", ridge_alpha=1.0))
        model.fit(self.X, self.y, feature_names=[f"f{i}" for i in range(8)])
        prediction = model.predict(sparse.csr_matrix(self.X[:5]))
        self.assertEqual(prediction.shape, (5,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(model.feature_importance().shape, (8,))
        metadata = model.metadata()
        self.assertFalse(metadata["early_stopping_supported"])
        self.assertEqual(metadata["early_stopping_rounds"], 0)
        with self.assertRaisesRegex(ValueError, "expected 8"):
            model.predict(np.ones((2, 7)))

    def test_string_classification_probability_contract(self):
        labels = np.where(self.X[:, 0] > 0, "premium", "budget")
        model = create_model(
            ModelConfig(
                engine="ridge",
                task="classification",
                ridge_alpha=1.0,
                max_iterations=500,
            )
        )
        model.fit(self.X, labels)
        prediction = model.predict(self.X[:6])
        probability = model.predict_proba(self.X[:6])
        self.assertTrue(set(prediction).issubset({"premium", "budget"}))
        self.assertEqual(probability.shape, (6, 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertEqual(model.classes_.tolist(), ["budget", "premium"])

    def test_unseen_validation_class_is_rejected(self):
        model = create_model(ModelConfig(engine="ridge", task="classification"))
        with self.assertRaisesRegex(ValueError, "absent from training"):
            model.fit(
                self.X[:20],
                np.array(["a", "b"] * 10),
                x_valid=self.X[20:24],
                y_valid=np.array(["a", "c", "a", "c"]),
            )

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not been fitted"):
            UnifiedModel(ModelConfig()).predict(self.X)


class TreeBackendTests(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(7)
        X = rng.normal(size=(180, 6)).astype(np.float32)
        y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.05, size=180)
        return X[:140], y[:140], X[140:], y[140:]

    def test_lightgbm_early_stopping_and_metadata(self):
        if not backend_status()["lightgbm"]["available"]:
            self.skipTest("lightgbm is unavailable")
        x_train, y_train, x_valid, y_valid = self.data()
        model = create_model(
            ModelConfig(
                engine="lightgbm",
                max_iterations=80,
                learning_rate=0.1,
                early_stopping_rounds=8,
                params={"num_leaves": 15},
            )
        )
        model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
        self.assertTrue(np.isfinite(model.predict(x_valid)).all())
        self.assertIsNotNone(model.best_iteration_)
        self.assertEqual(model.best_iteration_, model.estimator_.best_iteration_ - 1)
        self.assertEqual(model.metadata()["engine"], "lightgbm")

    def test_catboost_early_stopping_and_metadata(self):
        if not backend_status()["catboost"]["available"]:
            self.skipTest("catboost is unavailable")
        x_train, y_train, x_valid, y_valid = self.data()
        model = create_model(
            ModelConfig(
                engine="catboost",
                max_iterations=60,
                learning_rate=0.1,
                early_stopping_rounds=8,
                params={"depth": 4},
            )
        )
        model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
        self.assertTrue(np.isfinite(model.predict(x_valid)).all())
        self.assertIsNotNone(model.best_iteration_)
        self.assertEqual(model.metadata()["engine"], "catboost")

    def test_xgboost_early_stopping_and_metadata(self):
        if not backend_status()["xgboost"]["available"]:
            self.skipTest("xgboost is unavailable")
        x_train, y_train, x_valid, y_valid = self.data()
        model = create_model(
            ModelConfig(
                engine="xgboost",
                max_iterations=80,
                learning_rate=0.1,
                early_stopping_rounds=8,
                params={"max_depth": 4},
            )
        )
        model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
        self.assertTrue(np.isfinite(model.predict(x_valid)).all())
        self.assertIsNotNone(model.best_iteration_)
        self.assertEqual(model.metadata()["engine"], "xgboost")

    def test_xgboost_multiclass_native_dmatrix_inference(self):
        if not backend_status()["xgboost"]["available"]:
            self.skipTest("xgboost is unavailable")
        rng = np.random.default_rng(31)
        features = rng.normal(size=(150, 5)).astype(np.float32)
        labels = np.asarray(["budget", "standard", "premium"] * 50)
        model = create_model(
            ModelConfig(
                engine="xgboost",
                task="classification",
                max_iterations=15,
                early_stopping_rounds=3,
                params={"max_depth": 3},
            )
        )
        model.fit(
            features[:120], labels[:120],
            x_valid=features[120:], y_valid=labels[120:],
        )
        probabilities = model.predict_proba(features[120:])
        predictions = model.predict(features[120:])
        self.assertEqual(probabilities.shape, (30, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
        self.assertTrue(set(predictions) <= set(labels))

    def test_xgboost_missing_dependency_is_actionable(self):
        x_train, y_train, x_valid, y_valid = self.data()
        model = create_model(
            ModelConfig(engine="xgboost", max_iterations=10, early_stopping_rounds=2)
        )
        with patch("src.models.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "requires the optional package"):
                model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)


class FoldSerializationTests(unittest.TestCase):
    def fitted_model(self):
        X = np.arange(60, dtype=np.float32).reshape(20, 3)
        y = np.linspace(1.0, 5.0, 20)
        return create_model(ModelConfig(engine="ridge")).fit(X, y), X

    def test_atomic_round_trip_provenance_and_sidecar(self):
        model, X = self.fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold_000.joblib"
            model_path, metadata_path = save_fold_model(
                model,
                path,
                fold_id=0,
                run_signature="run-abc",
                valid_positions=[1, 3, 5],
                provenance={"dataset_sha256": "abc"},
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["fold_id"], 0)
            self.assertEqual(len(metadata["file_sha256"]), 64)
            loaded = load_fold_model(
                model_path,
                expected_fold_id=0,
                expected_run_signature="run-abc",
                expected_feature_count=3,
                expected_valid_positions=[1, 3, 5],
            )
            np.testing.assert_allclose(loaded.predict(X), model.predict(X))
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_fold_model(model, path, fold_id=0, run_signature="run-abc")
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                load_fold_model(path, expected_run_signature="wrong-run")

    def test_corruption_is_rejected_before_deserialization(self):
        model, _ = self.fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold_002.joblib"
            save_fold_model(model, path, fold_id=2, run_signature="run")
            with path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "authentication"):
                load_fold_model(path)

    def test_fold_store_names_lists_and_loads(self):
        model, X = self.fitted_model()
        with tempfile.TemporaryDirectory() as tmp:
            store = FoldModelStore(tmp)
            store.save(model, 4, "signature")
            self.assertEqual(store.completed_folds(), [4])
            loaded = store.load(4, "signature", expected_feature_count=3)
            np.testing.assert_allclose(loaded.predict(X[:2]), model.predict(X[:2]))
            self.assertEqual(store.path_for(4).name, "fold_004.joblib")


if __name__ == "__main__":
    unittest.main()
