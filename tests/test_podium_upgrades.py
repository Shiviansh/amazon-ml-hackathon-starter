"""Unit tests for the three Rank-1 podium upgrades:
1. Custom objective gradients for GBDTs (SMAPE/MAPE).
2. Deterministic domain extraction engine.
3. Upgraded text embeddings configuration.
"""

import unittest
import numpy as np
import pandas as pd

from src.models import (
    ModelConfig,
    create_model,
    smooth_log_smape_objective,
    smooth_raw_smape_objective,
    smooth_log_mape_objective,
    smooth_raw_mape_objective,
)
from src.features import extract_deterministic_domain_features, DeterministicDomainExtractor
from src.embeddings import DEFAULT_MODEL_NAME, load_embedding_model
from src.train import TrainingConfig, validate_config, _make_model


class TestPodiumUpgrades(unittest.TestCase):
    def test_smape_and_mape_gradients(self):
        y_true = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        y_pred = np.array([12.0, 48.0, 100.0], dtype=np.float32)

        # Test log SMAPE
        grad, hess = smooth_log_smape_objective(y_true, y_pred)
        self.assertEqual(grad.shape, (3,))
        self.assertEqual(hess.shape, (3,))
        self.assertTrue(np.all(hess > 0))
        self.assertTrue(np.all(np.isfinite(grad)))

        # Test raw SMAPE
        grad, hess = smooth_raw_smape_objective(y_true, y_pred)
        self.assertEqual(grad.shape, (3,))
        self.assertEqual(hess.shape, (3,))
        self.assertTrue(np.all(hess > 0))

        # Test log MAPE
        grad, hess = smooth_log_mape_objective(y_true, y_pred)
        self.assertEqual(grad.shape, (3,))
        self.assertEqual(hess.shape, (3,))
        self.assertTrue(np.all(hess > 0))

        # Test raw MAPE
        grad, hess = smooth_raw_mape_objective(y_true, y_pred)
        self.assertEqual(grad.shape, (3,))
        self.assertEqual(hess.shape, (3,))
        self.assertTrue(np.all(hess > 0))

    def test_model_with_custom_objective(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 4)).astype(np.float32)
        y = rng.uniform(5.0, 50.0, size=50).astype(np.float32)

        # Test LightGBM with custom SMAPE objective
        model = create_model(
            ModelConfig(
                engine="lightgbm",
                task="regression",
                objective="smape",
                early_stopping_rounds=0,
                max_iterations=10,
                random_state=42,
            )
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (50,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_deterministic_domain_extraction(self):
        sample = pd.DataFrame({
            "TITLE": [
                "Nivea Men Shower Gel 250 ml (Pack of 3)",
                "Stainless Steel Screw 10 x 20 cm 500g",
                "USB Flash Drive 64 GB 100W Fast Charger",
            ],
            "PACK_SIZE": ["Pack of 3", "1 count", None],
            "DESCRIPTION": [
                "Total net weight 750 ml revitalizing wash",
                "Heavy duty industrial hardware screws",
                "High speed storage memory device",
            ],
        })
        extracted = extract_deterministic_domain_features(sample)
        self.assertIsInstance(extracted, pd.DataFrame)
        self.assertEqual(len(extracted), 3)
        self.assertIn("pack_count", extracted.columns)
        self.assertIn("is_multipack", extracted.columns)
        self.assertIn("unit_volume_ml", extracted.columns)
        self.assertIn("total_volume_ml", extracted.columns)

        # Verify pack count for first row
        self.assertEqual(extracted.iloc[0]["pack_count"], 3.0)
        self.assertEqual(extracted.iloc[0]["is_multipack"], 1)
        self.assertEqual(extracted.iloc[0]["unit_volume_ml"], 250.0)

        # Verify alias
        self.assertEqual(DeterministicDomainExtractor.__name__, "CatalogFeatureExtractor")

    def test_training_config_and_model_resolution(self):
        config = TrainingConfig(
            target_column="PRICE",
            id_column="ID",
            numeric_columns=("feat_1",),
            metric="smape",
            model="lightgbm",
            objective="auto",
            extract_domain_features=True,
        )
        validate_config(config)
        self.assertEqual(config.objective, "auto")
        self.assertTrue(config.extract_domain_features)

        # Verify model builder resolves objective
        model = _make_model(config, fold_id=0)
        self.assertIsNotNone(model)

    def test_default_embedding_model_name(self):
        self.assertEqual(DEFAULT_MODEL_NAME, "BAAI/bge-large-en-v1.5")


if __name__ == "__main__":
    unittest.main()
