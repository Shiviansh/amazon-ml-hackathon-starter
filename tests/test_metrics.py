"""Unit tests for src.metrics using Python's built-in unittest runner."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from src.metrics import (
    METRIC_REGISTRY,
    accuracy,
    custom_log_smape_objective,
    evaluate_metric,
    f1_macro,
    f1_micro,
    f1_weighted,
    get_metric,
    get_metric_spec,
    mae,
    mape,
    mape_percent,
    rmse,
    smape,
    smape_percent,
)


class RegressionMetricTests(unittest.TestCase):
    def test_smape_hand_calculation(self):
        expected = (2 / 11 + 2 / 7 + 0.0) / 3.0
        self.assertAlmostEqual(smape([10, 20, 0], [12, 15, 0]), expected)

    def test_smape_perfect_and_both_zero(self):
        self.assertEqual(smape([0, 1, 2], [0, 1, 2]), 0.0)

    def test_smape_zero_versus_nonzero_is_two(self):
        self.assertEqual(smape([0], [5]), 2.0)

    def test_smape_tiny_nonzero_values_are_not_changed(self):
        self.assertAlmostEqual(smape([1e-12], [2e-12]), 2 / 3)

    def test_smape_percentage_is_explicit(self):
        self.assertAlmostEqual(smape_percent([1], [2]), smape([1], [2]) * 100)

    def test_smape_is_symmetric(self):
        self.assertEqual(smape([1, 5], [2, 3]), smape([2, 3], [1, 5]))

    def test_custom_log_smape_objective(self):
        y_true = np.array([5.0, 10.0, 15.0])
        y_pred = np.array([5.0, 12.0, 10.0])
        grad, hess = custom_log_smape_objective(y_true, y_pred)
        self.assertEqual(grad.shape, (3,))
        self.assertEqual(hess.shape, (3,))
        self.assertAlmostEqual(grad[0], 0.0, places=3)
        self.assertGreater(grad[1], 0.0)
        self.assertLess(grad[2], 0.0)
        self.assertTrue(np.all(hess > 0))

    def test_mape_hand_calculation(self):
        self.assertAlmostEqual(mape([10, 20, 50], [12, 15, 50]), 0.15)

    def test_mape_percentage_is_explicit(self):
        self.assertAlmostEqual(mape_percent([10], [12]), 20.0)

    def test_mape_zero_policies(self):
        self.assertEqual(mape([0, 10], [5, 12], zero_policy="ignore"), 0.2)
        with self.assertRaisesRegex(ValueError, "undefined"):
            mape([0, 10], [5, 12], zero_policy="raise")

    def test_mape_ignore_rejects_all_zero_targets(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            mape([0, 0], [0, 1], zero_policy="ignore")

    def test_mape_rejects_bad_options(self):
        with self.assertRaisesRegex(ValueError, "zero_policy"):
            mape([1], [1], zero_policy="bad")
        with self.assertRaisesRegex(ValueError, "positive finite"):
            mape([1], [1], eps=0)

    def test_mae_and_rmse_match_sklearn(self):
        y = [1, 4, 9]
        p = [2, 2, 10]
        self.assertEqual(mae(y, p), mean_absolute_error(y, p))
        self.assertEqual(rmse(y, p), root_mean_squared_error(y, p))

    def test_pandas_series_are_supported(self):
        y = pd.Series([1, 2, 3])
        p = pd.Series([2, 2, 3])
        self.assertAlmostEqual(mae(y, p), 1 / 3)

    def test_multidimensional_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "1-D"):
            smape(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            smape([1, 2, 3], [1, 2])

    def test_empty_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            smape([], [])

    def test_nan_and_infinity_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    smape([1, invalid], [1, 2])
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    smape([1, 2], [1, invalid])

    def test_non_numeric_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            mae(["one"], ["two"])


class ClassificationMetricTests(unittest.TestCase):
    def setUp(self):
        self.y = [0, 1, 1, 2]
        self.p = [0, 1, 0, 2]

    def test_classification_metrics_match_sklearn(self):
        self.assertEqual(accuracy(self.y, self.p), accuracy_score(self.y, self.p))
        self.assertEqual(
            f1_micro(self.y, self.p),
            f1_score(self.y, self.p, average="micro", zero_division=0),
        )
        self.assertEqual(
            f1_macro(self.y, self.p),
            f1_score(self.y, self.p, average="macro", zero_division=0),
        )
        self.assertEqual(
            f1_weighted(self.y, self.p),
            f1_score(self.y, self.p, average="weighted", zero_division=0),
        )

    def test_f1_macro_hand_calculation(self):
        self.assertAlmostEqual(f1_macro(self.y, self.p), 7 / 9)

    def test_string_labels_are_supported(self):
        self.assertAlmostEqual(f1_macro(["a", "b", "b"], ["a", "a", "b"]), 2 / 3)

    def test_empty_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            f1_macro([], [])

    def test_mismatched_and_multidimensional_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            f1_macro([0, 1], [0])
        with self.assertRaisesRegex(ValueError, "1-D"):
            f1_macro([[0], [1]], [0, 1])

    def test_missing_and_nonfinite_labels_are_rejected(self):
        for invalid in (None, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "missing or non-finite"):
                    f1_macro([0, invalid], [0, 1])

    def test_probabilities_are_not_accepted_as_class_labels(self):
        with self.assertRaises(ValueError):
            f1_macro([0, 1], [0.2, 0.8])


class RegistryTests(unittest.TestCase):
    def test_registry_directions(self):
        for name in ("smape", "smape_percent", "mape", "mape_percent", "mae", "rmse"):
            self.assertFalse(METRIC_REGISTRY[name].greater_is_better)
        for name in ("accuracy", "f1_micro", "f1_macro", "f1_weighted"):
            self.assertTrue(METRIC_REGISTRY[name].greater_is_better)

    def test_lookup_is_case_and_whitespace_insensitive(self):
        fn, greater_is_better = get_metric("  SMAPE ")
        self.assertIs(fn, smape)
        self.assertFalse(greater_is_better)

    def test_spec_contains_output_scale(self):
        self.assertEqual(
            get_metric_spec("smape_percent").output_scale,
            "percentage [0, 200]",
        )

    def test_original_dictionary_style_registry_access_still_works(self):
        self.assertIs(METRIC_REGISTRY["smape"]["fn"], smape)
        self.assertFalse(METRIC_REGISTRY["smape"]["greater_is_better"])

    def test_unknown_or_empty_metric_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            get_metric("not_a_metric")
        for invalid in ("", "   ", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    get_metric_spec(invalid)

    def test_evaluate_metric(self):
        self.assertEqual(evaluate_metric("accuracy", [0, 1], [0, 1]), 1.0)
        self.assertAlmostEqual(
            evaluate_metric("smape", [1], [2], percentage=True),
            200 / 3,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
