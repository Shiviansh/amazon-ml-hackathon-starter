"""Tests for the competition dataset audit."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit import audit_datasets, write_html_report, write_json_report


def _codes(report):
    return {item["code"] for item in report["findings"]}


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "id": np.arange(100),
            "text": [f"training product description {i}" for i in range(100)],
            "category": np.tile(["a", "b"], 50),
            "number": np.linspace(0.0, 1.0, 100),
            "target": np.linspace(10.0, 20.0, 100) + np.sin(np.arange(100)),
        })
        self.test = pd.DataFrame({
            "id": np.arange(100, 140),
            "text": [f"test product description {i}" for i in range(40)],
            "category": np.tile(["a", "b"], 20),
            "number": np.linspace(0.0, 1.0, 40),
        })

    def audit(self, train=None, test=None, **kwargs):
        return audit_datasets(
            self.train if train is None else train,
            self.test if test is None else test,
            target_column="target",
            id_columns="id",
            text_columns="text",
            task="regression",
            **kwargs,
        )

    def test_clean_data_has_no_errors(self):
        report = self.audit()
        self.assertEqual(report["summary"]["finding_counts"]["error"], 0)
        self.assertEqual(report["summary"]["train_rows"], 100)
        self.assertEqual(report["summary"]["test_rows"], 40)

    def test_input_frames_are_not_modified(self):
        train_copy = self.train.copy(deep=True)
        test_copy = self.test.copy(deep=True)
        self.audit()
        pd.testing.assert_frame_equal(self.train, train_copy)
        pd.testing.assert_frame_equal(self.test, test_copy)

    def test_missing_and_duplicate_ids_are_errors(self):
        train = self.train.copy()
        train.loc[1, "id"] = np.nan
        train.loc[3, "id"] = train.loc[2, "id"]
        report = self.audit(train=train)
        self.assertIn("ID_MISSING", _codes(report))
        self.assertIn("ID_DUPLICATE", _codes(report))
        self.assertEqual(report["status"], "error")

    def test_schema_mismatch_is_reported(self):
        test = self.test.drop(columns="category").assign(extra="x")
        report = self.audit(test=test)
        self.assertIn("TEST_MISSING_COLUMNS", _codes(report))
        self.assertIn("TEST_EXTRA_COLUMNS", _codes(report))

    def test_missing_target_is_an_error(self):
        train = self.train.copy()
        train.loc[0, "target"] = np.nan
        report = self.audit(train=train)
        self.assertIn("TARGET_MISSING", _codes(report))

    def test_target_clone_is_detected(self):
        train = self.train.assign(leaked=self.train["target"])
        test = self.test.assign(leaked=np.arange(len(self.test), dtype=float))
        report = self.audit(train=train, test=test)
        self.assertIn("TARGET_CLONE_FEATURE", _codes(report))

    def test_rare_class_warning(self):
        train = self.train.copy()
        train["target"] = ["common"] * 98 + ["rare_a", "rare_b"]
        report = audit_datasets(
            train,
            self.test,
            target_column="target",
            id_columns="id",
            task="classification",
        )
        self.assertIn("RARE_TARGET_CLASSES", _codes(report))

    def test_train_test_id_overlap_is_reported(self):
        test = self.test.copy()
        test.loc[0, "id"] = self.train.loc[0, "id"]
        report = self.audit(test=test)
        self.assertIn("TRAIN_TEST_ID_OVERLAP", _codes(report))

    def test_id_columns_are_not_treated_as_distribution_drift(self):
        report = self.audit()
        self.assertNotIn("id", report["drift"])
        self.assertNotIn("NUMERIC_DRIFT", {
            item["code"] for item in report["findings"] if item.get("column") == "id"
        })

    def test_auto_infers_high_cardinality_integer_target_as_regression(self):
        train = self.train.copy()
        train["target"] = np.arange(len(train))
        report = audit_datasets(
            train,
            self.test,
            target_column="target",
            id_columns="id",
            task="auto",
        )
        self.assertEqual(report["target"]["task"], "regression")

    def test_feature_overlap_ignores_id_and_target(self):
        test = self.test.copy()
        feature_columns = ["text", "category", "number"]
        test.loc[0, feature_columns] = self.train.loc[0, feature_columns].to_numpy()
        report = self.audit(test=test)
        self.assertIn("TRAIN_TEST_FEATURE_OVERLAP", _codes(report))

    def test_conflicting_duplicate_targets_are_reported(self):
        train = self.train.copy()
        train.loc[1, ["text", "category", "number"]] = train.loc[
            0, ["text", "category", "number"]
        ].to_numpy()
        train.loc[1, "target"] = train.loc[0, "target"] + 50.0
        report = self.audit(train=train)
        self.assertIn("CONFLICTING_DUPLICATE_TARGETS", _codes(report))

    def test_numeric_category_can_be_audited_as_categorical(self):
        train = self.train.assign(code=np.tile([1, 2], 50))
        test = self.test.assign(code=3)
        report = audit_datasets(
            train,
            test,
            target_column="target",
            id_columns="id",
            categorical_columns="code",
            task="regression",
        )
        self.assertEqual(
            report["profiles"]["train"]["code"]["semantic_type"],
            "categorical",
        )
        self.assertIn("UNSEEN_TEST_CATEGORIES", _codes(report))

    def test_memory_and_downcast_estimates_are_reported(self):
        report = self.audit()
        self.assertGreater(report["memory"]["train"]["current_bytes"], 0)
        self.assertIn("opportunities", report["memory"]["train"])
        self.assertGreater(report["profiles"]["train"]["text"]["memory_bytes"], 0)

    def test_text_oov_drift_is_reported(self):
        train = self.train.copy()
        test = self.test.copy()
        train["text"] = "alpha beta known"
        test["text"] = "unseen vocabulary token"
        report = self.audit(train=train, test=test, text_sample_size=100)
        self.assertIn("TEXT_VOCABULARY_DRIFT", _codes(report))
        self.assertEqual(report["drift"]["text"]["oov_token_rate"], 1.0)

    def test_categorical_class_mapping_leakage_is_reported(self):
        train = self.train.copy()
        train["target"] = np.tile(["no", "yes"], 50)
        train["leaky_code"] = train["target"].map({"no": "N", "yes": "Y"})
        test = self.test.assign(leaky_code=np.tile(["N", "Y"], 20))
        report = audit_datasets(
            train,
            test,
            target_column="target",
            id_columns="id",
            categorical_columns=["category", "leaky_code"],
            task="classification",
        )
        self.assertIn("CATEGORICAL_TARGET_MAPPING", _codes(report))
        self.assertEqual(
            report["target"]["categorical_mapping_checks"]["leaky_code"][
                "pure_mapping_rate"
            ],
            1.0,
        )

    def test_numeric_drift_is_reported(self):
        test = self.test.copy()
        test["number"] += 100.0
        report = self.audit(test=test)
        self.assertIn("NUMERIC_DRIFT", _codes(report))
        self.assertGreater(report["drift"]["number"]["ks_statistic"], 0.9)

    def test_categorical_unseen_values_are_reported(self):
        test = self.test.copy()
        test["category"] = "unseen"
        report = self.audit(test=test)
        self.assertIn("UNSEEN_TEST_CATEGORIES", _codes(report))
        self.assertEqual(report["drift"]["category"]["unseen_test_rate"], 1.0)

    def test_missingness_drift_is_reported(self):
        test = self.test.copy()
        test.loc[:19, "category"] = None
        report = self.audit(test=test)
        self.assertIn("MISSINGNESS_DRIFT", _codes(report))

    def test_reports_are_strict_json_and_html(self):
        report = self.audit()
        with tempfile.TemporaryDirectory() as directory:
            json_path = write_json_report(report, Path(directory) / "audit.json")
            html_path = write_html_report(report, Path(directory) / "audit.html")
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["audit_version"], "1.1")
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Dataset audit", html_text)
            self.assertIn("Column profiles", html_text)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.audit(drift_warning=1.1)

    def test_empty_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            audit_datasets(pd.DataFrame())


if __name__ == "__main__":
    unittest.main(verbosity=2)
