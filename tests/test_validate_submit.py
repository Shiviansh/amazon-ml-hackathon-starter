"""Break-tests for the fatal submission validator."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.validate_submit import (
    SubmissionValidationError,
    assert_valid_submission,
    validate_submission,
    validate_submission_file,
    write_validation_report,
)


def _error_codes(report):
    return {item["code"] for item in report["issues"] if item["severity"] == "error"}


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame({"ID": ["001", "002", "003"], "PRICE": [0.0, 0.0, 0.0]})
        self.valid = pd.DataFrame({"ID": ["001", "002", "003"], "PRICE": [10.0, 20.0, 30.0]})

    def validate(self, submission=None, **kwargs):
        return validate_submission(
            self.valid if submission is None else submission,
            self.sample,
            id_column="ID",
            target_columns="PRICE",
            task="regression",
            minimum_value=0.0,
            **kwargs,
        )

    def test_valid_submission_passes(self):
        report = self.validate()
        self.assertTrue(report["valid"])
        self.assertEqual(report["summary"]["errors"], 0)
        assert_valid_submission(report)

    def test_wrong_columns_fail(self):
        broken = self.valid.rename(columns={"PRICE": "price"})
        report = self.validate(broken)
        self.assertIn("COLUMN_SCHEMA_MISMATCH", _error_codes(report))

    def test_reversed_column_order_fails(self):
        report = self.validate(self.valid[["PRICE", "ID"]])
        self.assertIn("COLUMN_ORDER_MISMATCH", _error_codes(report))

    def test_missing_row_fails(self):
        report = self.validate(self.valid.iloc[:-1])
        self.assertIn("ROW_COUNT_MISMATCH", _error_codes(report))
        self.assertIn("ID_SET_MISMATCH", _error_codes(report))

    def test_duplicate_id_fails(self):
        broken = self.valid.copy()
        broken.loc[2, "ID"] = "002"
        report = self.validate(broken)
        self.assertIn("DUPLICATE_IDS", _error_codes(report))
        self.assertIn("ID_SET_MISMATCH", _error_codes(report))

    def test_reordered_ids_fail_by_default(self):
        broken = self.valid.iloc[::-1].reset_index(drop=True)
        report = self.validate(broken)
        self.assertIn("ID_ORDER_MISMATCH", _error_codes(report))

    def test_missing_prediction_fails(self):
        broken = self.valid.copy()
        broken.loc[1, "PRICE"] = np.nan
        report = self.validate(broken)
        self.assertIn("MISSING_PREDICTIONS", _error_codes(report))

    def test_non_numeric_prediction_fails(self):
        broken = self.valid.copy()
        broken["PRICE"] = broken["PRICE"].astype(object)
        broken.loc[1, "PRICE"] = "not-a-number"
        report = self.validate(broken)
        self.assertIn("NON_NUMERIC_PREDICTIONS", _error_codes(report))

    def test_infinite_prediction_fails(self):
        broken = self.valid.copy()
        broken.loc[1, "PRICE"] = np.inf
        report = self.validate(broken)
        self.assertIn("INFINITE_PREDICTIONS", _error_codes(report))

    def test_negative_price_fails(self):
        broken = self.valid.copy()
        broken.loc[0, "PRICE"] = -0.01
        report = self.validate(broken)
        self.assertIn("PREDICTION_BELOW_MINIMUM", _error_codes(report))

    def test_constant_predictions_warn_but_pass(self):
        broken = self.valid.copy()
        broken["PRICE"] = 5.0
        report = self.validate(broken)
        self.assertTrue(report["valid"])
        codes = {item["code"] for item in report["issues"]}
        self.assertIn("CONSTANT_PREDICTIONS", codes)

    def test_probability_submission_checks_range_and_sum(self):
        sample = pd.DataFrame({"ID": ["a", "b"], "p0": [0.0, 0.0], "p1": [0.0, 0.0]})
        broken = pd.DataFrame({"ID": ["a", "b"], "p0": [1.2, 0.4], "p1": [0.1, 0.4]})
        report = validate_submission(
            broken,
            sample,
            id_column="ID",
            target_columns=["p0", "p1"],
            task="classification",
            probabilities=True,
        )
        self.assertIn("PROBABILITY_OUT_OF_RANGE", _error_codes(report))
        self.assertIn("PROBABILITY_SUM_MISMATCH", _error_codes(report))

    def test_unknown_class_label_fails_with_train_labels(self):
        sample = pd.DataFrame({"ID": ["a", "b"], "LABEL": ["x", "x"]})
        submission = pd.DataFrame({"ID": ["a", "b"], "LABEL": ["x", "unknown"]})
        train = pd.DataFrame({"LABEL": ["x", "y", "x"]})
        report = validate_submission(
            submission,
            sample,
            id_column="ID",
            target_columns="LABEL",
            task="classification",
            train=train,
        )
        self.assertIn("UNKNOWN_CLASS_LABELS", _error_codes(report))

    def test_unknown_numeric_class_label_also_fails(self):
        sample = pd.DataFrame({"ID": ["a", "b"], "LABEL": [0, 0]})
        submission = pd.DataFrame({"ID": ["a", "b"], "LABEL": [0, 2]})
        train = pd.DataFrame({"LABEL": [0, 1, 0]})
        report = validate_submission(
            submission,
            sample,
            id_column="ID",
            target_columns="LABEL",
            task="classification",
            train=train,
        )
        self.assertIn("UNKNOWN_CLASS_LABELS", _error_codes(report))

    def test_assert_valid_raises_for_fatal_report(self):
        report = self.validate(self.valid.iloc[:-1])
        with self.assertRaises(SubmissionValidationError):
            assert_valid_submission(report)

    def test_raw_duplicate_header_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "sample.csv"
            submission_path = Path(directory) / "submission.csv"
            self.sample.to_csv(sample_path, index=False)
            submission_path.write_text("ID,PRICE,PRICE\n001,10,10\n002,20,20\n003,30,30\n", encoding="utf-8")
            report = validate_submission_file(
                submission_path,
                sample_path,
                id_column="ID",
                target_columns="PRICE",
                task="regression",
                minimum_value=0.0,
            )
            self.assertIn("DUPLICATE_COLUMNS", _error_codes(report))

    def test_file_validation_preserves_leading_zero_ids_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "sample.csv"
            submission_path = Path(directory) / "submission.csv"
            self.sample.to_csv(sample_path, index=False)
            self.valid.to_csv(submission_path, index=False)
            report = validate_submission_file(
                submission_path,
                sample_path,
                id_column="ID",
                target_columns="PRICE",
                task="regression",
                minimum_value=0.0,
            )
            self.assertTrue(report["valid"])
            self.assertTrue(report["file"]["round_trip_checked"])
            self.assertEqual(len(report["file"]["sha256"]), 64)

    def test_json_report_is_written(self):
        report = self.validate()
        with tempfile.TemporaryDirectory() as directory:
            path = write_validation_report(report, Path(directory) / "report.json")
            self.assertTrue(path.exists())
            self.assertIn('"valid": true', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
