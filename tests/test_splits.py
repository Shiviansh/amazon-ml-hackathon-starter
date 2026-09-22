"""Tests for leakage-aware fold assignment."""

import unittest

import numpy as np
import pandas as pd

from src.splits import (
    assign_folds,
    build_group_labels,
    fold_summary,
    iter_fold_indices,
    validate_fold_assignment,
)


class FoldAssignmentTests(unittest.TestCase):
    def test_kfold_assigns_every_row_and_preserves_input(self):
        frame = pd.DataFrame({"x": np.arange(101)}, index=np.arange(500, 601))
        original = frame.copy(deep=True)
        folded = assign_folds(frame, strategy="kfold", n_splits=5, random_state=7)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(folded.index.tolist(), frame.index.tolist())
        self.assertEqual(sorted(folded["fold"].unique().tolist()), [0, 1, 2, 3, 4])
        self.assertFalse((folded["fold"] < 0).any())
        self.assertLessEqual(folded["fold"].value_counts().max() - folded["fold"].value_counts().min(), 1)

    def test_kfold_is_deterministic_for_the_same_seed(self):
        frame = pd.DataFrame({"x": np.arange(50)})
        first = assign_folds(frame, strategy="kfold", n_splits=5, random_state=19)
        second = assign_folds(frame, strategy="kfold", n_splits=5, random_state=19)
        np.testing.assert_array_equal(first["fold"], second["fold"])

    def test_different_kfold_seeds_change_assignment(self):
        frame = pd.DataFrame({"x": np.arange(50)})
        first = assign_folds(frame, strategy="kfold", n_splits=5, random_state=1)
        second = assign_folds(frame, strategy="kfold", n_splits=5, random_state=2)
        self.assertFalse(np.array_equal(first["fold"], second["fold"]))

    def test_classification_stratification(self):
        frame = pd.DataFrame({"target": [0] * 50 + [1] * 25})
        folded = assign_folds(
            frame,
            target_column="target",
            task="classification",
            strategy="stratified",
            n_splits=5,
        )
        counts = folded.groupby(["fold", "target"]).size().unstack(fill_value=0)
        self.assertTrue((counts[0] == 10).all())
        self.assertTrue((counts[1] == 5).all())

    def test_rare_class_can_be_rejected_explicitly(self):
        frame = pd.DataFrame({"target": [0] * 20 + [1] * 3})
        with self.assertRaisesRegex(ValueError, "at least n_splits"):
            assign_folds(
                frame,
                target_column="target",
                task="classification",
                strategy="stratified",
                n_splits=5,
                rare_class_policy="error",
            )

    def test_rare_classes_are_pooled_for_stratification_only(self):
        frame = pd.DataFrame({
            "target": ["common"] * 30 + [f"rare_{i}" for i in range(10)],
        })
        with self.assertWarnsRegex(UserWarning, "Pooled"):
            folded = assign_folds(
                frame,
                target_column="target",
                task="classification",
                strategy="stratified",
                n_splits=5,
            )
        self.assertEqual(frame["target"].tolist(), folded["target"].tolist())
        self.assertTrue((folded.groupby("fold").size() == 8).all())

    def test_rare_class_fallback_records_actual_strategy(self):
        frame = pd.DataFrame({"target": [0] * 20 + [1] * 3})
        with self.assertWarnsRegex(UserWarning, "falling back"):
            folded = assign_folds(
                frame,
                target_column="target",
                task="classification",
                strategy="stratified",
                n_splits=5,
                rare_class_policy="fallback",
            )
        metadata = folded.attrs["split_metadata"]
        self.assertEqual(metadata["requested_strategy"], "stratified")
        self.assertEqual(metadata["strategy"], "kfold")

    def test_regression_stratification(self):
        rng = np.random.default_rng(10)
        frame = pd.DataFrame({"target": rng.lognormal(size=500)})
        folded = assign_folds(
            frame,
            target_column="target",
            task="regression",
            strategy="stratified",
            n_splits=5,
            regression_bins=10,
        )
        global_mean = frame["target"].mean()
        fold_means = folded.groupby("fold")["target"].mean()
        self.assertTrue(np.all(np.abs(fold_means - global_mean) < global_mean * 0.35))

    def test_group_split_has_no_group_leakage(self):
        frame = pd.DataFrame({
            "product_id": np.repeat(np.arange(20), 3),
            "x": np.arange(60),
        })
        folded = assign_folds(
            frame,
            strategy="group",
            group_columns="product_id",
            n_splits=5,
        )
        self.assertTrue((folded.groupby("product_id")["fold"].nunique() == 1).all())
        validate_fold_assignment(
            folded,
            fold_column="fold",
            n_splits=5,
            group_columns="product_id",
        )

    def test_multiple_group_columns_produce_stable_labels(self):
        frame = pd.DataFrame({
            "brand": ["ab", "a", "ab", None],
            "model": ["c", "bc", "c", "x"],
        })
        first = build_group_labels(frame, ["brand", "model"])
        second = build_group_labels(frame, ["brand", "model"])
        pd.testing.assert_series_equal(first, second)
        self.assertEqual(first.iloc[0], first.iloc[2])
        self.assertNotEqual(first.iloc[0], first.iloc[1])

    def test_missing_group_keys_are_unique_by_default(self):
        frame = pd.DataFrame({"brand": [None, None, "Sony", "Sony"]})
        labels = build_group_labels(frame, "brand")
        self.assertNotEqual(labels.iloc[0], labels.iloc[1])
        self.assertEqual(labels.iloc[2], labels.iloc[3])

    def test_missing_group_keys_can_be_grouped_or_rejected_explicitly(self):
        frame = pd.DataFrame({"brand": [None, None, "Sony"]})
        together = build_group_labels(
            frame,
            "brand",
            missing_group_policy="together",
        )
        self.assertEqual(together.iloc[0], together.iloc[1])
        with self.assertRaisesRegex(ValueError, "missing values"):
            build_group_labels(frame, "brand", missing_group_policy="error")

    def test_missing_group_marker_cannot_collide_with_real_value(self):
        frame = pd.DataFrame({"brand": [None, "<MISSING>"]})
        labels = build_group_labels(
            frame,
            "brand",
            missing_group_policy="together",
        )
        self.assertNotEqual(labels.iloc[0], labels.iloc[1])

    def test_stratified_group_split_has_no_leakage(self):
        groups = np.repeat(np.arange(30), 4)
        frame = pd.DataFrame({
            "group": groups,
            "target": np.tile([0, 0, 1, 1], 30),
        })
        folded = assign_folds(
            frame,
            target_column="target",
            task="classification",
            strategy="stratified_group",
            group_columns="group",
            n_splits=5,
            random_state=22,
        )
        self.assertTrue((folded.groupby("group")["fold"].nunique() == 1).all())
        per_fold = folded.groupby(["fold", "target"]).size().unstack(fill_value=0)
        self.assertTrue((per_fold[0] > 0).all())
        self.assertTrue((per_fold[1] > 0).all())

    def test_imperfect_stratified_group_warns_and_continues(self):
        frame = pd.DataFrame({
            "group": np.repeat(np.arange(10), 10),
            "target": np.repeat([0] * 8 + [1] * 2, 10),
        })
        with self.assertWarnsRegex(UserWarning, "best approximate"):
            folded = assign_folds(
                frame,
                target_column="target",
                task="classification",
                strategy="stratified_group",
                group_columns="group",
                n_splits=5,
            )
        self.assertTrue((folded.groupby("group")["fold"].nunique() == 1).all())

    def test_imperfect_stratified_group_can_be_strict(self):
        frame = pd.DataFrame({
            "group": np.repeat(np.arange(10), 10),
            "target": np.repeat([0] * 8 + [1] * 2, 10),
        })
        with self.assertRaisesRegex(ValueError, "best approximate"):
            assign_folds(
                frame,
                target_column="target",
                task="classification",
                strategy="stratified_group",
                group_columns="group",
                n_splits=5,
                infeasible_stratification="error",
            )

    def test_auto_selects_stratified_group_when_groups_and_target_exist(self):
        frame = pd.DataFrame({
            "group": np.repeat(np.arange(20), 4),
            "target": np.tile([0, 0, 1, 1], 20),
        })
        folded = assign_folds(
            frame,
            target_column="target",
            task="classification",
            strategy="auto",
            group_columns="group",
            n_splits=4,
        )
        self.assertEqual(folded.attrs["split_metadata"]["strategy"], "stratified_group")

    def test_time_split_marks_warmup_and_prevents_future_training(self):
        frame = pd.DataFrame({
            "event_time": pd.date_range("2026-01-01", periods=60, freq="h"),
            "x": np.arange(60),
        }).sample(frac=1, random_state=11)
        folded = assign_folds(
            frame,
            strategy="time",
            time_column="event_time",
            n_splits=5,
        )
        self.assertTrue((folded["fold"] == -1).any())
        chronological = folded.sort_values("event_time")
        assigned = chronological[chronological["fold"] >= 0]["fold"].to_numpy()
        self.assertTrue(np.all(assigned[:-1] <= assigned[1:]))

        times = pd.to_datetime(frame["event_time"], utc=True).to_numpy()
        for _, train, valid in iter_fold_indices(folded["fold"], temporal=True):
            self.assertLess(times[train].max(), times[valid].min())

    def test_identical_timestamps_are_never_split(self):
        frame = pd.DataFrame({
            "event_time": np.repeat(pd.date_range("2026-01-01", periods=30, freq="h"), 3),
            "x": np.arange(90),
        }).sample(frac=1, random_state=5)
        folded = assign_folds(
            frame,
            strategy="time",
            time_column="event_time",
            n_splits=5,
        )
        self.assertTrue((folded.groupby("event_time")["fold"].nunique() == 1).all())

    def test_time_and_group_constraints_require_an_explicit_choice(self):
        frame = pd.DataFrame({
            "event_time": pd.date_range("2026-01-01", periods=20, freq="h"),
            "group": np.repeat(np.arange(10), 2),
        })
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            assign_folds(
                frame,
                strategy="auto",
                time_column="event_time",
                group_columns="group",
                n_splits=3,
            )
        with self.assertRaisesRegex(ValueError, "cannot also guarantee group isolation"):
            assign_folds(
                frame,
                strategy="time",
                time_column="event_time",
                group_columns="group",
                n_splits=3,
            )

    def test_ordinary_iterator_uses_every_other_fold_for_training(self):
        folds = np.array([0, 0, 1, 1, 2, 2])
        generated = list(iter_fold_indices(folds))
        self.assertEqual(len(generated), 3)
        for fold_id, train, valid in generated:
            self.assertTrue(np.all(folds[valid] == fold_id))
            self.assertTrue(np.all(folds[train] != fold_id))

    def test_summary_reports_all_folds(self):
        frame = pd.DataFrame({"target": np.arange(20, dtype=float)})
        folded = assign_folds(frame, target_column="target", strategy="kfold", n_splits=4)
        summary = fold_summary(folded, target_column="target")
        self.assertEqual(summary["rows"].sum(), 20)
        self.assertIn("target_mean", summary.columns)

    def test_existing_fold_column_is_not_overwritten(self):
        frame = pd.DataFrame({"fold": [99, 99], "x": [1, 2]})
        with self.assertRaisesRegex(ValueError, "already exists"):
            assign_folds(frame, strategy="kfold", n_splits=2)

    def test_invalid_configuration_is_rejected(self):
        frame = pd.DataFrame({"x": [1, 2, 3, 4]})
        with self.assertRaisesRegex(ValueError, "at least 2"):
            assign_folds(frame, n_splits=1)
        with self.assertRaisesRegex(ValueError, "requires target_column"):
            assign_folds(frame, strategy="stratified", n_splits=2)
        with self.assertRaisesRegex(ValueError, "requires group_columns"):
            assign_folds(frame, strategy="group", n_splits=2)
        with self.assertRaisesRegex(ValueError, "requires time_column"):
            assign_folds(frame, strategy="time", n_splits=2)

    def test_validator_detects_manual_group_leakage(self):
        frame = pd.DataFrame({
            "group": ["a", "a", "b", "b"],
            "fold": [0, 1, 0, 1],
        })
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_fold_assignment(
                frame,
                n_splits=2,
                group_columns="group",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
