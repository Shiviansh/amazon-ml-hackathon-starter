"""Tests for the resumable end-to-end pipeline orchestrator."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline import (
    DatasetProfile,
    PipelineOptions,
    PipelineRunner,
    PipelineStage,
    _validate_profile,
    _verify_validation_gate,
    build_pipeline_stages,
)


class PipelineRunnerTests(unittest.TestCase):
    def _graph(self, root: Path):
        source = root / "source.txt"
        code = root / "worker.py"
        first = root / "first.txt"
        second = root / "second.txt"
        source.write_text("version-one", encoding="utf-8")
        code.write_text("# worker version one\n", encoding="utf-8")
        stages = [
            PipelineStage(
                "first",
                "first stage",
                (),
                ("fake", "first"),
                (source,),
                (first,),
                (code,),
            ),
            PipelineStage(
                "second",
                "second stage",
                ("first",),
                ("fake", "second"),
                (first,),
                (second,),
                (code,),
            ),
        ]
        return source, first, second, stages

    def test_executes_in_dependency_order_caches_and_invalidates_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _, stages = self._graph(root)
            calls = []

            def executor(stage, _repo_root, _log_path):
                calls.append(stage.name)
                payload = "|".join(path.read_text(encoding="utf-8") for path in stage.inputs)
                stage.outputs[0].write_text(f"{stage.name}:{payload}", encoding="utf-8")
                return 0

            runner = PipelineRunner(
                stages,
                repo_root=root,
                run_directory=root / "run",
                executor=executor,
            )
            first_run = runner.run()
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual([item.status for item in first_run], ["complete", "complete"])

            calls.clear()
            cached_run = runner.run()
            self.assertEqual(calls, [])
            self.assertEqual([item.status for item in cached_run], ["cached", "cached"])

            source.write_text("version-two", encoding="utf-8")
            calls.clear()
            changed_run = runner.run()
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual([item.status for item in changed_run], ["complete", "complete"])

    def test_dry_run_is_read_only_and_handles_missing_generated_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, stages = self._graph(root)
            run = root / "run"

            def forbidden_executor(*_args):
                raise AssertionError("dry-run must not execute a stage")

            runner = PipelineRunner(
                stages,
                repo_root=root,
                run_directory=run,
                executor=forbidden_executor,
            )
            result = runner.run(dry_run=True)
            self.assertEqual([item.status for item in result], ["planned", "planned"])
            self.assertFalse((run / ".pipeline").exists())
            self.assertFalse((root / "first.txt").exists())

    def test_failure_is_durable_and_stops_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, stages = self._graph(root)
            calls = []

            def executor(stage, _repo_root, _log_path):
                calls.append(stage.name)
                return 7

            runner = PipelineRunner(
                stages,
                repo_root=root,
                run_directory=root / "run",
                executor=executor,
            )
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                runner.run()
            self.assertEqual(calls, ["first"])
            manifest = json.loads(runner.manifest_path("first").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("exit code 7", manifest["error"])


class PipelineContractTests(unittest.TestCase):
    def test_minimal_graph_still_ends_in_mandatory_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            repo = Path(__file__).resolve().parents[1]
            profile = DatasetProfile(
                name="amazon",
                train=Path(directory) / "train.csv",
                test=Path(directory) / "test.csv",
                sample_submission=Path(directory) / "sample.csv",
            )
            options = PipelineOptions(
                repo_root=repo,
                run_directory=run,
                device="cpu",
                models=("linear",),
                use_images=False,
            )
            stages = build_pipeline_stages(profile, options)
            names = [stage.name for stage in stages]
            self.assertEqual(
                names,
                [
                    "folds",
                    "features",
                    "train_linear",
                    "ensemble",
                    "postprocess",
                    "validate_submission",
                ],
            )
            self.assertEqual(stages[-1].dependencies, ("postprocess",))
            self.assertIn("validate_submit.py", " ".join(stages[-1].command))

    def test_profile_schema_validation_accepts_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = pd.DataFrame({
                "PRODUCT_ID": [1],
                "PRICE": [10.0],
                "TITLE": ["item"],
                "DESCRIPTION": ["desc"],
                "PACK_SIZE": ["1 pack"],
                "CATEGORY": ["cat"],
                "BRAND": ["brand"],
            })
            test = train.drop(columns="PRICE")
            sample = pd.DataFrame({"PRODUCT_ID": [1], "PRICE": [0.0]})
            train_path = root / "train.parquet"
            test_path = root / "test.parquet"
            sample_path = root / "sample.csv"
            train.to_parquet(train_path, index=False)
            test.to_parquet(test_path, index=False)
            sample.to_csv(sample_path, index=False)
            _validate_profile(
                DatasetProfile("amazon", train_path, test_path, sample_path),
                use_images=False,
            )

    def test_validation_gate_rejects_any_nonpassing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "validation.json"
            report.write_text('{"valid": true, "status": "pass"}', encoding="utf-8")
            self.assertEqual(_verify_validation_gate(report)["status"], "pass")
            report.write_text('{"valid": false, "status": "fail"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                _verify_validation_gate(report)

    def test_invalid_runtime_options_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                PipelineOptions(
                    repo_root=root,
                    run_directory=root / "run",
                    max_image_failure_rate=1.1,
                )
            with self.assertRaisesRegex(ValueError, "unselected models"):
                PipelineOptions(
                    repo_root=root,
                    run_directory=root / "run",
                    models=("linear",),
                    model_params={"xgboost": root / "params.json"},
                )


if __name__ == "__main__":
    unittest.main()
