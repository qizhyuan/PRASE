import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[1] / "benchmarks" / "reproduce.py"
SPEC = importlib.util.spec_from_file_location("prase_benchmark_runner", RUNNER)
reproduce = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reproduce
SPEC.loader.exec_module(reproduce)


class BenchmarkRunnerTest(unittest.TestCase):
    def test_report_uses_later_root_and_groups_equivalent_round_one(self):
        with tempfile.TemporaryDirectory() as directory:
            older, newer = (Path(directory) / part for part in ("older", "newer"))
            base = reproduce.RunCase("prase-bootea", "D-W-15K-V2")
            zero_rounds = reproduce.RunCase(base.experiment, base.dataset, rounds=0)
            first_round = reproduce.RunCase(base.experiment, base.dataset, rounds=1)
            third_round = reproduce.RunCase(base.experiment, base.dataset, rounds=3)
            self.write_result(older, base, 0.61)
            self.write_result(newer, first_round, 0.72)
            self.write_result(newer, base, 0.83)
            self.write_result(newer, zero_rounds, 0.55)
            self.write_result(newer, third_round, 0.94)
            self.write_result(newer, reproduce.RunCase("supervised-bootea", base.dataset), 0.31)
            self.write_result(newer, reproduce.RunCase("se-bootea", base.dataset), 0.42)

            stdout = StringIO()
            with redirect_stdout(stdout):
                reproduce.report([older, newer])

        report = stdout.getvalue()
        self.assertEqual(report.count("## PRASE\n"), 1)
        self.assertIn("## SE (Embedding Only)", report)
        self.assertIn("## PARIS-Seeded Retrieval", report)
        self.assertIn("0.830", report)
        self.assertIn("0.940", report)
        self.assertIn("| prase-bootea | D-W-15K-V2 | 0 | 0 | 0.550 |", report)
        self.assertNotIn("0.610", report)
        self.assertNotIn("0.720", report)
        self.assertNotIn("Fold", report)

    def test_list_distinguishes_completed_failed_incomplete_and_pending_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = reproduce.RunCase("paris", "D-W-15K-V2")
            self.write_result(root, completed, 0.66)
            failed = reproduce.RunCase("paris", "D-W-100K-V2")
            incomplete = reproduce.RunCase("paris", "D-Y-100K-V2")
            for case, record in ((failed, {"failed": {"step": 1}}), (incomplete, {"records": []})):
                path = root / case.directory / "run.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(record), encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                reproduce.main(["list", "--method", "paris", "--output", str(root)])

        status = stdout.getvalue()
        self.assertIn("| paris | D-W-15K-V2 | Completed | F1 0.660 |", status)
        self.assertIn("| paris | D-W-100K-V2 | Failed |", status)
        self.assertIn("| paris | D-Y-100K-V2 | Incomplete |", status)
        self.assertIn("| paris | MED-BBK-9K | Pending |", status)
        self.assertNotIn("Rounds", status)

    def test_run_all_schedules_base_and_configured_sweeps(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                  patch.object(reproduce, "word_vectors"), patch.object(reproduce.logging, "basicConfig"),
                  redirect_stdout(StringIO())):
                self.assertEqual(reproduce.main(["run", "--all", "--output", directory]), 0)

        invoked = [(call.args[0], call.args[1], call.kwargs["rounds"]) for call in run.call_args_list]
        self.assertEqual(len(invoked), len(set(invoked)))
        self.assertEqual(len(invoked), len(reproduce.planned_runs()))
        self.assertEqual([rounds for experiment, dataset, rounds in invoked
                          if (experiment, dataset) == ("prase-bootea", "D-W-15K-V2")], [None, 2, 3, 4])
        self.assertIn(("supervised-gcnalign", "EN-FR-100K-V2", None), invoked)

    def test_run_method_selects_the_underlying_recipe(self):
        dataset = "D-W-15K-V2"
        with tempfile.TemporaryDirectory() as directory:
            for method, extra, expected, fold in (
                ("paris", [], "paris", None),
                ("se", ["--model", "bootea", "--fold", "1"], "supervised-bootea", 1),
                ("prase", ["--model", "bootea"], "prase-bootea", None),
            ):
                with (self.subTest(method=method),
                      patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                      patch.object(reproduce.logging, "basicConfig"), redirect_stdout(StringIO())):
                    self.assertEqual(reproduce.main(["run", "--method", method, "--dataset", dataset,
                                                     "--output", directory, *extra]), 0)
                    self.assertEqual(run.call_args.args[:2], (expected, dataset))
                    self.assertEqual(run.call_args.kwargs["fold"], fold)

    def test_records_replace_the_previous_file_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text('{"measured": {"f1": 0.1}}', encoding="utf-8")
            reproduce.write_record(path, {"measured": {"f1": 0.6}})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"measured": {"f1": 0.6}})
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["comparison.json"])

    def write_result(self, root: Path, case: reproduce.RunCase, f1: float) -> None:
        target = root / case.directory
        target.mkdir(parents=True)
        record = {"experiment": case.experiment, "dataset": case.dataset, "rounds": case.rounds, "seed": 0,
                  "measured": {"f1": f1}}
        (target / "comparison.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
