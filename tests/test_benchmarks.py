import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

    def test_saved_runs_keep_the_rounds_they_ran_with(self):
        one_round = {**reproduce.recipe_method("prase-bootea", "D-W-15K-V2"), "rounds": 1}
        catalog = copy.deepcopy(reproduce.CATALOG)
        catalog["experiments"]["prase-bootea"]["method"]["rounds"] = 2
        with tempfile.TemporaryDirectory() as directory, patch.object(reproduce, "CATALOG", catalog):
            root = Path(directory)
            # Saved while the recipe ran one round.
            self.write_result(root, reproduce.RunCase("prase-bootea", "D-W-15K-V2"), 0.7, method=one_round)
            stdout = StringIO()
            with redirect_stdout(stdout):
                reproduce.report([root])
                reproduce.main(["list", "--experiment", "prase-bootea", "--dataset", "D-W-15K-V2",
                                "--output", str(root)])
        output = stdout.getvalue()
        self.assertIn("| prase-bootea | D-W-15K-V2 | 1 | 0 | 0.700 |", output)
        # The recipe's two-round base run would write to the folder of the saved one-round run.
        self.assertIn("| prase-bootea | D-W-15K-V2 | 2 | Historical |", output)

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

    def test_recipes_merge_their_config_with_their_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prase").mkdir()
            (root / "prase" / "model.toml").write_text(
                '[method]\nname = "prase"\nmodel = "bootea"\n[method.options]\ndim = 50\n'
                '[dataset_options]\nD-W-15K-V2 = { epsilon = 0.98 }\n', encoding="utf-8")
            (root / "suite.toml").write_text(
                'datasets = ["D-W-15K-V2", "MED-BBK-9K"]\n'
                '[experiments.base]\nconfig = "prase/model.toml"\n'
                '[experiments.changed]\nconfig = "prase/model.toml"\ndatasets = ["D-W-15K-V2"]\n'
                'method = { propose = false, options = { k = 5 } }\n'
                'dataset_options = { D-W-15K-V2 = { batch_size = 10 } }\n'
                '[experiments.inline]\nmethod = { name = "str-match", threshold = 0.5 }\n', encoding="utf-8")
            experiments = reproduce.load_suite(root / "suite.toml")["experiments"]

        self.assertEqual(experiments["base"]["datasets"], ["D-W-15K-V2", "MED-BBK-9K"])
        self.assertEqual(experiments["base"]["method"], {"name": "prase", "model": "bootea", "options": {"dim": 50}})
        changed = experiments["changed"]
        self.assertEqual(changed["datasets"], ["D-W-15K-V2"])
        self.assertEqual(changed["method"], {"name": "prase", "model": "bootea", "options": {"dim": 50, "k": 5},
                                             "propose": False})
        self.assertEqual(changed["dataset_options"], {"D-W-15K-V2": {"epsilon": 0.98, "batch_size": 10}})
        self.assertEqual(experiments["inline"]["method"], {"name": "str-match", "threshold": 0.5})

    def test_invalid_recipes_name_the_recipe_and_the_problem(self):
        cases = (
            ('[experiments.a]\nconfig = "missing.toml"\n', "config missing.toml not found"),
            ('[experiments.a]\nmethod = { name = "paris" }\nmodels = 1\n', "unknown settings models"),
            ('[experiments.a]\nmethod = { model = "bootea" }\n', "needs a config or a [method] name"),
            ('[experiments.a]\nmethod = { name = "paris" }\ndatasets = ["D-W-15K"]\n', "unknown datasets D-W-15K"),
            ('[experiments.a]\nmethod = { name = "paris" }\nrounds = { D-W-15K-V2 = [2] }\n', "rounds"),
            ('[experiments.a]\nmethod = { name = "prase", model = "bootea" }\nevaluation = "fold-ranking"\n',
             "fold-ranking evaluates an embedding-only method"),
            ('[experiments.a]\nmethod = { name = "paris" }\nevaluation = "folds"\n', "evaluation must be one of"),
        )
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.toml"
            for text, message in cases:
                with self.subTest(message=message):
                    suite.write_text(text, encoding="utf-8")
                    with self.assertRaises(SystemExit) as raised:
                        reproduce.load_suite(suite)
                    self.assertIn("experiment a", str(raised.exception))
                    self.assertIn(message, str(raised.exception))

    def test_every_stable_config_is_documented_and_used_by_the_suite(self):
        folder = RUNNER.parent
        used = {recipe["config"] for recipe in reproduce._read_toml(reproduce.SUITE)["experiments"].values()
                if "config" in recipe}
        configs = {path.relative_to(folder).as_posix() for path in folder.glob("*/*.toml")}
        self.assertEqual(configs, used)
        for config in sorted(configs):
            with self.subTest(config=config):
                self.assertTrue((folder / config).read_text(encoding="utf-8").startswith("# "))

    def test_prase_and_supervised_configs_give_each_model_the_same_options(self):
        folder = RUNNER.parent
        for supervised in sorted((folder / "supervised").glob("*.toml")):
            with self.subTest(model=supervised.stem):
                prase = reproduce._read_toml(folder / "prase" / supervised.name)
                alone = reproduce._read_toml(supervised)
                self.assertEqual(prase["method"]["model"], alone["method"]["model"])
                self.assertEqual(prase["method"].get("options"), alone["method"].get("options"))
                self.assertEqual(prase.get("dataset_options"), alone.get("dataset_options"))

    def test_records_name_the_code_that_produced_them(self):
        import prase

        revision = reproduce.code_revision()
        self.assertEqual(set(revision), {"commit", "modified"})
        if revision["commit"] is not None:
            self.assertRegex(revision["commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(revision["modified"], bool)
        engine = reproduce.engine_identity()
        self.assertEqual(engine["prase"], prase.__version__)
        self.assertRegex(engine["source"], r"^[0-9a-f]{16}$")
        self.assertTrue(engine["native"] is None or len(engine["native"]) == 16)

    def test_runs_of_changed_recipes_or_versions_are_historical(self):
        case = reproduce.RunCase("prase-bootea", "D-W-15K-V2")
        method = reproduce.recipe_method(case.experiment, case.dataset)
        current = {"experiment": case.experiment, "dataset": case.dataset, "rounds": None, "seed": 0,
                   "evaluation": "alignment", "device": "cpu", "engine": reproduce.engine_identity()}
        on_gpu = {**method, "options": {**method.get("options", {}), "device": "cuda"}}
        self.assertEqual(reproduce.outdated({**current, "method": on_gpu}), [])
        changed = {**method, "options": {**method.get("options", {}), "dim": 7}}
        self.assertEqual(reproduce.outdated({**current, "method": changed}), ["changed method.options.dim"])
        self.assertEqual(reproduce.outdated({**current, "method": method, "engine": {"prase": "1.0.0"}}),
                         ["PRASE 1.0.0"])
        self.assertEqual(reproduce.outdated({**current, "method": method,
                                             "engine": {"prase": reproduce.engine_identity()["prase"]}}),
                         ["PRASE build unknown"])
        rebuilt = {**reproduce.engine_identity(), "source": "0" * 16}
        self.assertEqual(reproduce.outdated({**current, "method": method, "engine": rebuilt}), ["another PRASE build"])
        self.assertEqual(reproduce.outdated({**current, "method": method}, seed=7), ["seed 0"])
        self.assertEqual(reproduce.outdated({**current, "method": method}, device="cuda"), ["device cpu"])
        self.assertEqual(reproduce.outdated({**current, "method": method, "evaluation": "paris-seeded-ranking"}),
                         ["evaluation paris-seeded-ranking"])
        older_torch = {**reproduce.engine_identity(), "torch": "1.0"}
        self.assertEqual(reproduce.outdated({**current, "method": method, "engine": older_torch}), ["torch 1.0"])
        multike = reproduce.recipe_method("prase-multike", "D-W-15K-V2")
        resolved = {**multike, "options": {key: "data/vectors.vec" if value == "@word_vectors" else value
                                           for key, value in multike["options"].items()}}
        self.assertEqual(reproduce.outdated({**current, "experiment": "prase-multike", "method": resolved}), [])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, case, 0.5, method=changed)
            self.write_result(root, reproduce.RunCase("paris", "D-W-15K-V2"), 0.66)
            stdout = StringIO()
            with redirect_stdout(stdout):
                reproduce.report([root])
                reproduce.main(["list", "--experiment", "prase-bootea", "--dataset", "D-W-15K-V2",
                                "--output", str(root)])
        output = stdout.getvalue()
        self.assertIn("## Historical Runs", output)
        self.assertIn("| prase-bootea | D-W-15K-V2 | F1 0.500 |", output)
        self.assertIn("changed method.options.dim", output)
        self.assertIn("| prase-bootea | D-W-15K-V2 | 1 | Historical |", output)
        self.assertIn("0.660", output)

    def test_resume_skips_finished_runs_and_moves_outdated_ones(self):
        finished = reproduce.RunCase("paris", "D-W-15K-V2")
        outdated = reproduce.RunCase("paris", "D-Y-100K-V2")
        interrupted = reproduce.RunCase("paris", "EN-DE-100K-V2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, finished, 0.66)
            self.write_result(root, outdated, 0.5, method={"name": "paris", "iterations": 1})
            (root / interrupted.directory).mkdir(parents=True)
            (root / interrupted.directory / "run.json").write_text("{}", encoding="utf-8")
            datasets = [case.dataset for case in (finished, outdated, interrupted)]
            arguments = ["run", "--method", "paris", "--output", directory]
            for dataset in datasets:
                arguments += ["--dataset", dataset]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                reproduce.main(arguments)  # the folders hold runs, so --resume is required
            with (patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                  patch.object(reproduce.logging, "basicConfig"), redirect_stdout(StringIO())):
                self.assertEqual(reproduce.main([*arguments, "--resume"]), 0)
            self.assertEqual([call.args[1] for call in run.call_args_list], datasets[1:])
            self.assertTrue(all(call.kwargs["resume"] for call in run.call_args_list))
            moved = list(root.glob("history/*/paris/D-Y-100K-V2/comparison.json"))
            self.assertEqual(len(moved), 1)
            self.assertFalse((root / outdated.directory).exists())
            _, historical = reproduce._saved_runs([root])
            self.assertEqual([row["dataset"] for _, row, _ in historical], ["D-Y-100K-V2"])

    def test_resume_identifies_runs_by_seed_build_and_normalized_rounds(self):
        base = reproduce.RunCase("prase-bootea", "D-W-15K-V2")
        other = reproduce.RunCase("paris", "D-W-15K-V2")
        # --rounds with the recipe's own count names the base run and its folder.
        self.assertEqual(reproduce.planned_runs(["prase-bootea"], datasets=["D-W-15K-V2"], rounds=1), [base])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, base, 0.7)
            self.write_result(root, other, 0.66)
            # --rounds 1 is the recipe's own round count: the run in its folder is current.
            with (patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                  patch.object(reproduce.logging, "basicConfig"), redirect_stdout(StringIO())):
                reproduce.main(["run", "--method", "prase", "--model", "bootea", "--dataset", base.dataset,
                                "--rounds", "1", "--output", directory, "--resume"])
            run.assert_not_called()
            self.assertTrue((root / base.directory / "comparison.json").is_file())
            self.assertEqual(reproduce._status(base, [root]), "Completed")
            # Another seed replaces the run.
            with (patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                  patch.object(reproduce.logging, "basicConfig"), redirect_stdout(StringIO())):
                reproduce.main(["run", "--method", "paris", "--dataset", other.dataset, "--seed", "7",
                                "--output", directory, "--resume"])
            self.assertEqual(run.call_args.kwargs["seed"], 7)
            self.assertEqual(len(list(root.glob("history/*/paris/D-W-15K-V2/comparison.json"))), 1)
            # Another PRASE build replaces the run as well.
            self.write_result(root, other, 0.66)
            rebuilt = {**reproduce.engine_identity(), "source": "f" * 16}
            with (patch.object(reproduce, "engine_identity", return_value=rebuilt),
                  patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                  patch.object(reproduce.logging, "basicConfig"), redirect_stdout(StringIO())):
                self.assertEqual(reproduce._status(other, [root]), "Historical")
                reproduce.main(["run", "--method", "paris", "--dataset", other.dataset, "--output", directory,
                                "--resume"])
            run.assert_called_once()
            self.assertEqual(len(list(root.glob("history/*/paris/D-W-15K-V2*/comparison.json"))), 2)

    def test_unreadable_records_are_historical_and_never_hide_a_current_run(self):
        case = reproduce.RunCase("paris", "D-W-15K-V2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, case, 0.66)
            moved = root / "history" / "20260101-000000" / case.directory
            moved.mkdir(parents=True)
            (moved / "comparison.json").write_text("{not json", encoding="utf-8")
            broken = reproduce.RunCase("paris", "D-Y-100K-V2")
            (root / broken.directory).mkdir(parents=True)
            (root / broken.directory / "comparison.json").write_text('{"experiment": "paris"}', encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                reproduce.report([root])
                reproduce.main(["list", "--method", "paris", "--output", str(root)])
            output = stdout.getvalue()
            self.assertIn("0.660", output)
            self.assertEqual(output.count("unreadable record"), 2)
            self.assertIn("| paris | D-W-15K-V2 | Completed | F1 0.660 |", output)
            self.assertIn("| paris | D-Y-100K-V2 | Historical |", output)

    def test_resume_reruns_a_run_of_another_device_or_protocol(self):
        case = reproduce.RunCase("prase-bootea", "D-W-15K-V2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, case, 0.7)
            self.assertEqual(reproduce._status(case, [root], 0, "cpu"), "Completed")
            self.assertEqual(reproduce._status(case, [root], 0, "cuda"), "Historical")
            self.assertEqual(reproduce.planned_device("prase-bootea", "D-W-15K-V2", None), "cpu")
            self.assertEqual(reproduce.planned_device("prase-bootea", "D-W-15K-V2", "cuda"), "cuda")
            self.assertIsNone(reproduce.planned_device("paris", "D-W-15K-V2", "cuda"))
            with (patch.object(reproduce, "run", return_value={"measured": {"f1": 0.6}}) as run,
                  patch.object(reproduce.logging, "basicConfig"), redirect_stdout(StringIO())):
                reproduce.main(["run", "--method", "prase", "--model", "bootea", "--dataset", case.dataset,
                                "--device", "cuda", "--output", directory, "--resume"])
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["device"], "cuda")
            self.assertEqual(len(list(root.glob("history/*/prase-bootea/D-W-15K-V2/comparison.json"))), 1)
            # A recipe that changes its evaluation protocol makes its earlier results historical.
            self.write_result(root, case, 0.7)
            catalog = copy.deepcopy(reproduce.CATALOG)
            catalog["experiments"]["prase-bootea"]["evaluation"] = "paris-seeded-ranking"
            with patch.object(reproduce, "CATALOG", catalog):
                self.assertEqual(reproduce._status(case, [root]), "Historical")
            self.assertEqual(reproduce._status(case, [root]), "Completed")

    def test_checkpoints_stay_until_the_benchmark_record_is_written(self):
        from fixtures import graph_pair
        from prase import Dataset

        left, right, reference = graph_pair(40, seed=3)
        links = tuple((left.entities[i], right.entities[j]) for i, j in reference)
        graphs = Dataset("toy", left, right, links)
        original, attempts = reproduce.write_record, []

        def fails_once(path, record):
            attempts.append(path)
            if len(attempts) == 1:
                raise OSError("disk full")
            original(path, record)

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(reproduce, "load_benchmark", return_value=graphs),
              patch.object(reproduce, "write_record", fails_once)):
            root = Path(directory)
            target = root / "paris" / "D-W-15K-V2"
            options = dict(seed=0, rounds=None, device=None, vectors=None)
            with self.assertRaises(OSError):
                reproduce.run("paris", "D-W-15K-V2", None, root, **options)
            # The run finished, and its checkpoints wait for the record.
            self.assertTrue((target / "checkpoint" / "prase-checkpoints.json").is_file())
            with self.assertLogs("prase", "INFO") as logs:
                row = reproduce.run("paris", "D-W-15K-V2", None, root, resume=True, **options)
            self.assertTrue(any("reused from its checkpoint" in line for line in logs.output))
            self.assertFalse(list(root.glob("history/*")))
            self.assertFalse((target / "checkpoint").exists())
            self.assertEqual(json.loads((target / "comparison.json").read_text(encoding="utf-8"))["measured"],
                             row["measured"])

    def test_supervised_folds_are_reused_only_with_the_same_identity(self):
        import numpy as np

        from fixtures import graph_pair
        from prase import Dataset
        from prase.datasets import Split

        left, right, reference = graph_pair(40, seed=3)
        links = tuple((left.entities[i], right.entities[j]) for i, j in reference)
        splits = (Split(links[:20], links[20:25], links[25:]),)
        graphs = Dataset("toy", left, right, links, splits)
        method = {"name": "embedding-only", "model": "fixed_model:StableModel"}
        calls = []
        original = reproduce.Pipeline.run

        def counted(pipeline, *args, **kwargs):
            calls.append(kwargs["output"])
            return original(pipeline, *args, **kwargs)

        with (tempfile.TemporaryDirectory() as directory, patch.object(reproduce, "benchmark_folds", return_value=[1]),
              patch.object(reproduce.Pipeline, "run", counted)):
            root = Path(directory)
            target = root / "supervised-bootea" / "D-W-15K-V2"
            options = dict(seed=0, fold=None, experiment="supervised-bootea", output=root)
            first = reproduce.supervised_runs("D-W-15K-V2", graphs, target, method, **options)
            again = reproduce.supervised_runs("D-W-15K-V2", graphs, target, method, resume=True, **options)
            self.assertEqual(len(calls), 1)
            self.assertEqual(again, first)
            rebuilt = {**reproduce.engine_identity(), "native": "e" * 16}
            with patch.object(reproduce, "engine_identity", return_value=rebuilt):
                rerun = reproduce.supervised_runs("D-W-15K-V2", graphs, target, method, resume=True, **options)
            self.assertEqual(len(calls), 2)
            self.assertEqual(rerun[0]["measured"], first[0]["measured"])
            self.assertEqual(len(list(root.glob("history/*/supervised-bootea/D-W-15K-V2/fold-1/run.json"))), 1)
            # A fold trained on another device runs again.
            on_gpu = {**method, "options": {"device": "cuda"}}
            self.assertIsNone(reproduce._finished_fold(
                target / "fold-1", reproduce.chain_key(
                    "fold", reproduce.run_identity("supervised-bootea", "D-W-15K-V2", None, 0, 1,
                                                   reproduce.effective_device(on_gpu)),
                    graphs.left.fingerprint, graphs.right.fingerprint,
                    reproduce.array_digest(graphs.reference(splits[0].train)),
                    reproduce.array_digest(graphs.reference(splits[0].test)), 0)))
        self.assertTrue(np.isfinite(first[0]["measured"]["hits@1"]))

    def test_word_vectors_are_verified_on_every_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.vec"
            path.write_text("1 1\na 0.5\n", encoding="utf-8")
            with patch.object(reproduce, "verify_word_vectors", side_effect=lambda value: value) as verify:
                for _ in range(3):
                    self.assertEqual(reproduce.word_vectors(path), path)
            self.assertEqual(verify.call_count, 3)
            with self.assertRaises(SystemExit):
                reproduce.word_vectors(path)  # the content is not the pinned file

    def test_records_replace_the_previous_file_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text('{"measured": {"f1": 0.1}}', encoding="utf-8")
            reproduce.write_record(path, {"measured": {"f1": 0.6}})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"measured": {"f1": 0.6}})
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["comparison.json"])

    def write_result(self, root: Path, case: reproduce.RunCase, f1: float, method: dict | None = None,
                     seed: int = 0) -> None:
        target = root / case.directory
        target.mkdir(parents=True)
        method = method if method is not None else reproduce.recipe_method(case.experiment, case.dataset, case.rounds)
        record = {"experiment": case.experiment, "dataset": case.dataset, "rounds": case.rounds, "seed": seed,
                  "method": method, "evaluation": reproduce.evaluation_protocol(case.experiment),
                  "device": reproduce.effective_device(method), "engine": reproduce.engine_identity(),
                  "measured": {"f1": f1}}
        (target / "comparison.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
