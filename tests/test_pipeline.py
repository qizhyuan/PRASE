import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import call, patch

import numpy as np

from fixed_model import FixedModel, StableModel
from fixtures import graph_pair, write_openea
from prase import (
    Alignment, CheckpointPolicy, ConfigError, DataError, Decisions, Embed, Embeddings, Load, NameMatch, ParisConfig,
    ParisStep, Pipeline, Predict, evaluate, load_config, load_dataset, methods,
)
from prase.cli import main

FAST = ParisConfig(engine="python")


def _record_mutations(record):
    """Copies of a step checkpoint record with one field removed or replaced, and whether its digest is updated.

    With the old digest, a copy must be refused as altered. With an updated
    digest, the step must still run again or reuse a valid record.
    """
    import copy

    def walk(value, path):
        yield path, value
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from walk(item, (*path, index))

    for path, value in list(walk(record, ())):
        if not path or path == ("record",):
            continue
        for description, replacement in (("removed", None), ("replaced by text", "damaged"), ("negative", -1)):
            if description == "negative" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                continue
            mutated = copy.deepcopy(record)
            container = mutated
            for key in path[:-1]:
                container = container[key]
            if description == "removed":
                del container[path[-1]]
            else:
                container[path[-1]] = replacement
            label = f"{'/'.join(map(str, path))} {description}"
            yield label, mutated, False
            yield label, mutated, True


def _array_mutations(arrays, sizes):
    """Copies of a step checkpoint's arrays with one array removed, shortened, retyped, or out of range."""
    for name, value in arrays.items():
        others = {key: item for key, item in arrays.items() if key != name}
        yield f"{name} removed", others
        if len(value):
            yield f"{name} shortened", {**others, name: value[:-1]}
        yield f"{name} retyped", {**others, name: value.astype(np.float32 if value.dtype != np.float32 else np.int8)}
        if len(value) and value.dtype.kind == "f":
            broken = value.copy()
            broken.flat[0] = np.nan
            yield f"{name} not finite", {**others, name: broken}
        if len(value) and value.dtype.kind == "i":
            yield f"{name} out of range", {**others, name: value + max(sizes)}


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.left, self.right, self.reference = graph_pair(80, seed=5, named=0.4, drop=0.25)

    def test_prase_feeds_embedding_results_back_into_reasoning(self):
        model = FixedModel(reference=self.reference)
        pipeline = methods.prase(model, config=FAST)
        result = pipeline.run(self.left, self.right, reference=self.reference)
        self.assertEqual([step.name for step in result.steps], ["paris", "embed-fixed", "paris"])
        paris_alone, _, final = result.steps
        self.assertGreater(final.scores.f1, paris_alone.scores.f1)
        trained = model.seen[0]
        self.assertEqual(len(trained.seeds), paris_alone.values["matches"])
        self.assertTrue(np.isin(trained.candidates[0], trained.seeds[:, 0], invert=True).all())
        self.assertEqual(final.values["seeds"], result.steps[1].values["proposals"])
        self.assertEqual(len(paris_alone.observations), FAST.iterations)
        self.assertIn("scores", paris_alone.observations[-1])

    def test_feedback_channels_can_be_chosen(self):
        model = FixedModel(reference=self.reference)
        mappings_only = methods.prase(model, config=FAST, share_embeddings=False).run(self.left, self.right)
        self.assertFalse(mappings_only.steps[2].values["embeddings"])
        embeddings_only = methods.prase(model, config=FAST, propose=False).run(self.left, self.right)
        self.assertEqual(embeddings_only.steps[2].values["seeds"], 0)
        self.assertTrue(embeddings_only.steps[2].values["embeddings"])
        with self.assertRaises(ConfigError):
            Embed(model, propose=False, share_embeddings=False)

    def test_decisions_hold_through_every_step(self):
        result = methods.paris(FAST).run(self.left, self.right)
        i, j, _ = next(iter(result.alignment))
        k = int(self.reference[(self.reference[:, 0] != i)][0, 0])
        decisions = Decisions(
            frozenset({(self.left.entities[k], self.right.entities[j])}),
            frozenset({(self.left.entities[i], self.right.entities[j])}),
        )
        model = FixedModel(reference=self.reference)
        reviewed = methods.prase(model, config=FAST).run(self.left, self.right, decisions=decisions)
        pairs = reviewed.alignment.pairs()
        self.assertIn((k, j), pairs)
        self.assertNotIn((i, j), pairs)
        self.assertIn([k, j], model.seen[0].seeds.tolist())

    def test_rejected_seeds_do_not_train_the_model(self):
        model = FixedModel()
        seeds = Alignment(self.reference[:3, 0], self.reference[:3, 1])
        (i, j), kept = self.reference[0], self.reference[1:3].tolist()
        decisions = Decisions(rejected={(self.left.entities[i], self.right.entities[j])})
        Pipeline([Embed(model, seed_threshold=1.0)]).run(self.left, self.right, seeds=seeds, decisions=decisions)
        trained = model.seen[0]
        self.assertEqual(sorted(trained.seeds.tolist()), sorted(kept))
        self.assertIn(i, trained.candidates[0])
        self.assertIn(j, trained.candidates[1])
        self.assertEqual(trained.rejected.tolist(), [[i, j]])

    def test_stale_seeds_do_not_block_proposals(self):
        from fixed_model import no_literals

        (a, z), (b, x) = self.reference[0].tolist(), self.reference[1].tolist()

        class Propose:
            name = "propose"

            def run(self, state, context):
                return replace(state, proposals=Alignment([a], [z]))

        pipeline = Pipeline([Propose(), ParisStep(FAST, literal_matcher=no_literals)])
        left, right = self.left.entities, self.right.entities
        for decisions in (Decisions(rejected={(left[a], right[x])}), Decisions(approved={(left[b], right[x])})):
            with self.subTest(decisions=decisions):
                result = pipeline.run(self.left, self.right, seeds=Alignment([a], [x]), decisions=decisions)
                self.assertIn((a, z), result.alignment.pairs())

    def test_embedding_only_skips_rejected_neighbors(self):
        from prase import KG

        class Vectors:
            name = "vectors"

            def fit(self, data, *, seed):
                return Embeddings([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

        left = KG.from_triples([("l0", "p", "l1")])
        right = KG.from_triples([("r0", "q", "r1"), ("r1", "q", "r2")])
        decisions = Decisions(rejected={("l0", "r0")})
        result = methods.embedding_only(Vectors()).run(left, right, seeds=Alignment([1], [2]), decisions=decisions)
        self.assertEqual([pair[:2] for pair in result.alignment.to_ids(left, right)], [("l0", "r1")])

    def test_embedding_only_baseline_uses_given_seeds(self):
        model = FixedModel()
        seeds = Alignment(self.reference[:40, 0], self.reference[:40, 1])
        result = methods.embedding_only(model).run(self.left, self.right, seeds=seeds, reference=self.reference)
        self.assertEqual(len(model.seen[0].seeds), 40)
        self.assertGreaterEqual(result.steps[-1].values["added"], 1)

    def test_preset_rejects_options_for_a_model_instance(self):
        for build in (methods.embedding_only, methods.prase):
            with self.subTest(build=build.__name__), self.assertRaisesRegex(ConfigError, "given by name"):
                build(FixedModel(), options={"dim": 2})

    def test_output_supports_a_slotted_custom_model(self):
        class SlottedModel:
            __slots__ = ()
            name = "slotted"

            def fit(self, data, *, seed):
                return Embeddings(np.ones((data.left.num_entities, 2)), np.ones((data.right.num_entities, 2)))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            methods.embedding_only(SlottedModel()).run(self.left, self.right, output=output)
            record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["steps"][0]["model"], {"model": "slotted"})

    def test_custom_steps_follow_the_protocol(self):
        class KeepFirst:
            name = "keep-first"

            def run(self, state, context):
                context.report(kept=1)
                return replace(state, alignment=state.alignment.subset(np.arange(len(state.alignment)) < 1))

        result = Pipeline([ParisStep(FAST), KeepFirst()]).run(self.left, self.right)
        self.assertEqual(len(result.alignment), 1)
        self.assertEqual(result.steps[1].values, {"kept": 1})
        with self.assertRaises(ConfigError):
            Pipeline([object()])

    def test_run_directory_holds_every_intermediate_result(self):
        model = FixedModel(reference=self.reference)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            methods.prase(model, config=FAST).run(
                self.left, self.right, reference=self.reference, output=output, save_embeddings=True,
            )
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual([record["name"] for record in run["records"]], ["paris", "embed-fixed", "paris"])
            self.assertIn("f1", run["scores"])
            self.assertTrue((output / "alignment.tsv").is_file())
            self.assertTrue((output / "01-paris" / "predicates.tsv").is_file())
            self.assertTrue((output / "02-embed-fixed" / "proposals.tsv").is_file())
            self.assertTrue((output / "02-embed-fixed" / "embeddings.npz").is_file())
            embed = output / "02-embed-fixed"
            proposals = Alignment.read(embed / "proposals.tsv", self.left, self.right)
            proposal_precision = evaluate(proposals, self.reference).precision
            self.assertGreater(proposal_precision, 0)
            step = json.loads((embed / "step.json").read_text(encoding="utf-8"))
            self.assertEqual(step["proposal_precision"], proposal_precision)
            self.assertEqual(run["records"][1]["proposal_precision"], proposal_precision)
            step = json.loads((output / "03-paris" / "step.json").read_text(encoding="utf-8"))
            self.assertEqual(len(step["observations"]), FAST.iterations)
            self.assertIsNone(step["proposal_precision"])
            with self.assertRaises(ValueError):
                methods.paris(FAST).run(self.left, self.right, output=output)

    def test_saved_feedback_can_be_loaded(self):
        embeddings = Embeddings(np.ones((self.left.num_entities, 2)), np.ones((self.right.num_entities, 2)))
        proposals = Alignment(self.reference[:5, 0], self.reference[:5, 1], np.full(5, 0.9))
        with tempfile.TemporaryDirectory() as directory:
            embeddings.save(Path(directory) / "e.npz")
            proposals.write(Path(directory) / "p.tsv", self.left, self.right)
            pipeline = Pipeline([Load(Path(directory) / "p.tsv", Path(directory) / "e.npz"), ParisStep(FAST)])
            result = pipeline.run(self.left, self.right)
        self.assertEqual(result.steps[0].values, {"proposals": 5, "embeddings": True})
        self.assertEqual(result.steps[1].values["seeds"], 5)

    def test_predict_needs_proposals(self):
        with self.assertRaises(ConfigError):
            Pipeline([Predict()]).run(self.left, self.right)

    def test_an_interrupted_run_resumes_from_its_finished_steps(self):
        pipeline = methods.prase(StableModel(), config=FAST)
        plain = pipeline.run(self.left, self.right, reference=self.reference, seed=3)
        original, calls = ParisStep.run, []

        def crash_in_the_second_paris_step(step, state, context):
            calls.append(step)
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return original(step, state, context)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(ParisStep, "run", crash_in_the_second_paris_step), self.assertRaises(RuntimeError):
                pipeline.run(self.left, self.right, reference=self.reference, seed=3, output=output)
            saved = sorted(path.name for path in (output / "checkpoint" / "steps").iterdir())
            self.assertEqual(saved, ["01-paris", "02-embed-stable"])
            with self.assertRaisesRegex(DataError, "resume"):
                pipeline.run(self.left, self.right, reference=self.reference, seed=3, output=output)
            with self.assertLogs("prase", "INFO") as logs:
                resumed = pipeline.run(self.left, self.right, reference=self.reference, seed=3, output=output,
                                       resume=True)
            self.assertEqual([record.resumed for record in resumed.steps], [True, True, False])
            self.assertTrue(any("reused from its checkpoint" in line for line in logs.output))
            self.assertEqual(resumed.alignment.pairs(), plain.alignment.pairs())
            np.testing.assert_array_equal(resumed.alignment.score, plain.alignment.score)
            self.assertEqual([record.values for record in resumed.steps], [record.values for record in plain.steps])
            self.assertEqual([record.observations for record in resumed.steps],
                             [record.observations for record in plain.steps])
            self.assertFalse((output / "checkpoint").exists())
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(run["resumed"])
            self.assertNotIn("failed", run)
            self.assertEqual([record.get("resumed", False) for record in run["records"]], [True, True, False])
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["01-paris", "02-embed-stable", "03-paris", "alignment.tsv", "run.json"])

    def test_resume_reuses_the_steps_before_a_changed_setting(self):
        policy = CheckpointPolicy(keep=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = methods.prase(StableModel(), config=FAST).run(self.left, self.right, output=output,
                                                                    checkpoints=policy)
            self.assertTrue((output / "checkpoint" / "steps" / "03-paris" / "state.npz").is_file())
            again = methods.prase(StableModel(), config=FAST).run(self.left, self.right, output=output,
                                                                    checkpoints=policy, resume=True)
            self.assertEqual([record.resumed for record in again.steps], [True, True, True])
            self.assertEqual(again.alignment.pairs(), first.alignment.pairs())
            changed = methods.prase(StableModel(), config=FAST, proposal_weight=0.5)
            third = changed.run(self.left, self.right, output=output, checkpoints=policy, resume=True)
            self.assertEqual([record.resumed for record in third.steps], [True, True, False])
            reseeded = methods.prase(StableModel(), config=FAST).run(self.left, self.right, output=output, seed=1,
                                                                       checkpoints=policy, resume=True)
            self.assertEqual([record.resumed for record in reseeded.steps], [False, False, False])

    def test_checkpoint_policy_chooses_what_a_run_saves(self):
        pipeline = methods.prase(StableModel(), config=FAST)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pipeline.run(self.left, self.right, output=folder / "none", checkpoints=None)
            self.assertFalse((folder / "none" / "checkpoint").exists())
            elsewhere = folder / "elsewhere"
            pipeline.run(self.left, self.right, checkpoints=CheckpointPolicy(directory=elsewhere, keep=True))
            self.assertEqual(len(list((elsewhere / "steps").iterdir())), 3)
            with self.assertRaisesRegex(DataError, "earlier run"):
                pipeline.run(self.left, self.right, checkpoints=CheckpointPolicy(directory=elsewhere))
            with self.assertRaisesRegex(ConfigError, "output directory"):
                pipeline.run(self.left, self.right, resume=True)
            with self.assertRaisesRegex(ConfigError, "checkpoint policy"):
                pipeline.run(self.left, self.right, output=folder / "none", checkpoints=None, resume=True)
            with self.assertRaisesRegex(DataError, "no run to resume"):
                (folder / "other").mkdir()
                (folder / "other" / "notes.txt").write_text("mine", encoding="utf-8")
                pipeline.run(self.left, self.right, output=folder / "other", resume=True)
        for minutes in (-1, float("nan"), True):
            with self.subTest(minutes=minutes), self.assertRaises(ConfigError):
                CheckpointPolicy(training_minutes=minutes)

    def test_checkpoint_cleanup_removes_only_what_prase_wrote(self):
        pipeline = methods.prase(StableModel(), config=FAST)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            mine = folder / "mine"
            mine.mkdir()
            (mine / "notes.txt").write_text("keep me", encoding="utf-8")
            with self.assertRaisesRegex(DataError, "did not write"):
                pipeline.run(self.left, self.right, checkpoints=CheckpointPolicy(directory=mine))
            with self.assertRaisesRegex(DataError, "did not write"):
                pipeline.run(self.left, self.right, output=folder / "run", checkpoints=CheckpointPolicy(directory=mine),
                             resume=True)
            self.assertEqual([path.name for path in mine.iterdir()], ["notes.txt"])
            self.assertFalse((folder / "run").exists())
            for checkpoints, output in (
                (folder / "out", folder / "out"),  # the output directory itself
                (folder, folder / "out"),  # a folder that holds the output directory
                (folder / "out" / "01-paris", folder / "out"),  # a step folder of the run
            ):
                with self.subTest(checkpoints=checkpoints), self.assertRaisesRegex(ConfigError, "checkpoint directory"):
                    pipeline.run(self.left, self.right, output=output,
                                 checkpoints=CheckpointPolicy(directory=checkpoints))
            self.assertFalse((folder / "out").exists())
            # A file added to a checkpoint folder of PRASE survives the cleanup.
            shared = folder / "shared"
            original = ParisStep.run

            def add_a_file(step, state, context):
                (shared / "added.txt").write_text("mine", encoding="utf-8")
                return original(step, state, context)

            with patch.object(ParisStep, "run", add_a_file):
                pipeline.run(self.left, self.right, output=folder / "run",
                             checkpoints=CheckpointPolicy(directory=shared))
            self.assertEqual([path.name for path in shared.iterdir()], ["added.txt"])
            self.assertTrue((folder / "run" / "alignment.tsv").is_file())

    def test_a_rejected_resume_leaves_the_finished_run_untouched(self):
        pipeline = methods.prase(StableModel(), config=FAST)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            pipeline.run(self.left, self.right, output=output)
            before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            for checkpoints, error, message in (
                (None, ConfigError, "checkpoint policy"),
                (CheckpointPolicy(), DataError, "nothing to resume"),
                (CheckpointPolicy(directory=output), ConfigError, "checkpoint directory"),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(error, message):
                    pipeline.run(self.left, self.right, output=output, checkpoints=checkpoints, resume=True)
            after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(after, before)

    def test_resume_rereads_changed_files_and_needs_a_key_for_custom_steps(self):
        class Halve:
            name = "halve"

            def __init__(self):
                self.runs = 0

            def run(self, state, context):
                self.runs += 1
                return replace(state, alignment=state.alignment.scaled(0.5))

        class Keyed(Halve):
            factor = 0.5

            def checkpoint_key(self):
                return {"factor": self.factor}

        policy = CheckpointPolicy(keep=True)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            proposals = folder / "proposals.tsv"
            Alignment(self.reference[:1, 0], self.reference[:1, 1]).write(proposals, self.left, self.right)
            unkeyed, keyed = Halve(), Keyed()
            pipeline = Pipeline([Load(proposals), ParisStep(FAST), unkeyed, keyed])
            pipeline.run(self.left, self.right, output=folder / "run", checkpoints=policy)
            again = pipeline.run(self.left, self.right, output=folder / "run", checkpoints=policy, resume=True)
            # A step without checkpoint_key runs again; the next step is reused, as its input is unchanged.
            self.assertEqual([record.resumed for record in again.steps], [True, True, False, True])
            self.assertEqual((unkeyed.runs, keyed.runs), (2, 1))
            # The same proposals file with new content is read again.
            Alignment(self.reference[1:3, 0], self.reference[1:3, 1]).write(proposals, self.left, self.right)
            changed = Pipeline([Load(proposals), ParisStep(FAST), Keyed()]).run(
                self.left, self.right, output=folder / "other", checkpoints=policy)
            self.assertEqual(changed.steps[0].values["proposals"], 2)
            resumed = Pipeline([Load(proposals), ParisStep(FAST), Keyed()]).run(
                self.left, self.right, output=folder / "other", checkpoints=policy, resume=True)
            self.assertEqual([record.resumed for record in resumed.steps], [True, True, True])
            Alignment(self.reference[:1, 0], self.reference[:1, 1]).write(proposals, self.left, self.right)
            reread = Pipeline([Load(proposals), ParisStep(FAST), Keyed()]).run(
                self.left, self.right, output=folder / "other", checkpoints=policy, resume=True)
            self.assertEqual([record.resumed for record in reread.steps], [False, False, False])
            self.assertEqual(reread.steps[0].values["proposals"], 1)
            # A changed key reruns the custom step.
            stronger = Keyed()
            stronger.factor = 0.25
            rekeyed = Pipeline([Load(proposals), ParisStep(FAST), stronger]).run(
                self.left, self.right, output=folder / "other", checkpoints=policy, resume=True)
            self.assertEqual([record.resumed for record in rekeyed.steps], [True, True, False])

    def test_a_literal_matcher_needs_a_key_for_its_step_to_be_reused(self):
        policy = CheckpointPolicy(keep=True)
        names = {"value": "x"}

        def captured(left, right):  # reads a setting from outside its source
            index = {value: j for j, value in enumerate(right.literals)}
            pairs = [(i, index[value], 1.0) for i, value in enumerate(left.literals)
                     if value in index and value != names["value"]]
            return Alignment.from_pairs(pairs)

        def keyed(left, right):
            return captured(left, right)

        keyed.checkpoint_key = lambda: {"skip": names["value"]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for matcher, reused in ((captured, False), (keyed, True)):
                with self.subTest(matcher=matcher.__name__):
                    # A second, keyed step keeps a step checkpoint for the finished run to resume from.
                    pipeline = Pipeline([ParisStep(FAST, literal_matcher=matcher), ParisStep(FAST)])
                    pipeline.run(self.left, self.right, output=output, checkpoints=policy)
                    again = pipeline.run(self.left, self.right, output=output, checkpoints=policy, resume=True)
                    self.assertEqual(again.steps[0].resumed, reused)
                    names["value"] = self.left.literals[0]
                    changed = pipeline.run(self.left, self.right, output=output, checkpoints=policy, resume=True)
                    fresh = pipeline.run(self.left, self.right)
                    self.assertFalse(changed.steps[0].resumed)
                    self.assertEqual(changed.alignment.pairs(), fresh.alignment.pairs())
                    names["value"] = "x"
                    shutil.rmtree(output)

    def test_invalid_arguments_leave_a_finished_run_untouched(self):
        pipeline = methods.prase(StableModel(), config=FAST)
        policy = CheckpointPolicy(keep=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            pipeline.run(self.left, self.right, output=output, checkpoints=policy)
            before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            for arguments in ({"seed": -1}, {"seed": 1.5}, {"seed": True}, {"save_embeddings": "yes"},
                              {"seeds": Alignment([self.left.num_entities], [0])},
                              {"seeds": Alignment([0], [0], [np.nan])},
                              {"reference": np.array([[0, self.right.num_entities]])},
                              {"checkpoints": "keep"}):
                with self.subTest(arguments=arguments), self.assertRaises((ConfigError, DataError)):
                    pipeline.run(self.left, self.right, output=output, resume=True,
                                 **{"checkpoints": policy, **arguments})
            after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(after, before)
        with self.assertRaisesRegex(ConfigError, "plain folder name"):
            Pipeline([replace(ParisStep(FAST), name="../outside")])
        for flag in ("steps", "training", "keep"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ConfigError, flag):
                CheckpointPolicy(**{flag: "no"})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_a_failure_after_training_keeps_the_finished_fit(self):
        from prase.embedding.gcnalign import GCNAlign

        pipeline = methods.prase("gcnalign", options={"epochs": 30, "dim": 8}, config=FAST)
        plain = pipeline.run(self.left, self.right)
        original, calls = GCNAlign.propose, []

        def fails_once(model, embeddings, data):
            calls.append(model)
            if len(calls) == 1:
                raise RuntimeError("retrieval failed")
            return original(model, embeddings, data)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(GCNAlign, "propose", fails_once):
                with self.assertRaises(RuntimeError):
                    pipeline.run(self.left, self.right, output=output)
                self.assertEqual(len(list((output / "checkpoint" / "training").glob("*.pt"))), 1)
                with self.assertLogs("prase", "INFO") as logs:
                    resumed = pipeline.run(self.left, self.right, output=output, resume=True)
            self.assertTrue(any("reused the finished fit" in line for line in logs.output))
            self.assertEqual(resumed.alignment.pairs(), plain.alignment.pairs())
            self.assertFalse((output / "checkpoint").exists())

    def test_every_damaged_part_of_a_step_checkpoint_runs_the_step_again(self):
        """Damage each field of every step's record and arrays: a resumed run must match an uninterrupted one."""
        import io

        from prase.checkpoints import chain_key

        pipeline = methods.prase(StableModel(), config=FAST)
        policy = CheckpointPolicy(keep=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = pipeline.run(self.left, self.right, output=output, checkpoints=policy)
            steps = output / "checkpoint" / "steps"
            genuine = {path: path.read_bytes() for path in steps.rglob("*") if path.is_file()}

            def restore():
                for path, content in genuine.items():
                    path.write_bytes(content)

            def resumed_run():
                with self.assertNoLogs("prase", "ERROR"):
                    result = pipeline.run(self.left, self.right, output=output, checkpoints=policy, resume=True)
                self.assertEqual(result.alignment.pairs(), first.alignment.pairs())
                np.testing.assert_array_equal(result.alignment.score, first.alignment.score)
                restore()
                return result

            for folder in sorted(steps.iterdir()):
                record = json.loads((folder / "step.json").read_text(encoding="utf-8"))
                with np.load(folder / "state.npz") as saved:
                    arrays = {name: saved[name] for name in saved.files}
                for name, value, sealed in _record_mutations(record):
                    with self.subTest(step=folder.name, field=name, digests="kept" if not sealed else "updated"):
                        if sealed:
                            value = {key: item for key, item in value.items() if key != "record"}
                            value["record"] = chain_key("record", value)
                        (folder / "step.json").write_text(json.dumps(value), encoding="utf-8")
                        resumed_run()
                for name, value in _array_mutations(arrays, (self.left.num_entities, self.right.num_entities)):
                    with self.subTest(step=folder.name, array=name):
                        buffer = io.BytesIO()
                        np.savez(buffer, **value)
                        body = buffer.getvalue()
                        (folder / "state.npz").write_bytes(body)
                        sealed = {key: item for key, item in record.items() if key != "record"}
                        sealed["arrays"] = hashlib.sha256(body).hexdigest()
                        sealed["record"] = chain_key("record", sealed)
                        (folder / "step.json").write_text(json.dumps(sealed), encoding="utf-8")
                        resumed_run()
            # With the genuine checkpoints, every step is reused.
            again = pipeline.run(self.left, self.right, output=output, checkpoints=policy, resume=True)
            self.assertEqual([step.resumed for step in again.steps], [True, True, True])


class NameMatchTest(unittest.TestCase):
    def test_names_match_by_edit_similarity(self):
        from prase import KG
        left = KG.from_triples([("http://a/Barack_Obama", "p", "http://a/Hawaii")])
        right = KG.from_triples([("http://b/Hawai", "q", "http://b/Barak_Obama")])
        result = Pipeline([NameMatch(threshold=0.8)]).run(left, right)
        self.assertEqual(sorted((left.entities[i], right.entities[j]) for i, j, _ in result.alignment),
                         [("http://a/Barack_Obama", "http://b/Barak_Obama"), ("http://a/Hawaii", "http://b/Hawai")])

    def test_name_baselines_choose_the_best_eligible_target(self):
        from prase import KG, NameEmbeddingMatch
        left = KG.from_triples([("http://a/Paris", "p", "http://a/Lyon")])
        right = KG.from_triples([("http://b/Paris", "q", "http://b/Paris_France")])
        cases = (
            # The approved pair reserves b/Paris, so a/Paris takes the next eligible target.
            (Decisions(approved={("http://a/Lyon", "http://b/Paris")}),
             {"http://a/Lyon": "http://b/Paris", "http://a/Paris": "http://b/Paris_France"}),
            # Rejecting the best target leaves the next one.
            (Decisions(rejected={("http://a/Paris", "http://b/Paris")}), {"http://a/Paris": "http://b/Paris_France"}),
            # With every target ruled out, a/Paris stays unmatched.
            (Decisions(approved={("http://a/Lyon", "http://b/Paris")},
                       rejected={("http://a/Paris", "http://b/Paris_France")}), {"http://a/Lyon": "http://b/Paris"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("Paris 1 0\nLyon 0 1\nFrance 0.6 0.8\n", encoding="utf-8")
            for step in (NameMatch(threshold=0.3), NameEmbeddingMatch(vectors, threshold=0.6)):
                for decisions, expected in cases:
                    with self.subTest(step=step.name, decisions=decisions):
                        result = Pipeline([step]).run(left, right, decisions=decisions)
                        self.assertEqual({a: b for a, b, _ in result.alignment.to_ids(left, right)}, expected)


class ConfigTest(unittest.TestCase):
    def write(self, folder: Path, text: str) -> Path:
        left, right, reference = graph_pair(40, seed=1)
        write_openea(folder / "pair", left, right, reference)
        path = folder / "run.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_method_tables_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = self.write(folder, (
                'seed = 3\n[input]\ndataset = "pair"\n[method]\nname = "paris"\nparis = { engine = "python" }\n'
                '[output]\ndirectory = "out"\n'
            ))
            config = load_config(path)
            self.assertEqual((config.seed, config.pipeline.name), (3, "paris"))
            self.assertEqual(config.output, folder / "out")
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["run", str(path), "--quiet"]), 0)
            summary = json.loads(stdout.getvalue())
            self.assertGreater(summary["f1"], 0.8)
            stdout = StringIO()
            with redirect_stdout(stdout):
                main(["evaluate", str(folder / "out" / "alignment.tsv"), "--dataset", str(folder / "pair")])
            self.assertIn("F1=", stdout.getvalue())

    def test_cli_rejects_an_invalid_seed_before_touching_the_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = self.write(folder, (
                '[input]\ndataset = "pair"\n[method]\nname = "paris"\nparis = { engine = "python" }\n'
                '[output]\ndirectory = "out"\n'))
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["run", str(path), "--quiet", "--keep-checkpoints"]), 0)
            output = folder / "out"
            before = {item.relative_to(output): item.read_bytes() for item in output.rglob("*") if item.is_file()}
            with redirect_stderr(StringIO()) as stderr:
                self.assertEqual(main(["run", str(path), "--quiet", "--resume", "--seed", "-1"]), 2)
            self.assertIn("seed", stderr.getvalue())
            after = {item.relative_to(output): item.read_bytes() for item in output.rglob("*") if item.is_file()}
            self.assertEqual(after, before)

    def test_run_files_and_cli_choose_the_checkpoint_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            base = '[input]\ndataset = "pair"\n[method]\nname = "paris"\nparis = { engine = "python" }\n'
            self.assertEqual(load_config(self.write(folder, base)).checkpoints, CheckpointPolicy())
            off = load_config(self.write(folder / "off", base + '[output]\ncheckpoints = false\n'))
            self.assertIsNone(off.checkpoints)
            chosen = load_config(self.write(folder / "chosen", base + (
                '[output.checkpoints]\ntraining_minutes = 2.5\nkeep = true\ndirectory = "saved"\n')))
            self.assertEqual(chosen.checkpoints,
                             CheckpointPolicy(training_minutes=2.5, keep=True, directory=folder / "chosen" / "saved"))
            for table, error in (('keep = "yes"', "true or false"), ("every = 3", "unknown setting"),
                                 ("training_minutes = -1", "nonnegative")):
                with self.subTest(table=table), self.assertRaisesRegex(ConfigError, error):
                    load_config(self.write(folder / "bad", base + f'[output.checkpoints]\n{table}\n'))
            path = self.write(folder / "cli", base + '[output]\ndirectory = "out"\n')
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["run", str(path), "--quiet", "--keep-checkpoints"]), 0)
            self.assertTrue((folder / "cli" / "out" / "checkpoint" / "steps" / "01-paris").is_dir())
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["run", str(path), "--quiet", "--resume"]), 0)
            run = json.loads((folder / "cli" / "out" / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(run["resumed"])
            self.assertTrue(run["records"][0]["resumed"])
            self.assertFalse((folder / "cli" / "out" / "checkpoint").exists())
            with redirect_stderr(StringIO()) as stderr:
                self.assertEqual(main(["run", str(path), "--quiet", "--no-checkpoints", "--resume"]), 2)
            self.assertIn("--no-checkpoints", stderr.getvalue())

    def test_step_tables_and_custom_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), (
                '[input]\ndataset = "pair"\n'
                '[[steps]]\ntype = "paris"\nengine = "python"\niterations = 3\n'
                '[[steps]]\ntype = "embed"\nmodel = "fixed_model:FixedModel"\noptions = { dim = 3 }\n'
                '[[steps]]\ntype = "paris"\nengine = "python"\nproposal_weight = 0.5\n'
            ))
            config = load_config(path)
            self.assertEqual([step.name for step in config.pipeline.steps], ["paris", "embed-fixed", "paris"])
            no_literals = load_config(self.write(Path(directory) / "custom", (
                '[input]\ndataset = "pair"\n[method]\nname = "paris"\nliteral_matcher = "fixed_model:no_literals"\n'
                'paris = { engine = "python" }\n'
            )))
            self.assertEqual(len(no_literals.run().alignment), 0)
            self.assertEqual(config.pipeline.steps[2].proposal_weight, 0.5)
            self.assertGreater(evaluate(config.run().alignment, config.dataset.reference()).f1, 0.5)

    def test_run_file_loads_a_benchmark_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = self.write(folder, (
                '[input]\nbenchmark = "D-W-15K-V2"\n[method]\nname = "paris"\nparis = { engine = "python" }\n'
            ))
            dataset = load_dataset(folder / "pair")
            with patch("prase.config.load_benchmark", return_value=dataset) as load:
                config = load_config(path)
            load.assert_called_once_with("D-W-15K-V2")
            self.assertIs(config.dataset, dataset)
            self.assertGreater(config.run().scores.f1, 0.8)

    def test_benchmark_input_is_exclusive_and_requires_an_id(self):
        with tempfile.TemporaryDirectory() as directory, patch("prase.config.load_benchmark") as load:
            path = Path(directory) / "run.toml"
            for source, error in (
                ('benchmark = "D-W-15K-V2"\ndataset = "pair"\n', "exactly one"),
                ('benchmark = "D-W-15K-V2"\n[input.left]\nrelations = "left.tsv"\n', "exactly one"),
                ('benchmark = "./D-W-15K-V2"\n', "must be one of"),
                ('benchmark = 15\n', "must be one of"),
            ):
                with self.subTest(source=source):
                    path.write_text(f'[input]\n{source}[method]\nname = "paris"\n', encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, error):
                        load_config(path)
            load.assert_not_called()

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for text, message in (
                ('[input]\ndataset = "pair"\n[method]\nname = "paris"\nspeed = 1\n', "speed"),
                ('[input]\ndataset = "pair"\n[method]\nname = "paris"\nparis = { iteration = 2 }\n', "iteration"),
                ('[input]\ndataset = "pair"\n[method]\nname = "embedding-only"\nparis = { iterations = 2 }\n', "paris"),
                ('[input]\ndataset = "pair"\n[[steps]]\ntype = "magic"\n', "magic"),
                ('[input]\ndataset = "pair"\n', "method"),
            ):
                with self.subTest(message=message):
                    path = self.write(Path(directory) / message, text)
                    with self.assertRaisesRegex(ConfigError, message):
                        load_config(path)


class EvaluateCommandTest(unittest.TestCase):
    def test_evaluation_needs_reference_links(self):
        left, right, reference = graph_pair(20, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "pair"
            write_openea(folder, left, right, reference)
            alignment = Path(directory) / "alignment.tsv"
            Alignment(reference[:, 0], reference[:, 1]).write(alignment, left, right)
            with redirect_stdout(StringIO()) as stdout:
                self.assertEqual(main(["evaluate", str(alignment), "--dataset", str(folder)]), 0)
            self.assertIn("F1=1.0000", stdout.getvalue())

            (folder / "ent_links").unlink()
            typo = Path(directory) / "typo.tsv"
            a, b = left.entities[reference[0, 0]], right.entities[reference[0, 1]]
            typo.write_text(f"{a}\t{b}\n{a}x\t{b}\n", encoding="utf-8")
            for extra, message in (([], "evaluation needs reference links"),
                                   (["--reference", str(typo)], f"1 of 2 links name entities missing from the graphs, "
                                                                f"such as {a}x -> {b}")):
                with redirect_stderr(StringIO()) as stderr:
                    self.assertEqual(main(["evaluate", str(alignment), "--dataset", str(folder), *extra]), 2)
                self.assertIn(message, stderr.getvalue())


class DownloadCommandTest(unittest.TestCase):
    def test_download_all(self):
        cache = Path("benchmark-cache")
        stdout = StringIO()
        with patch("prase.cli.download_benchmark", return_value=cache) as download, redirect_stdout(stdout):
            self.assertEqual(main(["download"]), 0)
        download.assert_called_once_with()
        self.assertEqual(stdout.getvalue().splitlines(), [str(cache)])

    def test_download_selected_benchmarks(self):
        names = ["D-W-15K-V2", "MED-BBK-9K"]
        cache = Path("benchmark-cache")
        stdout = StringIO()
        with patch("prase.cli.download_benchmark", return_value=cache) as download, redirect_stdout(stdout):
            self.assertEqual(main(["download", *names, names[0]]), 0)
        self.assertEqual(download.call_args_list, [call(name) for name in names])
        self.assertEqual(stdout.getvalue().splitlines(), [str(cache / name) for name in names])

    def test_download_word_vectors_alone_or_with_benchmarks(self):
        with (patch("prase.cli.download_benchmark", return_value=Path("benchmark-cache")) as download,
              patch("prase.cli.download_word_vectors", return_value=Path("data/vectors.vec")) as vectors,
              redirect_stdout(StringIO())):
            self.assertEqual(main(["download", "--word-vectors"]), 0)
            download.assert_not_called()
            vectors.assert_called_once_with(Path("data"))
            self.assertEqual(main(["download", "D-W-15K-V2", "--word-vectors", "--vectors-dir", "vectors"]), 0)
        download.assert_called_once_with("D-W-15K-V2")
        self.assertEqual(vectors.call_args, call(Path("vectors")))

    def test_download_rejects_unknown_benchmarks_before_fetching(self):
        stderr = StringIO()
        with patch("prase.cli.download_benchmark") as download, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                main(["download", "D-W-15K-V2", "unknown"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("unknown", stderr.getvalue())
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
