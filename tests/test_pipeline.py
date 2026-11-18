import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import call, patch

import numpy as np

from fixed_model import FixedModel
from fixtures import graph_pair, write_openea
from prase import (
    Alignment, ConfigError, Decisions, Embed, Embeddings, Load, NameMatch, ParisConfig, ParisStep, Pipeline,
    Predict, evaluate, load_config, load_dataset, methods,
)
from prase.cli import main

FAST = ParisConfig(engine="python")


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
