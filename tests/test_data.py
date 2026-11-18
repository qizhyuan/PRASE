import hashlib
import io
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fixtures import graph_pair, write_openea
from prase import (
    KG, Alignment, DataError, Decisions, clean_literal, download_benchmark, download_word_vectors, evaluate,
    load_benchmark, load_dataset, threshold_scores,
)
from prase import datasets as dataset_api
from prase.decisions import Constraints


class GraphTest(unittest.TestCase):
    def test_from_triples_indexes_by_first_appearance_and_removes_duplicates(self):
        kg = KG.from_triples(
            [("a", "knows", "b"), ("a", "knows", "b"), ("b", "likes", "c")],
            [("a", "born", '"1990"^^<http://www.w3.org/2001/XMLSchema#gYear>'), ("c", "born", "1990")],
        )
        self.assertEqual(kg.entities, ("a", "b", "c"))
        self.assertEqual([p.kind for p in kg.predicates], ["relation", "relation", "attribute"])
        self.assertEqual(kg.relation_triples.tolist(), [[0, 0, 1], [1, 1, 2]])
        self.assertEqual(kg.literals, ("1990",))
        self.assertEqual(kg.attribute_triples.tolist(), [[0, 2, 0], [2, 2, 0]])
        self.assertFalse(kg.relation_triples.flags.writeable)

    def test_literal_cleanup_keeps_the_lexical_form(self):
        self.assertEqual(clean_literal('"Berlin"'), "Berlin")
        self.assertEqual(clean_literal("<http://x/y>"), "http://x/y")
        self.assertEqual(clean_literal('"3.5"^^<http://www.w3.org/2001/XMLSchema#double>'), "3.5")
        self.assertEqual(clean_literal('"1990-03-05"^^xsd:date'), "1990-03-05")
        self.assertEqual(clean_literal('"Stewart"@en'), "Stewart")
        self.assertEqual(clean_literal('"北京"@zh-Hans'), "北京")
        self.assertEqual(clean_literal('"He said "hi""@en'), 'He said "hi"')
        self.assertEqual(clean_literal("  plain text "), "plain text")

    def test_predicate_kinds_must_match_their_triples(self):
        from prase.kg import Predicate
        with self.assertRaises(DataError):
            KG(["a", "b"], [Predicate("p", "attribute")], [], [(0, 0, 1)], [])

    def test_fingerprint_depends_on_content_only(self):
        first = KG.from_triples([("a", "p", "b")], name="one")
        second = KG.from_triples([("a", "p", "b")], name="two")
        third = KG.from_triples([("a", "p", "c")])
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, third.fingerprint)

    def test_openea_folder_round_trip(self):
        left, right, reference = graph_pair(20)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "pair"
            write_openea(folder, left, right, reference)
            fold = folder / "721_5fold" / "1"
            fold.mkdir(parents=True)
            (fold / "train_links").write_text("L:e0\tR:e0\n", encoding="utf-8")
            dataset = load_dataset(folder)
        self.assertEqual(dataset.left.summary(), left.summary())
        self.assertEqual(len(dataset.links), 20)
        self.assertEqual(len(dataset.reference()), 20)
        self.assertEqual(dataset.splits[0].train, (("L:e0", "R:e0"),))

    def test_malformed_relation_rows_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rel"
            path.write_text("a\tp\n", encoding="utf-8")
            with self.assertRaisesRegex(DataError, "expected 3"):
                KG.from_tsv(path)


class BenchmarkTest(unittest.TestCase):
    name = "D-W-15K-V2"

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.folder = self.root / self.name
        left, right, reference = graph_pair(8)
        write_openea(self.folder, left, right, reference)
        fold = self.folder / "721_5fold" / "2"
        fold.mkdir(parents=True)
        for part in ("train", "valid", "test"):
            (fold / f"{part}_links").write_text("L:e0\tR:e0\n", encoding="utf-8")
        self.checksums = {
            path.relative_to(self.folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.folder.rglob("*")) if path.is_file()
        }
        manifest = dataset_api._BENCHMARKS["datasets"]
        self.enterContext(patch.dict(manifest, {self.name: {"sha256": self.checksums}}))
        self.card = self.root / "README.md"
        self.card.write_bytes(b"# Fixture benchmark\n")
        self.metadata = {"README.md": hashlib.sha256(self.card.read_bytes()).hexdigest()}
        self.enterContext(patch.dict(dataset_api._BENCHMARKS["source"], {"sha256": self.metadata}))

    def test_load_benchmark_verifies_a_pinned_snapshot(self):
        with patch("huggingface_hub.snapshot_download", return_value=str(self.root)) as download:
            dataset = load_benchmark(self.name)
        self.assertEqual(dataset.name, self.name)
        self.assertEqual(len(dataset.reference()), 8)
        self.assertEqual(dataset_api.benchmark_folds(self.name), (2,))
        self.assertEqual(dataset.splits[0].train, (("L:e0", "R:e0"),))
        source = dataset_api._BENCHMARKS["source"]
        download.assert_called_once_with(
            source["repository"], repo_type="dataset", revision=source["revision"], local_dir=None,
            allow_patterns=[*self.metadata, *(f"{self.name}/{file}" for file in self.checksums)],
        )

    def test_download_uses_explicit_destination_and_rejects_corruption(self):
        with patch("huggingface_hub.snapshot_download", return_value=str(self.root)) as download:
            self.assertEqual(download_benchmark(self.name, self.root), self.root)
            self.assertEqual(download.call_args.kwargs["local_dir"], self.root)
            (self.folder / "ent_links").write_bytes(b"L:e0\tR:changed\n")
            with self.assertRaisesRegex(DataError, r"ent_links: SHA-256"):
                load_benchmark(self.name)

    def test_default_download_checks_all_requested_datasets_and_metadata(self):
        second = "EN-DE-100K-V2"
        shutil.copytree(self.folder, self.root / second)
        with (patch.object(dataset_api, "BENCHMARK_NAMES", (self.name, second)),
              patch.dict(dataset_api._BENCHMARKS["datasets"], {second: {"sha256": self.checksums}}),
              patch("huggingface_hub.snapshot_download", return_value=str(self.root)) as download):
            self.assertEqual(download_benchmark(), self.root)
            self.assertIn(f"{self.name}/ent_links", download.call_args.kwargs["allow_patterns"])
            self.assertIn(f"{second}/ent_links", download.call_args.kwargs["allow_patterns"])
            self.card.write_bytes(b"Changed dataset card")
            with self.assertRaisesRegex(DataError, r"README.md: SHA-256"):
                download_benchmark()

    def test_load_local_benchmark_reads_only_pinned_folds(self):
        extra = self.folder / "721_5fold" / "1"
        extra.mkdir()
        (extra / "train_links").write_text("L:e1\tR:e1\n", encoding="utf-8")
        with patch("huggingface_hub.snapshot_download") as download:
            dataset = dataset_api.load_local_benchmark(self.name, self.root)
        download.assert_not_called()
        self.assertEqual(len(dataset.splits), 1)
        self.assertEqual(dataset.splits[0].train, (("L:e0", "R:e0"),))
        self.assertEqual(len(load_dataset(self.folder).splits), 2)

    def test_download_error_reported_as_data_error(self):
        with patch("huggingface_hub.snapshot_download", side_effect=OSError("authentication required")):
            with self.assertRaisesRegex(DataError, "authentication required"):
                download_benchmark(self.name)

    def test_missing_file_and_unknown_name_raise_data_errors(self):
        (self.folder / "721_5fold" / "2" / "test_links").unlink()
        with self.assertRaisesRegex(DataError, r"test_links: missing benchmark file"):
            dataset_api.verify_benchmark(self.name, self.root)
        with patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaisesRegex(DataError, "unknown benchmark"):
                download_benchmark("../other")
        download.assert_not_called()


class WordVectorDownloadTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.vectors = b"2 3\nparis 0.1 0.2 0.3\nfrance 0.4 0.5 0.6\n"
        self.pin = {"file": "vectors.vec", "url": "https://example.org/vectors.vec.zip",
                    "sha256": hashlib.sha256(self.vectors).hexdigest()}
        self.enterContext(patch.dict(dataset_api._BENCHMARKS, {"word_vectors": self.pin}))

    def _archive(self, content: bytes) -> io.BytesIO:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(self.pin["file"], content)
        response = io.BytesIO(archive.getvalue())
        response.headers = {"Content-Length": str(len(archive.getvalue()))}
        return response

    def test_download_extracts_verifies_and_reuses_the_vectors(self):
        with patch("urllib.request.urlopen", return_value=self._archive(self.vectors)) as fetch:
            path = download_word_vectors(self.root / "data")
        self.assertEqual(fetch.call_args.args[0].full_url, self.pin["url"])
        self.assertEqual(path, self.root / "data" / "vectors.vec")
        self.assertEqual(path.read_bytes(), self.vectors)
        self.assertEqual([item.name for item in path.parent.iterdir()], ["vectors.vec"])
        with patch("urllib.request.urlopen") as fetch:
            self.assertEqual(download_word_vectors(self.root / "data"), path)
        fetch.assert_not_called()

    def test_corrupt_or_failed_download_raises_data_error(self):
        with patch("urllib.request.urlopen", return_value=self._archive(b"changed")):
            with self.assertRaisesRegex(DataError, "SHA-256 does not match the pinned word vectors"):
                download_word_vectors(self.root)
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with self.assertRaisesRegex(DataError, "connection refused"):
                download_word_vectors(self.root)
        self.assertEqual(list(self.root.iterdir()), [])


class AlignmentTest(unittest.TestCase):
    def test_one_to_one_keeps_the_best_pair_per_entity(self):
        alignment = Alignment([0, 0, 1, 2], [5, 6, 5, 7], [0.9, 0.8, 0.95, 0.1])
        self.assertEqual(sorted(alignment.one_to_one()), [(0, 6, 0.8), (1, 5, 0.95), (2, 7, 0.1)])
        self.assertEqual(len(alignment.above(0.85)), 2)

    def test_write_and_read_round_trip(self):
        left = KG.from_triples([("a", "p", "b")])
        right = KG.from_triples([("x", "p", "y")])
        alignment = Alignment([0, 1], [1, 0], [0.25, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.tsv"
            alignment.write(path, left, right)
            loaded = Alignment.read(path, left, right)
        self.assertEqual(list(loaded), list(alignment))

    def test_evaluation(self):
        alignment = Alignment([0, 1, 2], [0, 2, 1], [0.9, 0.4, 0.2])
        scores = evaluate(alignment, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual((scores.correct, scores.predicted, scores.reference), (1, 3, 4))
        self.assertAlmostEqual(scores.f1, 2 / 7)
        by_threshold = threshold_scores(alignment, [(0, 0)], thresholds=(0.0, 0.5))
        self.assertEqual(by_threshold[0.5].precision, 1.0)


class DecisionsTest(unittest.TestCase):
    def test_read_write_and_resolve(self):
        left = KG.from_triples([("a", "p", "b")])
        right = KG.from_triples([("x", "p", "y")])
        decisions = Decisions(frozenset({("a", "x")}), frozenset({("b", "x")}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.tsv"
            decisions.write(path)
            self.assertEqual(Decisions.read(path), decisions)
        constraints = decisions.resolve(left, right)
        self.assertEqual(constraints.approved.tolist(), [[0, 0]])
        self.assertEqual(constraints.rejected, frozenset({(1, 0)}))
        self.assertFalse(Constraints())

    def test_approvals_must_be_one_to_one_and_known(self):
        with self.assertRaises(DataError):
            Decisions(frozenset({("a", "x"), ("b", "x")}))
        with self.assertRaises(DataError):
            Decisions(frozenset({("a", "x")}), frozenset({("a", "x")}))
        left = KG.from_triples([("a", "p", "b")])
        with self.assertRaises(DataError):
            Decisions(frozenset({("zzz", "a")})).resolve(left, left)

    def test_later_decisions_win(self):
        first = Decisions(frozenset({("a", "x")}), frozenset({("b", "y")}))
        second = Decisions(frozenset({("b", "y"), ("c", "x")}))
        merged = first.merged(second)
        self.assertEqual(merged.approved, {("b", "y"), ("c", "x")})
        self.assertEqual(merged.rejected, frozenset())

    def test_reading_skips_blank_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.tsv"
            path.write_text("left\tright\tdecision\na\tx\t0.9\tapprove\nb\ty\t0.3\t\n", encoding="utf-8")
            decisions = Decisions.read(path)
        self.assertEqual(decisions.approved, {("a", "x")})
        self.assertEqual(decisions.rejected, frozenset())


if __name__ == "__main__":
    unittest.main()
