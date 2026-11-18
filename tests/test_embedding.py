import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fixtures import graph_pair
from prase import ConfigError, Embeddings, TrainingData, available_models, create_model, ranking_scores

HAS_TORCH = importlib.util.find_spec("torch") is not None


class EmbeddingsTest(unittest.TestCase):
    def test_nearest_ranks_and_scores_follow_the_metric(self):
        left = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        right = np.array([[0.0, 2.0], [3.0, 0.1], [0.5, 0.5]])
        cosine = Embeddings(left, right, "cosine")
        found = cosine.nearest(np.arange(3), np.arange(3))
        self.assertEqual(found.right.tolist(), [1, 0, 2])
        self.assertAlmostEqual(found.score[2], 1.0, places=6)
        self.assertEqual(cosine.ranks(np.array([0, 1]), np.array([1, 0])).tolist(), [1, 1])
        manhattan = Embeddings(left * 0.1, right * 0.1, "l1")
        self.assertEqual(manhattan.nearest(np.array([2]), np.arange(3)).right.tolist(), [2])
        self.assertAlmostEqual(manhattan.pair_scores(np.array([2]), np.array([2]))[0], 0.9, places=6)

    def test_nearest_skips_excluded_pairs(self):
        devices = ["cpu"]
        if HAS_TORCH:
            import torch

            devices += ["cuda"] if torch.cuda.is_available() else []
        for device in devices:
            with self.subTest(device=device):
                embeddings = Embeddings([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], device=device)
                found = embeddings.nearest([0, 1], [0, 1, 2], excluded=[[0, 0]])
                self.assertEqual(sorted(found.pairs()), [(0, 1), (1, 2)])
                np.testing.assert_allclose(found.score, [0.8, 1.0], atol=1e-6)
                self.assertEqual(len(embeddings.nearest([0], [0, 1], excluded=[[0, 0], [0, 1]])), 0)
                mutual = embeddings.reciprocal_nearest([0, 1], [0, 1, 2], excluded=[[0, 0]])
                self.assertEqual(sorted(mutual.pairs()), [(0, 1), (1, 2)])

    def test_csls_penalizes_hubs(self):
        left = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]])
        right = np.array([[0.9, 0.44], [1.0, 0.0], [0.6, 0.8]])
        plain = Embeddings(left, right, "cosine").nearest(np.arange(3), np.arange(3)).right.tolist()
        csls = Embeddings(left, right, "csls", csls_k=1).nearest(np.arange(3), np.arange(3)).right.tolist()
        self.assertEqual(plain[1], 0)
        self.assertEqual(len(csls), 3)

    def test_ranking_scores(self):
        left = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1]])
        right = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        scores = ranking_scores(Embeddings(left, right), np.array([[0, 0], [1, 1], [2, 2]]), ks=(1, 2))
        self.assertAlmostEqual(scores["hits@1"], 2 / 3)
        self.assertAlmostEqual(scores["hits@2"], 1.0)
        self.assertAlmostEqual(scores["mrr"], (1 + 1 + 0.5) / 3)

    def test_save_and_load(self):
        embeddings = Embeddings(np.ones((2, 3)), np.zeros((4, 3)), "csls", csls_k=5)
        with tempfile.TemporaryDirectory() as directory:
            embeddings.save(Path(directory) / "e.npz")
            loaded = Embeddings.load(Path(directory) / "e.npz")
        self.assertEqual((loaded.metric, loaded.csls_k), ("csls", 5))
        np.testing.assert_array_equal(loaded.right, embeddings.right)

    def test_invalid_embeddings(self):
        with self.assertRaises(ValueError):
            Embeddings(np.ones((2, 3)), np.ones((2, 4)))
        with self.assertRaises(ValueError):
            Embeddings(np.ones((2, 3)), np.ones((2, 3)), "dot")


class RegistryTest(unittest.TestCase):
    def test_builtin_models_and_errors(self):
        self.assertEqual(available_models(), ("bootea", "gcnalign", "imuse", "iptranse", "mtranse", "multike"))
        with self.assertRaises(ConfigError):
            create_model("no-such-model")
        with self.assertRaises(ConfigError):
            create_model("gcnalign", dimensions=3)
        with self.assertRaises(ConfigError):
            create_model("gcnalign", dim=0)

    def test_import_path_models(self):
        model = create_model("fixed_model:FixedModel", dim=3)
        self.assertEqual(model.name, "fixed")


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TrainingTest(unittest.TestCase):
    def test_validation_holds_out_pairs_and_returns_best_checkpoint(self):
        import torch

        from prase.embedding.common import SeedValidation

        left, right, reference = graph_pair(40, seed=2)
        seeds = reference[:20]
        data = TrainingData(left, right, seeds, (np.arange(left.num_entities), np.arange(right.num_entities)))
        validation = SeedValidation(torch, data, seed=1, device="cpu", epochs=5, start=1, interval=1, patience=2)
        held_out = validation.indices.numpy()
        for side in range(2):
            self.assertFalse(np.isin(held_out[side], validation.training.seeds[:, side]).any())
            self.assertFalse(np.isin(held_out[side], validation.training.candidates[side]).any())
        vectors = (torch.zeros(left.num_entities, 2), torch.zeros(right.num_entities, 2))
        for table, ids in zip(vectors, held_out, strict=True):
            table[ids] = torch.eye(2)
        self.assertFalse(validation.stop(1, lambda: vectors))
        for table in vectors:
            table.zero_()
        self.assertFalse(validation.stop(2, lambda: vectors))
        self.assertTrue(validation.stop(3, lambda: vectors))
        best = validation.vectors(lambda: vectors)
        self.assertEqual(validation.best_epoch, 1)
        for table, ids in zip(best, held_out, strict=True):
            torch.testing.assert_close(table[ids], torch.eye(2))

    def test_validation_distractors_affect_only_the_expected_ranks(self):
        import torch

        from prase.embedding.common import SeedValidation

        left, right, reference = graph_pair(20, seed=2)
        seeds = reference[:10]
        extra = reference[10:11, 1]
        data = TrainingData(left, right, seeds, (np.array([], dtype=int), extra))
        validation = SeedValidation(torch, data, seed=1, device="cpu", epochs=1, distractors=1)
        vectors = (torch.zeros(left.num_entities, 2), torch.zeros(right.num_entities, 2))
        for table, ids in zip(vectors, validation.indices, strict=True):
            table[ids] = torch.eye(2)
        vectors[1][extra] = torch.tensor([1.0, 0.0])
        validation.stop(1, lambda: vectors)
        self.assertAlmostEqual(validation.best_score, 11 / 12, places=6)

    def test_every_builtin_model_trains_on_a_small_pair(self):
        left, right, reference = graph_pair(40, seed=2)
        seeds = reference[:25]
        data = TrainingData(left, right, seeds, (reference[25:, 0], reference[25:, 1]))
        quick = {
            "bootea": {"epochs": 2, "dim": 8, "bootstrap_every": 1},
            "gcnalign": {"epochs": 2, "dim": 8},
            "imuse": {"epochs": 2, "dim": 8},
            "iptranse": {"epochs": 2, "dim": 8},
            "mtranse": {"epochs": 2, "dim": 8},
        }
        for name, options in quick.items():
            with self.subTest(model=name):
                model = create_model(name, **options)
                embeddings = model.fit(data, seed=0)
                self.assertEqual(embeddings.left.shape[0], left.num_entities)
                self.assertEqual(embeddings.right.shape[0], right.num_entities)
                again = model.fit(data, seed=0)
                np.testing.assert_allclose(again.left, embeddings.left, rtol=1e-5, atol=1e-6)

    def test_training_neighbor_search_skips_excluded_positions(self):
        import torch

        from prase.embedding.common import nearest_neighbors

        left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        right = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])

        def search(excluded, reciprocal):
            found = nearest_neighbors(torch, left, right, lambda a, b: a @ b.T, 1, largest=True,
                                      reciprocal=reciprocal, excluded=excluded)
            return [(row, column) for row, column, _ in found]

        for reciprocal in (False, True):
            with self.subTest(reciprocal=reciprocal):
                self.assertEqual(search({(0, 0)}, reciprocal), [(0, 1), (1, 2)])
                self.assertEqual(search({(0, 0), (0, 1), (0, 2)}, reciprocal), [(1, 2)])

    def test_bootstrapping_models_never_train_on_rejected_pairs(self):
        from prase.embedding import bootea, imuse, iptranse

        left, right, reference = graph_pair(30, seed=4)
        pool = reference[20:]
        rejected = np.array([(i, j) for i in pool[:, 0] for j in pool[:, 1]])
        searches = (
            (bootea.BootEA, "_bootstrap", {"epochs": 3, "dim": 8, "bootstrap_every": 1, "sim_th": 0.0}),
            (iptranse.IPTransE, "_bootstrap",
             {"epochs": 3, "dim": 8, "bootstrap_start": 1, "bootstrap_every": 1, "sim_th": 0.0}),
            (imuse, "_entity_links", {"epochs": 1, "dim": 8}),
        )
        for owner, search, options in searches:
            name = getattr(owner, "name", "imuse")
            for blocked in (np.zeros((0, 2), dtype=np.int64), rejected):
                with self.subTest(model=name, rejected=len(blocked)):
                    found = []
                    original = getattr(owner, search)

                    def record(*args, original=original, found=found):
                        result = original(*args)
                        found.extend(result)
                        return result

                    data = TrainingData(left, right, reference[:20], (pool[:, 0], pool[:, 1]), blocked)
                    with patch.object(owner, search, record):
                        create_model(name, **options).fit(data, seed=0)
                    self.assertEqual(bool(found), not len(blocked))

    def test_multike_reads_word_vectors(self):
        left, right, reference = graph_pair(30, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("3 4\nEntity 0.1 0.2 0.3 0.4\n1 0.5 0.1 0.0 0.2\n2 0.3 0.3 0.3 0.3\n", encoding="utf-8")
            model = create_model("multike", word_vectors=str(vectors), epochs=2, dim=8, literal_epochs=1,
                                 char_epochs=1, literal_hidden=(8,), predicate_start=1, predicate_refresh=1)
            data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]))
            embeddings = model.fit(data, seed=0)
            again = model.fit(data, seed=0)
        self.assertEqual(embeddings.left.shape[0], left.num_entities)
        self.assertEqual(embeddings.info["epochs"], 2)
        np.testing.assert_allclose(again.left, embeddings.left, rtol=1e-5, atol=1e-6)

    def test_multike_names_come_from_attributes_or_local_names(self):
        import torch

        from prase import KG
        from prase.embedding.multike import _Inputs

        left = KG.from_triples([("http://l/Paris", "in", "http://l/France")], [("http://l/Paris", "label", "Paris")])
        right = KG.from_triples([("http://r/Q90", "in", "http://r/Q142")], [("http://r/Q90", "label", "Paris")])
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("Paris 0.1 0.2\n", encoding="utf-8")
            model = create_model("multike", word_vectors=str(vectors), left_names=("label",),
                                 right_names=("label",), right_local_names=False)
        data = TrainingData(left, right, np.array([[0, 0]]), (np.array([1]), np.array([1])))
        inputs = _Inputs.build(torch, data, model)
        names = [inputs.texts[i] if i >= 0 else None for i in inputs.names.tolist()]
        self.assertEqual(names, ["Paris", "France", "Paris", None])
        self.assertEqual(inputs.swapped_relations.tolist(), [[0, 2, 3], [2, 0, 1]])

    def test_multike_trains_with_a_float64_default_dtype(self):
        import torch

        left, right, reference = graph_pair(30, seed=4)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.addCleanup(torch.set_default_dtype, previous)
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("Entity 0.1 0.2 0.3 0.4\n", encoding="utf-8")
            model = create_model("multike", word_vectors=str(vectors), epochs=1, dim=8, literal_epochs=1,
                                 char_epochs=1, literal_hidden=(8,))
            data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]))
            embeddings = model.fit(data, seed=0)
        self.assertEqual(embeddings.left.shape, (left.num_entities, 8))

    def test_multike_trains_on_graphs_without_text(self):
        from prase import KG

        edges = [(i, (i + 1) % 12) for i in range(12)] + [(i, (5 * i + 3) % 12) for i in range(12)]
        left, right = (KG.from_triples([(f"{side}:e{h}", f"{side}:knows", f"{side}:e{t}") for h, t in edges])
                       for side in ("l", "r"))
        reference = np.array([[left.entity_index[f"l:e{i}"], right.entity_index[f"r:e{i}"]] for i in range(12)])
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("Entity 0.1 0.2 0.3 0.4\n", encoding="utf-8")
            model = create_model("multike", word_vectors=str(vectors), epochs=2, dim=8, literal_epochs=1,
                                 literal_hidden=(8,), left_local_names=False, right_local_names=False)
            data = TrainingData(left, right, reference[:8], (reference[8:, 0], reference[8:, 1]))
            embeddings = model.fit(data, seed=0)
        self.assertEqual(embeddings.left.shape, (left.num_entities, 8))
        self.assertTrue(np.isfinite(embeddings.right).all())


if __name__ == "__main__":
    unittest.main()
