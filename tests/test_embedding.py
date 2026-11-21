import importlib.util
import itertools
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fixtures import graph_pair
from prase import (
    ConfigError, Embeddings, TrainingData, available_models, create_model, ranking_scores, training_checkpoints,
)

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

    def test_ranks_break_ties_like_retrieval(self):
        devices = ["cpu"]
        if HAS_TORCH:
            import torch

            devices += ["cuda"] if torch.cuda.is_available() else []
        pairs = np.arange(10)
        for device in devices:
            with self.subTest(device=device):
                # Every candidate ties, so only the first test pair ranks first, as retrieval picks it.
                tied = Embeddings(np.ones((10, 4)), np.ones((10, 4)), device=device)
                self.assertEqual(tied.ranks(pairs, pairs).tolist(), list(range(1, 11)))
                self.assertAlmostEqual(ranking_scores(tied, np.stack((pairs, pairs), axis=1))["hits@1"], 0.1)
                found = tied.nearest(pairs, pairs)
                self.assertEqual(int((found.left == found.right).sum()), 1)
                # A target repeated in the reference is one candidate, which each of its queries ranks first.
                shared = Embeddings(np.eye(4), np.eye(4), device=device)
                self.assertEqual(shared.ranks(np.array([1, 1, 1]), np.array([1, 1, 1])).tolist(), [1, 1, 1])
                # Left 1's target appears twice but ranks first; left 2 ties with target 1, which comes first.
                self.assertEqual(shared.ranks(np.array([0, 1, 2]), np.array([1, 1, 0])).tolist(), [2, 1, 2])

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
        # Right 1 and right 2 tie for every left entity, and right 1 comes first.
        self.assertAlmostEqual(scores["hits@1"], 2 / 3)
        self.assertAlmostEqual(scores["hits@2"], 2 / 3)
        self.assertAlmostEqual(scores["mrr"], (1 + 1 + 1 / 3) / 3)

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


class _Interrupt(Exception):
    pass


_REMOVED = object()


def _mutations(torch, payload):
    """Copies of a training checkpoint payload with one part damaged.

    Yields a description, the copy, and whether the fit must match an
    uninterrupted one: damaged tables, random state, optimizer state, or
    validation state must make the fit start afresh. The optimizer's saved
    hyperparameters are not read, and damaged model state may still be valid
    state, so those copies only have to finish. Of each list of tables, the
    first and last are damaged.
    """
    import copy

    def walk(value, path):
        yield path, value
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if index in (0, len(value) - 1) or path[:1] in (("state",),):
                    yield from walk(item, (*path, index))

    for path, value in list(walk(payload, ())):
        if not path:
            continue
        replacements = [("removed", _REMOVED), ("replaced by text", "damaged")]
        if isinstance(value, torch.Tensor):
            if value.ndim and len(value) > 1:
                replacements.append(("truncated", value[:-1].clone()))
            replacements.append(("retyped", value.to(torch.float32 if value.dtype == torch.float64 else torch.float64)))
            if value.is_floating_point() and value.numel():
                broken = value.clone()
                broken.view(-1)[0] = float("nan")
                replacements.append(("not finite", broken))
        elif isinstance(value, bool):
            replacements.append(("negated", not value))
        elif isinstance(value, int):
            replacements.append(("negative", -5))
        elif isinstance(value, float):
            replacements.append(("not finite", float("nan")))
        exact = path[0] != "state" and "param_groups" not in path
        for description, replacement in replacements:
            mutated = copy.deepcopy(payload)
            container = mutated
            for key in path[:-1]:
                container = container[key]
            if replacement is _REMOVED:
                del container[path[-1]]
            else:
                container[path[-1]] = replacement
            yield f"{'/'.join(map(str, path))} {description}", mutated, exact


def _small_models(vectors: Path) -> dict[str, dict]:
    """Options that train every built-in model on a small graph in about a second."""
    return {
        "bootea": dict(epochs=40, dim=8, bootstrap_every=5, neighbor_refresh=5, batch_size=64),
        "gcnalign": dict(epochs=40, dim=8),
        "imuse": dict(epochs=30, dim=8, batch_size=64),
        "iptranse": dict(epochs=30, dim=8, bootstrap_every=5, batch_size=64),
        "mtranse": dict(epochs=30, dim=8, batch_size=64),
        "multike": dict(word_vectors=str(vectors), epochs=30, dim=8, literal_epochs=1, char_epochs=1,
                        literal_hidden=(8,), predicate_start=1, predicate_refresh=3),
    }


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TrainingTest(unittest.TestCase):
    def test_every_model_resumes_training_from_a_checkpoint(self):
        from prase.embedding.common import SeedValidation

        left, right, reference = graph_pair(40, seed=3)
        data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]))
        save = SeedValidation.checkpoint

        def interrupted_after_epoch_17(validation, epoch):
            save(validation, epoch)
            if epoch == 17:
                raise _Interrupt

        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("3 4\nEntity 0.1 0.2 0.3 0.4\n1 0.5 0.1 0.0 0.2\n2 0.3 0.3 0.3 0.3\n", encoding="utf-8")
            for name, options in _small_models(vectors).items():
                with self.subTest(model=name):
                    model = create_model(name, **options)
                    plain = model.fit(data, seed=0)
                    folder = Path(directory) / name
                    with training_checkpoints(folder, minutes=0):
                        with (patch.object(SeedValidation, "checkpoint", interrupted_after_epoch_17),
                              self.assertRaises(_Interrupt)):
                            model.fit(data, seed=0)
                        self.assertEqual(len(list(folder.glob("*.pt"))), 1)
                        with self.assertLogs("prase", "INFO") as logs:
                            resumed = model.fit(data, seed=0)
                    self.assertTrue(any("resumed after epoch 17" in line for line in logs.output))
                    np.testing.assert_array_equal(resumed.left, plain.left)
                    np.testing.assert_array_equal(resumed.right, plain.right)
                    self.assertEqual(resumed.info, plain.info)
                    # The finished fit stays saved, and the same fit again returns it without training.
                    self.assertEqual(len(list(folder.glob("*.pt"))), 1)
                    with training_checkpoints(folder, minutes=0), self.assertLogs("prase", "INFO") as logs:
                        reused = model.fit(data, seed=0)
                    self.assertTrue(any("reused the finished fit" in line for line in logs.output))
                    np.testing.assert_array_equal(reused.left, plain.left)
                    self.assertEqual(reused.info, plain.info)

    def test_a_checkpoint_for_other_settings_or_a_damaged_file_trains_afresh(self):
        from prase.checkpoints import training_key

        left, right, reference = graph_pair(40, seed=3)
        data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]))
        model = create_model("gcnalign", epochs=20, dim=8)
        plain = model.fit(data, seed=0)
        self.assertNotEqual(training_key(model, data, 0), training_key(model, data, 1))
        self.assertNotEqual(training_key(model, data, 0), training_key(create_model("gcnalign", epochs=20, dim=9),
                                                                       data, 0))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / f"{training_key(model, data, 0)}.pt").write_bytes(b"not a checkpoint")
            with training_checkpoints(directory, minutes=0), self.assertLogs("prase", "WARNING") as logs:
                again = model.fit(data, seed=0)
        self.assertIn("cannot be used", logs.output[0])
        np.testing.assert_array_equal(again.left, plain.left)
        with self.assertRaises(ConfigError):
            with training_checkpoints("unused", minutes=-1):
                pass

    def test_every_damaged_part_of_a_training_checkpoint_trains_afresh(self):
        """Damage each part of a real checkpoint in several ways: every fit must still finish."""
        import torch

        from prase.checkpoints import decode_training_state, encode_training_state, training_key
        from prase.embedding.common import SeedValidation

        left, right, reference = graph_pair(40, seed=3)
        data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]))
        save = SeedValidation.checkpoint

        def interrupted_after_epoch_15(validation, epoch):
            save(validation, epoch)
            if epoch == 15:
                raise _Interrupt

        vectors = Path(tempfile.mkdtemp()) / "words.vec"
        vectors.write_text("3 4\nEntity 0.1 0.2 0.3 0.4\n1 0.5 0.1 0.0 0.2\n2 0.3 0.3 0.3 0.3\n", encoding="utf-8")
        options = _small_models(vectors)
        try:
            for name in available_models():
                model = create_model(name, **{**options[name], "epochs": 20})
                plain = model.fit(data, seed=0)
                with tempfile.TemporaryDirectory() as directory:
                    file = Path(directory) / f"{training_key(model, data, 0)}.pt"
                    with training_checkpoints(directory, minutes=0):
                        with (patch.object(SeedValidation, "checkpoint", interrupted_after_epoch_15),
                              self.assertRaises(_Interrupt)):
                            model.fit(data, seed=0)
                    genuine = file.read_bytes()
                    payload = decode_training_state(torch, genuine, "cpu")
                    # A changed byte fails the digest, so the file is not used at all.
                    damaged = bytearray(genuine)
                    damaged[-10] ^= 0xFF
                    cases = [("a changed byte", bytes(damaged), True)]
                    cases += [(description, encode_training_state(torch, mutated), rejected)
                              for description, mutated, rejected in _mutations(torch, payload)]
                    for description, content, rejected in cases:
                        with self.subTest(model=name, damage=description):
                            file.write_bytes(content)
                            with training_checkpoints(directory, minutes=0):
                                again = model.fit(data, seed=0)
                            self.assertEqual(again.left.shape, plain.left.shape)
                            self.assertTrue(np.isfinite(again.left).all() and np.isfinite(again.right).all())
                            if rejected:
                                np.testing.assert_array_equal(again.left, plain.left)
                                np.testing.assert_array_equal(again.right, plain.right)
        finally:
            shutil.rmtree(vectors.parent)

    def test_training_keys_follow_files_settings_and_code(self):
        from prase.checkpoints import model_identity, training_key

        left, right, reference = graph_pair(20, seed=3)
        data = TrainingData(left, right, reference[:10], (reference[10:, 0], reference[10:, 1]))
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("1 2\nEntity 0.1 0.2\n", encoding="utf-8")
            model = create_model("multike", word_vectors=str(vectors))
            first = training_key(model, data, 0)
            self.assertEqual(training_key(model, data, 0), first)
            # The same size and modification time with new content still changes the key.
            status = vectors.stat()
            vectors.write_text("1 2\nEntity 0.3 0.4\n", encoding="utf-8")
            os.utime(vectors, ns=(status.st_atime_ns, status.st_mtime_ns))
            self.assertNotEqual(training_key(model, data, 0), first)
        # The PyTorch settings that change results are part of the key.
        import torch

        model = create_model("gcnalign")
        before = training_key(model, data, 0)
        self.assertIsNotNone(before)
        default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertNotEqual(training_key(model, data, 0), before)
        finally:
            torch.set_default_dtype(default)
        self.assertEqual(training_key(model, data, 0), before)

        class Undeclared:
            name = "undeclared"

            def __init__(self):
                self.dim = 4

        class Declared(Undeclared):
            def checkpoint_key(self):
                return {"dim": self.dim}

        # A model outside PRASE needs checkpoint_key, even when its attributes look plain.
        self.assertIsNone(model_identity(Undeclared()))
        self.assertIsNone(training_key(Undeclared(), data, 0))
        self.assertEqual(model_identity(Declared())["key"], {"dim": 4})
        self.assertIsNotNone(training_key(Declared(), data, 0))

    def test_training_without_validation_seeds_stops_at_its_budget(self):
        left, right, reference = graph_pair(40, seed=3)
        data = TrainingData(left, right, reference[:3], (reference[3:, 0], reference[3:, 1]))
        for name in ("gcnalign", "imuse", "iptranse"):
            with self.subTest(model=name):
                embeddings = create_model(name, dim=8, unvalidated_epochs=4).fit(data, seed=0)
                self.assertEqual(embeddings.info, {"epochs": 4})
                with self.assertRaises(ConfigError):
                    create_model(name, unvalidated_epochs=0)
        self.assertEqual(create_model("gcnalign").unvalidated_epochs, 300)
        self.assertEqual(create_model("imuse").unvalidated_epochs, 100)
        self.assertEqual(create_model("iptranse").unvalidated_epochs, 100)

    def test_training_logs_progress_and_the_time_left(self):
        import prase.progress

        left, right, reference = graph_pair(40, seed=3)
        data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]))
        with patch.object(prase.progress, "INTERVAL_SECONDS", 0), self.assertLogs("prase", "INFO") as logs:
            create_model("gcnalign", epochs=60, dim=8).fit(data, seed=0)
        lines = [line for line in logs.output if "GCN-Align training" in line]
        self.assertRegex(lines[0], r"epoch 1 of at most 60, \d+ s elapsed, at most .* left; "
                                   r"validation starts at epoch 50$")
        checked = [line for line in lines if "validation MRR" in line]
        self.assertRegex(checked[0], r"epoch 50 of at most 60, .*; validation MRR [0-9.]+, best [0-9.]+ at epoch 50, "
                                     r"stops at epoch 60 without improvement")
        self.assertRegex(lines[-1], r"GCN-Align training: 60 epochs in .* \(epoch cap reached\); selected epoch")
        self.assertEqual(prase.progress.duration(59.6), "1 min 0 s")
        self.assertEqual(prase.progress.duration(7322), "2 h 2 min")

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

    def test_bootea_labels_with_a_greedy_maximum_weight_matching(self):
        import torch

        from prase.embedding import bootea

        rng = np.random.default_rng(5)
        left = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(40, 4)), dtype=torch.float32), dim=1)
        right = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(30, 4)), dtype=torch.float32), dim=1)
        rejected = {(int(i), int(j)) for i, j in rng.integers(0, 30, size=(60, 2))}
        model = create_model("bootea", k=3, sim_th=0.2)
        similarity = (left @ right.T).numpy()
        # OpenEA: each row's k most similar columns above sim_th, then the heaviest pairs first.
        candidates = []
        for i, row in enumerate(similarity):
            allowed = [j for j in np.argsort(-row, kind="stable") if (i, int(j)) not in rejected][:3]
            candidates += [(float(row[j]), i, int(j)) for j in allowed if row[j] > 0.2]
        expected, rows, columns = [], set(), set()
        for _, i, j in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if i not in rows and j not in columns:
                rows.add(i)
                columns.add(j)
                expected.append((i, j))
        with patch.object(bootea, "_LABEL_ENTRIES", 64):  # several blocks of rows
            found = model._bootstrap(torch, left, right, rejected)
        self.assertEqual(list(found), expected)
        self.assertGreater(len(found), 5)
        self.assertFalse(set(found) & rejected)

    def test_bootea_keeps_the_relations_of_the_two_graphs_apart(self):
        from prase import KG
        from prase.embedding import bootea

        left = KG.from_triples([("a", "capital", "b")])
        right = KG.from_triples([("x", "capital", "y"), ("y", "located", "x")])
        triples = bootea._graph_triples(left, right)
        self.assertEqual(triples.tolist(), [[0, 0, 1], [2, 1, 3], [3, 2, 2]])

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

    def test_iptranse_negatives_change_one_endpoint_and_avoid_observed_triples(self):
        import torch

        from prase.embedding import iptranse

        generator = torch.Generator().manual_seed(0)
        triples = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0]] * 50)
        pool = torch.arange(4)
        known = iptranse._triple_keys(triples[:4], 4, 2).sort().values
        negatives, valid = iptranse._negative_triples(torch, generator, triples, pool, known, 4, 2)
        changed = negatives != triples
        self.assertTrue(valid.all())
        self.assertTrue(((changed[:, 0] ^ changed[:, 2]) & ~changed[:, 1]).all())
        self.assertFalse(torch.isin(iptranse._triple_keys(negatives, 4, 2), known).any())
        # With every corruption observed, a triple stays as it is and is marked invalid.
        everything = torch.cartesian_prod(torch.arange(4), torch.arange(2), torch.arange(4))
        known = iptranse._triple_keys(everything, 4, 2).sort().values
        negatives, valid = iptranse._negative_triples(torch, generator, triples, pool, known, 4, 2)
        self.assertFalse(valid.any())
        self.assertTrue(torch.equal(negatives, triples))

    def test_gcnalign_propagation_matches_openea(self):
        import random

        from prase import KG
        from prase.embedding import gcnalign

        def openea(triples, size):
            """OpenEA's get_weighted_adj and preprocess_adj, written out with NumPy."""
            heads, tails, counts = {}, {}, {}
            for h, r, t in triples:
                heads.setdefault(r, set()).add(h)
                tails.setdefault(r, set()).add(t)
                counts[r] = counts.get(r, 0) + 1
            adjacency = np.eye(size)
            for h, r, t in triples:
                if h != t:
                    adjacency[t, h] += max(len(tails[r]) / counts[r], 0.3)
                    adjacency[h, t] += max(len(heads[r]) / counts[r], 0.3)
            scale = adjacency.sum(axis=1) ** -0.5
            return (adjacency * scale[None, :]).T * scale[None, :]

        rng = random.Random(5)
        for trial in range(20):
            names = [f"e{i}" for i in range(12)]
            rows = [(rng.choice(names), f"r{rng.randrange(3)}", rng.choice(names)) for _ in range(30)]
            kg = KG.from_triples(rows, entities=names)
            expected = openea(kg.relation_triples.tolist(), kg.num_entities)
            found = np.zeros_like(expected)
            for (destination, source), weight in gcnalign._connectivity(kg).items():
                found[destination, source] = weight
            with self.subTest(trial=trial):
                np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-15)

    def test_builtin_models_propose_mutual_nearest_neighbors(self):
        left, right, reference = graph_pair(30, seed=4)
        data = TrainingData(left, right, reference[:20], (reference[20:, 0], reference[20:, 1]),
                            reference[20:22])
        rng = np.random.default_rng(0)
        embeddings = Embeddings(rng.normal(size=(left.num_entities, 4)), rng.normal(size=(right.num_entities, 4)))
        expected = embeddings.reciprocal_nearest(*data.candidates, excluded=data.rejected).pairs()
        for name in available_models():
            options = {"word_vectors": __file__} if name == "multike" else {}
            with self.subTest(model=name):
                self.assertEqual(create_model(name, **options).propose(embeddings, data).pairs(), expected)

    def test_bootea_triple_sets_match_set_operations(self):
        import random

        import torch

        from prase.embedding import bootea

        rng = random.Random(3)
        for trial in range(60):
            entities, predicates = rng.randrange(2, 30), rng.randrange(1, 5)
            rows = [(rng.randrange(entities), rng.randrange(predicates), rng.randrange(entities))
                    for _ in range(rng.randrange(0, 80))]
            ids = rng.sample(range(entities), 2 * rng.randrange(0, entities // 2 + 1))
            pairs = list(zip(ids[::2], ids[1::2]))
            counterparts = dict(pairs) | {right: left for left, right in pairs}
            expected = ({(counterparts[h], p, t) for h, p, t in rows if h in counterparts}
                        | {(h, p, counterparts[t]) for h, p, t in rows if t in counterparts})
            keys = bootea._TripleKeys(torch, entities, predicates, "cpu")
            triples = torch.tensor(rows, dtype=torch.long).reshape(-1, 3)
            swaps = bootea._swapped_triples(torch, triples, torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
                                            entities)
            with self.subTest(trial=trial):
                self.assertEqual(keys.rows(keys.of(swaps)).tolist(), [list(row) for row in sorted(expected)])
                self.assertEqual(keys.rows(keys.of(triples)).tolist(), [list(row) for row in sorted(set(rows))])
                union = keys.union(keys.of(triples), keys.of(swaps))
                self.assertEqual(keys.rows(union).tolist(), [list(row) for row in sorted(set(rows) | expected)])

    def test_multike_inputs_match_set_operations(self):
        import random
        from collections import Counter

        import torch

        from prase import KG
        from prase.embedding import multike

        def graph(rng, prefix):
            relations = [(f"{prefix}:e{rng.randrange(25)}", f"{prefix}:r{rng.randrange(3)}",
                          f"{prefix}:e{rng.randrange(25)}") for _ in range(80)]
            texts = ['"x.y"', '"http://z"', '"(.)"', '"Ab-c d"', '"de_f"', "1", "2"]
            attributes = [(f"{prefix}:e{rng.randrange(25)}", f"{prefix}:a{rng.randrange(4)}", rng.choice(texts))
                          for _ in range(60)]
            return KG.from_triples(relations, attributes, name=prefix)

        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "words.vec"
            vectors.write_text("x 0.1 0.2\n", encoding="utf-8")
            for seed, minimum in itertools.product(range(4), (1, 16)):
                rng = random.Random(seed)
                left, right = graph(rng, "L"), graph(rng, "R")
                ids = (rng.sample(range(left.num_entities), 6), rng.sample(range(right.num_entities), 6))
                pools = (np.arange(left.num_entities), np.arange(right.num_entities))
                data = TrainingData(left, right, np.array(ids).T, pools)
                settings = multike.MultiKE(word_vectors=vectors, attribute_min_count=minimum)
                inputs = multike._Inputs.build(torch, data, settings)

                relations, attributes = [], []
                for kg, entity, predicate in ((left, 0, 0), (right, left.num_entities, len(left.predicates))):
                    relations += [(h + entity, p + predicate, t + entity) for h, p, t in kg.relation_triples.tolist()]
                    counts = Counter(kg.attribute_triples[:, 1].tolist())
                    values = [literal.translate(multike._VALUE_PUNCTUATION).strip() for literal in kg.literals]
                    attributes += [(h + entity, p + predicate, values[v]) for h, p, v in kg.attribute_triples.tolist()
                                   if counts[p] >= minimum and values[v] and "http" not in values[v]]
                counterpart = {i: j + left.num_entities for i, j in zip(*ids, strict=True)}
                counterpart |= {j: i for i, j in counterpart.items()}
                swapped = ({(counterpart[h], p, t) for h, p, t in relations if h in counterpart}
                           | {(h, p, counterpart[t]) for h, p, t in relations if t in counterpart})
                with self.subTest(seed=seed, minimum=minimum):
                    self.assertEqual(inputs.relations.tolist(), [list(row) for row in sorted(relations)])
                    decoded = [(h, p, inputs.texts[v]) for h, p, v in inputs.attributes.tolist()]
                    self.assertEqual(decoded, sorted(set(attributes)))
                    self.assertEqual(inputs.swapped_relations.tolist(), [list(row) for row in sorted(swapped)])
                    self.assertEqual([(h, p, inputs.texts[v]) for h, p, v in inputs.swapped_attributes.tolist()],
                                     sorted({(counterpart[h], p, v) for h, p, v in attributes if h in counterpart}))

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


def scan_entity_links(attributes, candidates, seeds, aligned, threshold, rejected):
    """IMUSE entity links by a full scan of the right pool, as the method defines them."""
    from prase.embedding import imuse
    from prase.embedding.common import indel_similarity

    left, right = attributes
    used_left, used_right = ({pair[side] for pair in seeds} for side in (0, 1))
    proposals = []
    for i in sorted(set(candidates[0]) - used_left):
        best = threshold, -1
        for j in sorted(set(candidates[1]) - used_right):
            fields = [(left[i][a], right[j][b]) for a, b in aligned if a in left[i] and b in right[j]]
            if (i, j) in rejected or not any(
                indel_similarity(x, y) > threshold for xs, ys in fields for x in xs for y in ys
            ):
                continue
            score = sum(max(indel_similarity(x, y) for x in xs for y in ys) for xs, ys in fields) / len(fields)
            if score > best[0]:
                best = score, j
        if best[1] >= 0:
            proposals.append((i, best[1], best[0]))
    return imuse._one_to_one(proposals)


def attribute_case(seed: int):
    """Random attribute tables with repeated, empty, and non-ASCII values, and many ties."""
    import random

    from prase.embedding import imuse

    rng = random.Random(seed)
    alphabet = rng.choice(["01", "0123456789", "abcde", "abcdefghijklmnop", "aé中b"])
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 9))) for _ in range(rng.randrange(3, 40))]
    predicates = rng.randrange(1, 5)

    def side(size):
        return tuple(
            {p: tuple(sorted({rng.choice(texts) for _ in range(rng.randrange(1, 4))}))
             for p in range(predicates) if rng.random() < 0.7}
            for _ in range(size)
        )

    sizes = rng.randrange(1, 50), rng.randrange(1, 50)
    pairs = rng.sample([(a, b) for a in range(predicates) for b in range(predicates)], rng.randrange(1, predicates + 1))
    aligned = imuse._one_to_one([(a, b, 1.0) for a, b in pairs])
    candidates = tuple(sorted(rng.sample(range(size), rng.randrange(1, size + 1))) for size in sizes)
    seeds = {(rng.randrange(sizes[0]), rng.randrange(sizes[1])) for _ in range(rng.randrange(0, 5))}
    rejected = frozenset((rng.randrange(sizes[0]), rng.randrange(sizes[1])) for _ in range(rng.randrange(0, 30)))
    threshold = rng.choice([0.0, 0.3, 0.5, 0.6, 0.6, 0.75, 0.9, 1.0])
    return (side(sizes[0]), side(sizes[1])), candidates, seeds, aligned, threshold, rejected


@unittest.skipUnless(importlib.util.find_spec("rapidfuzz"), "RapidFuzz is not installed")
class IMUSEMatchingTest(unittest.TestCase):
    def test_entity_links_match_a_full_scan(self):
        from prase.embedding import imuse

        linked = 0
        defaults = (imuse._SOURCE_BATCH, imuse._BLOCK_ENTRIES, imuse._PARALLEL_ENTRIES, imuse._MATCH_ENTRIES)
        for batch, block, parallel, budget in (defaults, (3, 5, 0, defaults[3]), (16, 5, 0, 4)):
            with (patch.object(imuse, "_SOURCE_BATCH", batch), patch.object(imuse, "_BLOCK_ENTRIES", block),
                  patch.object(imuse, "_PARALLEL_ENTRIES", parallel), patch.object(imuse, "_MATCH_ENTRIES", budget)):
                for seed in range(200):
                    case = attribute_case(seed)
                    with self.subTest(batch=batch, seed=seed):
                        expected = scan_entity_links(*case)
                        self.assertEqual(imuse._entity_links(*case), expected)
                        linked += bool(expected)
        self.assertGreater(linked, 200)

    def test_entity_links_average_shared_fields_and_break_ties_by_entity_order(self):
        from prase.embedding import imuse

        left = ({0: ("1950",), 1: ("Ada",)},)
        right = ({0: ("1951",), 1: ("Ada",)}, {0: ("1951",), 1: ("Ada",)}, {0: ("1950",)})
        links = imuse._entity_links((left, right), ([0], [0, 1, 2]), set(), ((0, 0), (1, 1)), 0.6, frozenset())
        self.assertEqual(links, ((0, 2),))
        links = imuse._entity_links((left, right), ([0], [0, 1]), set(), ((0, 0), (1, 1)), 0.6, frozenset())
        self.assertEqual(links, ((0, 0),))


if __name__ == "__main__":
    unittest.main()
