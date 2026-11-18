"""Manual review guarantees, checked with random decisions against every method and neighbor search."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fixed_model import FixedModel
from fixtures import graph_pair
from prase import Alignment, Decisions, Embeddings, Load, ParisConfig, ParisStep, Pipeline, Predict, Session, methods

HAS_TORCH = importlib.util.find_spec("torch") is not None
FAST = ParisConfig(engine="python")
FORMATS = (np.array, list, set, lambda pairs: [], lambda pairs: None)


def brute_scores(left: np.ndarray, right: np.ndarray, metric: str, csls_k: int) -> np.ndarray:
    a, b = left.astype(np.float64), right.astype(np.float64)
    if metric in ("cosine", "csls"):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
    scores = -np.abs(a[:, None] - b[None]).sum(axis=2) if metric == "l1" else a @ b.T
    if metric == "csls":
        k = max(1, min(csls_k, len(a), len(b)))
        left_density = np.sort(scores, axis=1)[:, -k:].mean(axis=1)
        right_density = np.sort(scores.T, axis=1)[:, -k:].mean(axis=1)
        scores = 2 * scores - left_density[:, None] - right_density[None, :]
    return scores


def brute_nearest(scores, rows, columns, excluded, largest=True):
    """Pairs ``(rows[r], columns[c])`` of each row's best column that is not excluded."""
    found = []
    for r, row in enumerate(rows):
        values = np.array([(-np.inf if largest else np.inf) if (row, column) in excluded else scores[r, c]
                           for c, column in enumerate(columns)])
        best = int(np.argmax(values) if largest else np.argmin(values))
        if np.isfinite(values[best]):
            found.append((row, columns[best]))
    return found


def devices():
    if HAS_TORCH:
        import torch

        if torch.cuda.is_available():
            return ("cpu", "cuda")
    return ("cpu",)


class NeighborSearchTest(unittest.TestCase):
    def random_case(self, rng):
        left_vectors, right_vectors = rng.normal(size=(9, 3)), rng.normal(size=(8, 3))
        left = rng.choice(9, size=rng.integers(1, 10), replace=False)
        right = rng.choice(8, size=rng.integers(1, 9), replace=False)
        pairs = {(int(rng.choice(left)), int(rng.choice(right))) for _ in range(rng.integers(0, 12))}
        return left_vectors, right_vectors, left, right, pairs

    def test_embeddings_search_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for trial in range(40):
            left_vectors, right_vectors, left, right, pairs = self.random_case(rng)
            given = FORMATS[trial % len(FORMATS)](sorted(pairs))
            excluded = pairs if given is not None and len(given) else set()
            for metric in ("cosine", "inner", "l1", "csls"):
                scores = brute_scores(left_vectors[left], right_vectors[right], metric, 3)
                forward = brute_nearest(scores, left.tolist(), right.tolist(), excluded)
                backward = brute_nearest(scores.T, right.tolist(), left.tolist(), {(j, i) for i, j in excluded})
                mutual = sorted(set(forward) & {(i, j) for j, i in backward})
                for device in devices():
                    with self.subTest(trial=trial, metric=metric, device=device):
                        embeddings = Embeddings(left_vectors, right_vectors, metric, csls_k=3, device=device)
                        self.assertEqual(sorted(embeddings.nearest(left, right, excluded=given).pairs()),
                                         sorted(forward))
                        self.assertEqual(sorted(embeddings.reciprocal_nearest(left, right, excluded=given).pairs()),
                                         mutual)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_training_search_matches_brute_force(self):
        import torch

        from prase.embedding.common import nearest_neighbors

        rng = np.random.default_rng(1)
        for trial in range(40):
            left_vectors, right_vectors, _, _, _ = self.random_case(rng)
            rows, columns = len(left_vectors), len(right_vectors)
            excluded = {(int(rng.integers(rows)), int(rng.integers(columns))) for _ in range(rng.integers(0, 30))}
            given = FORMATS[trial % 3](sorted(excluded))
            scores = left_vectors.astype(np.float32).astype(np.float64) @ right_vectors.astype(np.float32).T
            for largest in (True, False):
                forward = brute_nearest(scores, list(range(rows)), list(range(columns)), excluded, largest)
                backward = dict(brute_nearest(scores.T, list(range(columns)), list(range(rows)),
                                              {(j, i) for i, j in excluded}, largest))
                for reciprocal in (False, True):
                    expected = [(i, j) for i, j in forward if not reciprocal or backward.get(j) == i]
                    for device in devices():
                        for block_size in (1, 2, 3, 7, 64):
                            with self.subTest(trial=trial, largest=largest, reciprocal=reciprocal, device=device,
                                              block_size=block_size):
                                found = nearest_neighbors(
                                    torch, torch.tensor(left_vectors, dtype=torch.float32, device=device),
                                    torch.tensor(right_vectors, dtype=torch.float32, device=device),
                                    lambda a, b: a @ b.T, block_size, largest=largest, reciprocal=reciprocal,
                                    excluded=given,
                                )
                                self.assertEqual([(i, j) for i, j, _ in found], expected)


class ReviewInvariantTest(unittest.TestCase):
    """Approved pairs stay, rejected pairs never appear, and no selection is lost to an excluded pair."""

    def scenario(self, trial):
        left, right, reference = graph_pair(30, seed=trial)
        rng = np.random.default_rng(trial)
        lefts, rights = rng.permutation(left.num_entities), rng.permutation(right.num_entities)
        approved = {(int(i), int(j)) for i, j in zip(lefts[:3], rights[:3])}
        rejected = {tuple(pair) for pair in reference[rng.choice(len(reference), 4, replace=False)].tolist()}
        rejected |= {(int(rng.integers(left.num_entities)), int(rng.integers(right.num_entities))) for _ in range(4)}
        rejected -= approved
        (a, x), stale = next(iter(sorted(approved))), int(rights[3])
        seeds = Alignment.from_pairs(
            [(a, stale, 1.0), (int(lefts[4]), x, 1.0), *(next(iter(sorted(rejected))) + (1.0,),),
             *((int(i), int(j), 1.0) for i, j in reference[rng.choice(len(reference), 4, replace=False)])],
        )
        proposals = Alignment.from_pairs(
            [(int(rng.integers(left.num_entities)), int(rng.integers(right.num_entities)), float(rng.random()))
             for _ in range(20)]
            # An approved entity's proposal must not hide a weaker proposal for the same target.
            + [(a, int(rights[5]), 0.95), (int(lefts[6]), int(rights[5]), 0.8), (*next(iter(sorted(rejected))), 0.99)],
        )
        decisions = Decisions(
            frozenset((left.entities[i], right.entities[j]) for i, j in approved),
            frozenset((left.entities[i], right.entities[j]) for i, j in rejected),
        )
        return left, right, reference, decisions, approved, rejected, seeds, proposals

    def check(self, result, approved, rejected, one_to_one):
        pairs = result.alignment.pairs()
        self.assertLessEqual(approved, pairs)
        self.assertFalse(pairs & rejected)
        approved_left, approved_right = {i for i, _ in approved}, {j for _, j in approved}
        for i, j in pairs - approved:
            self.assertNotIn(i, approved_left)
            self.assertNotIn(j, approved_right)
        if one_to_one:
            self.assertEqual(len({i for i, _ in pairs}), len(pairs))
            self.assertEqual(len({j for _, j in pairs}), len(pairs))
        return pairs

    def test_every_method_honors_random_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            for trial in range(12):
                left, right, reference, decisions, approved, rejected, seeds, proposals = self.scenario(trial)
                saved = Path(directory) / f"proposals-{trial}.tsv"
                proposals.write(saved, left, right)
                vectors = Path(directory) / f"vectors-{trial}.vec"
                rng = np.random.default_rng(100 + trial)
                tokens = {"Entity", "L", "R", *(f"e{k}" for k in range(30)), *(str(k) for k in range(30))}
                vectors.write_text("".join(f"{token} {' '.join(map(str, rng.normal(size=3)))}\n" for token in tokens),
                                   encoding="utf-8")
                trained = FixedModel(reference=reference)
                supervised = FixedModel()
                runs = {
                    "paris": (methods.paris(FAST), True),
                    "prase": (methods.prase(trained, config=FAST), True),
                    "embedding-only": (methods.embedding_only(supervised), True),
                    "load-predict": (Pipeline([Load(saved), Predict()]), True),
                    "load-paris": (Pipeline([Load(saved), ParisStep(FAST)]), True),
                    "str-match": (methods.name_match(0.3), False),
                    "emb-match": (methods.name_embedding_match(vectors, 0.3), False),
                }
                for name, (pipeline, one_to_one) in runs.items():
                    with self.subTest(trial=trial, method=name):
                        result = pipeline.run(left, right, seeds=seeds, decisions=decisions)
                        pairs = self.check(result, approved, rejected, one_to_one)
                        if name == "load-predict":
                            used_left, used_right = {i for i, _ in pairs}, {j for _, j in pairs}
                            for i, j, _ in proposals:
                                if (i, j) not in rejected:
                                    self.assertTrue(i in used_left or j in used_right, (i, j))
                for data in (*trained.seen, *supervised.seen):
                    with self.subTest(trial=trial, training=len(data.seeds)):
                        seed_pairs = set(data.seed_pairs)
                        self.assertLessEqual(approved, seed_pairs)
                        self.assertFalse(seed_pairs & rejected)
                        self.assertEqual(data.rejected_pairs, rejected)
                        self.assertFalse(set(data.candidates[0].tolist()) & {i for i, _ in seed_pairs})
                        self.assertFalse(set(data.candidates[1].tolist()) & {j for _, j in seed_pairs})

    def test_review_candidates_leave_out_impossible_counterparts(self):
        for trial in range(6):
            left, right, _, decisions, approved, rejected, _, _ = self.scenario(trial)
            session = Session(left, right, FAST)
            for a, b in decisions.approved:
                session.approve(a, b)
            for a, b in decisions.rejected:
                session.reject(a, b)
            session.run()
            owner = {j: i for i, j in approved}
            for i, entity in enumerate(left.entities):
                for counterpart, _ in session.candidates(entity, limit=50):
                    j = right.entity_index[counterpart]
                    with self.subTest(trial=trial, left=entity, right=counterpart):
                        self.assertNotIn((i, j), rejected)
                        self.assertEqual(owner.get(j, i), i)


if __name__ == "__main__":
    unittest.main()
