import unittest
from dataclasses import replace

import numpy as np

from fixtures import graph_pair
from prase import KG, Alignment, Decisions, Embeddings, Paris, ParisConfig, evaluate
from prase.paris import native

PYTHON = ParisConfig(engine="python")


def one_to_one(alignment: Alignment) -> bool:
    return len(set(alignment.left.tolist())) == len(alignment) == len(set(alignment.right.tolist()))


class ParisTest(unittest.TestCase):
    def setUp(self):
        self.left, self.right, self.reference = graph_pair(80, seed=3)
        self.paris = Paris(self.left, self.right, PYTHON)

    def test_finds_the_planted_alignment(self):
        result = self.paris.run()
        scores = evaluate(result.alignment, self.reference)
        self.assertGreater(scores.f1, 0.85)
        self.assertTrue(one_to_one(result.alignment))
        self.assertEqual(len(result.iterations), PYTHON.iterations)
        self.assertTrue(any(match.left == "L:rel0" and match.right == "R:rel0" for match in result.predicates))

    def test_literals_seed_the_first_iteration(self):
        self.assertGreater(len(self.paris.literal_pairs), 0)
        state = self.paris.initial_state()
        literal = self.left.num_entities + int(self.paris.literal_pairs.left[0])
        self.assertGreaterEqual(state.left_match[literal], self.right.num_entities)
        self.assertEqual(state.left_prob[literal], 1.0)

    def test_approved_pairs_stay_fixed_and_rejected_pairs_never_match(self):
        baseline = {i: j for i, j, _ in self.paris.run().alignment}
        i, j = next(iter(baseline.items()))
        wrong = next(k for k in range(self.right.num_entities) if k != baseline.get(1) and k != j)
        decisions = Decisions(
            frozenset({(self.left.entities[1], self.right.entities[wrong])}),
            frozenset({(self.left.entities[i], self.right.entities[j])}),
        )
        result = self.paris.run(constraints=decisions.resolve(self.left, self.right))
        pairs = {a: b for a, b, _ in result.alignment}
        self.assertEqual(pairs[1], wrong)
        self.assertNotEqual(pairs.get(i), j)
        self.assertEqual(result.alignment.score[result.alignment.left == 1][0], 1.0)
        self.assertTrue(one_to_one(result.alignment))

    def test_seeds_start_as_counterparts(self):
        seeds = Alignment([0], [int(self.reference[0, 1])], [0.7])
        state = self.paris.initial_state(seeds)
        self.assertEqual(state.left_match[0], self.reference[0, 1])
        self.assertEqual(state.left_prob[0], 0.7)
        self.assertEqual(state.right_match[self.reference[0, 1]], 0)

    def test_embeddings_only_count_after_warm_up(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(self.left.num_entities, 8)).astype(np.float32)
        right_vectors = np.zeros((self.right.num_entities, 8), dtype=np.float32)
        right_vectors[self.reference[:, 1]] = vectors[self.reference[:, 0]]
        embeddings = Embeddings(vectors, right_vectors)
        warm_only = Paris(self.left, self.right, ParisConfig(engine="python", iterations=2)).run(embeddings=embeddings)
        plain = Paris(self.left, self.right, ParisConfig(engine="python", iterations=2)).run()
        self.assertEqual(list(warm_only.alignment), list(plain.alignment))
        fused = self.paris.run(embeddings=embeddings)
        self.assertGreaterEqual(evaluate(fused.alignment, self.reference).f1,
                                evaluate(self.paris.run().alignment, self.reference).f1)

    def test_candidates_explain_the_chosen_match(self):
        result = self.paris.run()
        entity, counterpart, _ = next(iter(result.alignment))
        candidates = self.paris.candidates(result, entity)
        self.assertTrue(candidates)
        self.assertEqual(candidates, sorted(candidates, key=lambda c: (-c.score, c.right)))
        self.assertIn(counterpart, [c.right for c in candidates])

    def test_runs_are_deterministic(self):
        first, second = self.paris.run(), self.paris.run()
        self.assertEqual(list(first.alignment), list(second.alignment))
        self.assertEqual(first.predicates, second.predicates)

    def test_graphs_without_shared_literals_align_nothing(self):
        left = KG.from_triples([("a", "p", "b")])
        right = KG.from_triples([("x", "q", "y")])
        result = Paris(left, right, PYTHON).run()
        self.assertEqual(len(result.alignment), 0)


@unittest.skipUnless(native.available(), f"native engine unavailable: {native.load_error()}")
class NativeParityTest(unittest.TestCase):
    def compare(self, left, right, *, config=PYTHON, threads=3, **run):
        python = Paris(left, right, config).run(**run)
        compiled = Paris(left, right, replace(config, engine="native", threads=threads)).run(**run)
        self.assertEqual(compiled.engine, "native")
        self.assertEqual(python.alignment.left.tolist(), compiled.alignment.left.tolist())
        self.assertEqual(python.alignment.right.tolist(), compiled.alignment.right.tolist())
        np.testing.assert_allclose(python.alignment.score, compiled.alignment.score, rtol=1e-12, atol=1e-12)
        for a, b in zip(python.predicates, compiled.predicates, strict=True):
            self.assertEqual((a.left, a.right), (b.left, b.right))
            self.assertAlmostEqual(a.left_in_right, b.left_in_right, places=12)
            self.assertAlmostEqual(a.right_in_left, b.right_in_left, places=12)
        self.assertEqual([s["matched"] for s in python.iterations], [s["matched"] for s in compiled.iterations])

    def test_matches_the_python_engine(self):
        for seed in range(4):
            left, right, _ = graph_pair(70, seed=seed, drop=0.2)
            self.compare(left, right, threads=1 + seed)

    def test_matches_with_seeds_embeddings_and_decisions(self):
        left, right, reference = graph_pair(70, seed=9, drop=0.3, named=0.3)
        rng = np.random.default_rng(1)
        embeddings = Embeddings(rng.normal(size=(left.num_entities, 6)), rng.normal(size=(right.num_entities, 6)))
        seeds = Alignment(reference[:10, 0], reference[:10, 1], np.linspace(0.3, 0.9, 10))
        decisions = Decisions(
            frozenset({(left.entities[reference[20, 0]], right.entities[reference[21, 1]])}),
            frozenset({(left.entities[reference[30, 0]], right.entities[reference[30, 1]])}),
        )
        self.compare(left, right, seeds=seeds, embeddings=embeddings, constraints=decisions.resolve(left, right))

    def test_matches_with_fixed_predicate_equivalences(self):
        left = KG.from_triples([("a", "p", "b")], [("b", "label", "shared")])
        right = KG.from_triples([("x", "q", "y")], [("y", "name", "shared"), ("z", "extra", "value")])
        self.compare(left, right, config=replace(PYTHON, warmup=0), fixed_predicates=[(0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
