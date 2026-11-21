import unittest
from collections import defaultdict
from dataclasses import replace

import numpy as np

from fixtures import graph_pair
from prase import KG, Alignment, Decisions, Embeddings, Paris, ParisConfig, Predicate, evaluate
from prase.paris import exact_literals, native

PYTHON = ParisConfig(engine="python")


def one_to_one(alignment: Alignment) -> bool:
    return len(set(alignment.left.tolist())) == len(alignment) == len(set(alignment.right.tolist()))


def facts(kg: KG) -> list[tuple[int, int, int]]:
    """Facts with inverses, as PARIS reads them. Literals are numbered after the entities."""
    n, k = kg.num_entities, len(kg.predicates)
    rows = []
    for h, p, t in kg.relation_triples.tolist():
        rows += [(h, p, t), (t, p + k, h)]
    for e, p, v in kg.attribute_triples.tolist():
        rows += [(e, p, n + v), (n + v, p + k, e)]
    return rows


def functionality(rows: list[tuple[int, int, int]]) -> dict[int, float]:
    """Distinct heads per fact of each predicate."""
    heads, count = defaultdict(set), defaultdict(int)
    for h, p, _ in rows:
        heads[p].add(h)
        count[p] += 1
    return {p: len(heads[p]) / count[p] for p in count}


def equation_13(paris: Paris, result, entity: int, *, superrelation: bool = True) -> dict[int, float]:
    """Structural evidence for each counterpart of ``entity``, written out from PARIS Eq. 13.

    A left fact r(h, x) and a right fact s(h', y) with h ≡ h' support x ≡ y through
    P(r ⊆ s) with the functionality of s, and through P(s ⊆ r) with that of r.
    ``superrelation=False`` pairs each inclusion with its subrelation instead.
    """
    config, state = paris.config, result.state
    eps, theta = config.epsilon, config.theta
    left_facts, right_facts = facts(paris.left), facts(paris.right)
    fun_left, fun_right = functionality(left_facts), functionality(right_facts)
    left_in_right = {(a, b): v for a, b, v in state.left_relations.items()}
    right_in_left = {(a, b): v for b, a, v in state.right_relations.items()}
    by_head = defaultdict(list)
    for h, s, y in right_facts:
        if y < paris.right.num_entities:
            by_head[h].append((s, y))
    product = defaultdict(lambda: 1.0)
    for h, r, x in left_facts:
        anchor, m = state.left_match[h], state.left_prob[h]
        if x != entity or anchor < 0 or m < theta:
            continue
        for s, y in by_head[int(anchor)]:
            r_in_s, s_in_r = left_in_right.get((r, s), 0.0) / eps, right_in_left.get((r, s), 0.0) / eps
            if r_in_s < theta and s_in_r < theta:
                continue
            fun_s, fun_r = fun_right[s] / eps, fun_left[r] / eps
            if not superrelation:
                fun_s, fun_r = fun_r, fun_s
            factor = (1.0 - m * r_in_s * fun_s) * (1.0 - m * s_in_r * fun_r)
            if 1.0 - factor > config.delta:
                product[y] *= factor
    return {y: 1.0 - value for y, value in product.items()}


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

    def test_evidence_follows_equation_13(self):
        left, right, _ = graph_pair(60, seed=4, merged=True)
        paris = Paris(left, right, PYTHON)
        result = paris.run()
        checked = distinguished = 0
        for entity in range(left.num_entities):
            expected = equation_13(paris, result, entity)
            found = {c.right: c.structural for c in paris.candidates(result, entity, limit=right.num_entities)}
            self.assertEqual(found.keys(), expected.keys())
            for y, value in expected.items():
                self.assertAlmostEqual(found[y], value, places=12)
            swapped = equation_13(paris, result, entity, superrelation=False)
            checked += len(expected)
            distinguished += any(abs(swapped.get(y, 0.0) - v) > 1e-3 for y, v in expected.items())
        self.assertGreater(checked, 0)
        # The merged right relation makes inclusions and functionalities asymmetric,
        # so pairing an inclusion with its subrelation gives different evidence.
        self.assertGreater(distinguished, 0)

    def test_runs_are_deterministic(self):
        first, second = self.paris.run(), self.paris.run()
        self.assertEqual(list(first.alignment), list(second.alignment))
        self.assertEqual(first.predicates, second.predicates)

    def test_right_matches_follow_the_current_left_matches(self):
        from types import SimpleNamespace

        from prase.paris.python import _one_to_one

        # Left 0 held right 0 at 0.9 and has moved to right 1; left 1 now points at right 0 at 0.4.
        state = SimpleNamespace(
            left_match=np.array([1, 0, 1]), left_prob=np.array([0.9, 0.4, 0.5]),
            right_match=np.array([0, -1]), right_prob=np.array([0.9, 0.0]),
        )
        _one_to_one(state, 3, 2)
        self.assertEqual(state.right_match.tolist(), [1, 0])
        self.assertEqual(state.right_prob.tolist(), [0.4, 0.9])
        self.assertEqual(state.left_match.tolist(), [1, 0, -1])
        self.assertEqual(state.left_prob.tolist(), [0.9, 0.4, 0.0])

    def test_embeddings_need_one_row_per_entity(self):
        from prase import DataError

        engines = ["python", *(["native"] if native.available() else [])]
        short = Embeddings(np.ones((self.left.num_entities - 1, 4)), np.ones((self.right.num_entities, 4)))
        for name in engines:
            with self.subTest(engine=name), self.assertRaisesRegex(DataError, "rows"):
                Paris(self.left, self.right, ParisConfig(engine=name)).run(embeddings=short)

    def test_a_zero_probability_never_claims_a_counterpart(self):
        from types import SimpleNamespace

        from prase.paris.python import _one_to_one

        state = SimpleNamespace(left_match=np.array([0, 1]), left_prob=np.array([0.0, 0.5]),
                                right_match=np.array([-1, -1]), right_prob=np.zeros(2))
        _one_to_one(state, 2, 2)
        self.assertEqual(state.right_match.tolist(), [-1, 1])
        self.assertEqual(state.left_match.tolist(), [-1, 1])
        # On this pair, negative cosine similarity clips one entity's only candidates to zero.
        left, right, _ = graph_pair(16, seed=65, drop=0.45911108876979334, named=0.1874292816856471)
        rng = np.random.default_rng(65)
        rng.integers(12, 40), rng.uniform(0.2, 0.7), rng.uniform(0, 0.5)
        embeddings = Embeddings(rng.normal(size=(left.num_entities, 4)), rng.normal(size=(right.num_entities, 4)))
        for name in ["python", *(["native"] if native.available() else [])]:
            with self.subTest(engine=name):
                found = Paris(left, right, ParisConfig(engine=name)).run(embeddings=embeddings).alignment
                self.assertTrue((found.score > 0).all())
                self.assertEqual(len(found), 12)

    def test_seeds_and_decisions_must_name_entities_of_the_graphs(self):
        from prase import DataError
        from prase.decisions import Constraints

        right = int(self.reference[0, 1])
        for what, seeds, constraints in (
            ("seeds", Alignment([-1], [right]), None),
            ("seeds", Alignment([0], [self.right.num_entities]), None),
            ("approved", None, Constraints(approved=np.array([[self.left.num_entities, right]]))),
            ("rejected", None, Constraints(rejected=frozenset({(0, -1)}))),
        ):
            with self.subTest(what=what), self.assertRaisesRegex(DataError, f"{what}.*entity indices"):
                self.paris.run(seeds=seeds, constraints=constraints)

    def test_literal_matches_must_name_literals_with_probabilities(self):
        from prase import DataError

        for name in ["python", *(["native"] if native.available() else [])]:
            for pairs, message in (
                (Alignment([-1], [-1]), "literal indices"),
                (Alignment([0], [len(self.right.literals)]), "literal indices"),
                (Alignment([0], [0], [np.nan]), "probabilities"),
                (Alignment([0], [0], [1.5]), "probabilities"),
            ):
                with self.subTest(engine=name, pairs=pairs.pairs()), self.assertRaisesRegex(DataError, message):
                    Paris(self.left, self.right, ParisConfig(engine=name), literal_matcher=lambda left, right: pairs)
        with self.assertRaisesRegex(DataError, "Alignment"):
            Paris(self.left, self.right, PYTHON, literal_matcher=lambda left, right: [(0, 0, 1.0)])

    def test_zero_probability_evidence_keeps_estimates_finite(self):
        # theta = 0 and smoothing = 0 with a literal pair of probability 0 and a positive seed.
        right = int(self.reference[0, 1])
        config = replace(PYTHON, theta=0.0, smoothing=0.0)
        literals = exact_literals(self.left, self.right)
        zero = Alignment(literals.left, literals.right, np.zeros(len(literals)))
        seeds = Alignment([0], [right], [0.8])
        results = []
        for name in ["python", *(["native"] if native.available() else [])]:
            with self.subTest(engine=name):
                result = Paris(self.left, self.right, replace(config, engine=name),
                               literal_matcher=lambda left, right: zero).run(seeds=seeds)
                self.assertTrue(np.isfinite(result.alignment.score).all())
                self.assertTrue(np.isfinite(result.state.left_relations.values).all())
                self.assertTrue(all(np.isfinite([m.left_in_right, m.right_in_left]).all() for m in result.predicates))
                results.append(result)
        if len(results) == 2:
            self.assertEqual(results[0].alignment.pairs(), results[1].alignment.pairs())

    def test_candidates_need_a_left_entity_index_and_a_positive_limit(self):
        from prase import ConfigError, DataError

        result = self.paris.run()
        for entity in (-1, -2, self.left.num_entities, 1.0, True, "0"):
            with self.subTest(entity=entity), self.assertRaisesRegex(DataError, "left entity index"):
                self.paris.candidates(result, entity)
        for limit in (0, -3, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ConfigError, "limit"):
                self.paris.candidates(result, 0, limit=limit)
        self.assertEqual(self.paris.candidates(result, np.int64(3)), self.paris.candidates(result, 3))

    def test_epsilon_below_one_is_rejected(self):
        from prase import ConfigError

        for value in (0.5, 0.999, 1e-9):
            with self.subTest(epsilon=value), self.assertRaisesRegex(ConfigError, "at least 1"):
                ParisConfig(epsilon=value)
        for name in ["python", *(["native"] if native.available() else [])]:
            with self.subTest(engine=name):
                result = Paris(self.left, self.right, ParisConfig(engine=name, epsilon=1.0)).run()
                self.assertTrue(np.isfinite(result.alignment.score).all())
                self.assertTrue(((result.alignment.score >= 0) & (result.alignment.score <= 1)).all())

    def test_graphs_keep_each_triple_once(self):
        graph = KG(["a", "b"], [Predicate("p", "relation"), Predicate("q", "attribute")], ["x"],
                   [(0, 0, 1), (1, 0, 0), (0, 0, 1)], [(0, 1, 0), (0, 1, 0)])
        self.assertEqual(graph.relation_triples.tolist(), [[0, 0, 1], [1, 0, 0]])
        self.assertEqual(graph.attribute_triples.tolist(), [[0, 1, 0]])
        twice = KG(self.left.entities, self.left.predicates, self.left.literals,
                   np.concatenate((self.left.relation_triples, self.left.relation_triples)),
                   self.left.attribute_triples)
        self.assertEqual(twice.fingerprint, self.left.fingerprint)
        for name in ["python", *(["native"] if native.available() else [])]:
            with self.subTest(engine=name):
                once = Paris(self.left, self.right, ParisConfig(engine=name)).run()
                again = Paris(twice, self.right, ParisConfig(engine=name)).run()
                self.assertEqual(once.alignment.pairs(), again.alignment.pairs())

    def test_settings_must_be_finite_numbers(self):
        from prase import ConfigError

        for name in ("theta", "delta", "epsilon", "smoothing", "embedding_weight"):
            for value in (float("nan"), float("inf"), "0.5", True):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ConfigError, name):
                    ParisConfig(**{name: value})
        self.assertEqual(ParisConfig(smoothing=np.float64(5.0)).smoothing, 5.0)

    def test_zero_iterations_return_one_to_one_positive_matches(self):
        a, b = (int(value) for value in self.reference[:2, 1])
        seeds = Alignment([0, 1, 2, 3], [a, a, b, int(self.reference[3, 1])], [0.6, 0.9, 0.0, 0.4])
        for name in ["python", *(["native"] if native.available() else [])]:
            with self.subTest(engine=name):
                found = Paris(self.left, self.right, ParisConfig(engine=name, iterations=0)).run(seeds=seeds)
                self.assertEqual(sorted(found.alignment.pairs()), [(1, a), (3, int(self.reference[3, 1]))])
                self.assertTrue((found.alignment.score > 0).all())

    def test_seed_scores_must_be_finite(self):
        from prase import DataError

        seeds = Alignment([0, 1], [int(self.reference[0, 1]), int(self.reference[1, 1])], [np.nan, 0.8])
        with self.assertRaisesRegex(DataError, "finite"):
            self.paris.run(seeds=seeds)

    def test_an_empty_right_graph_aligns_nothing(self):
        left = KG.from_triples([], [("a", "name", "x")])
        right = KG.from_triples([], [], entities=[])
        for name in ["python", *(["native"] if native.available() else [])]:
            with self.subTest(engine=name):
                self.assertEqual(len(Paris(left, right, ParisConfig(engine=name)).run().alignment), 0)

    def test_graphs_without_shared_literals_align_nothing(self):
        left = KG.from_triples([("a", "p", "b")])
        right = KG.from_triples([("x", "q", "y")])
        result = Paris(left, right, PYTHON).run()
        self.assertEqual(len(result.alignment), 0)


@unittest.skipUnless(native.available(), f"native engine unavailable: {native.load_error()}")
class NativeParityTest(unittest.TestCase):
    def compare(self, left, right, *, config=PYTHON, threads=3, **run):
        """Both engines, at one and at ``threads`` threads, give the same results to the last bit."""
        python = Paris(left, right, config).run(**run)
        for count in sorted({1, threads}):
            compiled = Paris(left, right, replace(config, engine="native", threads=count)).run(**run)
            self.assertEqual(compiled.engine, "native")
            self.assertEqual(python.alignment.left.tolist(), compiled.alignment.left.tolist())
            self.assertEqual(python.alignment.right.tolist(), compiled.alignment.right.tolist())
            np.testing.assert_array_equal(python.alignment.score, compiled.alignment.score)
            self.assertEqual(python.predicates, compiled.predicates)
            np.testing.assert_array_equal(python.state.left_relations.values, compiled.state.left_relations.values)
            np.testing.assert_array_equal(python.state.right_relations.values, compiled.state.right_relations.values)
            self.assertEqual([s["matched"] for s in python.iterations], [s["matched"] for s in compiled.iterations])

    def test_matches_the_python_engine(self):
        for seed in range(4):
            left, right, _ = graph_pair(70, seed=seed, drop=0.2, merged=seed % 2 == 1)
            self.compare(left, right, threads=1 + seed)
        # A larger, sparser pair in which left entities often move to another counterpart.
        left, right, _ = graph_pair(200, seed=7, drop=0.4, merged=True)
        self.compare(left, right, threads=4)

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

    def test_matches_with_tied_embeddings_at_every_thread_count(self):
        # Right entities share vectors in pairs, so cosine scores tie exactly and break by the lower index.
        left, right, reference = graph_pair(90, seed=11, drop=0.35, named=0.3)
        rng = np.random.default_rng(3)
        shared = rng.normal(size=((right.num_entities + 1) // 2, 5))
        right_vectors = np.repeat(shared, 2, axis=0)[:right.num_entities]
        left_vectors = rng.normal(size=(left.num_entities, 5))
        left_vectors[reference[:, 0]] = right_vectors[reference[:, 1]]
        embeddings = Embeddings(left_vectors, right_vectors)
        for threads in (2, 3, 4):
            with self.subTest(threads=threads):
                self.compare(left, right, config=replace(PYTHON, embedding_weight=0.5), threads=threads,
                             embeddings=embeddings)

    def test_matches_with_fixed_predicate_equivalences(self):
        left = KG.from_triples([("a", "p", "b")], [("b", "label", "shared")])
        right = KG.from_triples([("x", "q", "y")], [("y", "name", "shared"), ("z", "extra", "value")])
        self.compare(left, right, config=replace(PYTHON, warmup=0), fixed_predicates=[(0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
