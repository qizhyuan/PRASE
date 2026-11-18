import unittest

import numpy as np

from fixtures import graph_pair
from prase import DataError, Embeddings, KG, ParisConfig, Predicate, PredicateMatch, Session


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.left, self.right, self.reference = graph_pair(60, seed=8)
        self.session = Session(self.left, self.right, ParisConfig(engine="python"))

    def test_review_loop(self):
        with self.assertRaises(DataError):
            self.session.matches()
        self.session.run()
        matches = self.session.matches()
        left_id, right_id, _ = matches[0]
        other = next(entity for entity in self.right.entities if entity != right_id)

        self.session.reject(left_id, right_id)
        self.session.approve(self.left.entities[0], other)
        self.session.run()
        pairs = {(a, b) for a, b, _ in self.session.matches()}
        self.assertNotIn((left_id, right_id), pairs)
        self.assertIn((self.left.entities[0], other), pairs)
        self.assertEqual(self.session.decisions.rejected, {(left_id, right_id)})

        self.session.forget(left_id, right_id)
        self.session.forget(self.left.entities[0], other)
        self.session.run()
        self.assertIn((left_id, right_id), {(a, b) for a, b, _ in self.session.matches()})

    def test_queries(self):
        self.session.seed(self.left.entities[1], self.right.entities[2], 0.4)
        result = self.session.run()
        left_unaligned, right_unaligned = self.session.unaligned()
        self.assertEqual(len(left_unaligned), self.left.num_entities - len(result.alignment))
        self.assertEqual(len(right_unaligned), self.right.num_entities - len(result.alignment))
        entity, counterpart, _ = self.session.matches()[0]
        self.assertIn(counterpart, [right for right, _ in self.session.candidates(entity)])
        self.assertTrue(self.session.predicate_matches(0.1))
        with self.assertRaises(DataError):
            self.session.seed("missing", self.right.entities[0])
        with self.assertRaisesRegex(DataError, "unknown left entity"):
            self.session.candidates("missing")

    def test_embeddings_add_similarity_to_candidates(self):
        with self.assertRaisesRegex(DataError, "one vector per entity"):
            self.session.set_embeddings(Embeddings(np.ones((2, 3)), np.ones((2, 3))))
        self.session.set_embeddings(Embeddings(np.ones((self.left.num_entities, 3)),
                                               np.ones((self.right.num_entities, 3))))
        self.session.run()
        entity = self.session.matches()[0][0]
        self.assertTrue(all(candidate.similarity == 1.0 for _, candidate in self.session.candidates(entity)))
        self.session.set_embeddings(None)
        self.session.run()
        self.assertTrue(all(candidate.similarity is None for _, candidate in self.session.candidates(entity)))

    def test_fixed_predicates_supply_evidence_and_can_be_forgotten(self):
        left = KG.from_triples([("a", "p", "b")])
        right = KG.from_triples([("x", "q", "y")])
        session = Session(left, right, ParisConfig(engine="python", warmup=0, iterations=3))
        session.seed("b", "y")
        self.assertEqual(len(session.run().alignment), 1)

        session.fix_predicate("p", "q")
        result = session.run()
        self.assertEqual([step["matched"] for step in result.iterations], [2, 2, 2])
        self.assertEqual(session.predicate_matches(), (
            PredicateMatch("p", "q", 1.0, 1.0),
            PredicateMatch("p^-1", "q^-1", 1.0, 1.0),
        ))
        self.assertEqual([name for name, _ in session.candidates("a")], ["x"])
        self.assertEqual(session.run().predicates, result.predicates)

        session.forget_predicate("p", "q")
        self.assertEqual(len(session.run().alignment), 1)
        self.assertFalse(session.predicate_matches())

    def test_fixed_predicates_use_names_or_typed_predicates(self):
        left = KG.from_triples([("a", "p", "b")], [("a", "p", "value")])
        right = KG.from_triples([("x", "q", "y")], [("x", "label", "value")])
        session = Session(left, right, ParisConfig(engine="python", warmup=0))
        with self.assertRaisesRegex(DataError, "ambiguous left predicate"):
            session.fix_predicate("p", "q")
        with self.assertRaisesRegex(DataError, "unknown right predicate"):
            session.fix_predicate(Predicate("p", "relation"), "missing")
        with self.assertRaisesRegex(DataError, "same kind"):
            session.fix_predicate(Predicate("p", "relation"), "label")

        session.fix_predicate(Predicate("p", "attribute"), "label")
        self.assertIn(("a", "x"), {(a, b) for a, b, _ in session.run().alignment.to_ids(left, right)})
        self.assertIn(PredicateMatch("p^-1", "label^-1", 1.0, 1.0), session.predicate_matches())


if __name__ == "__main__":
    unittest.main()
