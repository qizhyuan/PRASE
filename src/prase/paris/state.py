"""PARIS state and graph indexes shared by the Python and native engines.

Terms of a graph are its entities followed by its literals, so literal ``k``
is term ``num_entities + k``. Each predicate ``p`` also has an inverse
``p + P``, where ``P`` is the predicate count, so every fact can be read in
both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..kg import KG


class RelationTable:
    """Sparse predicate-subsumption probabilities ``P(a ⊆ b)``, keyed by ``a * width + b``."""

    __slots__ = ("keys", "values", "width")

    def __init__(self, keys: np.ndarray, values: np.ndarray, width: int) -> None:
        self.keys = np.asarray(keys, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)
        self.width = width

    @classmethod
    def empty(cls, width: int) -> RelationTable:
        return cls(np.zeros(0, dtype=np.int64), np.zeros(0), width)

    def lookup(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not len(self.keys):
            return np.zeros(len(a))
        keys = a * self.width + b
        index = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[index] == keys, self.values[index], 0.0)

    def with_fixed(self, fixed: np.ndarray) -> RelationTable:
        """Include every fixed key at probability 1, overriding any estimate."""
        if not len(fixed):
            return self
        keys = np.union1d(self.keys, fixed)
        values = np.ones(len(keys))
        values[np.searchsorted(keys, self.keys)] = self.values
        values[np.searchsorted(keys, fixed)] = 1.0
        return RelationTable(keys, values, self.width)

    def items(self) -> list[tuple[int, int, float]]:
        return [(key // self.width, key % self.width, value) for key, value in zip(
            self.keys.tolist(), self.values.tolist(), strict=True)]


class GraphIndex:
    """Fact tables of one graph, sorted for the two roles a graph plays in a pass.

    As the *source*, the entity being scored is the tail of a fact, so facts
    are grouped by tail. As the *target*, candidates are tails reached from a
    matched head, so facts are grouped by ``(head, predicate)``. Facts whose
    tail is a literal are never needed in either role.
    """

    def __init__(self, kg: KG) -> None:
        self.num_entities = kg.num_entities
        self.num_terms = kg.num_entities + kg.num_literals
        self.num_predicates = 2 * len(kg.predicates)
        offset = len(kg.predicates)
        relations, attributes = kg.relation_triples, kg.attribute_triples
        literal_terms = attributes[:, 2] + kg.num_entities
        heads = np.concatenate((relations[:, 0], relations[:, 2], attributes[:, 0], literal_terms))
        preds = np.concatenate((relations[:, 1], relations[:, 1] + offset, attributes[:, 1], attributes[:, 1] + offset))
        tails = np.concatenate((relations[:, 2], relations[:, 0], literal_terms, attributes[:, 0]))

        # Functionality of q: distinct heads per fact. With inverses this also
        # gives the inverse functionality of every original predicate.
        counts = np.bincount(preds, minlength=self.num_predicates)
        distinct = np.unique(preds * self.num_terms + heads) // self.num_terms
        distinct_heads = np.bincount(distinct, minlength=self.num_predicates)
        self.functionality = np.divide(
            distinct_heads, counts, out=np.zeros(self.num_predicates), where=counts > 0,
        )

        entity_tail = tails < self.num_entities
        heads, preds, tails = heads[entity_tail], preds[entity_tail], tails[entity_tail]

        order = np.lexsort((heads, preds, tails))
        self.source_tail = tails[order]
        self.source_pred = preds[order]
        self.source_head = heads[order]
        self.source_ptr = _pointers(self.source_tail, self.num_entities)

        order = np.lexsort((tails, preds, heads))
        self.target_head = heads[order]
        self.target_pred = preds[order]
        self.target_tail = tails[order]
        group = np.r_[True, (np.diff(self.target_head) != 0) | (np.diff(self.target_pred) != 0)] \
            if len(order) else np.zeros(0, dtype=bool)
        self.group_start = np.flatnonzero(group)
        self.group_end = np.r_[self.group_start[1:], len(order)].astype(np.int64)
        self.group_pred = self.target_pred[self.group_start]
        self.head_groups = _pointers(self.target_head[self.group_start], self.num_terms)

        pair_keys = self.target_head * self.num_entities + self.target_tail
        order = np.lexsort((self.target_pred, pair_keys))
        self.pair_keys = pair_keys[order]
        self.pair_pred = self.target_pred[order]


def _pointers(sorted_values: np.ndarray, size: int) -> np.ndarray:
    """CSR offsets: rows with value ``v`` are ``ptr[v]:ptr[v + 1]``."""
    return np.searchsorted(sorted_values, np.arange(size + 1)).astype(np.int64)


@dataclass
class ParisState:
    """Mutable state of a PARIS run.

    Every term has at most one counterpart (``-1`` for none) with a
    probability. ``left_relations`` holds ``P(left q ⊆ right q')`` and
    ``right_relations`` holds ``P(right q' ⊆ left q)``. ``fixed_left`` and
    ``fixed_right`` mark entities in approved pairs, whose counterpart never
    changes, and ``rejected`` holds the sorted keys ``left * R + right`` of
    rejected pairs, where ``R`` is the right entity count. The fixed relation
    keys are kept at probability 1 in their respective relation tables.
    """

    left_match: np.ndarray
    left_prob: np.ndarray
    right_match: np.ndarray
    right_prob: np.ndarray
    left_relations: RelationTable
    right_relations: RelationTable
    fixed_left: np.ndarray
    fixed_right: np.ndarray
    rejected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fixed_left_relations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fixed_right_relations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
