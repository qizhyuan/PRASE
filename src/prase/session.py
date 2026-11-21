"""Interactive PARIS runs for reviewing entity and predicate matches."""

from __future__ import annotations

import numpy as np

from .alignment import Alignment, PredicateMatch
from .decisions import Decisions
from .embedding.base import Embeddings
from .errors import DataError
from .kg import KG, Predicate
from .paris import Candidate, LiteralMatcher, Paris, ParisConfig, ParisResult, exact_literals


class Session:
    """Keep graph indexes, seeds, and manual decisions across reasoning runs.

    Use entity identifiers to review matches. Changes to seeds and decisions
    take effect on the next ``run()``.
    """

    def __init__(
        self, left: KG, right: KG, config: ParisConfig | None = None,
        *, literal_matcher: LiteralMatcher = exact_literals,
    ) -> None:
        self.left, self.right = left, right
        self.paris = Paris(left, right, config, literal_matcher=literal_matcher)
        self._seeds: dict[tuple[str, str], float] = {}
        self._approved: set[tuple[str, str]] = set()
        self._rejected: set[tuple[str, str]] = set()
        self._fixed_predicates: set[tuple[int, int]] = set()
        self._embeddings: Embeddings | None = None
        self._run_embeddings: Embeddings | None = None
        self.result: ParisResult | None = None

    def seed(self, left_id: str, right_id: str, probability: float = 1.0) -> None:
        """Initialize this pair at ``probability`` and let reasoning update it."""
        self._pair(left_id, right_id)
        if not 0.0 <= probability <= 1.0:
            raise DataError("a seed probability must be between 0 and 1")
        self._seeds[left_id, right_id] = probability

    def approve(self, left_id: str, right_id: str) -> None:
        """Fix this pair and replace earlier approvals for either entity."""
        self._pair(left_id, right_id)
        self._approved = {(a, b) for a, b in self._approved if a != left_id and b != right_id}
        self._approved.add((left_id, right_id))
        self._rejected.discard((left_id, right_id))

    def reject(self, left_id: str, right_id: str) -> None:
        """Never match this pair."""
        self._pair(left_id, right_id)
        self._rejected.add((left_id, right_id))
        self._approved.discard((left_id, right_id))
        self._seeds.pop((left_id, right_id), None)

    def forget(self, left_id: str, right_id: str) -> None:
        """Remove any seed, approval, or rejection of this pair."""
        pair = (left_id, right_id)
        self._seeds.pop(pair, None)
        self._approved.discard(pair)
        self._rejected.discard(pair)

    def fix_predicate(self, left: str | Predicate, right: str | Predicate) -> None:
        """Fix both subsumption probabilities at 1, including those of the inverse pair.

        Use a predicate name or a ``Predicate`` with its kind if a relation and
        an attribute share the same name.
        """
        a, b = self._predicate_pair(left, right)
        if self.left.predicates[a].kind != self.right.predicates[b].kind:
            raise DataError("fixed predicates must have the same kind")
        self._fixed_predicates.add((a, b))

    def forget_predicate(self, left: str | Predicate, right: str | Predicate) -> None:
        """Remove a fixed predicate equivalence for subsequent runs."""
        self._fixed_predicates.discard(self._predicate_pair(left, right))

    def set_embeddings(self, embeddings: Embeddings | None) -> None:
        """Set or clear the embeddings used for reasoning and candidate scores."""
        if embeddings is not None and (
            len(embeddings.left) != self.left.num_entities or len(embeddings.right) != self.right.num_entities
        ):
            raise DataError("embeddings must have one vector per entity of each graph")
        self._embeddings = embeddings

    @property
    def decisions(self) -> Decisions:
        """Current entity approvals and rejections as an immutable snapshot."""
        return Decisions(frozenset(self._approved), frozenset(self._rejected))

    def run(self) -> ParisResult:
        """Recompute the alignment using the current review inputs."""
        rows = [
            (self.left.entity_index[a], self.right.entity_index[b], p) for (a, b), p in sorted(self._seeds.items())
        ]
        embeddings = self._embeddings
        self.result = self.paris.run(
            seeds=Alignment.from_pairs(rows), embeddings=embeddings,
            constraints=self.decisions.resolve(self.left, self.right),
            fixed_predicates=self._fixed_predicates,
        )
        self._run_embeddings = embeddings
        return self.result

    def matches(self, minimum: float = 0.0) -> list[tuple[str, str, float]]:
        """Return entity identifier pairs and scores from the last run."""
        return self._finished().alignment.above(minimum).to_ids(self.left, self.right)

    def predicate_matches(self, minimum: float = 0.0) -> tuple[PredicateMatch, ...]:
        """Return predicate matches from the last run."""
        return self.paris.predicate_matches(self._finished().state, minimum)

    def unaligned(self) -> tuple[list[str], list[str]]:
        """Return the unmatched entity identifiers on the left and right."""
        alignment = self._finished().alignment
        left, right = set(alignment.left.tolist()), set(alignment.right.tolist())
        return (
            [entity for index, entity in enumerate(self.left.entities) if index not in left],
            [entity for index, entity in enumerate(self.right.entities) if index not in right],
        )

    def match(self, left_id: str) -> tuple[str, float] | None:
        """The right entity matched to a left entity by the last run and its probability, or None."""
        result = self._finished()
        left = self._left_index(left_id)
        found = result.alignment.left == left
        if not found.any():
            return None
        position = int(np.flatnonzero(found)[0])
        return self.right.entities[int(result.alignment.right[position])], float(result.alignment.score[position])

    def candidates(self, left_id: str, limit: int = 10) -> list[tuple[str, Candidate]]:
        """Score right-side alternatives for a left entity against the final state of the last run.

        The scores repeat the evidence computation with the final relation and
        entity matches, so they can differ from the probability that the run
        recorded for the accepted match, which :meth:`match` returns. They use
        the embeddings of the last run, not ones set since. Rejected
        pairs and right entities approved for another left entity are left out.
        """
        result = self._finished()
        left = self._left_index(left_id)
        found = self.paris.candidates(
            result, left, embeddings=self._run_embeddings, limit=limit,
        )
        return [(self.right.entities[candidate.right], candidate) for candidate in found]

    def _left_index(self, left_id: str) -> int:
        left = self.left.entity_index.get(left_id)
        if left is None:
            raise DataError(f"unknown left entity {left_id!r}")
        return left

    def _finished(self) -> ParisResult:
        if self.result is None:
            raise DataError("call run() first")
        return self.result

    def _pair(self, left_id: str, right_id: str) -> None:
        if left_id not in self.left.entity_index or right_id not in self.right.entity_index:
            raise DataError(f"unknown entity pair ({left_id!r}, {right_id!r})")

    def _predicate_pair(self, left: str | Predicate, right: str | Predicate) -> tuple[int, int]:
        return _predicate_index(self.left, left, "left"), _predicate_index(self.right, right, "right")


def _predicate_index(kg: KG, identifier: str | Predicate, side: str) -> int:
    matches = [i for i, predicate in enumerate(kg.predicates) if identifier in (predicate, predicate.name)]
    if not matches:
        raise DataError(f"unknown {side} predicate {identifier!r}")
    if len(matches) != 1:
        raise DataError(f"ambiguous {side} predicate {identifier!r}; specify a Predicate with its kind")
    return matches[0]
