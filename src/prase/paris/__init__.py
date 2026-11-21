"""PARIS reasoning shared by the standalone and hybrid pipelines."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from operator import index

import numpy as np

from ..alignment import Alignment, PredicateMatch
from ..decisions import Constraints
from ..errors import ConfigError, DataError
from ..kg import KG
from ..progress import Progress
from . import native
from .python import PythonEngine, _candidates, _one_to_one
from .state import GraphIndex, ParisState, RelationTable

__all__ = ["Candidate", "LiteralMatcher", "Paris", "ParisConfig", "ParisResult", "exact_literals"]

ENGINES = ("auto", "native", "python")
LiteralMatcher = Callable[[KG, KG], Alignment]


@dataclass(frozen=True)
class ParisConfig:
    """Settings for reasoning, embedding feedback, and engine selection.

    ``theta`` is the minimum probability of a matched neighbor to count as
    evidence, which always needs a positive probability, and the relation
    prior during warm-up. ``delta`` is the minimum evidence a fact must
    contribute. Relation and functionality scores are divided by ``epsilon``,
    which is at least 1 so that every evidence factor stays a probability,
    and ``smoothing`` is added to the denominator of every relation estimate.
    The first ``warmup`` iterations use the prior for unknown relations and
    ignore embeddings. A candidate's score is
    ``(1 - embedding_weight) * structural + embedding_weight * cosine``.
    """

    iterations: int = 10
    theta: float = 0.1
    delta: float = 0.01
    epsilon: float = 1.01
    smoothing: float = 10.0
    warmup: int = 2
    embedding_weight: float = 0.2
    engine: str = "auto"
    threads: int = 0

    def __post_init__(self) -> None:
        for name in ("iterations", "warmup", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ConfigError(f"paris.{name} must be a nonnegative integer")
        for name in ("theta", "delta", "epsilon", "smoothing", "embedding_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigError(f"paris.{name} must be a finite number")
        for name in ("theta", "delta", "embedding_weight"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ConfigError(f"paris.{name} must be between 0 and 1")
        if self.epsilon < 1:
            raise ConfigError("paris.epsilon must be at least 1")
        if self.smoothing < 0:
            raise ConfigError("paris.smoothing must be nonnegative")
        if self.engine not in ENGINES:
            raise ConfigError(f"paris.engine must be one of {', '.join(ENGINES)}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A scored counterpart of one left entity, as computed by the last iteration."""

    right: int
    score: float
    structural: float
    similarity: float | None


@dataclass(frozen=True, eq=False)
class ParisResult:
    alignment: Alignment
    predicates: tuple[PredicateMatch, ...]
    state: ParisState
    iterations: tuple[dict[str, int], ...]
    engine: str


def exact_literals(left: KG, right: KG) -> Alignment:
    """Pair literals with equal values at probability 1."""
    index = {value: position for position, value in enumerate(right.literals)}
    pairs = [(i, index[value]) for i, value in enumerate(left.literals) if value in index]
    return Alignment.from_pairs((i, j, 1.0) for i, j in pairs)


class Paris:
    """A PARIS reasoner for one graph pair. Indexes are built once and reused across runs."""

    def __init__(
        self, left: KG, right: KG, config: ParisConfig | None = None,
        *, literal_matcher: LiteralMatcher = exact_literals,
    ) -> None:
        self.left, self.right = left, right
        self.config = config or ParisConfig()
        self.left_index, self.right_index = GraphIndex(left), GraphIndex(right)
        literals = literal_matcher(left, right)
        if not isinstance(literals, Alignment):
            raise DataError("a literal matcher must return an Alignment of literal index pairs")
        _check_pairs("literal pairs", literals.left, literals.right, len(left.literals), len(right.literals),
                     kind="literal")
        if len(literals) and not ((literals.score >= 0) & (literals.score <= 1)).all():
            raise DataError("literal pair scores must be probabilities between 0 and 1")
        # Only a positive probability makes literals counterparts.
        self.literal_pairs = literals.subset(literals.score > 0).one_to_one()
        self.engine = _engine(self.left_index, self.right_index, self.config)

    def initial_state(
        self, seeds: Alignment | None = None, constraints: Constraints | None = None,
        *, fixed_predicates: Iterable[tuple[int, int]] = (),
    ) -> ParisState:
        """Literal matches, then ``seeds`` as starting counterparts, then approved pairs.

        A seed's score becomes its starting probability. Rejected seeds and
        seeds that conflict with an approved pair are dropped. Fixed predicate
        pairs use graph predicate indices and have probability 1 in both
        directions, including the corresponding pair of inverse predicates.
        """
        left, right = self.left_index, self.right_index
        state = ParisState(
            np.full(left.num_terms, -1, dtype=np.int64), np.zeros(left.num_terms),
            np.full(right.num_terms, -1, dtype=np.int64), np.zeros(right.num_terms),
            RelationTable.empty(right.num_predicates), RelationTable.empty(left.num_predicates),
            np.zeros(left.num_entities, dtype=bool), np.zeros(right.num_entities, dtype=bool),
        )
        predicate_pairs = set()
        left_count, right_count = len(self.left.predicates), len(self.right.predicates)
        for a, b in fixed_predicates:
            a, b = index(a), index(b)
            if not (0 <= a < left_count and 0 <= b < right_count):
                raise DataError(f"unknown predicate index pair ({a}, {b})")
            if self.left.predicates[a].kind != self.right.predicates[b].kind:
                raise DataError("fixed predicates must have the same kind")
            predicate_pairs.update(((a, b), (a + left_count, b + right_count)))
        state.fixed_left_relations = np.array(sorted(a * right.num_predicates + b for a, b in predicate_pairs),
                                              dtype=np.int64)
        state.fixed_right_relations = np.array(sorted(b * left.num_predicates + a for a, b in predicate_pairs),
                                               dtype=np.int64)
        state.left_relations = state.left_relations.with_fixed(state.fixed_left_relations)
        state.right_relations = state.right_relations.with_fixed(state.fixed_right_relations)
        literals = self.literal_pairs
        state.left_match[left.num_entities + literals.left] = right.num_entities + literals.right
        state.right_match[right.num_entities + literals.right] = left.num_entities + literals.left
        state.left_prob[left.num_entities + literals.left] = literals.score
        state.right_prob[right.num_entities + literals.right] = literals.score

        constraints = constraints or Constraints()
        approved = constraints.approved
        _check_pairs("approved pairs", approved[:, 0], approved[:, 1], left.num_entities, right.num_entities)
        if constraints.rejected:
            rejected = np.array(sorted(constraints.rejected), dtype=np.int64).reshape(-1, 2)
            _check_pairs("rejected pairs", rejected[:, 0], rejected[:, 1], left.num_entities, right.num_entities)
        if seeds is not None and len(seeds):
            _check_pairs("seeds", seeds.left, seeds.right, left.num_entities, right.num_entities)
        state.fixed_left[approved[:, 0]] = True
        state.fixed_right[approved[:, 1]] = True
        state.rejected = np.unique(np.array(
            [i * right.num_entities + j for i, j in constraints.rejected], dtype=np.int64,
        ))
        if seeds is not None and len(seeds):
            if not np.isfinite(seeds.score).all():
                raise DataError("seed scores must be finite")
            keep = ~state.fixed_left[seeds.left] & ~state.fixed_right[seeds.right]
            if len(state.rejected):
                keep &= ~np.isin(seeds.left * right.num_entities + seeds.right, state.rejected)
            seeds = seeds.subset(keep)
            score = np.clip(seeds.score, 0.0, 1.0)
            for own, other, match, prob in (
                (seeds.left, seeds.right, state.left_match, state.left_prob),
                (seeds.right, seeds.left, state.right_match, state.right_prob),
            ):
                # Each entity starts from its best seed; ties go to the lower index.
                best = _first_per_group(own, np.lexsort((other, -score, own)))
                match[own[best]] = other[best]
                prob[own[best]] = score[best]
        state.left_match[approved[:, 0]] = approved[:, 1]
        state.left_prob[approved[:, 0]] = 1.0
        state.right_match[approved[:, 1]] = approved[:, 0]
        state.right_prob[approved[:, 1]] = 1.0
        return state

    def run(
        self,
        *,
        seeds: Alignment | None = None,
        embeddings: object | None = None,
        constraints: Constraints | None = None,
        fixed_predicates: Iterable[tuple[int, int]] = (),
        on_iteration: Callable[[int, Alignment, dict[str, int]], None] | None = None,
    ) -> ParisResult:
        """Run ``config.iterations`` iterations from a fresh state.

        ``embeddings`` (an :class:`~prase.embedding.Embeddings`) adds cosine
        similarity to candidate scores after warm-up. ``fixed_predicates``
        supplies predicate index pairs to keep equivalent throughout the run.
        ``on_iteration`` receives the iteration number, the current alignment,
        and iteration statistics.
        """
        self._check_embeddings(embeddings)
        state = self.initial_state(seeds, constraints, fixed_predicates=fixed_predicates)
        history = []
        progress = Progress("PARIS", self.config.iterations, unit="iteration")
        for iteration in range(self.config.iterations):
            stats = self.engine.iterate(state, iteration < self.config.warmup, embeddings)
            history.append(stats)
            progress.update(iteration + 1, f"; {stats.get('matched', 0)} matches")
            if on_iteration is not None:
                on_iteration(iteration + 1, self.alignment(state), stats)
        if not self.config.iterations:
            # Starting matches obey the rules of an iteration's result: one to one, positive probability.
            _one_to_one(state, self.left_index.num_entities, self.right_index.num_entities)
        return ParisResult(
            self.alignment(state), self.predicate_matches(state), state, tuple(history), self.engine.name,
        )

    def _check_embeddings(self, embeddings: object | None) -> None:
        """Require one embedding row for every entity of each graph."""
        if embeddings is None:
            return
        rows = (np.shape(embeddings.left)[0], np.shape(embeddings.right)[0])
        if rows != (self.left.num_entities, self.right.num_entities):
            raise DataError(
                f"embeddings have {rows[0]} and {rows[1]} rows, but the graphs have "
                f"{self.left.num_entities} and {self.right.num_entities} entities",
            )

    def alignment(self, state: ParisState) -> Alignment:
        count = self.left_index.num_entities
        match, prob = state.left_match[:count], state.left_prob[:count]
        matched = np.flatnonzero(match >= 0)
        return Alignment(matched, match[matched], prob[matched])

    def predicate_matches(self, state: ParisState, minimum: float = 0.0) -> tuple[PredicateMatch, ...]:
        """Predicate pairs with either subsumption probability above ``minimum``."""
        forward = {(a, b): value for a, b, value in state.left_relations.items()}
        backward = {(b, a): value for a, b, value in state.right_relations.items()}
        names = (_predicate_names(self.left), _predicate_names(self.right))
        rows = []
        for a, b in sorted(forward.keys() | backward.keys()):
            values = forward.get((a, b), 0.0), backward.get((a, b), 0.0)
            if max(values) > minimum:
                rows.append(PredicateMatch(names[0][a], names[1][b], *values))
        return tuple(rows)

    def candidates(
        self, result: ParisResult, entity: int, *, embeddings: object | None = None, limit: int = 10,
    ) -> list[Candidate]:
        """Score the counterparts of one left entity against the final state.

        This repeats the last iteration's evidence computation for ``entity``,
        which is what a reviewer needs to compare alternatives. Rejected pairs
        and right entities approved for another left entity are left out.
        """
        self._check_embeddings(embeddings)
        source, target, state = self.left_index, self.right_index, result.state
        start, stop = source.source_ptr[entity], source.source_ptr[entity + 1]
        head = source.source_head[start:stop]
        match, prob = state.left_match[head], state.left_prob[head]
        keep = (match >= 0) & (prob > 0.0) & (prob >= self.config.theta)
        found = _candidates(
            source, target, source.source_tail[start:stop][keep], source.source_pred[start:stop][keep],
            match[keep], prob[keep], state.left_relations, state.right_relations, self.config,
            self.config.iterations <= self.config.warmup, None,
        )
        if found is None:
            return []
        _, y, factor = found
        order = np.argsort(y, kind="stable")
        y, factor = y[order], factor[order]
        starts = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
        right = y[starts]
        structural = 1.0 - np.multiply.reduceat(factor, starts)
        similarity = None
        score = structural
        if embeddings is not None and self.config.iterations > self.config.warmup:
            similarity = embeddings.cosine(np.full(len(right), entity), right)
            weight = self.config.embedding_weight
            score = (1.0 - weight) * structural + weight * similarity
        score = np.clip(score, 0.0, 1.0)
        allowed = ~(state.fixed_right[right] & (state.right_match[right] != entity))
        if len(state.rejected):
            allowed &= ~np.isin(entity * target.num_entities + right, state.rejected)
        order = np.lexsort((right, -score))
        ranked = order[allowed[order]][:limit]
        return [
            Candidate(int(right[k]), float(score[k]), float(structural[k]),
                      None if similarity is None else float(similarity[k]))
            for k in ranked
        ]


def _check_pairs(what: str, left: np.ndarray, right: np.ndarray, left_count: int, right_count: int,
                 *, kind: str = "entity") -> None:
    """Require indices of the left and the right graph's entities, or of their literals."""
    if len(left) and (left.min() < 0 or left.max() >= left_count or right.min() < 0 or right.max() >= right_count):
        raise DataError(f"{what} need {kind} indices below {left_count} on the left and {right_count} on the right")


def _first_per_group(keys: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Positions of the first row of each run of equal ``keys[order]``."""
    if not len(order):
        return order
    ordered = keys[order]
    return order[np.r_[True, ordered[1:] != ordered[:-1]]]


def _predicate_names(kg: KG) -> list[str]:
    names = [predicate.name for predicate in kg.predicates]
    return names + [f"{name}^-1" for name in names]


def _engine(left: GraphIndex, right: GraphIndex, config: ParisConfig):
    if config.engine != "python" and native.available():
        return native.NativeEngine(left, right, config)
    if config.engine == "native":
        raise ConfigError(f"the native PARIS engine is unavailable: {native.load_error()}")
    if config.engine == "auto":
        native.report_fallback()
    return PythonEngine(left, right, config)

