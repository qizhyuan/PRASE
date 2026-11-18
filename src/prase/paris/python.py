"""NumPy implementation of one PARIS iteration.

The native engine in ``cpp/paris.cpp`` follows the same sequence.

1. **Left pass.** Score candidates for every left entity from the facts it
   shares with matched neighbors, keep its best candidate if it is at least as
   probable as its current counterpart, and re-estimate ``P(left q ⊆ right q')``.
2. **One-to-one step.** Each right entity keeps the most probable left entity
   pointing at it. The other left entities lose their counterpart.
3. **Right pass.** Re-estimate ``P(right q' ⊆ left q)`` from the right side.

Updates within a pass read the state from the start of the pass, so results
do not depend on processing order or thread count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .state import GraphIndex, ParisState, RelationTable

if TYPE_CHECKING:
    from ..embedding.base import Embeddings
    from . import ParisConfig

_CHUNK_FACTS = 1 << 20


class PythonEngine:
    name = "python"

    def __init__(self, left: GraphIndex, right: GraphIndex, config: ParisConfig) -> None:
        self.left, self.right, self.config = left, right, config

    def iterate(self, state: ParisState, warm: bool, embeddings: Embeddings | None) -> dict[str, int]:
        stats: dict[str, int] = {}
        relations, updates = _pass(
            self.left, self.right, state.left_match, state.left_prob,
            state.left_relations, state.right_relations, self.config, warm,
            fixed=state.fixed_left_relations, embeddings=None if warm else embeddings, state=state, stats=stats,
        )
        state.left_relations = relations
        if updates is not None:
            entity, match, prob = updates
            state.left_match[entity] = match
            state.left_prob[entity] = prob
        _one_to_one(state, self.left.num_entities)
        state.right_relations, _ = _pass(
            self.right, self.left, state.right_match, state.right_prob,
            state.right_relations, state.left_relations, self.config, warm,
            fixed=state.fixed_right_relations,
        )
        stats["matched"] = int((state.left_match[: self.left.num_entities] >= 0).sum())
        return stats


def _expand(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(row, position)`` for every position in ``start[row]:end[row]``."""
    counts = end - start
    total = int(counts.sum())
    row = np.repeat(np.arange(len(start)), counts)
    position = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(start, counts)
    return row, position


def _pass(
    source: GraphIndex,
    target: GraphIndex,
    match: np.ndarray,
    prob: np.ndarray,
    forward: RelationTable,
    backward: RelationTable,
    config: ParisConfig,
    warm: bool,
    *,
    fixed: np.ndarray,
    embeddings: Embeddings | None = None,
    state: ParisState | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[RelationTable, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """One directional pass from ``source`` to ``target``.

    Returns the re-estimated ``forward`` table with ``fixed`` keys preserved
    and, when ``state`` is given, the entity updates
    ``(entities, counterparts, probabilities)``.
    """
    theta = config.theta
    relation_keys, relation_weights = [], []
    norm = np.zeros(source.num_predicates)
    updates = []
    scored = []
    for first, last in _chunks(source):
        x = source.source_tail[first:last]
        q = source.source_pred[first:last]
        head = source.source_head[first:last]
        head_match, head_prob = match[head], prob[head]
        keep = (head_match >= 0) & (head_prob >= theta)
        x, q, head_match, head_prob = x[keep], q[keep], head_match[keep], head_prob[keep]
        x_match, x_prob = match[x], prob[x]

        # Relation statistics: weighted support for q ⊆ q' over matched facts.
        has = x_match >= 0
        weight = head_prob * x_prob
        norm += np.bincount(q[has], weights=weight[has], minlength=source.num_predicates)
        sel = np.flatnonzero(has & (x_prob > 0.0))
        keys = head_match[sel] * target.num_entities + x_match[sel]
        row, position = _expand(
            np.searchsorted(target.pair_keys, keys, "left"), np.searchsorted(target.pair_keys, keys, "right"),
        )
        relation_keys.append(q[sel][row] * target.num_predicates + target.pair_pred[position])
        relation_weights.append(weight[sel][row])

        if state is None:
            continue
        candidates = _candidates(
            source, target, x, q, head_match, head_prob, forward, backward, config, warm, stats,
        )
        if candidates is not None:
            *update, considered = _best(candidates, prob, target, embeddings, config, state)
            updates.append(update)
            scored.append(considered)

    keys = np.concatenate(relation_keys) if relation_keys else np.zeros(0, dtype=np.int64)
    weights = np.concatenate(relation_weights) if relation_weights else np.zeros(0)
    unique, inverse = np.unique(keys, return_inverse=True)
    support = np.bincount(inverse, weights=weights, minlength=len(unique))
    table = RelationTable(
        unique, support / (config.smoothing + norm[unique // target.num_predicates]), target.num_predicates,
    ).with_fixed(fixed)
    if stats is not None:
        stats["relation_pairs"] = len(table.keys)
    if state is None:
        return table, None

    entities = np.concatenate([u[0] for u in updates]) if updates else np.zeros(0, dtype=np.int64)
    counterparts = np.concatenate([u[1] for u in updates]) if updates else np.zeros(0, dtype=np.int64)
    probabilities = np.concatenate([u[2] for u in updates]) if updates else np.zeros(0)
    # An entity without any allowed candidate keeps its counterpart only if
    # that counterpart has a positive probability.
    lonely = np.ones(source.num_entities, dtype=bool)
    if scored:
        lonely[np.concatenate(scored)] = False
    lonely &= (prob[: source.num_entities] <= 0.0) & ~state.fixed_left
    lonely_entities = np.flatnonzero(lonely)
    return table, (
        np.concatenate((entities, lonely_entities)),
        np.concatenate((counterparts, np.full(len(lonely_entities), -1, dtype=np.int64))),
        np.concatenate((probabilities, np.zeros(len(lonely_entities)))),
    )


def _chunks(source: GraphIndex):
    """Fact ranges that end on an entity boundary and hold about ``_CHUNK_FACTS`` facts."""
    ptr = source.source_ptr
    start = 0
    while start < source.num_entities:
        stop = int(np.searchsorted(ptr, ptr[start] + _CHUNK_FACTS, "right")) - 1
        stop = min(max(stop, start + 1), source.num_entities)
        yield int(ptr[start]), int(ptr[stop])
        start = stop


def _candidates(source, target, x, q, head_match, head_prob, forward, backward, config, warm, stats):
    """Evidence factors for ``(x, candidate)`` pairs, as parallel arrays."""
    theta, epsilon = config.theta, config.epsilon
    row, group = _expand(target.head_groups[head_match], target.head_groups[head_match + 1])
    q_source, q_target, p = q[row], target.group_pred[group], head_prob[row]
    sub = forward.lookup(q_source, q_target) / epsilon
    sup = backward.lookup(q_target, q_source) / epsilon
    weak = (sub < theta) & (sup < theta)
    if warm:
        sub = np.where(weak, theta, sub)
        sup = np.where(weak, theta, sup)
    else:
        strong = ~weak
        row, group, q_source, q_target, p = row[strong], group[strong], q_source[strong], q_target[strong], p[strong]
        sub, sup = sub[strong], sup[strong]
    factor = (1.0 - p * sup * (target.functionality[q_target] / epsilon)) * \
             (1.0 - p * sub * (source.functionality[q_source] / epsilon))
    admitted = 1.0 - factor > config.delta
    if stats is not None:
        stats["evidence"] = stats.get("evidence", 0) + int(admitted.sum())
    row, group, factor = row[admitted], group[admitted], factor[admitted]
    if not len(row):
        return None
    k, position = _expand(target.group_start[group], target.group_end[group])
    return x[row][k], target.target_tail[position], factor[k]


def _best(candidates, prob, target, embeddings, config, state):
    """Keep each entity's best candidate when it is at least as probable as now."""
    x, y, factor = candidates
    keys = x * target.num_entities + y
    order = np.argsort(keys, kind="stable")
    keys, factor = keys[order], factor[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    score = 1.0 - np.multiply.reduceat(factor, starts)
    pair = keys[starts]
    x, y = pair // target.num_entities, pair % target.num_entities
    if embeddings is not None:
        weight = config.embedding_weight
        score = (1.0 - weight) * score + weight * embeddings.cosine(x, y)
    score = np.clip(score, 0.0, 1.0)

    allowed = ~state.fixed_left[x] & ~(state.fixed_right[y] & (state.right_match[y] != x))
    if len(state.rejected):
        allowed &= ~np.isin(pair, state.rejected)
    x, y, score = x[allowed], y[allowed], score[allowed]
    considered = np.unique(x)

    order = np.lexsort((y, -score, x))
    first = np.r_[True, x[order][1:] != x[order][:-1]] if len(order) else np.zeros(0, dtype=bool)
    best = order[first]
    x, y, score = x[best], y[best], score[best]
    improved = score >= prob[x]
    return x[improved], y[improved], score[improved], considered


def _one_to_one(state: ParisState, num_entities: int) -> None:
    """Each right entity keeps the most probable left entity that points at it."""
    left = np.arange(num_entities)
    match, prob = state.left_match[:num_entities], state.left_prob[:num_entities]
    valid = np.flatnonzero(match >= 0)
    if len(valid):
        order = valid[np.lexsort((valid, -prob[valid], match[valid]))]
        first = np.r_[True, match[order][1:] != match[order][:-1]]
        winners = order[first]
        right = match[winners]
        better = prob[winners] > state.right_prob[right]
        state.right_match[right[better]] = winners[better]
        state.right_prob[right[better]] = prob[winners[better]]
    lost = (match >= 0) & (state.right_match[np.maximum(match, 0)] != left)
    state.left_match[:num_entities][lost] = -1
    state.left_prob[:num_entities][lost] = 0.0
