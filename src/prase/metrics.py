"""Evaluation of alignments against reference links."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .alignment import Alignment

if TYPE_CHECKING:
    from .embedding.base import Embeddings

THRESHOLDS = tuple(round(0.1 * step, 1) for step in range(10))


@dataclass(frozen=True, slots=True)
class Scores:
    precision: float
    recall: float
    f1: float
    correct: int
    predicted: int
    reference: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate(
    alignment: Alignment, reference: np.ndarray | Iterable[tuple[int, int]], *, threshold: float = 0.0,
) -> Scores:
    """Precision, recall, and F1 of the pairs scoring at least ``threshold``."""
    gold = _pair_set(reference)
    predicted = alignment.above(threshold).pairs() if threshold > 0 else alignment.pairs()
    correct = len(predicted & gold)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(precision, recall, f1, correct, len(predicted), len(gold))


def threshold_scores(
    alignment: Alignment, reference: np.ndarray | Iterable[tuple[int, int]], thresholds: Sequence[float] = THRESHOLDS,
) -> dict[float, Scores]:
    gold = _pair_set(reference)
    return {threshold: evaluate(alignment, gold, threshold=threshold) for threshold in thresholds}


def ranking_scores(
    embeddings: Embeddings, pairs: np.ndarray, ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Hits@k, mean rank, and MRR of each left entity among the right entities of ``pairs``.

    This is the OpenEA test protocol: the candidates of a test entity are the
    distinct counterparts of all test links, ranked by the model's own metric.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return {**{f"hits@{k}": 0.0 for k in ks}, "mr": 0.0, "mrr": 0.0}
    ranks = embeddings.ranks(pairs[:, 0], pairs[:, 1])
    result = {f"hits@{k}": float((ranks <= k).mean()) for k in ks}
    result["mr"] = float(ranks.mean())
    result["mrr"] = float((1.0 / ranks).mean())
    return result


def _pair_set(reference: np.ndarray | Iterable[tuple[int, int]] | set) -> set[tuple[int, int]]:
    if isinstance(reference, set):
        return reference
    array = np.asarray(list(reference) if not isinstance(reference, np.ndarray) else reference, dtype=np.int64)
    return set(map(tuple, array.reshape(-1, 2).tolist()))
