"""Scored entity and predicate alignments between two graphs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .datasets import read_pairs_with_scores
from .errors import DataError
from .kg import KG
from .tsv import PathLike, write_rows


class Alignment:
    """Scored ``(left entity, right entity)`` pairs stored as parallel arrays.

    ``left`` and ``right`` hold entity indices. ``score`` holds the pair's
    probability or similarity. These arrays are read-only. Transformations
    return new alignments.
    """

    __slots__ = ("left", "right", "score")

    def __init__(self, left: object = (), right: object = (), score: object = None) -> None:
        left = np.array(left, dtype=np.int64).reshape(-1)
        right = np.array(right, dtype=np.int64).reshape(-1)
        score = np.ones(len(left)) if score is None else np.array(score, dtype=np.float64).reshape(-1)
        if not len(left) == len(right) == len(score):
            raise DataError("alignment arrays must have equal lengths")
        for array in (left, right, score):
            array.setflags(write=False)
        self.left, self.right, self.score = left, right, score

    @classmethod
    def from_pairs(cls, rows: Iterable[tuple[int, int, float]]) -> Alignment:
        rows = list(rows)
        if not rows:
            return cls()
        left, right, score = zip(*rows, strict=True)
        return cls(left, right, score)

    @classmethod
    def read(cls, path: PathLike, left: KG, right: KG) -> Alignment:
        """Read ``left_id, right_id[, score]`` rows. Unknown identifiers are an error."""
        rows = []
        for a, b, score in read_pairs_with_scores(path):
            try:
                rows.append((left.entity_index[a], right.entity_index[b], score))
            except KeyError as error:
                raise DataError(f"{path}: unknown entity {error.args[0]!r}") from error
        return cls.from_pairs(rows)

    def write(self, path: PathLike, left: KG, right: KG) -> None:
        write_rows(path, ("left", "right", "score"), self.to_ids(left, right))

    def to_ids(self, left: KG, right: KG) -> list[tuple[str, str, float]]:
        return [(left.entities[i], right.entities[j], s) for i, j, s in self]

    def __len__(self) -> int:
        return len(self.left)

    def __iter__(self) -> Iterator[tuple[int, int, float]]:
        return zip(self.left.tolist(), self.right.tolist(), self.score.tolist(), strict=True)

    def __repr__(self) -> str:
        return f"Alignment({len(self)} pairs)"

    def subset(self, mask: np.ndarray) -> Alignment:
        return Alignment(self.left[mask], self.right[mask], self.score[mask])

    def above(self, threshold: float) -> Alignment:
        """Pairs whose score is at least ``threshold``."""
        return self.subset(self.score >= threshold)

    def sorted(self) -> Alignment:
        """Pairs by descending score, then ascending left and right index."""
        return self.subset(np.lexsort((self.right, self.left, -self.score)))

    def one_to_one(self) -> Alignment:
        """Greedy one-to-one matching: keep the best remaining pair for each entity."""
        ranked = self.sorted()
        used_left, used_right = set(), set()
        keep = np.zeros(len(ranked), dtype=bool)
        for position, (i, j) in enumerate(zip(ranked.left.tolist(), ranked.right.tolist(), strict=True)):
            if i not in used_left and j not in used_right:
                used_left.add(i)
                used_right.add(j)
                keep[position] = True
        return ranked.subset(keep)

    def pairs(self) -> set[tuple[int, int]]:
        return set(zip(self.left.tolist(), self.right.tolist(), strict=True))

    def scaled(self, factor: float) -> Alignment:
        return Alignment(self.left, self.right, np.clip(self.score * factor, 0.0, 1.0))

    def without(self, pairs: Iterable[tuple[int, int]]) -> Alignment:
        """These pairs minus ``pairs``, given as an array, list, or set of ``(left, right)`` indices."""
        excluded = np.array(pairs if isinstance(pairs, np.ndarray) else list(pairs), dtype=np.int64).reshape(-1, 2)
        if not len(excluded) or not len(self):
            return self
        width = int(max(self.right.max(), excluded[:, 1].max())) + 1
        return self.subset(~np.isin(self.left * width + self.right, excluded[:, 0] * width + excluded[:, 1]))

    def concat(self, other: Alignment) -> Alignment:
        return Alignment(
            np.concatenate((self.left, other.left)), np.concatenate((self.right, other.right)),
            np.concatenate((self.score, other.score)),
        )


@dataclass(frozen=True, slots=True)
class PredicateMatch:
    """Subsumption probabilities between a left and a right predicate.

    ``left_in_right`` estimates P(left ⊆ right) and ``right_in_left`` the
    reverse. A name ending in ``^-1`` is the inverse of that predicate.
    """

    left: str
    right: str
    left_in_right: float
    right_in_left: float


def write_predicate_matches(path: PathLike, matches: Iterable[PredicateMatch]) -> None:
    write_rows(
        path, ("left", "right", "left_in_right", "right_in_left"),
        ((m.left, m.right, m.left_in_right, m.right_in_left) for m in matches),
    )
