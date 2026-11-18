"""Replace the alignment with matches based on entity names, keeping approved pairs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .alignment import Alignment
from .datasets import read_word_vectors
from .embedding.base import Embeddings
from .errors import ConfigError
from .kg import KG, entity_names
from .pipeline import Context, State, _approved


def levenshtein(a: str, b: str) -> int:
    """Edit distance with Hyyrö's bit-parallel algorithm."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    masks: dict[str, int] = {}
    for position, char in enumerate(b):
        masks[char] = masks.get(char, 0) | (1 << position)
    full = (1 << len(b)) - 1
    top = 1 << (len(b) - 1)
    positive, negative, distance = full, 0, len(b)
    for char in a:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        horizontal_positive = negative | ~(horizontal | positive) & full
        horizontal_negative = positive & horizontal
        if horizontal_positive & top:
            distance += 1
        elif horizontal_negative & top:
            distance -= 1
        horizontal_positive = ((horizontal_positive << 1) | 1) & full
        horizontal_negative = (horizontal_negative << 1) & full
        positive = horizontal_negative | ~(vertical | horizontal_positive) & full
        negative = horizontal_positive & vertical
    return distance


def name_similarity(a: str, b: str) -> float:
    """One minus the edit distance divided by the longer length."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


@dataclass
class NameMatch:
    """Match names by normalized edit similarity.

    For each left entity, rank up to ``candidates`` eligible right entities by
    TF-IDF cosine over character bigrams and include exact name matches.
    Select the greatest edit similarity at or above ``threshold``. Break ties
    by the lower right index.
    """

    threshold: float = 0.5
    left_names: Sequence[str] = ()
    right_names: Sequence[str] = ()
    lowercase: bool = False
    candidates: int = 20
    name: str = "names"

    def run(self, state: State, context: Context) -> State:
        left = self._names(state.left, self.left_names)
        right = self._names(state.right, self.right_names)
        review = _Review(state)
        rows = []
        for i, choices in enumerate(_bigram_candidates(left, right, self.candidates, review.excluded)):
            if review.fixed_left[i]:
                continue
            best_score, best_j = -1.0, -1
            for j in choices:
                score = name_similarity(left[i], right[j])
                if score > best_score or (score == best_score and j < best_j):
                    best_score, best_j = score, j
            if best_j >= 0 and best_score >= self.threshold:
                rows.append((i, best_j, best_score))
        alignment = review.approved.concat(Alignment.from_pairs(rows))
        context.report(matches=len(alignment))
        return replace(state, alignment=alignment)

    def _names(self, kg: KG, predicates: Sequence[str]) -> list[str]:
        names = entity_names(kg, predicates)
        return [name.lower() for name in names] if self.lowercase else names


@dataclass
class NameEmbeddingMatch:
    """Match names by cosine similarity of their averaged word vectors.

    Each left entity takes its most similar eligible right entity when the
    cosine is at least ``threshold``.
    """

    word_vectors: str | Path
    threshold: float = 0.5
    left_names: Sequence[str] = ()
    right_names: Sequence[str] = ()
    name: str = "name-embeddings"

    def run(self, state: State, context: Context) -> State:
        names = (entity_names(state.left, self.left_names), entity_names(state.right, self.right_names))
        tokens = [[_tokens(name) for name in side] for side in names]
        vocabulary = {token for side in tokens for row in side for token in row}
        vectors, width = read_word_vectors(self.word_vectors, vocabulary)
        matrices = [_average(side, vectors, width) for side in tokens]
        embeddings = Embeddings(*matrices, "cosine")
        review = _Review(state)
        left, right = (np.flatnonzero(np.abs(matrix).sum(axis=1) > 0) for matrix in matrices)
        found = embeddings.nearest(left[~review.fixed_left[left]], right[~review.fixed_right[right]],
                                   excluded=review.rejected_pairs)
        alignment = review.approved.concat(found.above(self.threshold))
        context.report(words=len(vectors), vocabulary=len(vocabulary), matches=len(alignment))
        return replace(state, alignment=alignment)


class _Review:
    """Manual decisions as the name matchers apply them.

    Approved pairs are kept as they are, their entities match nothing else,
    and rejected pairs are never chosen, so each entity's best eligible
    counterpart is selected.
    """

    def __init__(self, state: State) -> None:
        self.approved = _approved(state.constraints)
        self.fixed_left = np.zeros(state.left.num_entities, dtype=bool)
        self.fixed_left[self.approved.left] = True
        self.fixed_right = np.zeros(state.right.num_entities, dtype=bool)
        self.fixed_right[self.approved.right] = True
        self.rejected_pairs = np.array(sorted(state.constraints.rejected), dtype=np.int64).reshape(-1, 2)
        self.rejected: dict[int, set[int]] = {}
        for left, right in state.constraints.rejected:
            self.rejected.setdefault(left, set()).add(right)

    def excluded(self, left: int, right: np.ndarray) -> np.ndarray:
        """Which of the ``right`` entities the left entity may not match."""
        blocked = self.fixed_right[right]
        if left in self.rejected:
            blocked |= np.isin(right, list(self.rejected[left]))
        return blocked


def _tokens(name: str) -> list[str]:
    return re.findall(r"\w+", name)


def _average(rows: list[list[str]], vectors: dict[str, np.ndarray], width: int) -> np.ndarray:
    result = np.zeros((len(rows), width), dtype=np.float32)
    for index, row in enumerate(rows):
        found = [vectors[token] for token in row if token in vectors]
        if found:
            result[index] = np.mean(found, axis=0)
    return result


def _bigram_candidates(left: list[str], right: list[str], limit: int, excluded):
    """For each left name, the right names sharing the most character bigrams (TF-IDF cosine).

    ``excluded(i, columns)`` marks the right entities that left entity ``i``
    may not match. They are removed before the ``limit`` best are chosen.
    """
    try:
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as error:
        raise ConfigError("name matching requires SciPy: pip install scipy") from error
    vocabulary: dict[str, int] = {}

    def matrix(names: list[str], grow: bool) -> csr_matrix:
        indptr, indices = [0], []
        for name in names:
            grams = {name[k:k + 2] for k in range(max(1, len(name) - 1))}
            row = []
            for gram in grams:
                column = vocabulary.setdefault(gram, len(vocabulary)) if grow else vocabulary.get(gram)
                if column is not None:
                    row.append(column)
            indices.extend(sorted(row))
            indptr.append(len(indices))
        return csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(names), max(1, len(vocabulary))))

    right_matrix = matrix(right, True)
    left_matrix = matrix(left, False)
    left_matrix.resize((len(left), right_matrix.shape[1]))
    frequency = np.asarray((right_matrix > 0).sum(axis=0)).ravel()
    idf = np.log((1 + len(right)) / (1 + frequency)) + 1.0
    right_matrix = _normalize(right_matrix.multiply(idf).tocsr())
    left_matrix = _normalize(left_matrix.multiply(idf).tocsr())
    exact: dict[str, list[int]] = {}
    for j, name in enumerate(right):
        exact.setdefault(name, []).append(j)
    for start in range(0, len(left), 2048):
        block = (left_matrix[start:start + 2048] @ right_matrix.T).tocsr()
        for offset in range(block.shape[0]):
            i = start + offset
            row = block.getrow(offset)
            keep = ~excluded(i, row.indices)
            columns, values = row.indices[keep], row.data[keep]
            if len(columns) > limit:
                columns = columns[np.argpartition(-values, limit)[:limit]]
            matches = np.array(exact.get(left[i], ()), dtype=np.int64)
            matches = matches[~excluded(i, matches)]
            yield sorted(set(columns.tolist()) | set(matches.tolist()))


def _normalize(matrix):
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    from scipy.sparse import diags
    return diags(1.0 / norms) @ matrix
