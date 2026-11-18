"""The contract between PRASE and embedding models.

A model receives :class:`TrainingData` (both graphs, seed pairs from
reasoning or from the user, and the entities still to be aligned) and returns
:class:`Embeddings`: one vector per entity of each graph in a shared space,
and the metric that ranks counterparts. PRASE derives proposals and
embedding feedback from those vectors, so a model only has to train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..alignment import Alignment
from ..errors import DataError

if TYPE_CHECKING:
    from ..kg import KG
    from ..tsv import PathLike

METRICS = ("cosine", "inner", "l1", "csls")
_BLOCK_ENTRIES = 1 << 24


@dataclass(frozen=True, eq=False)
class TrainingData:
    """Inputs for one training run.

    ``seeds`` is an ``(n, 2)`` array of aligned ``(left, right)`` entity
    indices. ``candidates`` holds the entity indices eligible for retrieval or
    bootstrapping on each side. The pipeline excludes training-pair endpoints.
    ``rejected`` is an ``(m, 2)`` array of pairs that a reviewer rejected,
    which a model must not use as training pairs.
    """

    left: KG
    right: KG
    seeds: np.ndarray
    candidates: tuple[np.ndarray, np.ndarray]
    rejected: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected", _pair_array(self.rejected))

    @property
    def seed_pairs(self) -> list[tuple[int, int]]:
        return [tuple(pair) for pair in self.seeds.tolist()]

    @property
    def rejected_pairs(self) -> frozenset[tuple[int, int]]:
        return frozenset(map(tuple, self.rejected.tolist()))


class Embeddings:
    """Entity vectors of both graphs in one space, and the metric that ranks them.

    ``metric`` is ``cosine``, ``inner``, ``l1`` (Manhattan distance), or
    ``csls`` (cross-domain similarity local scaling over ``csls_k`` neighbors).
    Pair scores use cosine, inner product, or one minus Manhattan distance,
    clipped to [0, 1]. CSLS ranks counterparts and assigns a cosine score.
    ``device`` selects where retrieval and ranking run. The vectors are
    stored as NumPy arrays for PARIS. ``info`` holds a model's training
    summary, such as the selected epoch, for the run records.
    """

    def __init__(
        self, left: object, right: object, metric: str = "cosine", *, csls_k: int = 10, device: str = "cpu",
        info: dict | None = None,
    ) -> None:
        left = np.ascontiguousarray(left, dtype=np.float32)
        right = np.ascontiguousarray(right, dtype=np.float32)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or left.shape[1] == 0:
            raise DataError("embeddings need two matrices with the same nonzero width")
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise DataError("embeddings must be finite")
        if metric not in METRICS:
            raise DataError(f"unknown embedding metric {metric!r}; expected one of {', '.join(METRICS)}")
        self.left, self.right, self.metric, self.csls_k = left, right, metric, csls_k
        self.device = device
        self.info = dict(info or {})
        for array in (left, right):
            array.setflags(write=False)

    @property
    def dim(self) -> int:
        return self.left.shape[1]

    def cosine(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Cosine similarity of each ``(left[k], right[k])`` pair. Zero vectors score 0."""
        result = np.empty(len(left), dtype=np.float64)
        rows = max(1, _BLOCK_ENTRIES // (4 * self.dim))
        for start in range(0, len(left), rows):
            a = self.left[left[start:start + rows]].astype(np.float64)
            b = self.right[right[start:start + rows]].astype(np.float64)
            norms = np.sqrt(np.einsum("ij,ij->i", a, a) * np.einsum("ij,ij->i", b, b))
            dots = np.einsum("ij,ij->i", a, b)
            result[start:start + rows] = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        return result

    def pair_scores(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Scores in [0, 1] for each ``(left[k], right[k])`` pair."""
        if self.metric == "l1":
            distance = np.abs(self.left[left].astype(np.float64) - self.right[right]).sum(axis=1)
            return np.clip(1.0 - distance, 0.0, 1.0)
        if self.metric == "inner":
            values = np.einsum("ij,ij->i", self.left[left].astype(np.float64), self.right[right])
            return np.clip(values, 0.0, 1.0)
        return np.clip(self.cosine(left, right), 0.0, 1.0)

    def nearest(self, left: np.ndarray, right: np.ndarray, *, excluded: np.ndarray | None = None) -> Alignment:
        """For each entity in ``left``, its best counterpart among ``right``.

        Right entities can be selected more than once. Ties go to the first
        candidate in ``right``. ``excluded`` lists ``(left, right)`` entity
        pairs that are never selected, so an entity takes its best remaining
        counterpart, and none when every candidate is excluded.
        """
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        if not len(left) or not len(right):
            return Alignment()
        rows, columns = _positions(left, right, _pair_array(excluded))
        best = np.empty(len(left), dtype=np.int64)
        found = np.ones(len(left), dtype=bool)
        cuda = self.device != "cpu"
        if cuda:
            import torch
        for start, block in (self._cuda_blocks if cuda else self._blocks)(left, right):
            stop = start + len(block)
            low, high = np.searchsorted(rows, (start, stop))
            if high > low:
                cells = rows[low:high] - start, columns[low:high]
                if cuda:
                    cells = tuple(torch.as_tensor(side, device=self.device) for side in cells)
                block[cells] = float("-inf")
            if cuda:
                values, indices = block.max(dim=1)
                best[start:stop] = indices.cpu().numpy()
                found[start:stop] = torch.isfinite(values).cpu().numpy()
            else:
                indices = np.argmax(block, axis=1)
                best[start:stop] = indices
                found[start:stop] = np.isfinite(block[np.arange(len(block)), indices])
        left, counterparts = left[found], right[best[found]]
        return Alignment(left, counterparts, self.pair_scores(left, counterparts))

    def reciprocal_nearest(self, left: np.ndarray, right: np.ndarray, *,
                           excluded: np.ndarray | None = None) -> Alignment:
        """Keep pairs where both entities rank each other first, never selecting ``excluded`` pairs."""
        excluded = _pair_array(excluded)
        forward = self.nearest(left, right, excluded=excluded)
        if not len(forward):
            return forward
        reverse = Embeddings(self.right, self.left, self.metric, csls_k=self.csls_k, device=self.device)
        backward = reverse.nearest(right, left, excluded=excluded[:, ::-1])
        best = np.full(len(self.right), -1, dtype=np.int64)
        best[backward.left] = backward.right
        return forward.subset(best[forward.right] == forward.left)

    def ranks(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Rank (from 1) of ``right[k]`` among all of ``right`` for each ``left[k]``."""
        ranks = np.empty(len(left), dtype=np.int64)
        if self.device == "cpu":
            for start, block in self._blocks(left, right):
                gold = block[np.arange(len(block)), np.arange(start, start + len(block))]
                ranks[start:start + len(block)] = (block > gold[:, None]).sum(axis=1) + 1
        else:
            for start, block in self._cuda_blocks(left, right):
                gold = block[:, start:start + len(block)].diagonal()
                ranks[start:start + len(block)] = ((block > gold[:, None]).sum(dim=1) + 1).cpu().numpy()
        return ranks

    def _blocks(self, left: np.ndarray, right: np.ndarray):
        """Yield ``(start, similarity block)`` over ``left`` against all of ``right``."""
        a, b = self._prepared(self.left[left]), self._prepared(self.right[right])
        left_density = right_density = None
        if self.metric == "csls":
            k = max(1, min(self.csls_k, len(right), len(left)))
            left_density = _neighbor_density(a, b, k)
            right_density = _neighbor_density(b, a, k)
        rows = max(1, _BLOCK_ENTRIES // max(1, len(right) * (self.dim if self.metric == "l1" else 1)))
        for start in range(0, len(a), rows):
            block = a[start:start + rows]
            if self.metric == "l1":
                values = -np.abs(block[:, None, :] - b[None, :, :]).sum(axis=2)
            else:
                values = block @ b.T
            if left_density is not None:
                values = 2.0 * values - left_density[start:start + rows, None] - right_density[None, :]
            yield start, values

    def _prepared(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric in ("cosine", "csls"):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        return vectors

    def _cuda_blocks(self, left: np.ndarray, right: np.ndarray):
        import torch

        a = torch.as_tensor(self._prepared(self.left[left]), device=self.device)
        b = torch.as_tensor(self._prepared(self.right[right]), device=self.device)
        rows = max(1, _BLOCK_ENTRIES // max(1, len(b)))
        left_density = right_density = None
        if self.metric == "csls":
            k = max(1, min(self.csls_k, len(a), len(b)))
            left_density = _cuda_neighbor_density(a, b, k)
            right_density = _cuda_neighbor_density(b, a, k)
        for start in range(0, len(a), rows):
            block = a[start:start + rows]
            if self.metric == "l1":
                values = -torch.cdist(block, b, p=1)
            else:
                values = block @ b.T
            if left_density is not None:
                values = 2.0 * values - left_density[start:start + rows, None] - right_density[None, :]
            yield start, values

    def save(self, path: PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream, left=self.left, right=self.right, metric=np.array(self.metric),
                csls_k=np.array(self.csls_k),
            )

    @classmethod
    def load(cls, path: PathLike) -> Embeddings:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["left"], data["right"], str(data["metric"]), csls_k=int(data["csls_k"]))


def _pair_array(pairs: object) -> np.ndarray:
    """``(left, right)`` pairs given as an array, list, or set, as an ``(n, 2)`` integer array."""
    if pairs is None:
        return np.zeros((0, 2), dtype=np.int64)
    if not isinstance(pairs, np.ndarray):
        pairs = list(pairs)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _positions(left: np.ndarray, right: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column positions of the entity ``pairs`` within ``left`` and ``right``, sorted by row."""
    cells = []
    if len(pairs):
        left_positions, right_positions = {}, {}
        for position in np.flatnonzero(np.isin(left, pairs[:, 0])).tolist():
            left_positions.setdefault(int(left[position]), []).append(position)
        for position in np.flatnonzero(np.isin(right, pairs[:, 1])).tolist():
            right_positions.setdefault(int(right[position]), []).append(position)
        cells = sorted((row, column) for a, b in pairs.tolist()
                       for row in left_positions.get(a, ()) for column in right_positions.get(b, ()))
    cells = np.array(cells, dtype=np.int64).reshape(-1, 2)
    return cells[:, 0], cells[:, 1]


def _neighbor_density(queries: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine of each query's ``k`` nearest targets (vectors are unit length)."""
    density = np.empty(len(queries), dtype=np.float32)
    rows = max(1, _BLOCK_ENTRIES // max(1, len(targets)))
    for start in range(0, len(queries), rows):
        values = queries[start:start + rows] @ targets.T
        top = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
        density[start:start + rows] = top.mean(axis=1)
    return density


def _cuda_neighbor_density(queries, targets, k: int):
    density = queries.new_empty(len(queries))
    rows = max(1, _BLOCK_ENTRIES // max(1, len(targets)))
    for start in range(0, len(queries), rows):
        values = queries[start:start + rows] @ targets.T
        density[start:start + rows] = values.topk(k, dim=1, sorted=False).values.mean(dim=1)
    return density


@runtime_checkable
class EmbeddingModel(Protocol):
    """A model that produces entity vectors for a graph pair.

    ``fit`` returns one row per entity in each graph's index order. A model
    may define ``propose(embeddings, data) -> Alignment`` to replace default
    nearest-neighbor retrieval. Its proposals should skip ``data.rejected``,
    for example with ``excluded=data.rejected``; the pipeline removes any
    rejected pair that remains.
    """

    name: str

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings: ...
