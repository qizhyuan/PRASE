"""Helpers shared by the built-in PyTorch models."""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from dataclasses import replace
from functools import lru_cache
from math import isfinite

import numpy as np

from ..errors import ConfigError
from .base import Embeddings, TrainingData


_LOGGER = logging.getLogger("prase")


def positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def nonnegative_int(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{name} must be a nonnegative integer")
    return value


def finite_number(name: str, value: float) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    return float(value)


def fraction(name: str, value: float) -> float:
    value = finite_number(name, value)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be between 0 and 1")
    return value


def check_device(device: str) -> str:
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ConfigError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    return device


def training_runtime(device: str, seed: int):
    """Import PyTorch and return it with a generator seeded for this run."""
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise ImportError("embedding models require PyTorch: pip install 'prase[train]'") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"device {device!r} requires a CUDA-enabled PyTorch")
    return torch, torch.Generator(device=device).manual_seed(seed)


def to_embeddings(left, right, metric: str = "cosine", *, csls_k: int = 10, info: dict | None = None) -> Embeddings:
    """Convert two PyTorch matrices to :class:`Embeddings`."""
    return Embeddings(
        left.detach().float().cpu().numpy(), right.detach().float().cpu().numpy(), metric,
        csls_k=csls_k, device=str(left.device), info=info,
    )


class SeedValidation:
    """Select a checkpoint by bidirectional MRR on held-out training seeds.

    Optional distractors come from the graph's unmatched candidate pools. The
    held-out endpoints cannot be selected for bootstrapping.
    """

    def __init__(self, torch, data: TrainingData, seed: int, device: str, epochs: int, *,
                 start: int = 10, interval: int = 10, patience: int = 5, metric: str = "cosine", csls_k: int = 10,
                 distractors: int = 0):
        self.torch = torch
        self.training = data
        self.indices = None
        self.epochs, self.start, self.interval, self.patience = epochs, start, interval, patience
        self.metric = metric
        self.csls_k = csls_k
        self.best = None
        self.best_score = -1.0
        self.best_epoch = 0
        self.last_epoch = 0
        self.stale = 0
        seeds = data.seeds
        if len(seeds) < 4:
            return
        # Only hold out pairs whose endpoints cannot occur in another training pair.
        unique = [np.unique(side, return_inverse=True, return_counts=True) for side in seeds.T]
        eligible = np.flatnonzero(np.logical_and.reduce([counts[inverse] == 1 for _, inverse, counts in unique]))
        count = min(len(eligible), max(2, len(seeds) // 10), 512)
        if count < 2:
            return
        rng = np.random.default_rng(seed)
        indices = rng.choice(eligible, count, replace=False)
        keep = np.ones(len(seeds), dtype=bool)
        keep[indices] = False
        held_out = seeds[indices]
        candidates = tuple(pool[~np.isin(pool, side)] for pool, side in zip(data.candidates, held_out.T, strict=True))
        self.training = replace(data, seeds=seeds[keep], candidates=candidates)
        self.indices = torch.as_tensor(held_out.T.copy(), dtype=torch.long, device=device)
        self._retrieval_indices = tuple(self.indices)
        if distractors:
            self._retrieval_indices = tuple(
                torch.as_tensor(np.concatenate((known, rng.choice(pool, min(distractors, len(pool)), replace=False))),
                                dtype=torch.long, device=device)
                for known, pool in zip(held_out.T, candidates, strict=True)
            )

    def stop(self, epoch: int, encode, *, parameters=()) -> bool:
        """Evaluate on schedule. ``encode`` returns the current two vector tables."""
        self.last_epoch = epoch
        if self.indices is None or not (
            epoch == self.epochs or (epoch >= self.start and (epoch - self.start) % self.interval == 0)
        ):
            return False
        torch = self.torch
        with torch.no_grad():
            vectors = encode()
            left, right = (table[ids] for table, ids in zip(vectors, self._retrieval_indices, strict=True))
            if self.metric == "l1":
                similarity = -torch.cdist(left, right, p=1)
            else:
                normalize = torch.nn.functional.normalize
                similarity = normalize(left, dim=1) @ normalize(right, dim=1).T
                if self.metric == "csls":
                    k = min(self.csls_k, len(left), len(right))
                    left_density = similarity.topk(k, dim=1).values.mean(dim=1, keepdim=True)
                    right_density = similarity.topk(k, dim=0).values.mean(dim=0, keepdim=True)
                    similarity = 2.0 * similarity - left_density - right_density
            count = self.indices.shape[1]
            matches = similarity[:count, :count].diag()
            # Use average rank for ties so a constant representation cannot score perfectly.
            row_ranks = _validation_ranks(similarity[:count], matches[:, None], 1)
            col_ranks = _validation_ranks(similarity[:, :count], matches[None, :], 0)
            score = float((row_ranks.reciprocal().mean() + col_ranks.reciprocal().mean()) * 0.5)
            _LOGGER.debug("Validation MRR at epoch %d: %.5f", epoch, score)
            if score > self.best_score + 1e-5:
                self.best = tuple(table.detach().clone() for table in (parameters or vectors))
                self.best_score = score
                self.best_epoch = epoch
                self.stale = 0
            else:
                self.stale += 1
        return self.stale >= self.patience

    def vectors(self, encode):
        self._report()
        with self.torch.no_grad():
            return self.best if self.best is not None else encode()

    def restore(self, parameters):
        self._report()
        if self.best is not None:
            with self.torch.no_grad():
                for parameter, value in zip(parameters, self.best, strict=True):
                    parameter.copy_(value)

    def summary(self) -> dict:
        """Epochs trained, the selected epoch, and its validation score, for the run records."""
        result = {"epochs": self.last_epoch}
        if self.best is not None:
            result.update(selected_epoch=self.best_epoch, validation_mrr=round(self.best_score, 6),
                          validation_pairs=int(self.indices.shape[1]))
        return result

    def _report(self):
        if self.best is not None:
            reason = "patience exhausted" if self.stale >= self.patience else "epoch cap reached"
            _LOGGER.info("Selected epoch %d at validation MRR %.4f on %d seed pairs after %d epochs (%s)",
                         self.best_epoch, self.best_score, self.indices.shape[1], self.last_epoch, reason)
        else:
            _LOGGER.info("Used the final checkpoint at epoch %d with insufficient validation seeds", self.last_epoch)


def _validation_ranks(similarity, matches, dimension):
    greater = (similarity > matches).sum(dim=dimension)
    equal = (similarity == matches).sum(dim=dimension)
    return greater + (equal + 1) * 0.5


def nearest_neighbors(torch, left, right, score, block_size: int, *, largest: bool, reciprocal: bool,
                      excluded: Collection[tuple[int, int]] = ()):
    """Find one nearest index per row, keeping the lower index on equal scores.

    Used inside training loops for bootstrapping; returns ``(row, column, score)``.
    ``excluded`` holds ``(row, column)`` positions that are never selected, so
    a row takes its best remaining column, and none when all are excluded.
    """
    if len(left) == 0 or len(right) == 0:
        return ()
    fill = float("-inf") if largest else float("inf")
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, column in excluded:
        row, column = int(row), int(column)
        groups.setdefault((row // block_size, column // block_size), []).append(
            (row % block_size, column % block_size))
    masks = {key: torch.tensor(cells, dtype=torch.long, device=left.device).T for key, cells in groups.items()}
    left_scores = left.new_full((len(left),), fill)
    left_indices = left.new_zeros((len(left),), dtype=torch.long)
    right_scores = right.new_full((len(right),), fill) if reciprocal else None
    right_indices = right.new_zeros((len(right),), dtype=torch.long) if reciprocal else None
    for left_start in range(0, len(left), block_size):
        left_block = slice(left_start, left_start + block_size)
        for right_start in range(0, len(right), block_size):
            right_block = slice(right_start, right_start + block_size)
            scores = score(left[left_block], right[right_block])
            cells = masks.get((left_start // block_size, right_start // block_size))
            if cells is not None:
                scores[cells[0], cells[1]] = fill
            _update_nearest(scores, 1, right_start, left_scores[left_block], left_indices[left_block], largest)
            if right_indices is not None:
                _update_nearest(scores, 0, left_start, right_scores[right_block], right_indices[right_block], largest)
    matches = zip(left_indices.cpu().tolist(), left_scores.cpu().tolist())
    reverse = None if right_indices is None else right_indices.cpu().tolist()
    return tuple(
        (index, counterpart, value)
        for index, (counterpart, value) in enumerate(matches)
        if isfinite(value) and (reverse is None or reverse[counterpart] == index)
    )


def _update_nearest(scores, axis: int, offset: int, best_scores, best_indices, largest: bool):
    values, indices = scores.max(dim=axis) if largest else scores.min(dim=axis)
    improved = values > best_scores if largest else values < best_scores
    best_scores.copy_(values.where(improved, best_scores))
    best_indices.copy_((offset + indices).where(improved, best_indices))


@lru_cache(maxsize=8_192)
def _character_positions(text):
    positions = {}
    for index, char in enumerate(text):
        positions[char] = positions.get(char, 0) | (1 << index)
    return positions


@lru_cache(maxsize=32_768)
def indel_similarity(left: str, right: str) -> float:
    """Normalized indel similarity via a bit-parallel longest common subsequence."""
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) < len(right):
        left, right = right, left
    positions = _character_positions(left)
    state = 0
    for char in right:
        matched = state | positions.get(char, 0)
        state = matched & ~(matched - ((state << 1) | 1))
    return 2 * state.bit_count() / (len(left) + len(right))
