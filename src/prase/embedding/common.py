"""Helpers shared by the built-in PyTorch models."""

from __future__ import annotations

import copy
import logging
import math
import re
from collections.abc import Collection
from dataclasses import replace
from functools import lru_cache
from math import isfinite

import numpy as np

from ..checkpoints import TrainingCheckpoint
from ..errors import ConfigError
from ..progress import Progress
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
    """Select the best epoch by bidirectional MRR on held-out training seeds.

    Optional distractors come from the graph's unmatched candidate pools. The
    held-out endpoints cannot be selected for bootstrapping. Without enough
    seeds to hold out, training runs for ``unvalidated_epochs`` epochs, or
    ``epochs`` when it is None, and keeps the final vectors.

    The validation also logs training progress with an estimate of the time
    left, and saves and restores training checkpoints when
    :func:`prase.training_checkpoints` is active. A model's loop runs::

        first = validation.resume(generator, tensors, optimizers, best=encode, check=state_fits, state=state)
        for epoch in range(first, epochs + 1):
            ...  # one epoch
            if validation.stop(epoch, encode):
                break
            validation.checkpoint(epoch)
        return validation.vectors(encode)  # saves the finished fit

    A checkpoint whose parts do not fit the fit, or whose model state fails
    ``check``, is discarded before anything is restored, and training starts
    from the first epoch.
    """

    def __init__(self, torch, data: TrainingData, seed: int, device: str, epochs: int, *,
                 start: int = 10, interval: int = 10, patience: int = 5, metric: str = "cosine", csls_k: int = 10,
                 distractors: int = 0, model: object = None, unvalidated_epochs: int | None = None):
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
        self.last_check = 0
        self.stale = 0
        self.score = None
        self._objects = None  # the generator, tables, optimizers, and model state that resume registers
        self._finished = False
        name = getattr(model, "name", "embedding")
        self.label = f"{_MODEL_TITLES.get(name, name)} training"
        self._checkpoint = None if model is None else TrainingCheckpoint.active(torch, model, data, seed, device)
        self._hold_out(data, seed, device, distractors)
        self.cap = epochs
        if self.indices is None and unvalidated_epochs is not None:
            self.cap = min(epochs, unvalidated_epochs)
        self.progress = Progress(self.label, self.cap, unit="epoch")

    def _hold_out(self, data, seed, device, distractors):
        torch = self.torch
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

    def saved_state(self, check=None) -> dict | None:
        """The model's own state in the checkpoint this fit resumes from, or None.

        ``check(state)`` returns whether the state fits this fit. A state that
        fails it, or makes it raise, discards the checkpoint.
        """
        payload = None if self._checkpoint is None else self._checkpoint.payload
        if payload is None:
            return None
        state = payload.get("state")
        try:
            fits = isinstance(state, dict) and (check is None or bool(check(state)))
        except Exception:  # a part of the wrong type or shape
            fits = False
        if not fits:
            self._checkpoint.discard("holds model state that does not fit this fit")
            return None
        return state

    def resume(self, generator, tensors, optimizers=(), *, best, check=None, state=None) -> int:
        """Register the training objects, restore a saved state into them, and return the first epoch to train.

        ``best`` returns the tables that the validation keeps for the best
        epoch, to check the saved ones against, and ``state`` returns the
        model's other training state for :meth:`checkpoint`. Every part of a
        checkpoint is read and checked before anything changes, and a failure
        while applying it undoes what was applied. A checkpoint that does not
        fit is discarded, and training starts from the first epoch. A finished
        fit returns its vectors without training.
        """
        self._objects = (generator, list(tensors), list(optimizers), state)
        if self.saved_state(check) is None:
            return 1
        try:
            restored = self._read(self._checkpoint.payload, best)
        except Exception as error:  # a missing, mistyped, or inconsistent part
            self._checkpoint.discard(f"does not fit this fit ({type(error).__name__}: {error})")
            return 1
        snapshot = self._snapshot()
        try:
            self._apply(restored)
        except Exception as error:
            self._apply(snapshot)
            self._checkpoint.discard(f"could not be applied ({type(error).__name__}: {error})")
            return 1
        self.progress.resume(self.last_epoch)
        if restored["finished"]:
            self._finished = True
            _LOGGER.info("%s: reused the finished fit from %s", self.label, self._checkpoint.path)
            return self.epochs + 1
        _LOGGER.info("%s: resumed after epoch %d from %s", self.label, self.last_epoch, self._checkpoint.path)
        return self.last_epoch + 1

    def _read(self, payload: dict, best) -> dict:
        """The saved state as values to apply, or an error naming the part that does not fit."""
        torch = self.torch
        generator, tensors, optimizers, _ = self._objects
        epoch, finished = payload["epoch"], payload.get("finished", False)
        if type(epoch) is not int or not isinstance(finished, bool):
            raise ValueError("the epoch is not a number")
        if not (1 <= epoch <= self.cap if finished else 1 <= epoch < self.cap):
            raise ValueError(f"epoch {epoch} is outside this training")
        saved = payload["tensors"]
        if len(saved) != len(tensors) or any(
            not isinstance(value, torch.Tensor) or value.shape != tensor.shape or value.dtype != tensor.dtype
            or not bool(torch.isfinite(value).all()) for value, tensor in zip(saved, tensors, strict=False)
        ):
            raise ValueError("the tables differ in number, shape, or type, or are not finite")
        random_state = payload["generator"]
        if not isinstance(random_state, torch.Tensor) or random_state.dtype != torch.uint8:
            raise ValueError("the random state is not a byte tensor")
        torch.Generator(device=generator.device).set_state(random_state.cpu())  # raises for an invalid state
        optimizer_states = payload["optimizers"]
        if len(optimizer_states) != len(optimizers):
            raise ValueError("the number of optimizers differs")
        merged = [_optimizer_state(torch, optimizer, saved_state)
                  for optimizer, saved_state in zip(optimizers, optimizer_states, strict=True)]
        validation = payload["validation"]
        numbers = (int, float)
        best_tables = validation["best"]
        best_score, best_epoch = validation["best_score"], validation["best_epoch"]
        last_check, stale, score = validation["last_check"], validation["stale"], validation["score"]
        for name, value in (("best_epoch", best_epoch), ("last_check", last_check), ("stale", stale)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} is not a count")
        for name, value in (("best_score", best_score), ("score", score)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, numbers)
                                      or not math.isfinite(value)):
                raise ValueError(f"{name} is not a number")
        checked = self.indices is not None and best_epoch > 0  # a best epoch exists once validation ran a check
        # Training stops when stale reaches patience, so only a finished fit can have that count, and a
        # finished fit stopped there or at the epoch cap.
        if best_epoch > epoch or last_check > epoch or stale > self.patience or (
                stale == self.patience and not finished):
            raise ValueError("the validation state is inconsistent")
        if finished and epoch != self.cap and stale != self.patience:
            raise ValueError("a finished fit stops at its epoch cap or when patience runs out")
        if (best_tables is not None) != checked or (score is None) != (self.indices is None or last_check == 0):
            raise ValueError("the validation state is inconsistent")
        if best_tables is not None:
            with torch.no_grad():
                expected = [(table.shape, table.dtype) for table in best()]
            if not isinstance(best_tables, list) or [
                (table.shape, table.dtype) if isinstance(table, torch.Tensor) else None for table in best_tables
            ] != expected or not all(bool(torch.isfinite(table).all()) for table in best_tables):
                raise ValueError("the best epoch has tables of another shape, or is not finite")
            best_tables = tuple(best_tables)
        return {
            "tensors": saved, "optimizers": merged, "generator": random_state.cpu(), "best": best_tables,
            "best_score": float(best_score), "best_epoch": best_epoch, "last_check": last_check, "stale": stale,
            "score": None if score is None else float(score), "last_epoch": epoch, "finished": finished,
        }

    def _snapshot(self) -> dict:
        generator, tensors, optimizers, _ = self._objects
        return {
            "tensors": [tensor.detach().clone() for tensor in tensors],
            "optimizers": [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers],
            "generator": generator.get_state(), "best": self.best, "best_score": self.best_score,
            "best_epoch": self.best_epoch, "last_check": self.last_check, "stale": self.stale, "score": self.score,
            "last_epoch": self.last_epoch, "finished": False,
        }

    def _apply(self, values: dict) -> None:
        generator, tensors, optimizers, _ = self._objects
        with self.torch.no_grad():
            for tensor, value in zip(tensors, values["tensors"], strict=True):
                tensor.copy_(value)
        for optimizer, state in zip(optimizers, values["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        generator.set_state(values["generator"])
        self.best, self.best_score, self.best_epoch = values["best"], values["best_score"], values["best_epoch"]
        self.last_check, self.stale, self.score = values["last_check"], values["stale"], values["score"]
        self.last_epoch = values["last_epoch"]

    def checkpoint(self, epoch: int) -> None:
        """Save the state after ``epoch`` when a checkpoint is due."""
        if self._checkpoint is not None and self._objects is not None and self._checkpoint.due():
            self._save(epoch, finished=False)

    def _save(self, epoch: int, *, finished: bool) -> None:
        generator, tensors, optimizers, state = self._objects
        self._checkpoint.save({
            "epoch": epoch,
            "finished": finished,
            "generator": generator.get_state(),
            "tensors": [tensor.detach() for tensor in tensors],
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "validation": {
                "best": None if self.best is None else list(self.best), "best_score": self.best_score,
                "best_epoch": self.best_epoch, "last_check": self.last_check, "stale": self.stale,
                "score": self.score,
            },
            "state": {} if state is None else state(),
        })

    def stop(self, epoch: int, encode, *, parameters=()) -> bool:
        """Evaluate on schedule and report whether training should stop after ``epoch``.

        ``encode`` returns the current two vector tables.
        """
        self.last_epoch = epoch
        if self.indices is None:
            self._log_progress(epoch)
            return epoch >= self.cap
        if not (epoch == self.epochs or (epoch >= self.start and (epoch - self.start) % self.interval == 0)):
            self._log_progress(epoch)
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
        self.score, self.last_check = score, epoch
        self._log_progress(epoch)
        return self.stale >= self.patience

    def _log_progress(self, epoch: int) -> None:
        if self.indices is None:
            self.progress.update(epoch)
            return
        if self.score is None:
            self.progress.update(epoch, f"; validation starts at epoch {min(self.start, self.epochs)}", bound=True)
            return
        # Without improvement, training stops after `patience - stale` more checks.
        early = min(self.cap, self.last_check + (self.patience - self.stale) * self.interval)
        self.progress.update(
            epoch, f"; validation MRR {self.score:.4f}, best {self.best_score:.4f} at epoch {self.best_epoch}, "
                   f"stops at epoch {early} without improvement", bound=True,
        )

    def vectors(self, encode):
        self._finish()
        with self.torch.no_grad():
            return self.best if self.best is not None else encode()

    def restore(self, parameters):
        self._finish()
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

    def _finish(self):
        # The finished fit stays saved until its result is: a pipeline removes it after saving the step.
        if self._checkpoint is not None and self._objects is not None and not self._finished:
            self._save(self.last_epoch, finished=True)
        if self.best is not None:
            reason = "patience exhausted" if self.stale >= self.patience else "epoch cap reached"
            self.progress.update(self.last_epoch, f" ({reason}); selected epoch {self.best_epoch} at validation MRR "
                                 f"{self.best_score:.4f} on {self.indices.shape[1]} seed pairs", final=True)
        else:
            self.progress.update(self.last_epoch, "; kept the final vectors, with too few seeds to validate",
                                 final=True)


_MODEL_TITLES = {
    "bootea": "BootEA", "gcnalign": "GCN-Align", "imuse": "IMUSE", "iptranse": "IPTransE", "mtranse": "MTransE",
    "multike": "MultiKE",
}


def _optimizer_state(torch, optimizer, saved: dict) -> dict:
    """The saved per-parameter state with the optimizer's own hyperparameters, or an error if they differ in form.

    The saved state must hold the entries, shapes, and types that the
    optimizer keeps itself, as Adagrad and SGD do from their construction. The
    hyperparameters follow the model settings, which the checkpoint key covers,
    so the saved ones are not read.
    """
    current = optimizer.state_dict()
    state = saved["state"]
    if not isinstance(state, dict) or set(state) != set(current["state"]):
        raise ValueError("the optimizer state has other entries")
    for index, entry in current["state"].items():
        values = state[index]
        if not isinstance(values, dict) or set(values) != set(entry):
            raise ValueError("the optimizer state has other fields")
        for name, value in entry.items():
            other = values[name]
            if isinstance(value, torch.Tensor):
                if not isinstance(other, torch.Tensor) or other.shape != value.shape or other.dtype != value.dtype \
                        or (other.is_floating_point() and not bool(torch.isfinite(other).all())):
                    raise ValueError(f"the optimizer's {name} has another shape or type")
            elif type(other) is not type(value):
                raise ValueError(f"the optimizer's {name} has another type")
    return {"state": state, "param_groups": current["param_groups"]}


def is_tensor(torch, value, dtype, ndim: int, columns: int | None = None, limits=None, rows: int | None = None) -> bool:
    """Whether ``value`` is a finite tensor of this type and rank with these rows and columns.

    ``limits`` bounds the values from above, per column when it is a sequence,
    and requires them to be nonnegative.
    """
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != ndim:
        return False
    if (columns is not None and value.shape[-1] != columns) or (rows is not None and value.shape[0] != rows):
        return False
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        return False
    if limits is not None and value.numel():
        limits = torch.as_tensor(limits, device=value.device)
        if bool((value < 0).any()) or bool((value >= limits).any()):
            return False
    return True


def is_sorted_keys(torch, value, limit: int) -> bool:
    """Whether ``value`` is a strictly increasing one-dimensional ``int64`` tensor of keys below ``limit``."""
    return is_tensor(torch, value, torch.long, 1, limits=limit) and (
        len(value) < 2 or bool((value[1:] > value[:-1]).all()))


def _validation_ranks(similarity, matches, dimension):
    greater = (similarity > matches).sum(dim=dimension)
    equal = (similarity == matches).sum(dim=dimension)
    return greater + (equal + 1) * 0.5


# Bounds each score block of the training neighbor search, in matrix entries.
_NEIGHBOR_ENTRIES = 1 << 25


def nearest_neighbors(torch, left, right, score, block_size: int, *, largest: bool, reciprocal: bool,
                      excluded: Collection[tuple[int, int]] | None = ()):
    """Find one nearest index per row, keeping the lower index on equal scores.

    Used inside training loops for bootstrapping; returns ``(row, column, score)``.
    ``excluded`` holds ``(row, column)`` positions that are never selected, so
    a row takes its best remaining column, and none when all are excluded.
    NaN scores are never selected. Blocks hold at least ``block_size`` rows and
    columns and grow up to ``_NEIGHBOR_ENTRIES`` scores, so that a large
    search runs as a few large matrix products.
    """
    if len(left) == 0 or len(right) == 0:
        return ()
    fill = float("-inf") if largest else float("inf")
    columns = min(len(right), max(block_size, _NEIGHBOR_ENTRIES // block_size))
    rows = min(len(left), max(block_size, _NEIGHBOR_ENTRIES // columns))
    masks = _block_masks(torch, excluded, (len(left), len(right)), (rows, columns), left.device)
    left_scores = left.new_full((len(left),), fill)
    left_indices = left.new_zeros((len(left),), dtype=torch.long)
    right_scores = right.new_full((len(right),), fill) if reciprocal else None
    right_indices = right.new_zeros((len(right),), dtype=torch.long) if reciprocal else None
    for left_start in range(0, len(left), rows):
        left_block = slice(left_start, left_start + rows)
        for right_start in range(0, len(right), columns):
            right_block = slice(right_start, right_start + columns)
            scores = score(left[left_block], right[right_block])
            scores.masked_fill_(scores.isnan(), fill)
            cells = masks.get((left_start // rows, right_start // columns))
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


def _block_masks(torch, excluded, shape, block, device):
    """Group excluded ``(row, column)`` positions by score block, as local index tensors on ``device``."""
    if excluded is None:
        return {}
    if hasattr(excluded, "detach"):
        excluded = excluded.detach().cpu().numpy()
    elif not isinstance(excluded, np.ndarray):
        excluded = list(excluded)
    cells = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    inside = (cells >= 0).all(axis=1) & (cells[:, 0] < shape[0]) & (cells[:, 1] < shape[1])
    cells = cells[inside]
    if not len(cells):
        return {}
    keys = cells[:, 0] // block[0] * (-(-shape[1] // block[1])) + cells[:, 1] // block[1]
    order = np.argsort(keys, kind="stable")
    keys, local = keys[order], cells[order] % np.asarray(block)
    local = torch.as_tensor(local.T.copy(), dtype=torch.long, device=device)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    columns = -(-shape[1] // block[1])
    return {(int(keys[s]) // columns, int(keys[s]) % columns): local[:, s:e]
            for s, e in zip(starts.tolist(), ends.tolist(), strict=True)}


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
