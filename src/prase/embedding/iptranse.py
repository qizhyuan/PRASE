"""IPTransE with parameter sharing, additive paths, and iterative soft alignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil, erf, sqrt
from typing import TYPE_CHECKING

from ..errors import ConfigError, DataError
from .base import Embeddings, TrainingData
from .common import (
    SeedValidation, check_device, finite_number, nearest_neighbors, positive_int, to_embeddings, training_runtime,
)

if TYPE_CHECKING:
    from torch import Tensor


class IPTransE:
    """IPTransE (Zhu et al., IJCAI 2017) with the OpenEA training schedule.

    Seed pairs share one entity row. TransE trains on the triples of both
    graphs and on two-hop paths composed into a shortcut relation. Every
    ``bootstrap_every`` epochs from ``bootstrap_start``, nearest neighbors with
    a cosine above ``sim_th`` exchange their triples, weighted by that cosine.
    Rejected pairs never exchange triples.
    """

    name = "iptranse"

    def __init__(
        self,
        *,
        device: str = "cpu",
        dim: int = 100,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 5000,
        block_size: int = 512,
        negative_samples: int = 1,
        margin: float = 1.5,
        path_weight: float = 0.1,
        path_branch_limit: int = 100,
        bootstrap_every: int = 100,
        bootstrap_start: int | None = None,
        sim_th: float = 0.7,
    ):
        self.device = check_device(device)
        self.dim = positive_int("dim", dim)
        self.epochs = positive_int("epochs", epochs)
        self.batch_size = positive_int("batch_size", batch_size)
        self.block_size = positive_int("block_size", block_size)
        self.negative_samples = positive_int("negative_samples", negative_samples)
        self.path_branch_limit = positive_int("path_branch_limit", path_branch_limit)
        self.bootstrap_every = positive_int("bootstrap_every", bootstrap_every)
        self.bootstrap_start = positive_int(
            "bootstrap_start", self.bootstrap_every if bootstrap_start is None else bootstrap_start,
        )
        self.learning_rate = finite_number("learning_rate", learning_rate)
        self.margin = finite_number("margin", margin)
        self.path_weight = finite_number("path_weight", path_weight)
        self.sim_th = finite_number("sim_th", sim_th)
        if min(self.learning_rate, self.margin) <= 0:
            raise ConfigError("learning_rate and margin must be positive")
        if self.path_weight < 0 or not 0 <= self.sim_th <= 1:
            raise ConfigError("path_weight must be nonnegative and sim_th must be in [0, 1]")

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        """Train one shared table in which seed pairs are a single entity."""
        if not len(data.seeds):
            raise DataError("IPTransE requires at least one seed pair")
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(torch, data, seed, self.device, self.epochs,
                                    start=max(10, min(self.bootstrap_start, self.epochs // 2)))
        data = validation.training
        rows, count = _shared_entity_rows(data)
        row_indices = tuple(torch.tensor(side, dtype=torch.long, device=self.device) for side in rows)
        left, right = _prepare_training_graphs((data.left, data.right), rows)
        if not (left.triples or right.triples):
            raise DataError("IPTransE requires relation triples")
        pools = tuple(sorted(pool.tolist()) for pool in data.candidates)
        candidates = tuple(
            torch.tensor([side[i] for i in pool], device=self.device, dtype=torch.long)
            for side, pool in zip(rows, pools, strict=True)
        )
        left_position, right_position = ({entity: k for k, entity in enumerate(pool)} for pool in pools)
        rejected = {(left_position[i], right_position[j]) for i, j in data.rejected_pairs
                    if i in left_position and j in right_position}
        left_vectors, right_vectors = self._train(torch, generator, (left, right), count, candidates,
                                                  row_indices, validation, rejected)
        return to_embeddings(left_vectors, right_vectors, "cosine", info=validation.summary())

    def _train(self, torch, generator, sides, entity_count, candidates, row_indices, validation, rejected):
        predicate_count = sum(len(side.predicates) for side in sides)
        if entity_count * predicate_count * entity_count > 2**63:
            raise DataError("IPTransE's triple index exceeds the supported integer range")
        entities = _embedding_table(torch, entity_count, self.dim, self.device, generator)
        relations = _embedding_table(torch, predicate_count, self.dim, self.device, generator)
        parameters = [entities.weight, relations.weight]
        optimizer = torch.optim.Adagrad(parameters, lr=self.learning_rate, initial_accumulator_value=0.1)
        alignment_optimizer = torch.optim.Adagrad(parameters, lr=self.learning_rate, initial_accumulator_value=0.1)

        normal = tuple(self._prepare_side(torch, side, entity_count, predicate_count) for side in sides)
        incidents = None
        if self.bootstrap_start <= self.epochs and all(len(pool) for pool in candidates):
            incidents = tuple(
                _incident_triples(side.triples, set(pool.tolist()))
                for side, pool in zip(sides, candidates, strict=True)
            )
            full_entities = torch.arange(entity_count, device=self.device)
            all_keys = torch.cat(tuple(side.positives for side in normal)).unique(sorted=True)
        steps = ceil(sum(len(side.triples) for side in normal) / self.batch_size)

        def encode():
            return tuple(_normalized_rows(torch, entities, side) for side in row_indices)

        for epoch in range(1, self.epochs + 1):
            batch_streams = tuple(_epoch_batches(torch, side, steps, generator) for side in normal)
            for batches in zip(*batch_streams, strict=True):
                terms = []
                for side, (triples, paths, weights) in zip(normal, batches, strict=True):
                    if len(triples):
                        terms.append(self._triple_loss(
                            torch, generator, entities, relations, triples, side.entities, side.positives,
                            entity_count, predicate_count,
                        ))
                    if len(paths):
                        terms.append(self.path_weight * _path_loss(
                            torch, relations, paths, weights, side.predicates, generator, self.margin,
                        ))
                if terms:
                    _step(optimizer, sum(terms))

            if (incidents is not None and epoch >= self.bootstrap_start
                    and (epoch - self.bootstrap_start) % self.bootstrap_every == 0):
                selected = self._bootstrap(torch, entities, candidates, rejected)
                swapped = _swapped_triples(incidents, selected)
                if swapped:
                    triples = _long_tensor(torch, (row[:3] for row in swapped), self.device)
                    weights = torch.tensor([row[3] for row in swapped], device=self.device, dtype=torch.float32)
                    order = torch.randperm(len(swapped), device=self.device, generator=generator)
                    for batch in order.split(self.batch_size):
                        loss = self._triple_loss(
                            torch, generator, entities, relations, triples[batch], full_entities, all_keys,
                            entity_count, predicate_count, weights=weights[batch],
                        )
                        _step(alignment_optimizer, loss)
            if validation.stop(epoch, encode):
                break
        return validation.vectors(encode)

    def _prepare_side(self, torch, side, entity_count, predicate_count):
        triples = _long_tensor(torch, side.triples, self.device)
        paths = _two_hop_paths(side.triples, self.path_branch_limit) if self.path_weight else ()
        return _TrainingTensors(
            triples=triples,
            entities=torch.tensor(side.entities, dtype=torch.long, device=self.device),
            predicates=torch.tensor(side.predicates, dtype=torch.long, device=self.device),
            positives=_triple_keys(triples, entity_count, predicate_count).unique(sorted=True),
            paths=_long_tensor(torch, (path[:3] for path in paths), self.device),
            path_weights=torch.tensor([path[3] for path in paths], dtype=torch.float32, device=self.device),
        )

    def _triple_loss(self, torch, generator, entities, relations, positives, pool, keys,
                     entity_count, predicate_count, *, weights=None):
        positives = positives.repeat_interleave(self.negative_samples, dim=0)
        if weights is not None:
            weights = weights.repeat_interleave(self.negative_samples)
        negatives, valid = _negative_triples(
            torch, generator, positives, pool, keys, entity_count, predicate_count,
        )
        costs = _energy(torch, entities, relations, positives)
        negative_costs = _energy(torch, entities, relations, negatives)
        terms = (self.margin + costs - negative_costs).relu() * valid
        return (terms if weights is None else terms * weights).sum()

    def _bootstrap(self, torch, entities, candidates, rejected):
        """Nearest right candidate of each left candidate, kept when the cosine exceeds ``sim_th``.

        ``rejected`` holds candidate positions that are never selected.
        """
        with torch.no_grad():
            left, right = (_normalized_rows(torch, entities, rows) for rows in candidates)
            matches = nearest_neighbors(torch, left, right, _cosine_block, self.block_size,
                                        largest=True, reciprocal=False, excluded=rejected)
        left_ids, right_ids = (rows.tolist() for rows in candidates)
        return [(left_ids[i], right_ids[j], score) for i, j, score in matches if score > self.sim_th]


def _shared_entity_rows(data):
    left_size, right_size = data.left.num_entities, data.right.num_entities
    left = tuple(range(left_size))
    shared = {}
    seen_left = set()
    for i, j in data.seed_pairs:
        if not (0 <= i < left_size and 0 <= j < right_size) or i in seen_left or j in shared:
            raise DataError("IPTransE seeds must be one-to-one pairs of valid entity indices")
        shared[j] = i
        seen_left.add(i)
    right = []
    next_id = left_size
    for j in range(right_size):
        if j in shared:
            right.append(shared[j])
        else:
            right.append(next_id)
            next_id += 1
    return (left, tuple(right)), next_id


@dataclass(frozen=True, slots=True)
class _TrainingGraph:
    triples: tuple[tuple[int, int, int], ...]
    entities: tuple[int, ...]
    predicates: tuple[int, ...]


def _prepare_training_graphs(graphs, rows):
    predicate_offset = 0
    sides = []
    for graph, side in zip(graphs, rows, strict=True):
        triples = graph.relation_triples.tolist()
        relation_ids = sorted({p for _, p, _ in triples})
        predicates = {p: predicate_offset + i for i, p in enumerate(relation_ids)}
        predicate_offset += len(predicates)
        sides.append(_TrainingGraph(
            tuple(sorted((side[h], predicates[p], side[t]) for h, p, t in triples)),
            tuple(side),
            tuple(predicates.values()),
        ))
    return tuple(sides)


def _two_hop_paths(triples, max_product):
    """Find shortcut relations closed by two directed edges in the same graph."""
    outgoing = defaultdict(lambda: defaultdict(list))
    for head, relation, tail in triples:
        outgoing[head][relation].append(tail)
    paths = []
    for head, branches in outgoing.items():
        shortcuts = defaultdict(list)
        for relation, tails in branches.items():
            for tail in tails:
                shortcuts[tail].append(relation)
        candidates = []
        for first, next_entities in branches.items():
            for intermediate in next_entities:
                for second, targets in outgoing.get(intermediate, {}).items():
                    product = len(next_entities) * len(targets)
                    if product > max_product:
                        continue
                    weight = 1 / product
                    for target in targets:
                        if target not in shortcuts:
                            continue
                        candidates.append((first, second, target, weight))
        for first, second, target, weight in candidates:
            paths.extend((first, second, relation, weight) for relation in shortcuts[target])
    return tuple(paths)


@dataclass(slots=True)
class _TrainingTensors:
    triples: Tensor
    entities: Tensor
    predicates: Tensor
    positives: Tensor
    paths: Tensor
    path_weights: Tensor


def _long_tensor(torch, rows, device):
    return torch.tensor(list(rows), dtype=torch.long, device=device).reshape(-1, 3)


def _epoch_batches(torch, side, steps, generator):
    triple_order = torch.randperm(len(side.triples), device=side.triples.device, generator=generator)
    path_order = torch.randperm(len(side.paths), device=side.paths.device, generator=generator)
    for triples, paths in zip(triple_order.tensor_split(steps), path_order.tensor_split(steps), strict=True):
        yield side.triples[triples], side.paths[paths], side.path_weights[paths]


def _incident_triples(triples, candidates):
    incidence = defaultdict(list)
    for head, relation, tail in triples:
        if head in candidates:
            incidence[head].append((head, relation, tail, 0))
        if tail in candidates:
            incidence[tail].append((head, relation, tail, 2))
    return incidence


def _swapped_triples(incidents, pairs):
    rows = set()
    for left, right, weight in pairs:
        for side, entity, counterpart in ((0, left, right), (1, right, left)):
            for head, relation, tail, column in incidents[side].get(entity, ()):
                rows.add((counterpart if column == 0 else head, relation, counterpart if column == 2 else tail, weight))
    return sorted(rows)


def _embedding_table(torch, rows, dim, device, generator):
    """Truncated normal initialization within two standard deviations, as in OpenEA."""
    values = torch.empty((rows, dim), dtype=torch.float32, device=device)
    limit = erf(sqrt(2))
    values.uniform_(-limit, limit, generator=generator).erfinv_().mul_(sqrt(2 / dim))
    return torch.nn.Embedding.from_pretrained(values, freeze=False, sparse=device == "cpu")


def _normalized_rows(torch, table, indices):
    return torch.nn.functional.normalize(table(indices), dim=-1)


def _energy(torch, entities, relations, triples):
    head, predicate, tail = triples.T
    delta = (
        _normalized_rows(torch, entities, head)
        + _normalized_rows(torch, relations, predicate)
        - _normalized_rows(torch, entities, tail)
    )
    return delta.square().sum(dim=1)


def _triple_keys(triples, entity_count, predicate_count):
    head, predicate, tail = triples.T
    return (head * predicate_count + predicate) * entity_count + tail


def _negative_triples(torch, generator, triples, pool, known, entity_count, predicate_count):
    """Corrupt the head or tail of each triple, retrying observed triples up to eight times."""
    negatives = triples.clone()
    unresolved = torch.arange(len(triples), device=triples.device)
    choices = ((0, pool), (2, pool))
    valid = torch.zeros(len(triples), dtype=torch.bool, device=triples.device)
    for _ in range(8):
        if len(unresolved) == 0:
            break
        trial = triples[unresolved].clone()
        columns = torch.randint(len(choices), (len(unresolved),), device=triples.device, generator=generator)
        for option, (column, candidates) in enumerate(choices):
            chosen = columns == option
            samples = torch.randint(len(candidates), (len(trial),), device=triples.device, generator=generator)
            trial[:, column] = torch.where(chosen, candidates[samples], trial[:, column])
        different = (trial != triples[unresolved]).any(dim=1)
        if len(known):
            keys = _triple_keys(trial, entity_count, predicate_count)
            nearest = torch.searchsorted(known, keys).clamp(max=len(known) - 1)
            different &= known[nearest] != keys
        resolved = unresolved[different]
        valid[resolved] = True
        negatives[resolved] = trial[different]
        unresolved = unresolved[~different]
    return negatives, valid


def _path_loss(torch, relations, paths, weights, pool, generator, margin):
    first, second, direct = paths.T
    samples = torch.randint(len(pool), (len(paths),), device=paths.device, generator=generator)
    composition = _normalized_rows(torch, relations, first) + _normalized_rows(torch, relations, second)
    positive = (composition - _normalized_rows(torch, relations, direct)).square().sum(dim=1)
    negative = (composition - _normalized_rows(torch, relations, pool[samples])).square().sum(dim=1)
    return (weights * (margin + positive - negative).relu()).sum()


def _cosine_block(left, right):
    return (left @ right.T).clamp(-1, 1)


def _step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
