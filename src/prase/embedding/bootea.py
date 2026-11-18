"""BootEA training and bootstrapping (Sun et al., IJCAI 2018)."""

from __future__ import annotations

from dataclasses import replace
from math import ceil

import numpy as np

from ..alignment import Alignment
from ..errors import ConfigError, DataError
from ..kg import RELATION
from .base import Embeddings, TrainingData
from .common import (
    SeedValidation, check_device, finite_number, nearest_neighbors, positive_int, to_embeddings, training_runtime,
)


class BootEA:
    """TransE-style training with iterative bootstrapping of new alignment labels.

    Training uses the graph triples, the seed pairs, and triples swapped
    through the seeds and the bootstrapped labels. Every ``bootstrap_every``
    epochs, mutual nearest neighbors with a cosine above ``sim_th`` become
    labels, except rejected pairs. Counterparts are ranked by cross-domain similarity local scaling
    (CSLS) over ``k`` neighbors, and proposals are mutual nearest neighbors.

    ``epochs`` caps training. Validation MRR is checked every ten epochs,
    starting at epoch 50 or after two bootstrap updates, whichever comes later.
    ``patience`` is the number of checks without an improvement before stopping.
    Hard-negative search refreshes every ``neighbor_refresh`` epochs. The
    nearest fraction, ``1 - epsilon``, is taken from at most
    ``neighbor_pool_size`` entities sampled from the same graph.
    """

    name = "bootea"

    def __init__(
        self,
        *,
        device: str = "cpu",
        dim: int = 100,
        epochs: int = 2000,
        patience: int = 10,
        learning_rate: float = 0.01,
        batch_size: int = 4096,
        block_size: int = 512,
        negative_samples: int = 10,
        positive_margin: float = 0.01,
        negative_margin: float = 2.0,
        negative_weight: float = 0.2,
        epsilon: float = 0.9,
        neighbor_refresh: int = 10,
        neighbor_pool_size: int = 16_384,
        bootstrap_every: int = 10,
        sim_th: float = 0.7,
        k: int = 10,
    ):
        self.device = check_device(device)
        self.dim = positive_int("dim", dim)
        self.epochs = positive_int("epochs", epochs)
        self.patience = positive_int("patience", patience)
        self.batch_size = positive_int("batch_size", batch_size)
        self.block_size = positive_int("block_size", block_size)
        self.negative_samples = positive_int("negative_samples", negative_samples)
        self.neighbor_refresh = positive_int("neighbor_refresh", neighbor_refresh)
        self.neighbor_pool_size = positive_int("neighbor_pool_size", neighbor_pool_size)
        self.bootstrap_every = positive_int("bootstrap_every", bootstrap_every)
        self.k = positive_int("k", k)
        self.learning_rate = finite_number("learning_rate", learning_rate)
        self.positive_margin = finite_number("positive_margin", positive_margin)
        self.negative_margin = finite_number("negative_margin", negative_margin)
        self.negative_weight = finite_number("negative_weight", negative_weight)
        self.epsilon = finite_number("epsilon", epsilon)
        self.sim_th = finite_number("sim_th", sim_th)
        if not 0 <= self.positive_margin < self.negative_margin:
            raise ConfigError("margins must satisfy 0 <= positive_margin < negative_margin")
        if min(self.learning_rate, self.negative_weight) <= 0:
            raise ConfigError("learning_rate and negative_weight must be positive")
        if not 0 <= self.epsilon < 1:
            raise ConfigError("epsilon must be in [0, 1)")
        if not 0 <= self.sim_th <= 1:
            raise ConfigError("sim_th must be in [0, 1]")

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        data = _validated_seeds(data)
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(
            torch, data, seed, self.device, self.epochs, start=max(50, 2 * self.bootstrap_every),
            patience=self.patience, metric="csls", csls_k=self.k, distractors=2048,
        )
        left_indices, right_indices = (np.sort(pool) for pool in validation.training.candidates)
        left, right = self._train(torch, generator, validation, left_indices, right_indices)
        return to_embeddings(left, right, "csls", csls_k=self.k, info=validation.summary())

    def propose(self, embeddings: Embeddings, data: TrainingData) -> Alignment:
        """Propose mutual nearest neighbors among the candidates."""
        return embeddings.reciprocal_nearest(*data.candidates, excluded=data.rejected)

    def _train(self, torch, generator, validation, left_indices, right_indices):
        data = validation.training
        left_graph, right_graph = data.left, data.right
        left_size, right_size = left_graph.num_entities, right_graph.num_entities
        entity_count = left_size + right_size
        predicate_count = len(left_graph.predicates) + len(right_graph.predicates)
        if entity_count * entity_count * predicate_count > 2**63:
            raise DataError("BootEA's triple index exceeds the supported integer range")

        shared_relations = {predicate.name: index for index, predicate in enumerate(left_graph.predicates)
                            if predicate.kind == RELATION}
        right_relations = [shared_relations.get(predicate.name, len(left_graph.predicates) + index)
                           for index, predicate in enumerate(right_graph.predicates)]
        triples = set(map(tuple, left_graph.relation_triples.tolist()))
        triples.update(
            (head + left_size, right_relations[predicate], tail + left_size)
            for head, predicate, tail in right_graph.relation_triples.tolist()
        )
        if not triples:
            raise DataError("BootEA requires relation triples")
        seed_pairs = [(i, j + left_size) for i, j in data.seed_pairs]
        seed_indices = torch.tensor(seed_pairs, dtype=torch.long, device=self.device)
        base_triples = triples | _swapped_triples(triples, seed_pairs)
        candidate_rows = set(left_indices) | {left_size + index for index in right_indices}
        bootstrap_triples = {row for row in triples if row[0] in candidate_rows or row[2] in candidate_rows}
        left_rows = torch.tensor(left_indices, dtype=torch.long, device=self.device)
        right_rows = torch.tensor(right_indices, dtype=torch.long, device=self.device) + left_size
        left_position = {entity: row for row, entity in enumerate(left_indices.tolist())}
        right_position = {entity: row for row, entity in enumerate(right_indices.tolist())}
        rejected = {(left_position[i], right_position[j]) for i, j in data.rejected_pairs
                    if i in left_position and j in right_position}
        training_triples = _triples_tensor(torch, base_triples, self.device)
        known_positives = _positive_keys(torch, training_triples, entity_count, predicate_count)

        entities = _unit_parameters(torch, entity_count, self.dim, self.device, generator)
        relations = _unit_parameters(torch, predicate_count, self.dim, self.device, generator)
        tables = (entities, relations)
        optimizer_options = dict(lr=self.learning_rate, initial_accumulator_value=0.1,
                                 fused=self.device.startswith("cuda"))
        optimizer = torch.optim.Adagrad(tables, **optimizer_options)
        alignment_optimizer = torch.optim.Adagrad(tables, **optimizer_options)
        neighbors = (None, None)
        pseudo_labels = {}
        previous_swaps = set()

        def encode():
            return entities[:left_size].detach(), entities[left_size:].detach()

        for epoch in range(1, self.epochs + 1):
            order = torch.randperm(len(training_triples), device=self.device, generator=generator)
            for batch in order.split(self.batch_size):
                positive = training_triples[batch]
                negative, valid = self._sample_negatives(
                    torch, generator, positive, (left_size, right_size), neighbors,
                    known_positives, predicate_count,
                )
                loss = self._triple_loss(torch, entities, relations, positive, negative, valid)
                _gradient_step(torch, optimizer, loss, entities, relations)

            order = torch.randperm(len(seed_indices), device=self.device, generator=generator)
            for batch in order.split(self.batch_size):
                # Swapping triples gives structure-level supervision. A direct distance
                # term also aligns seeds with few or no incident relation triples.
                source, target = torch.nn.functional.embedding(seed_indices[batch], entities).unbind(dim=1)
                loss = (source - target).square().sum()
                _gradient_step(torch, alignment_optimizer, loss, entities, relations)

            if epoch % self.bootstrap_every == 0:
                with torch.no_grad():
                    pseudo_labels = self._refresh_labels(
                        torch, entities[left_rows], entities[right_rows], pseudo_labels, rejected,
                    )
                pseudo_pairs = [(left_indices[l], left_size + right_indices[r]) for l, r in pseudo_labels.items()]
                swaps = _swapped_triples(bootstrap_triples, pseudo_pairs)
                if swaps:
                    swapped = _triples_tensor(torch, swaps, self.device)
                    order = torch.randperm(len(swapped), device=self.device, generator=generator)
                    for batch in order.split(self.batch_size):
                        cost = _triple_cost(torch, entities, relations, swapped[batch])
                        loss = torch.nn.functional.softplus(cost).sum()
                        _gradient_step(torch, alignment_optimizer, loss, entities, relations)
                if swaps != previous_swaps:
                    known = _triples_tensor(torch, base_triples | swaps, self.device)
                    known_positives = _positive_keys(torch, known, entity_count, predicate_count)
                    previous_swaps = swaps
            if validation.stop(epoch, encode) or epoch == self.epochs:
                break
            if self.epsilon > 0 and epoch % self.neighbor_refresh == 0:
                with torch.no_grad():
                    neighbors = (self._hard_neighbors(torch, generator, entities[:left_size]),
                                 self._hard_neighbors(torch, generator, entities[left_size:]))

        return validation.vectors(encode)

    def _refresh_labels(self, torch, left, right, previous, rejected):
        additions = self._bootstrap(torch, left, right, rejected)
        if not additions:
            return previous
        pairs = (*previous.items(), *additions)
        sources, targets = torch.as_tensor(pairs, device=self.device).T
        weights = (left[sources] * right[targets]).sum(dim=1).clamp(-1.0, 1.0).cpu().tolist()
        scores = dict(zip(pairs, weights, strict=True))
        return _edit_alignments(previous, additions, lambda l, r: scores[l, r])

    def _hard_neighbors(self, torch, generator, vectors):
        pool_size = min(len(vectors), self.neighbor_pool_size)
        if pool_size <= 1:
            return None
        if pool_size == len(vectors):
            pool = torch.arange(pool_size, device=self.device)
            targets = vectors
        else:
            pool = torch.randperm(len(vectors), device=self.device, generator=generator)[:pool_size]
            targets = vectors[pool]
        count = min(pool_size - 1, ceil((1.0 - self.epsilon) * pool_size))
        result = torch.empty((len(vectors), count), dtype=torch.int32, device=self.device)
        for start, indices in _nearest_k(torch, vectors, targets, count, self.block_size, pool):
            result[start:start + len(indices)] = pool[indices]
        return result

    def _triple_loss(self, torch, entities, relations, positive, negative, valid):
        # Gather endpoints once so autograd accumulates into each table once per batch.
        triples = torch.cat((positive, negative))
        costs = _triple_cost(torch, entities, relations, triples)
        positive_cost, negative_cost = costs.split((len(positive), len(negative)))
        positive_loss = (positive_cost - self.positive_margin).relu().sum()
        negative_loss = ((self.negative_margin - negative_cost).relu() * valid).sum()
        return positive_loss + self.negative_weight * negative_loss

    def _sample_negatives(self, torch, generator, positive, sizes, neighbors, positives, predicate_count):
        negative = positive.repeat_interleave(self.negative_samples, dim=0)
        corrupt_head = torch.randint(2, (len(negative),), device=self.device, generator=generator).bool()
        original = torch.where(corrupt_head, negative[:, 0], negative[:, 2])
        replacements = []
        offset = 0
        for size, neighbor_table in zip(sizes, neighbors, strict=True):
            local = (original - offset).clamp(0, size - 1)
            if size <= 1:
                choices = local
            else:
                available = size - 1 if neighbor_table is None else neighbor_table.shape[1]
                draws = torch.randint(available, local.shape, device=self.device, generator=generator)
                if neighbor_table is None:
                    choices = draws + (draws >= local)
                else:
                    choices = neighbor_table[local, draws].long()
            replacements.append(choices + offset)
            offset += size
        replacement = torch.where(original < sizes[0], *replacements)
        negative[:, 0] = torch.where(corrupt_head, replacement, negative[:, 0])
        negative[:, 2] = torch.where(corrupt_head, negative[:, 2], replacement)

        keys = _triple_keys(negative, sum(sizes), predicate_count)
        locations = torch.searchsorted(positives, keys).clamp(max=len(positives) - 1)
        # Mask observed positives without a variable-size GPU gather.
        return negative, keys != positives[locations]

    def _bootstrap(self, torch, left, right, rejected):
        if len(left) == 0 or len(right) == 0:
            return ()
        matches = nearest_neighbors(torch, left, right, _cosine_block, self.block_size,
                                    largest=True, reciprocal=True, excluded=rejected)
        return tuple((row, col) for row, col, score in matches if score > self.sim_th)


def _validated_seeds(data: TrainingData) -> TrainingData:
    seeds = np.asarray(data.seeds)
    if seeds.ndim != 2 or seeds.shape[1] != 2 or not len(seeds):
        raise DataError("BootEA requires at least one seed pair in a two-column array")
    sizes = (data.left.num_entities, data.right.num_entities)
    if not np.issubdtype(seeds.dtype, np.integer) or (seeds < 0).any() or (seeds >= sizes).any():
        raise DataError("BootEA seeds must use valid entity indices")
    _, first = np.unique(seeds, axis=0, return_index=True)
    seeds = seeds[np.sort(first)]
    if any(len(np.unique(side)) != len(seeds) for side in seeds.T):
        raise DataError("BootEA seed pairs must be one-to-one")
    candidates = tuple(pool[~np.isin(pool, side)] for pool, side in zip(data.candidates, seeds.T, strict=True))
    return replace(data, seeds=seeds, candidates=candidates)


def _swapped_triples(triples, pairs):
    counterparts = dict(pairs)
    counterparts.update((right, left) for left, right in pairs)
    swaps = set()
    for head, predicate, tail in triples:
        if head in counterparts:
            swaps.add((counterparts[head], predicate, tail))
        if tail in counterparts:
            swaps.add((head, predicate, counterparts[tail]))
    return swaps


def _triples_tensor(torch, triples, device):
    return torch.tensor(sorted(triples), dtype=torch.long, device=device).reshape(-1, 3)


def _triple_keys(triples, entity_count, predicate_count):
    head, predicate, tail = triples.T
    return (head * predicate_count + predicate) * entity_count + tail


def _positive_keys(torch, triples, entity_count, predicate_count):
    return torch.sort(_triple_keys(triples, entity_count, predicate_count)).values


def _triple_cost(torch, entities, relations, triples):
    gather = torch.nn.functional.embedding
    head, tail = gather(triples[:, ::2], entities).unbind(dim=1)
    predicate = gather(triples[:, 1], relations)
    return (head + predicate - tail).square().sum(dim=1)


def _unit_parameters(torch, count, dim, device, generator):
    values = torch.randn((count, dim), dtype=torch.float32, device=device, generator=generator)
    return torch.nn.Parameter(torch.nn.functional.normalize(values, dim=1))


def _project(torch, *tables):
    with torch.no_grad():
        for table in tables:
            table.copy_(torch.nn.functional.normalize(table, dim=1))


def _gradient_step(torch, optimizer, loss, *tables):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    _project(torch, *tables)


def _cosine_block(left, right):
    return (left @ right.T).clamp(-1, 1)


def _nearest_k(torch, left, right, k, block_size, shared_rows):
    for i in range(0, len(left), block_size):
        source = left[i:i + block_size]
        # Cover the default hard-negative pool in one pass while bounding the score matrix.
        target_block = max(1, (1 << 23) // len(source))
        best_scores = best_indices = None
        row_indices = torch.arange(i, i + len(source), device=source.device)[:, None]
        for j in range(0, len(right), target_block):
            target = right[j:j + target_block]
            scores = _cosine_block(source, target)
            scores.masked_fill_(row_indices == shared_rows[j:j + len(target)], float("-inf"))
            values, indices = scores.topk(min(k, len(target)), dim=1, sorted=False)
            if best_scores is None:
                best_scores, best_indices = values, indices + j
            else:
                scores = torch.cat((best_scores, values), dim=1)
                indices = torch.cat((best_indices, indices + j), dim=1)
                best_scores, order = scores.topk(min(k, scores.shape[1]), dim=1, sorted=False)
                best_indices = indices.gather(1, order)
        yield i, best_indices


def _edit_alignments(previous, additions, score):
    """Resolve competing labels by score, with stable index-based tie breaks."""
    by_left = {left: (right, score(left, right)) for left, right in previous.items()}
    for left, right in additions:
        weight = score(left, right)
        old_right, old_weight = by_left.get(left, (right, float("-inf")))
        if (weight, -right) > (old_weight, -old_right):
            by_left[left] = right, weight

    by_right = {}
    for left, (right, weight) in by_left.items():
        old_left, old_weight = by_right.get(right, (left, float("-inf")))
        if (weight, -left) > (old_weight, -old_left):
            by_right[right] = left, weight
    return {left: right for right, (left, _) in by_right.items()}
