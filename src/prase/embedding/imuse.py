"""IMUSE attribute induction and relational alignment in PyTorch."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from math import ceil, erf, sqrt, ulp

from ..errors import ConfigError, DataError
from .base import Embeddings, TrainingData
from .common import (
    SeedValidation, check_device, finite_number, fraction, nonnegative_int, positive_int, to_embeddings,
    training_runtime,
)
from .common import indel_similarity as _indel_similarity


class IMUSE:
    """IMUSE (He et al., DASFAA 2019): a translation model with attribute-induced alignment.

    Seed pairs share one entity row. Additional training pairs come from
    alternating attribute and entity alignment over entity names and
    attribute values, never from rejected pairs, and a separate alignment
    step pulls them together.
    """

    name = "imuse"

    def __init__(
        self,
        *,
        device: str = "cpu",
        dim: int = 100,
        epochs: int = 100,
        batch_size: int = 5_000,
        learning_rate: float = 0.01,
        margin: float = 1.5,
        discovery_rounds: int = 1,
        name_threshold: float = 0.6,
        entity_threshold: float = 0.6,
        attribute_threshold: float = 0.6,
        max_name_pairs: int = 10,
        csls_k: int = 0,
    ):
        self.device = check_device(device)
        self.dim = positive_int("dim", dim)
        self.epochs = positive_int("epochs", epochs)
        self.batch_size = positive_int("batch_size", batch_size)
        self.discovery_rounds = positive_int("discovery_rounds", discovery_rounds)
        self.max_name_pairs = positive_int("max_name_pairs", max_name_pairs)
        self.csls_k = nonnegative_int("csls_k", csls_k)
        self.learning_rate = finite_number("learning_rate", learning_rate)
        self.margin = finite_number("margin", margin)
        if self.learning_rate <= 0 or self.margin <= 0:
            raise ConfigError("learning_rate and margin must be positive")
        self.name_threshold = fraction("name_threshold", name_threshold)
        self.entity_threshold = fraction("entity_threshold", entity_threshold)
        self.attribute_threshold = fraction("attribute_threshold", attribute_threshold)

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(torch, data, seed, self.device, self.epochs)
        data = validation.training
        shared = tuple(data.seed_pairs)
        alignment_pairs = self._discover_seeds(data, shared)
        if not shared and not alignment_pairs:
            raise DataError("IMUSE requires seed pairs or attribute-induced pairs")
        left, right = self._train(torch, generator, (data.left, data.right), shared, alignment_pairs, validation)
        if self.csls_k:
            return to_embeddings(left, right, "csls", csls_k=self.csls_k, info=validation.summary())
        return to_embeddings(left, right, "cosine", info=validation.summary())

    def _discover_seeds(self, data, seeds):
        left, right = data.left, data.right
        candidates = tuple(pool.tolist() for pool in data.candidates)
        if not len(left.attribute_triples) or not len(right.attribute_triples) or not all(candidates):
            return ()
        attributes = (_attribute_values(left), _attribute_values(right))
        aligned = _name_links(left, right, self.name_threshold, self.max_name_pairs)
        known = set(seeds)
        discovered = []
        for index in range(self.discovery_rounds):
            pairs = _entity_links(attributes, candidates, known, aligned, self.entity_threshold, data.rejected_pairs)
            discovered.extend(pairs)
            known.update(pairs)
            if index + 1 == self.discovery_rounds:
                break
            additional = _infer_attributes(attributes, known, aligned, self.attribute_threshold)
            if not additional:
                break
            aligned = tuple(sorted((*aligned, *additional)))
        return tuple(discovered)

    def _train(self, torch, generator, graphs, shared, alignment_pairs, validation):
        sizes = tuple(len(graph.entities) for graph in graphs)
        predicate_count = sum(len(graph.predicates) for graph in graphs)
        entity_count = sum(sizes) - len(shared)
        if entity_count * entity_count * predicate_count > 2**63:
            raise DataError("IMUSE's triple index exceeds the supported integer range")
        rows = _entity_rows(torch, sizes, shared, self.device)
        triples = []
        predicate_offset = 0
        for graph, row_ids in zip(graphs, rows, strict=True):
            side = _index_tensor(torch, graph.relation_triples, 3, self.device)
            side[:, 0] = row_ids[side[:, 0]]
            side[:, 1] += predicate_offset
            side[:, 2] = row_ids[side[:, 2]]
            triples.append(side)
            predicate_offset += len(graph.predicates)
        if not any(len(side) for side in triples):
            raise DataError("IMUSE requires relation triples")
        positives = torch.cat(triples)
        known = _triple_codes(positives, entity_count, predicate_count).sort().values
        seed_rows = _index_tensor(torch, alignment_pairs, 2, self.device)
        pairs = torch.stack((rows[0][seed_rows[:, 0]], rows[1][seed_rows[:, 1]]), dim=1)

        entities = _parameter(torch, generator, entity_count, self.dim, self.device)
        relations = _parameter(torch, generator, predicate_count, self.dim, self.device)
        optimizer = torch.optim.SGD((entities, relations), lr=self.learning_rate)

        def encode():
            normalized = torch.nn.functional.normalize(entities, dim=1)
            return tuple(normalized[side] for side in rows)

        for epoch in range(1, self.epochs + 1):
            order = torch.randperm(len(positives), generator=generator, device=self.device)
            for indices in order.split(self.batch_size):
                batch = positives[indices]
                positive, negative = _negative_batch(
                    torch, generator, batch, indices < len(triples[0]), rows, known, entity_count, predicate_count,
                )
                if len(negative):
                    loss = _margin_loss(torch, entities, relations, positive, negative, self.margin)
                    _step(optimizer, loss)
            for seed_batch in pairs.split(self.batch_size):
                if len(seed_batch):
                    loss = _alignment_loss(torch, entities, seed_batch)
                    _step(optimizer, loss)
            if validation.stop(epoch, encode):
                break
        return validation.vectors(encode)


def _entity_rows(torch, sizes, shared, device):
    left_size, right_size = sizes
    mapping = {j: i for i, j in shared}
    if len(mapping) != len(shared) or len(set(mapping.values())) != len(shared):
        raise DataError("IMUSE shared seed pairs must be one-to-one")
    if any(not 0 <= i < left_size or not 0 <= j < right_size for i, j in shared):
        raise DataError("IMUSE shared seed pair has an invalid entity index")
    next_row = left_size
    right = []
    for index in range(right_size):
        if index in mapping:
            right.append(mapping[index])
        else:
            right.append(next_row)
            next_row += 1
    return torch.arange(left_size, device=device), torch.tensor(right, dtype=torch.long, device=device)


def _index_tensor(torch, rows, width, device):
    return torch.tensor(rows, dtype=torch.long, device=device).reshape(-1, width)


def _parameter(torch, generator, size, dim, device):
    uniform = torch.rand((size, dim), generator=generator, device=device)
    values = ((uniform * 2 - 1) * erf(sqrt(2))).erfinv() * sqrt(2 / dim)
    return torch.nn.Parameter(values)


def _triple_codes(triples, entity_count, predicate_count):
    return (triples[:, 0] * predicate_count + triples[:, 1]) * entity_count + triples[:, 2]


def _negative_batch(torch, generator, positive, is_left, rows, known, entity_count, predicate_count):
    device = positive.device
    negative = positive.clone()
    pending = torch.ones(len(positive), dtype=torch.bool, device=device)
    for side, pool in zip((is_left, ~is_left), rows, strict=True):
        for _ in range(10):
            indices = (side & pending).nonzero(as_tuple=True)[0]
            if not len(indices):
                break
            trial = positive[indices].clone()
            replacements = pool[torch.randint(len(pool), (len(indices),), generator=generator, device=device)]
            ends = torch.randint(2, (len(indices),), generator=generator, device=device) * 2
            trial[torch.arange(len(indices), device=device), ends] = replacements
            codes = _triple_codes(trial, entity_count, predicate_count)
            locations = torch.searchsorted(known, codes).clamp(max=len(known) - 1)
            valid = codes != known[locations]
            selected = indices[valid]
            negative[selected] = trial[valid]
            pending[selected] = False
    return positive[~pending], negative[~pending]


def _margin_loss(torch, entities, relations, positive, negative, margin):
    normalize = torch.nn.functional.normalize

    def costs(triples):
        h, r, t = triples.T
        difference = normalize(entities[h], dim=1) + normalize(relations[r], dim=1) - normalize(entities[t], dim=1)
        return difference.square().sum(dim=1)

    return (margin + costs(positive) - costs(negative)).relu().sum()


def _alignment_loss(torch, entities, pairs):
    left, right = pairs.T
    normalize = torch.nn.functional.normalize
    return (normalize(entities[left], dim=1) - normalize(entities[right], dim=1)).square().sum()


def _step(optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _attribute_values(graph):
    by_entity = [defaultdict(set) for _ in graph.entities]
    for entity, predicate, literal in graph.attribute_triples.tolist():
        by_entity[entity][predicate].add(graph.literals[literal])
    return tuple({key: tuple(sorted(values)) for key, values in attributes.items()} for attributes in by_entity)


def _name_links(left, right, threshold, limit):
    left_counts = Counter(left.attribute_triples[:, 1].tolist())
    right_counts = Counter(right.attribute_triples[:, 1].tolist())

    def names(graph, counts):
        return {key: graph.predicates[key].name.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                for key in sorted(counts)}

    right_names = names(right, right_counts)
    proposals = []
    for i, a in names(left, left_counts).items():
        ranked = ((_indel_similarity(a, b), j) for j, b in right_names.items())
        score, j = max(ranked, key=lambda hit: (hit[0], -hit[1]))
        if score > threshold:
            proposals.append((i, j, score))
    pairs = _one_to_one(proposals)
    pairs = sorted(pairs, key=lambda pair: (-sum(counts[k] for counts, k in zip(
        (left_counts, right_counts), pair, strict=True,
    )), pair))
    return tuple(sorted(pairs[:limit]))


def _one_to_one(proposals):
    selected = []
    used_left, used_right = set(), set()
    for i, j, _ in sorted(proposals, key=lambda row: (-row[2], row[0], row[1])):
        if i not in used_left and j not in used_right:
            selected.append((i, j))
            used_left.add(i)
            used_right.add(j)
    return tuple(sorted(selected))


def _entity_links(attributes, candidates, seeds, aligned, threshold, rejected):
    try:
        from rapidfuzz.distance import Indel
        from rapidfuzz.process import extract_iter
    except ImportError as error:
        raise ImportError("IMUSE attribute induction requires RapidFuzz: pip install 'prase[train]'") from error

    left, right = attributes
    used_left, used_right = ({pair[side] for pair in seeds} for side in (0, 1))
    target_predicates = {j for _, j in aligned}
    postings = {j: defaultdict(list) for j in target_predicates}
    for j in sorted(set(candidates[1]) - used_right):
        for predicate in target_predicates & right[j].keys():
            for value in right[j][predicate]:
                postings[predicate][value].append(j)

    # If the mean exceeds the threshold, at least one aligned field must
    # exceed it. The indel length bound prunes incompatible text lengths.
    indexed = {}
    for predicate, values in postings.items():
        ordered = sorted(values, key=lambda value: (len(value), value))
        indexed[predicate] = ordered, [len(value) for value in ordered]
    cutoff = max(0.0, threshold - ulp(1.0))
    proposals = []
    for i in sorted(set(candidates[0]) - used_left):
        possible = set()
        for a, b in aligned:
            words, lengths = indexed[b]
            for value in left[i].get(a, ()):
                if threshold:
                    size = len(value)
                    start = bisect_left(lengths, int(size * threshold / (2 - threshold)))
                    end = bisect_right(lengths, ceil(size * (2 - threshold) / threshold))
                else:
                    start, end = 0, len(words)
                matches = extract_iter(value, words[start:end], scorer=Indel.normalized_similarity,
                                       score_cutoff=cutoff)
                for target, _, _ in matches:
                    # Preserve the exact strict threshold after the native prefilter.
                    if _indel_similarity(value, target) > threshold:
                        possible.update(postings[b][target])
        best = threshold, -1
        for j in sorted(j for j in possible if (i, j) not in rejected):
            scores = [_value_score(left[i][a], right[j][b])
                      for a, b in aligned if a in left[i] and b in right[j]]
            score = sum(scores) / len(scores)
            if score > best[0]:
                best = score, j
        if best[1] >= 0:
            proposals.append((i, best[1], best[0]))
    return _one_to_one(proposals)


def _infer_attributes(attributes, seeds, aligned, threshold):
    left, right = attributes
    used_left, used_right = ({pair[side] for pair in aligned} for side in (0, 1))
    totals = defaultdict(float)
    counts = Counter()
    for i, j in sorted(seeds):
        for a in sorted(left[i].keys() - used_left):
            for b in sorted(right[j].keys() - used_right):
                totals[a, b] += _value_score(left[i][a], right[j][b])
                counts[a, b] += 1
    scores = ((a, b, value / counts[a, b]) for (a, b), value in totals.items())
    return _one_to_one(row for row in scores if row[2] > threshold)


def _value_score(left, right):
    return max(_indel_similarity(a, b) for a in left for b in right)
