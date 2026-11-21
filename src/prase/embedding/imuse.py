"""IMUSE attribute induction and relational alignment in PyTorch."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from math import ceil, erf, sqrt

import numpy as np

from ..alignment import Alignment
from ..errors import ConfigError, DataError
from .base import Embeddings, TrainingData
from .common import (
    SeedValidation, check_device, finite_number, fraction, is_tensor, nonnegative_int, positive_int, to_embeddings,
    training_runtime,
)
from .common import indel_similarity as _indel_similarity


class IMUSE:
    """IMUSE (He et al., DASFAA 2019): a translation model with attribute-induced alignment.

    Seed pairs share one entity row. Additional training pairs come from
    alternating attribute and entity alignment over entity names and
    attribute values, never from rejected pairs, and a separate alignment
    step pulls them together. Training runs for up to ``epochs`` epochs and stops
    when validation stops improving. With fewer than four seed pairs, too few
    to validate, it runs for ``unvalidated_epochs``. Proposals are mutual
    nearest neighbors.
    """

    name = "imuse"

    def __init__(
        self,
        *,
        device: str = "cpu",
        dim: int = 100,
        epochs: int = 2000,
        unvalidated_epochs: int = 100,
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
        self.unvalidated_epochs = positive_int("unvalidated_epochs", unvalidated_epochs)
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

    def propose(self, embeddings: Embeddings, data: TrainingData) -> Alignment:
        """Propose mutual nearest neighbors among the candidates."""
        return embeddings.reciprocal_nearest(*data.candidates, excluded=data.rejected)

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(torch, data, seed, self.device, self.epochs, model=self,
                                    unvalidated_epochs=self.unvalidated_epochs)
        data = validation.training
        shared = tuple(data.seed_pairs)
        sizes = (data.left.num_entities, data.right.num_entities)
        saved = validation.saved_state(
            lambda state: is_tensor(torch, state["discovered"], torch.long, 2, 2, limits=sizes))
        if saved is not None:
            alignment_pairs = tuple(map(tuple, saved["discovered"].tolist()))
        else:
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

        discovered = torch.tensor(alignment_pairs, dtype=torch.long).reshape(-1, 2)
        first = validation.resume(generator, (entities, relations), (optimizer,), best=encode,
                                  state=lambda: {"discovered": discovered})
        for epoch in range(first, self.epochs + 1):
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
            validation.checkpoint(epoch)
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


# Bounds the cdist blocks of the value search, in matrix entries.
_BLOCK_ENTRIES = 1 << 22
# Smaller blocks run on one thread, since starting the workers costs more.
_PARALLEL_ENTRIES = 1 << 16
# Left entities linked per batch. A batch whose similar texts exceed _MATCH_ENTRIES
# indices is split in half, which bounds the memory of the value search.
_SOURCE_BATCH = 4_096
_MATCH_ENTRIES = 1 << 24
# Bounds the gap between the vectorized and the exact mean field score.
_SCORE_TOLERANCE = 1e-9


def _entity_links(attributes, candidates, seeds, aligned, threshold, rejected):
    """Link each unseeded left entity to the right entity with the best mean aligned-field score.

    A right entity is a candidate when one of its aligned values is more
    similar than ``threshold`` to a value of the left entity. Its score is the
    mean, over the aligned fields both entities have, of the best indel
    similarity between their values. Field scores are computed in batches, and
    the near-best candidates are rescored exactly.
    """
    try:
        import rapidfuzz  # noqa: F401
    except ImportError as error:
        raise ImportError("IMUSE attribute induction requires RapidFuzz: pip install 'prase[train]'") from error

    left, right = attributes
    used_left, used_right = ({pair[side] for pair in seeds} for side in (0, 1))
    pool = sorted(set(candidates[1]) - used_right)
    fields = {b: _ValueIndex(right, pool, b) for b in sorted({b for _, b in aligned})}
    sources = sorted(set(candidates[0]) - used_left)
    blocked = defaultdict(set)
    for i, j in rejected:
        blocked[i].add(j)
    proposals = []
    # A stack of batches in source order; a batch over the match budget is split in half.
    pending = [sources[offset:offset + _SOURCE_BATCH] for offset in range(0, len(sources), _SOURCE_BATCH)][::-1]
    while pending:
        batch = pending.pop()
        matches = _batch_matches(left, batch, aligned, fields, threshold,
                                 None if len(batch) == 1 else _MATCH_ENTRIES)
        if matches is None:
            pending.extend((batch[len(batch) // 2:], batch[:len(batch) // 2]))
            continue
        for i in batch:
            link = _best_link(left[i], aligned, fields, matches, blocked.get(i), threshold)
            if link is not None:
                proposals.append((i, *link))
    return _one_to_one(proposals)


def _batch_matches(left, batch, aligned, fields, threshold, budget):
    """The similar texts of each value of ``batch``, or None when they exceed ``budget`` indices.

    Their entities are expanded one left entity at a time.
    """
    matches = {}
    for b, field in fields.items():
        values = {value for a, target in aligned if target == b for i in batch for value in left[i].get(a, ())}
        found = field.search(values, threshold, budget)
        if found is None:
            return None
        if budget is not None:
            budget -= sum(len(texts) for texts in found.values())
        matches.update(((b, value), texts) for value, texts in found.items())
    return matches


def _best_link(attributes, aligned, fields, matches, blocked, threshold):
    """The best right entity for one left entity and its score, or None."""
    found = []
    for a, b in aligned:
        texts = [matches[b, value] for value in attributes.get(a, ())]
        texts = [similar for similar in texts if len(similar)]
        if texts:
            found.append(fields[b].holders_of(np.unique(np.concatenate(texts))))
    if not found:
        return None
    possible = np.unique(np.concatenate(found))
    if blocked:
        possible = possible[~np.isin(possible, sorted(blocked))]
    if not len(possible):
        return None
    # Field scores in the order of ``aligned``, for the candidates that have the field.
    field_scores = []
    for a, b in aligned:
        values = attributes.get(a)
        if values:
            present, score = fields[b].best(values, possible)
            if present.any():
                field_scores.append((present, score))
    total = np.zeros(len(possible))
    count = np.zeros(len(possible))
    for present, score in field_scores:
        total += score
        count += present
    mean = total / count
    top = mean.max()
    if not top > threshold - _SCORE_TOLERANCE:
        return None
    # Rescore the near-best candidates exactly, in entity order, so that
    # ties and the strict threshold resolve as in a full scan.
    best = threshold, -1
    near = (mean >= top - _SCORE_TOLERANCE) & (mean > threshold - _SCORE_TOLERANCE)
    for index in np.flatnonzero(near).tolist():
        shared = [float(score[index]) for present, score in field_scores if present[index]]
        exact = sum(shared) / len(shared)
        if exact > best[0]:
            best = exact, int(possible[index])
    return (best[1], best[0]) if best[1] >= 0 else None


def _similarities(values, texts, lengths):
    """The indel similarity of each value to each text, exactly as ``indel_similarity`` computes it."""
    from rapidfuzz.distance import LCSseq
    from rapidfuzz.process import cdist

    common = cdist(values, texts, scorer=LCSseq.similarity, dtype=np.int64)
    total = np.array([len(value) for value in values], dtype=np.int64)[:, None] + lengths
    # 2 * LCS / total, where two empty texts are equal.
    return np.divide(2 * common, total, out=np.ones(common.shape), where=total > 0)


def _segments(starts, sizes):
    """Indices of the concatenated ranges ``[start, start + size)``."""
    ends = np.cumsum(sizes)
    return np.repeat(starts - ends + sizes, sizes) + np.arange(ends[-1] if len(ends) else 0)


class _ValueIndex:
    """The values of one predicate over a pool of entities, for searching and scoring."""

    def __init__(self, attributes, pool, predicate):
        ids = {}
        sizes = np.zeros(len(attributes), dtype=np.int64)
        flat = []
        for entity in pool:
            values = attributes[entity].get(predicate, ())
            sizes[entity] = len(values)
            flat.extend(ids.setdefault(value, len(ids)) for value in values)
        self.offsets = np.concatenate(([0], np.cumsum(sizes)))
        self.ids = np.asarray(flat, dtype=np.int64)
        self.texts = list(ids)
        self.lengths = np.array([len(text) for text in self.texts], dtype=np.int64)
        # The entities that have each text, in entity order.
        self.holders = np.repeat(np.arange(len(attributes)), sizes)[np.argsort(self.ids, kind="stable")]
        self.holder_offsets = np.concatenate(([0], np.cumsum(np.bincount(self.ids, minlength=len(self.texts)))))
        # Texts by length, for the length bound of the search.
        order = sorted(range(len(self.texts)), key=lambda k: (len(self.texts[k]), self.texts[k]))
        self.by_length = np.asarray(order, dtype=np.int64)
        self.sorted_lengths = self.lengths[self.by_length].tolist()

    def holders_of(self, texts):
        """The pool entities that have any of ``texts``, given by text index."""
        starts = self.holder_offsets[texts]
        return self.holders[_segments(starts, self.holder_offsets[texts + 1] - starts)]

    def search(self, values, threshold, budget=None):
        """Map each value to the indices of the texts more similar to it than ``threshold``.

        Returns None as soon as the indices exceed ``budget``.
        """
        from rapidfuzz.distance import Indel
        from rapidfuzz.process import cdist

        by_size = defaultdict(list)
        for value in sorted(values):
            by_size[len(value)].append(value)
        found = {}
        stored = 0
        for size, group in by_size.items():
            # Indel similarity is at most 2 * min / total, which bounds the lengths of similar texts.
            if threshold:
                start = bisect_left(self.sorted_lengths, int(size * threshold / (2 - threshold)))
                end = bisect_right(self.sorted_lengths, ceil(size * (2 - threshold) / threshold))
            else:
                start, end = 0, len(self.sorted_lengths)
            window = self.by_length[start:end].astype(np.int32)
            if not len(window):
                found.update((value, window) for value in group)
                continue
            texts = [self.texts[k] for k in window.tolist()]
            total = size + self.lengths[window]
            # Similarity 1 - distance / total exceeds the threshold only below this distance.
            bound = int(int(total[-1]) * (1 - threshold)) + 1
            rows = max(1, _BLOCK_ENTRIES // len(window))
            for offset in range(0, len(group), rows):
                block = group[offset:offset + rows]
                workers = -1 if len(block) * len(window) >= _PARALLEL_ENTRIES else 1
                distance = cdist(block, texts, scorer=Indel.distance, score_cutoff=bound, dtype=np.int64,
                                 workers=workers)
                # total - distance is 2 * LCS, so this is exactly the indel similarity.
                similarity = np.divide(total - distance, total, out=np.ones(distance.shape), where=total > 0)
                for value, row in zip(block, (distance <= bound) & (similarity > threshold), strict=True):
                    found[value] = window[row]
                    stored += len(found[value])
                if budget is not None and stored > budget:
                    return None
        return found

    def best(self, values, entities):
        """Which ``entities`` have values, and the best similarity of each to ``values``."""
        starts = self.offsets[entities]
        sizes = self.offsets[entities + 1] - starts
        present = sizes > 0
        score = np.zeros(len(entities))
        if not present.any():
            return present, score
        sizes = sizes[present]
        unique, inverse = np.unique(self.ids[_segments(starts[present], sizes)], return_inverse=True)
        best = np.empty(len(unique))
        width = max(1, _BLOCK_ENTRIES // len(values))
        for first in range(0, len(unique), width):
            texts = unique[first:first + width]
            similarity = _similarities(values, [self.texts[k] for k in texts.tolist()], self.lengths[texts])
            best[first:first + width] = similarity.max(axis=0)
        score[present] = np.maximum.reduceat(best[inverse], np.cumsum(sizes) - sizes)
        return present, score


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
