"""MultiKE (Zhang et al., IJCAI 2019) with in-training view combination, as in OpenEA.

Three views describe each entity. The name view encodes the entity's name
from pretrained word vectors with a literal autoencoder. The relation view
trains TransE with a logistic loss. The attribute view scores each
``(attribute, value)`` pair with a small convolutional network. A shared
entity table is trained towards all three views and gives the final vectors.

Seed pairs exchange their entities in the triples of both graphs. Predicates
whose names nearly match exchange triples as well, weighted by a similarity
that is refreshed from their embeddings during training.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

import numpy as np

from ..alignment import Alignment
from ..datasets import read_word_vectors
from ..errors import ConfigError, DataError
from ..kg import ATTRIBUTE, RELATION, entity_names, local_name
from .base import Embeddings, TrainingData
from .common import (
    SeedValidation, check_device, finite_number, fraction, indel_similarity, nearest_neighbors, positive_int,
    to_embeddings, training_runtime,
)

_VALUE_PUNCTUATION = str.maketrans({".": "", "(": "", ")": "", ",": "", '"': "", "_": " ", "-": " ", "/": " "})


class MultiKE:
    """Multi-view entity embeddings from names, relation triples, and attribute triples.

    ``word_vectors`` is a text file of ``word v1 v2 ...`` lines, such as
    FastText's ``wiki-news-300d-1M.vec``. Tokens without a vector are built
    from character vectors trained on those tokens.

    ``left_names`` and ``right_names`` list the attributes that hold entity
    names, in order of preference. Other entities are named by the local name
    of their identifier. Set ``left_local_names`` or ``right_local_names`` to
    false when identifiers are opaque, such as Wikidata Q-numbers, so those
    entities have no name. Name vectors are centered on their mean so that
    names, not the encoder's offset, determine their similarity.

    Training stops when retrieval on held-out seed pairs stops improving, and
    the checkpoint with the best score is returned. Proposals are mutual
    nearest neighbors by cosine similarity.
    """

    name = "multike"
    path_options = ("word_vectors",)

    def __init__(
        self, *, word_vectors: str | Path, left_names: tuple[str, ...] = (), right_names: tuple[str, ...] = (),
        left_local_names: bool = True, right_local_names: bool = True, device: str = "cpu",
        dim: int = 100, epochs: int = 2000, learning_rate: float = 0.001, view_learning_rate: float = 0.004,
        batch_size: int = 5000, negative_samples: int = 10, literal_tokens: int = 5,
        literal_hidden: tuple[int, ...] = (1024, 512), literal_epochs: int = 100, char_epochs: int = 5,
        attribute_min_count: int = 10, predicate_init_threshold: float = 0.9, predicate_threshold: float = 0.8,
        predicate_name_weight: float = 0.7, predicate_start: int = 10, predicate_refresh: int = 10,
    ):
        self.word_vectors = Path(word_vectors)
        if not self.word_vectors.is_file():
            raise ConfigError(f"{self.word_vectors}: word vector file not found")
        self.names = (_predicates("left_names", left_names), _predicates("right_names", right_names))
        if not all(isinstance(flag, bool) for flag in (left_local_names, right_local_names)):
            raise ConfigError("left_local_names and right_local_names must be true or false")
        self.local_names = (left_local_names, right_local_names)
        self.device = check_device(device)
        self.dim = positive_int("dim", dim)
        self.epochs = positive_int("epochs", epochs)
        self.learning_rate = _positive("learning_rate", learning_rate)
        self.view_learning_rate = _positive("view_learning_rate", view_learning_rate)
        self.batch_size = positive_int("batch_size", batch_size)
        self.negative_samples = positive_int("negative_samples", negative_samples)
        self.literal_tokens = positive_int("literal_tokens", literal_tokens)
        if not isinstance(literal_hidden, (list, tuple)):
            raise ConfigError("literal_hidden must be a list of layer widths")
        self.literal_hidden = tuple(positive_int("literal_hidden", width) for width in literal_hidden)
        self.literal_epochs = positive_int("literal_epochs", literal_epochs)
        self.char_epochs = positive_int("char_epochs", char_epochs)
        self.attribute_min_count = positive_int("attribute_min_count", attribute_min_count)
        self.predicate_init_threshold = fraction("predicate_init_threshold", predicate_init_threshold)
        self.predicate_threshold = fraction("predicate_threshold", predicate_threshold)
        if self.predicate_threshold >= 1:
            raise ConfigError("predicate_threshold must be below 1")
        self.predicate_name_weight = fraction("predicate_name_weight", predicate_name_weight)
        self.predicate_start = positive_int("predicate_start", predicate_start)
        self.predicate_refresh = positive_int("predicate_refresh", predicate_refresh)

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        if not len(data.seeds):
            raise DataError("MultiKE requires at least one seed pair")
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(torch, data, seed, self.device, self.epochs)
        inputs = _Inputs.build(torch, validation.training, self)
        literals = _literal_vectors(torch, generator, inputs.texts, self)
        training = _Training(torch, generator, inputs, literals, self)
        training.run(validation)
        left, right = training.vectors().split(inputs.sizes)
        return to_embeddings(left, right, "cosine", info=validation.summary())

    def propose(self, embeddings: Embeddings, data: TrainingData) -> Alignment:
        """Propose mutual nearest neighbors among the candidates."""
        return embeddings.reciprocal_nearest(*data.candidates, excluded=data.rejected)


def _predicates(option: str, values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) and value for value in values):
        raise ConfigError(f"{option} must be a list of predicate identifiers")
    return tuple(dict.fromkeys(values))


def _positive(option: str, value: float) -> float:
    value = finite_number(option, value)
    if value <= 0:
        raise ConfigError(f"{option} must be positive")
    return value


# Inputs ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Inputs:
    """Triples and texts of both graphs with entities and predicates in one index space.

    Left entities come first, then right entities; predicates likewise.
    ``names`` holds each entity's text index, or -1 for an unnamed entity.
    Attribute triples refer to texts instead of literals.
    """

    sizes: tuple[int, int]
    texts: tuple[str, ...]
    names: object
    predicate_names: tuple[str, ...]
    predicates: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    relations: object
    attributes: object
    swapped_relations: object
    swapped_attributes: object

    @classmethod
    def build(cls, torch, data: TrainingData, settings: MultiKE) -> _Inputs:
        graphs = (data.left, data.right)
        sizes = (data.left.num_entities, data.right.num_entities)
        entity_offsets, predicate_offsets = (0, sizes[0]), (0, len(data.left.predicates))
        names = [
            name for graph, predicates, local in zip(graphs, settings.names, settings.local_names, strict=True)
            for name in entity_names(graph, predicates, local_names=local)
        ]
        relations, attributes, sides = [], [], []
        for graph, entity, predicate in zip(graphs, entity_offsets, predicate_offsets, strict=True):
            sides.append(tuple(
                tuple(index + predicate for index, p in enumerate(graph.predicates) if p.kind == kind)
                for kind in (RELATION, ATTRIBUTE)
            ))
            relations.extend((h + entity, p + predicate, t + entity) for h, p, t in graph.relation_triples.tolist())
            counts = Counter(graph.attribute_triples[:, 1].tolist())
            values = [literal.translate(_VALUE_PUNCTUATION).strip() for literal in graph.literals]
            attributes.extend(
                (h + entity, p + predicate, values[v]) for h, p, v in graph.attribute_triples.tolist()
                if counts[p] >= settings.attribute_min_count and values[v] and "http" not in values[v]
            )
        texts = tuple(sorted({*(name for name in names if name), *(value for _, _, value in attributes)}))
        index = {text: position for position, text in enumerate(texts)}
        attributes = sorted({(h, p, index[value]) for h, p, value in attributes})

        counterpart = {}
        for left, right in data.seed_pairs:
            right += sizes[0]
            if left in counterpart or right in counterpart:
                raise DataError("MultiKE requires one-to-one seed pairs")
            counterpart[left], counterpart[right] = right, left
        swapped_relations = {(counterpart[h], p, t) for h, p, t in relations if h in counterpart}
        swapped_relations |= {(h, p, counterpart[t]) for h, p, t in relations if t in counterpart}
        swapped_attributes = {(counterpart[h], p, v) for h, p, v in attributes if h in counterpart}

        def table(rows):
            return torch.tensor(sorted(rows), dtype=torch.long, device=settings.device).reshape(-1, 3)

        return cls(
            sizes, texts, torch.tensor([index[name] if name else -1 for name in names], device=settings.device),
            tuple(local_name(p.name) for graph in graphs for p in graph.predicates), tuple(sides),
            table(relations), table(attributes), table(swapped_relations), table(swapped_attributes),
        )


# Literal vectors ------------------------------------------------------------

def _literal_vectors(torch, generator, texts, settings):
    """Encode each text's first tokens with the literal autoencoder; rows have unit length."""
    if not texts:
        return torch.zeros(0, settings.dim, device=settings.device)
    tokenized = [text.split()[:settings.literal_tokens] for text in texts]
    tokens = sorted({token for row in tokenized for token in row})
    known, width = read_word_vectors(settings.word_vectors, set(tokens))
    token_ids = {token: position for position, token in enumerate(tokens, 1)}
    table = torch.zeros(len(tokens) + 1, width, device=settings.device)
    if known:
        vectors = torch.from_numpy(np.stack(list(known.values()))).to(table)
        table[[token_ids[token] for token in known]] = vectors
    unknown = [token for token in tokens if token not in known]
    if unknown:
        table[[token_ids[token] for token in unknown]] = _character_composed(torch, generator, unknown, width,
                                                                             settings)
    rows = torch.tensor(
        [[token_ids[token] for token in row] + [0] * (settings.literal_tokens - len(row)) for row in tokenized],
        dtype=torch.long, device=settings.device,
    )
    return torch.nn.functional.normalize(_autoencode(torch, generator, table, rows, settings), dim=1)


def _character_composed(torch, generator, tokens, width, settings):
    """Vectors for tokens without a word vector: the mean of their characters' vectors.

    Character vectors are trained with skip-gram and negative sampling on the
    character sequences of these tokens, in a window of five characters.
    Returns one row per token.
    """
    characters = sorted({character for token in tokens for character in token})
    ids = {character: position for position, character in enumerate(characters)}
    words = {token: [ids[character] for character in token] for token in tokens}
    frequency = Counter(character for word in words.values() for character in word)
    pairs = Counter(
        (word[center], word[context]) for word in words.values() for center in range(len(word))
        for context in range(max(0, center - 5), min(len(word), center + 6)) if context != center
    )
    vectors = _parameter(torch, (len(characters), width), generator, settings.device)
    if pairs:
        functional = torch.nn.functional
        contexts = torch.nn.Parameter(torch.zeros_like(vectors))
        optimizer = torch.optim.Adagrad((vectors, contexts), lr=0.05)
        rows = torch.tensor(tuple(pairs), device=settings.device, dtype=torch.long)
        counts = torch.tensor(tuple(pairs.values()), device=settings.device, dtype=vectors.dtype)
        noise = torch.tensor([frequency[c] ** 0.75 for c in range(len(characters))], device=settings.device)
        for _ in range(settings.char_epochs):
            for batch in _shuffled(torch, torch.arange(len(rows), device=settings.device), settings.batch_size,
                                   generator):
                center, context = rows[batch].T
                negatives = torch.multinomial(noise, len(batch) * 5, replacement=True, generator=generator)
                inputs = vectors[center]
                positive = (inputs * contexts[context]).sum(1)
                negative = (inputs[:, None, :] * contexts[negatives.reshape(-1, 5)]).sum(2)
                loss = functional.softplus(-positive) + functional.softplus(negative).sum(1)
                _step(optimizer, (loss * counts[batch]).sum() / counts[batch].sum())
    starts = torch.tensor([0, *accumulate(len(word) for word in words.values())][:-1], device=settings.device)
    flat = torch.tensor([character for word in words.values() for character in word], device=settings.device,
                        dtype=torch.long)
    return torch.nn.functional.embedding_bag(flat, vectors.detach(), starts, mode="mean")


def _autoencode(torch, generator, table, rows, settings):
    """Compress literals with a linear autoencoder, as OpenEA's MultiKE does.

    A literal's input is the concatenation of its token vectors, the rows of
    ``table`` that ``rows`` lists, scaled to unit length. Inputs are built one
    batch at a time. Weights and biases start from a standard normal
    distribution, and each batch of codes is scaled to unit norm as a whole
    before decoding, so the encoder stays close to a random linear projection
    of its inputs. Returns the encoder's output for every literal.
    """
    def batches():
        for batch in rows.split(settings.batch_size):
            yield torch.nn.functional.normalize(table[batch].flatten(1), dim=1)

    widths = (rows.shape[1] * table.shape[1], *settings.literal_hidden, settings.dim)

    def layer(rows, columns):
        weight = torch.randn(rows, columns, device=settings.device, generator=generator)
        bias = torch.randn(columns, device=settings.device, generator=generator)
        return torch.nn.Parameter(weight), torch.nn.Parameter(bias)

    encoder = [layer(a, b) for a, b in zip(widths, widths[1:])]
    decoder = [layer(b, a) for a, b in reversed(tuple(zip(widths, widths[1:])))]
    optimizer = torch.optim.Adagrad([p for pair in (*encoder, *decoder) for p in pair], lr=0.01,
                                    initial_accumulator_value=0.1)

    def apply(rows, layers):
        for weight, bias in layers:
            rows = rows @ weight + bias
        return rows

    for _ in range(settings.literal_epochs):
        for batch in batches():
            code = apply(batch, encoder)
            _step(optimizer, (apply(code / code.norm(), decoder) - batch).square().mean())
    with torch.no_grad():
        return torch.cat([apply(batch, encoder) for batch in batches()])


# Training -------------------------------------------------------------------

class _ValueScorer:
    """Map an ``(attribute, value)`` pair to a point that its entity should be near.

    Two convolution layers read the stacked attribute and value vectors, and
    a dense layer projects the result. As in OpenEA, the projections of a
    batch are scaled to unit norm together.
    """

    def __init__(self, torch, generator, dim, device):
        self.torch, self.functional = torch, torch.nn.functional
        kernel = min(dim, 4)
        self.scale = torch.nn.Parameter(torch.ones(dim, device=device))
        self.shift = torch.nn.Parameter(torch.zeros(dim, device=device))
        self.first = _parameter(torch, (2, 1, 2, kernel), generator, device)
        self.second = _parameter(torch, (2, 2, 2, kernel), generator, device)
        self.biases = [torch.nn.Parameter(torch.zeros(2, device=device)) for _ in range(2)]
        self.dense = _parameter(torch, (4 * dim, dim), generator, device)
        self.dense_bias = torch.nn.Parameter(torch.zeros(dim, device=device))
        self.padding = ((kernel - 1) // 2, kernel // 2, 0, 1)

    def parameters(self):
        return (self.scale, self.shift, self.first, self.second, *self.biases, self.dense, self.dense_bias)

    def __call__(self, attributes, values):
        F = self.functional
        pair = (self.torch.stack((attributes, values), dim=1) * self.scale + self.shift).unsqueeze(1)
        hidden = F.conv2d(F.pad(pair, self.padding), self.first, self.biases[0]).tanh()
        hidden = F.conv2d(F.pad(hidden, self.padding), self.second, self.biases[1]).tanh()
        hidden = F.normalize(hidden, dim=3)
        projected = (hidden.flatten(1) @ self.dense + self.dense_bias).tanh()
        return projected / projected.norm().clamp_min(1e-12)


class _Training:
    """The parameters of the three views and the shared table, and one training schedule."""

    def __init__(self, torch, generator, inputs: _Inputs, literals, settings: MultiKE):
        self.torch, self.functional = torch, torch.nn.functional
        self.generator, self.inputs, self.settings = generator, inputs, settings
        device, dim = settings.device, settings.dim
        entities = sum(inputs.sizes)
        predicates = len(inputs.predicate_names)
        self.values = literals
        named = inputs.names >= 0
        self.names = torch.zeros(entities, dim, device=device)
        if named.any():
            names = literals[inputs.names[named]]
            self.names[named] = self.functional.normalize(names - names.mean(0), dim=1)
        self.relation_view, self.attribute_view, self.shared = (
            _parameter(torch, (entities, dim), generator, device) for _ in range(3)
        )
        self.relations = _parameter(torch, (predicates, dim), generator, device)
        self.attributes = _parameter(torch, (predicates, dim), generator, device)
        # OpenEA builds a separate scorer for each loss that reads attribute triples.
        self.scorers = {role: _ValueScorer(torch, generator, dim, device)
                        for role in ("view", "shared", "swapped", "matched")}
        parameters = [self.relation_view, self.attribute_view, self.shared, self.relations, self.attributes]
        for scorer in self.scorers.values():
            parameters.extend(scorer.parameters())
        self.optimizer = torch.optim.SGD(parameters, lr=settings.learning_rate)
        self.view_optimizer = torch.optim.SGD(parameters[:3], lr=settings.view_learning_rate)
        self.entity_ids = torch.arange(entities, device=device)
        self.attribute_weights = torch.ones(predicates, device=device)
        self.predicate_count = max(1, predicates)
        if entities * entities * self.predicate_count >= 2 ** 63:
            raise DataError("the graphs are too large for MultiKE's triple index")
        self.known_triples = self._keys(inputs.relations).sort().values

    # Losses ----------------------------------------------------------------

    def _lookup(self, table, ids):
        return self.functional.normalize(self.functional.embedding(ids, table, sparse=True), dim=1)

    def _translation(self, rows, head=None, tail=None):
        """Squared TransE distance ``|h + r - t|^2`` with unit-length vectors."""
        h, p, t = rows.T
        head = self._lookup(self.relation_view if head is None else head, h)
        tail = self._lookup(self.relation_view if tail is None else tail, t)
        return (head + self._lookup(self.relations, p) - tail).square().sum(1)

    def _attribute(self, rows, role, entities=None):
        """Squared distance between an entity and the scorer's point for its attribute value."""
        h, p, v = rows.T
        point = self.scorers[role](self.functional.embedding(p, self.attributes, sparse=True), self.values[v])
        return (self._lookup(self.attribute_view if entities is None else entities, h) - point).square().sum(1)

    def _name_pull(self, ids):
        return (self._lookup(self.shared, ids) - self.names[ids]).square().sum()

    def _keys(self, rows):
        h, p, t = rows.T
        return (h * self.predicate_count + p) * len(self.entity_ids) + t

    def _corrupt(self, rows):
        """``negative_samples`` corruptions of each triple's head or tail within the same graph."""
        torch = self.torch
        rows = rows.repeat_interleave(self.settings.negative_samples, dim=0)
        head = torch.randint(2, (len(rows),), device=rows.device, generator=self.generator).bool()
        original = torch.where(head, rows[:, 0], rows[:, 2])
        replacement = original.clone()
        start = 0
        for size in self.inputs.sizes:
            if size > 1:
                side = (rows[:, 0] >= start) & (rows[:, 0] < start + size)
                draw = torch.randint(size - 1, original.shape, device=rows.device, generator=self.generator) + start
                replacement = torch.where(side, draw + (draw >= original), replacement)
            start += size
        rows[:, 0] = torch.where(head, replacement, rows[:, 0])
        rows[:, 2] = torch.where(head, rows[:, 2], replacement)
        keys = self._keys(rows)
        found = torch.searchsorted(self.known_triples, keys).clamp_max(len(self.known_triples) - 1)
        return rows, self.known_triples[found] != keys

    # Epoch steps -----------------------------------------------------------

    def _batches(self, rows):
        return _shuffled(self.torch, rows, self.settings.batch_size, self.generator)

    def _relation_view(self):
        F = self.functional
        for batch in self._batches(self.inputs.relations):
            negatives, valid = self._corrupt(batch)
            loss = F.softplus(self._translation(batch)).sum() + (F.softplus(-self._translation(negatives)) * valid).sum()
            loss += self._translation(batch, head=self.shared).sum() + self._translation(batch, tail=self.shared).sum()
            loss += 0.5 * self._name_pull(batch[:, [0, 2]].flatten())
            _step(self.optimizer, loss)

    def _attribute_view(self):
        F = self.functional
        for batch in self._batches(self.inputs.attributes):
            loss = (F.softplus(self._attribute(batch, "view")) * self.attribute_weights[batch[:, 1]]).sum()
            loss += F.softplus(self._attribute(batch, "shared", self.shared)).sum()
            loss += 0.5 * self._name_pull(batch[:, 0])
            _step(self.optimizer, loss)

    def _common_space(self):
        """Pull the shared table towards the name, relation, and attribute views."""
        for ids in self._batches(self.entity_ids):
            shared = self._lookup(self.shared, ids)
            loss = (shared - self.names[ids]).square().sum()
            loss += (shared - self._lookup(self.relation_view, ids)).square().sum()
            loss += (shared - self._lookup(self.attribute_view, ids)).square().sum()
            _step(self.view_optimizer, loss)

    def _fit(self, rows, cost, weights=None):
        if not len(rows):
            return
        for batch in self._batches(self.torch.arange(len(rows), device=rows.device)):
            losses = cost(rows[batch])
            _step(self.optimizer, (losses if weights is None else losses * weights[batch]).sum())

    def run(self, validation: SeedValidation) -> None:
        F, inputs, settings = self.functional, self.inputs, self.settings
        relation_pairs, attribute_pairs = (
            _matching_names(inputs.predicates[0][kind], inputs.predicates[1][kind], inputs.predicate_names,
                            settings.predicate_init_threshold)
            for kind in (0, 1)
        )
        matched_relations = _exchange(self.torch, inputs.relations, relation_pairs, self.predicate_count)
        matched_attributes = _exchange(self.torch, inputs.attributes, attribute_pairs, self.predicate_count)
        self._weigh_attributes(attribute_pairs)

        def encode():
            return F.normalize(self.shared, dim=1).split(inputs.sizes)

        for epoch in range(1, settings.epochs + 1):
            predicates_on = epoch > settings.predicate_start
            self._relation_view()
            self._common_space()
            self._fit(inputs.swapped_relations, lambda rows: 2 * self._translation(rows))
            if predicates_on:
                self._fit(matched_relations[0], lambda rows: 2 * F.softplus(self._translation(rows)),
                          matched_relations[1])
            self._attribute_view()
            self._common_space()
            self._fit(inputs.swapped_attributes, lambda rows: 2 * F.softplus(self._attribute(rows, "swapped")))
            if predicates_on:
                self._fit(matched_attributes[0], lambda rows: F.softplus(self._attribute(rows, "matched")),
                          matched_attributes[1])
            if validation.stop(epoch, encode, parameters=(self.shared,)):
                break
            if epoch >= settings.predicate_start and epoch % settings.predicate_refresh == 0:
                relations = self._refresh(self.relations, relation_pairs, 0)
                attributes = self._refresh(self.attributes, attribute_pairs, 1)
                matched_relations = _exchange(self.torch, inputs.relations, relations, self.predicate_count)
                matched_attributes = _exchange(self.torch, inputs.attributes, attributes, self.predicate_count)
                self._weigh_attributes(attributes)
        validation.restore((self.shared,))

    def vectors(self):
        with self.torch.no_grad():
            return self.functional.normalize(self.shared, dim=1)

    def _refresh(self, embeddings, lexical, kind):
        """Rescore the name-matched predicate pairs with their embeddings' mutual nearest neighbors."""
        if not lexical:
            return ()
        settings = self.settings
        left, right = (side[kind] for side in self.inputs.predicates)
        with self.torch.no_grad():
            rows = self.functional.normalize(embeddings[list(left)], dim=1)
            columns = self.functional.normalize(embeddings[list(right)], dim=1)
            latent = {(left[i], right[j]): score for i, j, score in nearest_neighbors(
                self.torch, rows, columns, lambda a, b: a @ b.T, 512, largest=True, reciprocal=True,
            )}
        weight = settings.predicate_name_weight
        pairs = []
        for i, j, similarity in lexical:
            if (i, j) in latent:
                similarity = weight * similarity + (1 - weight) * latent[i, j]
            if similarity > settings.predicate_threshold:
                pairs.append((i, j, similarity))
        return pairs

    def _weigh_attributes(self, pairs):
        """Weigh attribute triples by how well their attribute matches one in the other graph."""
        self.attribute_weights.fill_(0.2)
        floor = self.settings.predicate_threshold
        for left, right, similarity in pairs:
            weight = min(1.0, max(0.0, 1 - 0.5 * (1 - similarity) / (1 - floor)))
            self.attribute_weights[left] = self.attribute_weights[right] = weight


def _matching_names(left, right, names, threshold):
    """Mutually best predicate pairs by name similarity above ``threshold``."""
    best_left, best_right = {}, {}
    for i in left:
        for j in right:
            a, b = names[i], names[j]
            if 2 * min(len(a), len(b)) <= threshold * (len(a) + len(b)):
                continue  # The similarity cannot exceed the threshold.
            score = indel_similarity(a, b)
            if score > best_left.get(i, (threshold, -1))[0]:
                best_left[i] = score, j
            if score > best_right.get(j, (threshold, -1))[0]:
                best_right[j] = score, i
    return tuple((i, j, score) for i, (score, j) in best_left.items() if best_right[j][1] == i)


def _exchange(torch, facts, pairs, predicate_count):
    """Copies of the facts of each matched predicate with its counterpart, weighted by their similarity."""
    if not pairs:
        return facts[:0], torch.zeros(0, device=facts.device)
    order = facts[:, 1].argsort(stable=True)
    counts = torch.bincount(facts[:, 1], minlength=predicate_count).tolist()
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    rows, weights = [], []
    for left, right, weight in pairs:
        for source, target in ((left, right), (right, left)):
            if counts[source]:
                copied = facts[order[starts[source]:starts[source + 1]]]
                copied[:, 1] = target
                rows.append(copied)
                weights.append(torch.full((len(copied),), weight, device=facts.device))
    if not rows:
        return facts[:0], torch.zeros(0, device=facts.device)
    return torch.cat(rows), torch.cat(weights)


def _parameter(torch, shape, generator, device):
    weights = torch.empty(shape, device=device)
    torch.nn.init.xavier_uniform_(weights, generator=generator)
    return torch.nn.Parameter(weights)


def _shuffled(torch, rows, size, generator):
    order = torch.randperm(len(rows), generator=generator, device=rows.device)
    for batch in order.split(size):
        yield rows[batch]


def _step(optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
