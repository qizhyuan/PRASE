"""GCNAlign (Wang et al., EMNLP 2018), implemented with sparse graph views."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt

from ..errors import ConfigError, DataError
from .base import Embeddings, TrainingData
from .common import SeedValidation, check_device, finite_number, fraction, positive_int, to_embeddings, training_runtime


class GCNAlign:
    """Two graph convolution layers over relation structure, optionally with an attribute view.

    Counterparts are ranked by the Manhattan distance of their vectors, as in
    OpenEA. The first layer learns featureless entity representations and the
    second propagates those representations without another learned transform.
    """

    name = "gcnalign"

    def __init__(
        self,
        *,
        features: str = "structure",
        device: str = "cpu",
        dim: int = 100,
        epochs: int = 300,
        learning_rate: float = 8.0,
        margin: float = 3.0,
        negative_samples: int = 5,
        attribute_dim: int = 64,
        attribute_weight: float = 0.1,
    ):
        if features not in ("structure", "structure+attributes"):
            raise ConfigError("features must be 'structure' or 'structure+attributes'")
        self.features = features
        self.device = check_device(device)
        self.dim = positive_int("dim", dim)
        self.epochs = positive_int("epochs", epochs)
        self.learning_rate = finite_number("learning_rate", learning_rate)
        self.margin = finite_number("margin", margin)
        if self.learning_rate <= 0 or self.margin <= 0:
            raise ConfigError("learning_rate and margin must be positive")
        self.negative_samples = positive_int("negative_samples", negative_samples)
        self.attribute_dim = positive_int("attribute_dim", attribute_dim)
        self.attribute_weight = fraction("attribute_weight", attribute_weight)

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        if not len(data.seeds):
            raise DataError("GCNAlign requires at least one seed pair")
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(torch, data, seed, self.device, self.epochs, start=50, metric="l1")
        left, right = self._train(torch, generator, validation)
        return to_embeddings(left, right, "l1", info=validation.summary())

    def _train(self, torch, generator, validation):
        data = validation.training
        graphs = (data.left, data.right)
        node_counts = tuple(graph.num_entities for graph in graphs)
        adjacencies = tuple(_adjacency(torch, graph, self.device) for graph in graphs)
        features = tuple(_parameter(torch, (size, self.dim), self.device, generator) for size in node_counts)

        attribute_matrices = ()
        attribute_first_weights = ()
        parameters = list(features)
        if self.features == "structure+attributes":
            attribute_matrices = tuple(_attribute_matrix(torch, graph, self.device) for graph in graphs)
            attribute_first_weights = tuple(
                _parameter(torch, (matrix.shape[1], self.attribute_dim), self.device, generator)
                for matrix in attribute_matrices
            )
            parameters.extend(attribute_first_weights)

        def forward():
            structure = tuple(
                _propagate(torch, adjacency, torch.nn.functional.normalize(feature, dim=1))
                for adjacency, feature in zip(adjacencies, features, strict=True)
            )
            if not attribute_matrices:
                return (structure,)
            attributes = tuple(
                _propagate(torch, adjacency, torch.sparse.mm(matrix, first))
                for adjacency, matrix, first in zip(adjacencies, attribute_matrices, attribute_first_weights,
                                                    strict=True)
            )
            return structure, attributes

        def encode():
            structure, *attributes = forward()
            if not attributes:
                return structure
            weight = self.attribute_weight
            return tuple(torch.cat((relations * (1.0 - weight), attrs * weight), dim=1)
                         for relations, attrs in zip(structure, attributes[0], strict=True))

        anchors = torch.tensor(data.seeds, dtype=torch.long, device=self.device).T
        if max(node_counts) > 1:
            optimizer = torch.optim.SGD(parameters, lr=self.learning_rate)
            for epoch in range(1, self.epochs + 1):
                if (epoch - 1) % 10 == 0:
                    corruptions = tuple(
                        _corrupt(torch, seed_side, size, self.negative_samples, generator)
                        for seed_side, size in zip(anchors, node_counts, strict=True)
                    )
                optimizer.zero_grad(set_to_none=True)
                loss = sum(_ranking_loss(left, right, anchors, corruptions, self.margin) for left, right in forward())
                loss.backward()
                optimizer.step()
                if validation.stop(epoch, encode):
                    break
        return validation.vectors(encode)


def _connectivity(graph):
    triples = graph.relation_triples.tolist()
    counts = Counter()
    heads = defaultdict(set)
    tails = defaultdict(set)
    for head, predicate, tail in triples:
        counts[predicate] += 1
        heads[predicate].add(head)
        tails[predicate].add(tail)

    functionality = {predicate: len(heads[predicate]) / count for predicate, count in counts.items()}
    inverse = {predicate: len(tails[predicate]) / count for predicate, count in counts.items()}
    weights = defaultdict(float)
    for head, predicate, tail in triples:
        # Row j aggregates contributions made by source i to destination j.
        weights[tail, head] += inverse[predicate]
        weights[head, tail] += functionality[predicate]
    for index in range(graph.num_entities):
        weights[index, index] += 1.0

    degree = [0.0] * graph.num_entities
    for (destination, _), weight in weights.items():
        degree[destination] += weight
    return {
        (destination, source): weight / sqrt(degree[destination] * degree[source])
        for (destination, source), weight in weights.items()
    }


def _adjacency(torch, graph, device):
    weights = _connectivity(graph)
    indices = torch.tensor(list(weights), dtype=torch.long, device=device).reshape(-1, 2).T
    values = torch.tensor(list(weights.values()), dtype=torch.float32, device=device)
    size = graph.num_entities
    return torch.sparse_coo_tensor(indices, values, (size, size), check_invariants=True).coalesce()


def _attribute_matrix(torch, graph, device):
    rows = graph.attribute_triples.tolist()
    features = sorted({predicate for _, predicate, _ in rows})
    if not features:
        raise DataError("features = 'structure+attributes' requires attribute triples in both graphs")
    columns = {predicate: index for index, predicate in enumerate(features)}
    pairs = sorted({(entity, columns[predicate]) for entity, predicate, _ in rows})
    indices = torch.tensor(pairs, dtype=torch.long, device=device).T
    values = torch.ones(len(pairs), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, (graph.num_entities, len(columns)), check_invariants=True,
    ).coalesce()


def _parameter(torch, shape, device, generator):
    bound = sqrt(6.0 / sum(shape))
    values = torch.empty(shape, dtype=torch.float32, device=device)
    return torch.nn.Parameter(values.uniform_(-bound, bound, generator=generator))


def _propagate(torch, adjacency, features):
    hidden = torch.sparse.mm(adjacency, features).relu()
    return torch.sparse.mm(adjacency, hidden)


def _corrupt(torch, anchors, size, quantity, generator):
    if size == 1:
        return None
    draws = torch.randint(size - 1, (len(anchors), quantity), device=anchors.device, generator=generator)
    return draws + (draws >= anchors[:, None])


def _ranking_loss(left, right, anchors, corruptions, margin):
    left_positive = left[anchors[0]][:, None, :]
    right_positive = right[anchors[1]][:, None, :]
    positive_distances = (left_positive - right_positive).abs().sum(dim=2)
    negative_left, negative_right = corruptions
    losses = []
    if negative_left is not None:
        distance = (left[negative_left] - right_positive).abs().sum(dim=2)
        losses.append((positive_distances + margin - distance).relu().mean())
    if negative_right is not None:
        distance = (left_positive - right[negative_right]).abs().sum(dim=2)
        losses.append((positive_distances + margin - distance).relu().mean())
    return sum(losses)
