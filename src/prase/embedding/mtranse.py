"""MTransE (Chen et al., IJCAI 2017) with separate graph embeddings and a learned linear alignment."""

from __future__ import annotations

from ..errors import ConfigError, DataError
from .base import Embeddings, TrainingData
from .common import SeedValidation, check_device, finite_number, positive_int, to_embeddings, training_runtime


class MTransE:
    """TransE in each graph and a linear map between the two spaces.

    Training minimizes the distance of positive triples in both graphs and the
    distance between mapped seed pairs. ``alpha`` weights the alignment
    objective, and an orthogonality penalty keeps the map close to a rotation.
    """

    name = "mtranse"

    def __init__(
        self,
        *,
        device: str = "cpu",
        dim: int = 100,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 5000,
        alpha: float = 5.0,
        block_size: int = 512,
    ):
        self.device = check_device(device)
        self.dim = positive_int("dim", dim)
        self.epochs = positive_int("epochs", epochs)
        self.batch_size = positive_int("batch_size", batch_size)
        self.block_size = positive_int("block_size", block_size)
        self.learning_rate = finite_number("learning_rate", learning_rate)
        self.alpha = finite_number("alpha", alpha)
        if self.learning_rate <= 0 or self.alpha <= 0:
            raise ConfigError("learning_rate and alpha must be positive")

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        if not len(data.seeds):
            raise DataError("MTransE requires at least one seed pair")
        torch, generator = training_runtime(self.device, seed)
        validation = SeedValidation(torch, data, seed, self.device, self.epochs)
        left, right = self._train(torch, generator, validation)
        return to_embeddings(left, right, "cosine", info=validation.summary())

    def _train(self, torch, generator, validation):
        data = validation.training
        left, right = data.left, data.right
        left_size = left.num_entities
        entity_count = left_size + right.num_entities
        predicate_offset = len(left.predicates)
        predicate_count = predicate_offset + len(right.predicates)
        edges = left.relation_triples.tolist()
        edges.extend(
            (head + left_size, predicate + predicate_offset, tail + left_size)
            for head, predicate, tail in right.relation_triples.tolist()
        )
        if not edges:
            raise DataError("MTransE requires relation triples")
        triples = torch.tensor(edges, dtype=torch.long, device=self.device)
        anchors = torch.tensor(
            [(i, j + left_size) for i, j in data.seed_pairs], dtype=torch.long, device=self.device,
        )

        normalize = torch.nn.functional.normalize
        entities = _unit_parameters(torch, normalize, entity_count, self.dim, self.device, generator)
        relations = _unit_parameters(torch, normalize, predicate_count, self.dim, self.device, generator)
        transform = torch.nn.Parameter(torch.empty(self.dim, self.dim, dtype=torch.float32, device=self.device))
        torch.nn.init.orthogonal_(transform, generator=generator)
        eye = torch.eye(self.dim, dtype=torch.float32, device=self.device)
        triple_optimizer = torch.optim.Adagrad(
            [entities, relations], lr=self.learning_rate, initial_accumulator_value=0.1,
        )
        mapping_optimizer = torch.optim.Adagrad(
            [entities, transform], lr=self.learning_rate, initial_accumulator_value=0.1,
        )
        triple_steps = (len(triples) + self.batch_size - 1) // self.batch_size
        # Each epoch interleaves one mapping batch, of at least one seed pair, per triple batch.
        seeds_per_step = max(1, len(anchors) // triple_steps)

        def encode():
            mapped_left = normalize(normalize(entities[:left_size], dim=1) @ transform, dim=1)
            return mapped_left, normalize(entities[left_size:], dim=1)

        for epoch in range(1, self.epochs + 1):
            order = torch.randperm(len(triples), device=self.device, generator=generator)
            for batch in order.split(self.batch_size):
                head, predicate, tail = triples[batch].T
                residual = normalize(entities[head], dim=1) + normalize(relations[predicate], dim=1)
                residual = residual - normalize(entities[tail], dim=1)
                triple_optimizer.zero_grad(set_to_none=True)
                residual.square().sum().backward()
                triple_optimizer.step()

            for _ in range(triple_steps):
                samples = torch.randperm(len(anchors), device=self.device, generator=generator)[:seeds_per_step]
                source, target = anchors[samples].T
                loss = _mapping_loss(
                    normalize(entities[source], dim=1), normalize(entities[target], dim=1), transform, eye,
                )
                mapping_optimizer.zero_grad(set_to_none=True)
                (self.alpha * loss).backward()
                mapping_optimizer.step()
            if validation.stop(epoch, encode):
                break

        return validation.vectors(encode)


def _unit_parameters(torch, normalize, count, dim, device, generator):
    samples = torch.randn(count, dim, dtype=torch.float32, device=device, generator=generator)
    return torch.nn.Parameter(normalize(samples, dim=1))


def _mapping_loss(left, right, transform, eye):
    alignment = (left @ transform - right).square().sum()
    orthogonality = (transform @ transform.T - eye).square().sum()
    return alignment + orthogonality
