"""A deterministic embedding model for tests, loadable as ``fixed_model:FixedModel``."""

import numpy as np

from prase import Alignment, Embeddings, TrainingData


def no_literals(left, right) -> Alignment:
    """A literal matcher that pairs nothing, so reasoning has no starting evidence."""
    return Alignment()


class FixedModel:
    """Give aligned seed entities equal vectors and everything else a vector from its name hash.

    When ``reference`` is set, unmatched candidates also get the vectors of
    their true counterparts, which makes proposals predictable in tests.
    """

    name = "fixed"

    def __init__(self, dim: int = 4, reference=None):
        self.dim = dim
        self.reference = reference
        self.seen: list[TrainingData] = []

    def fit(self, data: TrainingData, *, seed: int) -> Embeddings:
        self.seen.append(data)
        rng = np.random.default_rng(seed)
        left = rng.normal(size=(data.left.num_entities, self.dim))
        right = rng.normal(size=(data.right.num_entities, self.dim))
        pairs = data.seeds if self.reference is None else np.asarray(self.reference)
        right[pairs[:, 1]] = left[pairs[:, 0]]
        return Embeddings(left, right)
