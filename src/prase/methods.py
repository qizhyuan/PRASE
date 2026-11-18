"""Presets for reasoning, embedding, hybrid alignment, and name matching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .baselines import NameEmbeddingMatch, NameMatch
from .embedding import EmbeddingModel
from .errors import ConfigError
from .paris import LiteralMatcher, ParisConfig, exact_literals
from .pipeline import Embed, ParisStep, Pipeline, Predict


def paris(config: ParisConfig | None = None, *, literal_matcher: LiteralMatcher = exact_literals) -> Pipeline:
    """Build a single reasoning step based on graph facts and literal matches."""
    return Pipeline([ParisStep(config or ParisConfig(), literal_matcher=literal_matcher)], name="paris")


def prase(
    model: EmbeddingModel | str = "gcnalign",
    *,
    rounds: int = 1,
    options: dict[str, Any] | None = None,
    config: ParisConfig | None = None,
    seed_threshold: float = 0.1,
    propose: bool = True,
    share_embeddings: bool = True,
    proposal_weight: float = 1.0,
    literal_matcher: LiteralMatcher = exact_literals,
) -> Pipeline:
    """Run PARIS, then alternate embedding training and PARIS for ``rounds`` rounds.

    ``propose`` and ``share_embeddings`` choose the feedback channels: proposed
    pairs, embedding similarity, or both. A model name creates a separate model
    for each round using ``options``. A model instance is reused across rounds.
    """
    if type(rounds) is not int or rounds < 0:
        raise ConfigError("rounds must be a nonnegative integer")
    if options and not isinstance(model, str):
        raise ConfigError("options apply only when the model is given by name")
    config = config or ParisConfig()
    steps: list = [ParisStep(config, literal_matcher=literal_matcher)]
    for _ in range(rounds):
        steps.append(Embed(model, options=dict(options or {}), seed_threshold=seed_threshold,
                           propose=propose, share_embeddings=share_embeddings))
        steps.append(ParisStep(config, proposal_weight=proposal_weight, literal_matcher=literal_matcher))
    label = getattr(model, "name", model)
    return Pipeline(steps, name=f"prase-{label}")


def embedding_only(model: EmbeddingModel | str, *, options: dict[str, Any] | None = None) -> Pipeline:
    """Train on supplied seeds and turn the model's proposals into an alignment.

    Run with ``seeds=`` to supply known pairs. Their endpoints are excluded
    from prediction. The default proposals use nearest neighbors.
    """
    embed = Embed(model, options=dict(options or {}), seed_threshold=1.0, share_embeddings=False)
    return Pipeline([embed, Predict()], name=f"embedding-{getattr(embed.model, 'name', model)}")


def name_match(
    threshold: float = 0.5, *, left_names: Sequence[str] = (), right_names: Sequence[str] = (),
) -> Pipeline:
    """STR-Match: normalized edit similarity of entity names."""
    return Pipeline([NameMatch(threshold=threshold, left_names=left_names, right_names=right_names)], name="str-match")


def name_embedding_match(
    word_vectors: str | Path, threshold: float = 0.5, *,
    left_names: Sequence[str] = (), right_names: Sequence[str] = (),
) -> Pipeline:
    """EMB-Match: cosine of averaged word vectors of entity names."""
    step = NameEmbeddingMatch(word_vectors, threshold=threshold, left_names=left_names, right_names=right_names)
    return Pipeline([step], name="emb-match")
