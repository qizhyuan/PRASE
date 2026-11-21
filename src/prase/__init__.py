"""PRASE: probabilistic reasoning and semantic embedding for knowledge graph alignment.

The main pieces are graphs (:class:`KG`), alignments (:class:`Alignment`),
the PARIS reasoner (:class:`Paris`), embedding models
(:func:`create_model`), and pipelines that combine them
(:class:`Pipeline`, with presets in :mod:`prase.methods`).
"""

__version__ = "1.0.1"

from . import methods
from .alignment import Alignment, PredicateMatch
from .baselines import NameEmbeddingMatch, NameMatch
from .checkpoints import CheckpointPolicy, training_checkpoints
from .config import RunConfig, load_config
from .datasets import Dataset, download_benchmark, download_word_vectors, load_benchmark, load_dataset, read_links
from .decisions import Decisions
from .embedding import EmbeddingModel, Embeddings, TrainingData, available_models, create_model
from .errors import ConfigError, DataError, PraseError
from .kg import KG, Predicate, clean_literal
from .metrics import Scores, evaluate, ranking_scores, threshold_scores
from .paris import Candidate, Paris, ParisConfig, ParisResult
from .pipeline import Context, Embed, Load, ParisStep, Pipeline, Predict, RunResult, State, Step
from .session import Session

__all__ = [
    "KG", "Alignment", "Candidate", "CheckpointPolicy", "ConfigError", "Context", "DataError", "Dataset",
    "Decisions", "Embed", "EmbeddingModel", "Embeddings", "Load", "NameEmbeddingMatch", "NameMatch", "Paris",
    "ParisConfig", "ParisResult", "ParisStep", "Pipeline", "Predicate", "PredicateMatch", "Predict", "PraseError",
    "RunConfig", "RunResult", "Scores", "Session", "State", "Step", "TrainingData", "available_models",
    "clean_literal", "create_model", "download_benchmark", "download_word_vectors", "evaluate", "load_benchmark",
    "load_config", "load_dataset", "methods", "ranking_scores", "read_links", "threshold_scores",
    "training_checkpoints",
]
