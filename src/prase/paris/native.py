"""Adapter for the compiled PARIS iteration in ``prase._paris``.

Graph indexes are built in NumPy (:class:`~prase.paris.state.GraphIndex`) and
shared with the extension, so both engines read identical inputs. The
extension runs the same iteration as
:meth:`prase.paris.python.PythonEngine.iterate`.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

from .state import GraphIndex, ParisState, RelationTable

if TYPE_CHECKING:
    from ..embedding.base import Embeddings
    from . import ParisConfig

_LOGGER = logging.getLogger("prase")
_LOAD_ERROR: ImportError | None = None
_REPORTED = False
_API = 7

try:
    _paris = import_module(f"{__package__.rpartition('.')[0]}._paris")
    if getattr(_paris, "API_VERSION", None) != _API:
        _LOAD_ERROR = ImportError(f"prase._paris implements API {getattr(_paris, 'API_VERSION', 1)}, expected {_API}")
        _paris = None
except ImportError as error:
    _LOAD_ERROR = None if isinstance(error, ModuleNotFoundError) and error.name == "prase._paris" else error
    _paris = None


def available() -> bool:
    return _paris is not None


def load_error() -> ImportError | str:
    return _LOAD_ERROR or "the extension is not installed"


def report_fallback() -> None:
    """Note once per process that the NumPy engine is in use."""
    global _REPORTED
    if not _REPORTED:
        _REPORTED = True
        _LOGGER.info("Using the NumPy PARIS engine (%s)", load_error())


def _arrays(index: GraphIndex) -> dict[str, object]:
    return {
        "num_entities": index.num_entities,
        "num_terms": index.num_terms,
        "num_predicates": index.num_predicates,
        "functionality": index.functionality,
        "source_ptr": index.source_ptr,
        "source_pred": index.source_pred,
        "source_head": index.source_head,
        "target_tail": index.target_tail,
        "group_start": index.group_start,
        "group_end": index.group_end,
        "group_pred": index.group_pred,
        "head_groups": index.head_groups,
        "pair_keys": index.pair_keys,
        "pair_pred": index.pair_pred,
    }


class NativeEngine:
    name = "native"

    def __init__(self, left: GraphIndex, right: GraphIndex, config: ParisConfig) -> None:
        threads = config.threads or (getattr(os, "process_cpu_count", os.cpu_count)() or 1)
        self._engine = _paris.Engine(
            _arrays(left), _arrays(right), config.theta, config.delta, config.epsilon,
            config.smoothing, config.embedding_weight, threads,
        )

    def iterate(self, state: ParisState, warm: bool, embeddings: Embeddings | None) -> dict[str, int]:
        left_vectors = right_vectors = None
        if embeddings is not None and not warm:
            left_vectors, right_vectors = embeddings.left, embeddings.right
        (left_keys, left_values, right_keys, right_values, stats) = self._engine.iterate(
            state.left_match, state.left_prob, state.right_match, state.right_prob,
            state.left_relations.keys, state.left_relations.values,
            state.right_relations.keys, state.right_relations.values,
            state.fixed_left_relations, state.fixed_right_relations,
            state.fixed_left, state.fixed_right, state.rejected, warm, left_vectors, right_vectors,
        )
        state.left_relations = RelationTable(np.asarray(left_keys), np.asarray(left_values), state.left_relations.width)
        state.right_relations = RelationTable(
            np.asarray(right_keys), np.asarray(right_values), state.right_relations.width,
        )
        return dict(stats)
