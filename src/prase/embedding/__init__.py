"""Embedding models and their registry.

``create_model`` accepts a built-in name, an import path ``"module:Factory"``,
or the name of a ``prase.models`` entry point. Built-in models import
PyTorch only when they train.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points

from ..errors import ConfigError
from .base import METRICS, EmbeddingModel, Embeddings, TrainingData

__all__ = [
    "METRICS", "EmbeddingModel", "Embeddings", "TrainingData", "available_models", "create_model",
    "model_path_options",
]

_BUILTIN = {
    "bootea": ("bootea", "BootEA"),
    "gcnalign": ("gcnalign", "GCNAlign"),
    "imuse": ("imuse", "IMUSE"),
    "iptranse": ("iptranse", "IPTransE"),
    "mtranse": ("mtranse", "MTransE"),
    "multike": ("multike", "MultiKE"),
}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_BUILTIN))


def _factory(name: str) -> Callable[..., EmbeddingModel]:
    if name in _BUILTIN:
        module, attribute = _BUILTIN[name]
        return getattr(import_module(f".{module}", __name__), attribute)
    if ":" in name:
        module, _, attribute = name.partition(":")
        return getattr(import_module(module), attribute)
    matches = list(entry_points(group="prase.models", name=name))
    if len(matches) != 1:
        raise ConfigError(f"unknown embedding model {name!r}; built-in models: {', '.join(available_models())}")
    return matches[0].load()


def create_model(name: str, **options: object) -> EmbeddingModel:
    """Construct an embedding model from its name and options."""
    try:
        factory = _factory(name)
    except (ImportError, AttributeError) as error:
        raise ConfigError(f"cannot load embedding model {name!r}: {error}") from error
    try:
        model = factory(**options)
    except TypeError as error:
        raise ConfigError(f"invalid options for embedding model {name!r}: {error}") from error
    if not callable(getattr(model, "fit", None)):
        raise ConfigError(f"embedding model {name!r} has no fit(data, seed=...) method")
    return model


def model_path_options(name: str) -> tuple[str, ...]:
    """Names of the options that a model reads as file paths, declared as path_options."""
    try:
        factory = _factory(name)
    except (ConfigError, ImportError, AttributeError):
        return ()
    return tuple(getattr(factory, "path_options", ()))
