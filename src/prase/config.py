"""Read TOML run files and build pipelines.

A run file holds the settings below. Relative paths are resolved against the
run file's directory, and unknown settings are an error.

``seed``
    The random seed of the run (default 0).

``[input]``
    Exactly one source of graphs: ``benchmark`` (a benchmark ID), ``dataset``
    (a folder in the OpenEA layout), or the tables ``[input.left]`` and
    ``[input.right]``. A graph table holds a ``relations`` TSV file and an
    optional ``attributes`` TSV file, or an ``rdf`` file with an optional
    RDFLib ``format``. ``reference`` (evaluation pairs), ``seeds`` (known
    pairs), and ``decisions`` (approve and reject rows) are optional TSV files.

``[method]``
    A preset ``name``:

    - ``paris``: a ``literal_matcher`` import path and a ``[method.paris]``
      table of :class:`~prase.paris.ParisConfig` fields.
    - ``prase``: the ``paris`` settings, plus ``model``, ``options``,
      ``rounds``, ``seed_threshold``, ``propose``, ``share_embeddings``, and
      ``proposal_weight``.
    - ``embedding-only``: ``model`` and ``options``.
    - ``str-match``: the fields of :class:`~prase.baselines.NameMatch`.
    - ``emb-match``: the fields of :class:`~prase.baselines.NameEmbeddingMatch`.

    ``model`` is a built-in model name, a ``"module:Factory"`` import path, or
    a ``prase.models`` entry point, and ``options`` its keyword arguments.

``[[steps]]``
    A custom pipeline in place of ``[method]``. Each step has a ``type``:

    - ``paris``: :class:`~prase.paris.ParisConfig` fields, ``proposal_weight``,
      ``use_proposals``, ``use_embeddings``, and ``literal_matcher``.
    - ``embed``: ``model``, ``options``, ``seed_threshold``, ``propose``, and
      ``share_embeddings``.
    - ``predict``: no settings.
    - ``load``: ``proposals`` (TSV) and ``embeddings`` (``.npz``) files.
    - ``names`` and ``name_embeddings``: the fields of the two name matchers.
    - A ``"module:Factory"`` import path: the other fields become keyword
      arguments to the factory.

``[output]``
    ``directory`` for the run records, and ``embeddings = true`` to save the
    entity vectors there as well.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from importlib import import_module
from pathlib import Path
from typing import Any

from . import methods
from .alignment import Alignment
from .baselines import NameEmbeddingMatch, NameMatch
from .datasets import BENCHMARK_NAMES, Dataset, load_benchmark, load_dataset, read_links
from .decisions import Decisions
from .embedding import model_path_options
from .errors import ConfigError
from .kg import KG
from .paris import ParisConfig
from .pipeline import Embed, Load, ParisStep, Pipeline, Predict, RunResult, Step


@dataclass
class RunConfig:
    """A pipeline, its inputs, and the settings for one run."""

    pipeline: Pipeline
    dataset: Dataset
    reference: tuple[tuple[str, str], ...]
    decisions: Decisions | None = None
    seeds: Alignment | None = None
    seed: int = 0
    output: Path | None = None
    save_embeddings: bool = False

    def run(self) -> RunResult:
        """Run the pipeline and score it when reference links are available."""
        return self.pipeline.run(
            self.dataset.left, self.dataset.right, decisions=self.decisions, seeds=self.seeds, seed=self.seed,
            reference=self.dataset.reference(self.reference) if self.reference else None,
            output=self.output, save_embeddings=self.save_embeddings,
        )


def load_config(path: str | Path) -> RunConfig:
    """Read a TOML run file and load its pipeline and graph pair."""
    path = Path(path)
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path}: {error}") from error
    base = path.absolute().parent
    _known(values, {"seed", "input", "method", "steps", "output"}, "run file")
    if ("method" in values) == ("steps" in values):
        raise ConfigError("a run file needs either a [method] table or [[steps]] tables")
    pipeline = build_method(values["method"], base) if "method" in values else build_steps(values["steps"], base)

    source = _table(values, "input")
    _known(source, {"benchmark", "dataset", "left", "right", "reference", "decisions", "seeds"}, "[input]")
    dataset = _dataset(source, base)
    reference = dataset.links
    if "reference" in source:
        reference = read_links(_path(source["reference"], base))
    decisions = Decisions.read(_path(source["decisions"], base)) if "decisions" in source else None
    seeds = Alignment.read(_path(source["seeds"], base), dataset.left, dataset.right) if "seeds" in source else None

    output = _table(values, "output")
    _known(output, {"directory", "embeddings"}, "[output]")
    seed = values.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ConfigError("seed must be a nonnegative integer")
    return RunConfig(
        pipeline, dataset, tuple(reference), decisions, seeds, seed,
        _path(output["directory"], base) if "directory" in output else None, bool(output.get("embeddings", False)),
    )


def build_method(table: Mapping[str, Any], base: Path) -> Pipeline:
    """Build a preset pipeline from a ``[method]`` table."""
    if not isinstance(table, Mapping):
        raise ConfigError("[method] must be a table")
    table = dict(table)
    name = table.pop("name", None)
    if name in ("paris", "prase"):
        paris_settings = ParisConfig(**_settings(table.pop("paris", {}), ParisConfig, "[method.paris]"))
        literals = {"literal_matcher": _imported(table.pop("literal_matcher"))} if "literal_matcher" in table else {}
        if name == "paris":
            _known(table, set(), "[method]")
            return methods.paris(paris_settings, **literals)
        _known(table, {"model", "rounds", "options", "seed_threshold", "propose", "share_embeddings",
                       "proposal_weight"}, "[method]")
        model = table.pop("model", "gcnalign")
        options = _model_options(model, table.pop("options", {}), base)
        return methods.prase(model, options=options, config=paris_settings, **literals, **table)
    if name == "embedding-only":
        _known(table, {"model", "options"}, "[method]")
        model = table.pop("model", "gcnalign")
        return methods.embedding_only(model, options=_model_options(model, table.pop("options", {}), base))
    if name == "str-match":
        return Pipeline([_build("names", table, base)], name="str-match")
    if name == "emb-match":
        return Pipeline([_build("name_embeddings", table, base)], name="emb-match")
    raise ConfigError("[method] name must be paris, prase, embedding-only, str-match, or emb-match")


def build_steps(tables: list[Mapping[str, Any]], base: Path) -> Pipeline:
    """Build a pipeline from ``[[steps]]`` tables."""
    if not isinstance(tables, list) or not tables:
        raise ConfigError("[[steps]] must list at least one step")
    steps = []
    for index, table in enumerate(tables, 1):
        if not isinstance(table, Mapping):
            raise ConfigError(f"step {index} must be a table")
        table = dict(table)
        kind = table.pop("type", None)
        if not isinstance(kind, str):
            raise ConfigError(f"step {index} needs a type")
        try:
            steps.append(_build(kind, table, base))
        except ConfigError as error:
            raise ConfigError(f"step {index} ({kind}): {error}") from error
    return Pipeline(steps, name="custom")


def _build(kind: str, table: dict[str, Any], base: Path) -> Step:
    if kind == "paris":
        own = {key: table.pop(key) for key in ("proposal_weight", "use_proposals", "use_embeddings") if key in table}
        if "literal_matcher" in table:
            own["literal_matcher"] = _imported(table.pop("literal_matcher"))
        return ParisStep(ParisConfig(**_settings(table, ParisConfig, "paris step")), **own)
    if kind == "embed":
        _known(table, {"model", "options", "seed_threshold", "propose", "share_embeddings"}, "embed step")
        model = table.pop("model", None)
        if not isinstance(model, str):
            raise ConfigError("an embed step needs a model name")
        return Embed(model, _model_options(model, table.pop("options", {}), base), **table)
    if kind == "predict":
        _known(table, set(), "predict step")
        return Predict()
    if kind == "load":
        _known(table, {"proposals", "embeddings"}, "load step")
        return Load(**{key: _path(value, base) for key, value in table.items()})
    if kind == "names":
        return NameMatch(**_settings(table, NameMatch, "names step"))
    if kind == "name_embeddings":
        settings = _settings(table, NameEmbeddingMatch, "name_embeddings step")
        if "word_vectors" not in settings:
            raise ConfigError("name_embeddings needs word_vectors")
        settings["word_vectors"] = _path(settings["word_vectors"], base)
        return NameEmbeddingMatch(**settings)
    if ":" in kind:
        factory: Callable[..., Step] = _imported(kind)
        if not callable(factory):
            raise ConfigError(f"{kind!r} is not a step factory")
        try:
            return factory(**table)
        except TypeError as error:
            raise ConfigError(f"invalid options for step factory {kind!r}: {error}") from error
    raise ConfigError(f"unknown step type {kind!r}")


def _imported(path: Any) -> Any:
    """The object named by a ``module:attribute`` import path."""
    if not isinstance(path, str) or ":" not in path:
        raise ConfigError(f"expected a module:attribute import path, not {path!r}")
    module, _, attribute = path.partition(":")
    try:
        return getattr(import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise ConfigError(f"cannot import {path!r}: {error}") from error


def _model_options(model: str, options: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve the options a model declares as file paths against the run file's folder."""
    if not isinstance(model, str) or not model:
        raise ConfigError("expected a model name or module:Factory path")
    if not isinstance(options, dict):
        raise ConfigError("model options must be a table")
    paths = model_path_options(model)
    return {key: _path(value, base) if key in paths and isinstance(value, str) else value
            for key, value in options.items()}


def _dataset(source: dict[str, Any], base: Path) -> Dataset:
    selection = source.keys() & {"benchmark", "dataset", "left", "right"}
    if selection == {"benchmark"}:
        name = source["benchmark"]
        if name not in BENCHMARK_NAMES:
            raise ConfigError(f"[input].benchmark must be one of {', '.join(BENCHMARK_NAMES)}")
        return load_benchmark(name)
    if selection == {"dataset"}:
        return load_dataset(_path(source["dataset"], base))
    if selection != {"left", "right"}:
        raise ConfigError("[input] requires exactly one of benchmark, dataset, or both left and right")
    return Dataset("input", _graph(source["left"], base, "left"), _graph(source["right"], base, "right"), ())


def _graph(table: Any, base: Path, side: str) -> KG:
    if not isinstance(table, dict):
        raise ConfigError(f"[input] {side} must be a table")
    if "rdf" in table:
        _known(table, {"rdf", "format"}, f"[input.{side}]")
        return KG.from_rdf(_path(table["rdf"], base), format=table.get("format"), name=side)
    _known(table, {"relations", "attributes"}, f"[input.{side}]")
    if "relations" not in table:
        raise ConfigError(f"[input.{side}] needs relations, or rdf")
    attributes = _path(table["attributes"], base) if "attributes" in table else None
    return KG.from_tsv(_path(table["relations"], base), attributes, name=side)


def _settings(table: Any, cls: type, label: str) -> dict[str, Any]:
    if not isinstance(table, dict):
        raise ConfigError(f"{label} must be a table")
    allowed = {item.name for item in fields(cls) if not item.name.startswith("_") and item.name != "name"}
    _known(table, allowed, label)
    return dict(table)


def _table(values: dict[str, Any], key: str) -> dict[str, Any]:
    table = values.get(key, {})
    if not isinstance(table, dict):
        raise ConfigError(f"[{key}] must be a table")
    return table


def _known(table: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"unknown setting {unknown[0]!r} in {label}")


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("expected a file path")
    path = Path(value)
    return path if path.is_absolute() else base / path
