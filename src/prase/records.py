"""Write a run's intermediate and final results to a directory.

Layout::

    run.json                 settings, graph summaries, timings, and scores of every step
    alignment.tsv            the final alignment
    01-paris/
        alignment.tsv        the alignment after this step
        predicates.tsv       predicate subsumption probabilities
        step.json            timing, counts, scores, and per-iteration scores
    02-embed-gcnalign/
        proposals.tsv        pairs proposed for the next reasoning step
        embeddings.npz       entity vectors, when saving embeddings is enabled
        step.json            timing, counts, the training summary, and proposal precision with a reference
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import __version__
from .alignment import write_predicate_matches
from .errors import DataError

if TYPE_CHECKING:
    from .kg import KG
    from .pipeline import Pipeline, RunResult, State, StepRecord


class RunWriter:
    def __init__(
        self, directory: str | Path, pipeline: Pipeline, left: KG, right: KG, seed: int, save_embeddings: bool,
    ) -> None:
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise DataError(f"{self.directory}: choose a new or empty output directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.left, self.right = left, right
        self.save_embeddings = save_embeddings
        self.run: dict[str, Any] = {
            "prase_version": __version__,
            "python": platform.python_version(),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline": pipeline.name,
            "steps": [describe(step) for step in pipeline.steps],
            "seed": seed,
            "graphs": [
                {"name": kg.name, "fingerprint": kg.fingerprint, **kg.summary()} for kg in (left, right)
            ],
            "records": [],
        }
        self._write_run()

    def step(self, record: StepRecord, before: State, after: State) -> None:
        folder = self.directory / f"{record.index:02d}-{record.name}"
        folder.mkdir()
        if after.alignment is not before.alignment:
            after.alignment.write(folder / "alignment.tsv", self.left, self.right)
        if after.predicates is not before.predicates and after.predicates:
            write_predicate_matches(folder / "predicates.tsv", after.predicates)
        if after.proposals is not None and after.proposals is not before.proposals:
            after.proposals.write(folder / "proposals.tsv", self.left, self.right)
        if after.embeddings is not None and after.embeddings is not before.embeddings and self.save_embeddings:
            after.embeddings.save(folder / "embeddings.npz")
        summary = _record(record)
        _write_json(folder / "step.json", summary)
        self.run["records"].append({key: value for key, value in summary.items() if key != "observations"})
        self._write_run()

    def fail(self, index: int, name: str, error: Exception) -> None:
        self.run["failed"] = {"step": index, "name": name, "error": f"{type(error).__name__}: {error}"}
        self._write_run()

    def finish(self, result: RunResult) -> None:
        result.alignment.write(self.directory / "alignment.tsv", self.left, self.right)
        self.run["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.run["seconds"] = result.seconds
        self.run["matches"] = len(result.alignment)
        if result.scores is not None:
            self.run["scores"] = result.scores.as_dict()
        self._write_run()

    def _write_run(self) -> None:
        _write_json(self.directory / "run.json", self.run)


def describe(step: object) -> dict[str, Any]:
    """A JSON-friendly description of a step's settings."""
    fields = getattr(step, "__dataclass_fields__", None)
    values: dict[str, Any] = {"type": type(step).__name__, "name": getattr(step, "name", "")}
    if fields:
        for name in fields:
            if name.startswith("_") or name == "name":
                continue
            values[name] = _plain(getattr(step, name))
    return values


def _plain(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if callable(getattr(value, "fit", None)):
        settings = {
            key: _plain(item) for key, item in getattr(value, "__dict__", {}).items()
            if not key.startswith("_") and key != "labels"
        }
        return {"model": getattr(value, "name", type(value).__name__), **settings}
    if isinstance(value, Path):
        return str(value)
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _record(record: StepRecord) -> dict[str, Any]:
    return {
        "index": record.index,
        "name": record.name,
        "seconds": record.seconds,
        "values": _plain(record.values),
        "scores": None if record.scores is None else record.scores.as_dict(),
        "proposal_precision": record.proposal_precision,
        "observations": _plain(record.observations),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
