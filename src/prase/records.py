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
    checkpoint/              step results and training state for resuming the run, removed when it finishes
"""

from __future__ import annotations

import json
import platform
import re
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
        *, resume: bool = False,
    ) -> None:
        self.directory = Path(directory)
        if self.directory.exists() and not self.directory.is_dir():
            raise DataError(f"{self.directory} is a file, not an output directory")
        resumed = self.directory.exists() and any(self.directory.iterdir())
        if resumed:
            if not resume:
                raise DataError(f"{self.directory}: choose a new or empty output directory, or resume the run in it")
            if not (self.directory / "run.json").is_file():
                raise DataError(f"{self.directory}: there is no run to resume here")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.resumed = resumed
        self.written: set[str] = set()
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
        if resumed:
            self.run["resumed"] = True
        self._write_run()

    def step(self, record: StepRecord, before: State, after: State) -> None:
        folder = self.directory / f"{record.index:02d}-{record.name}"
        # A resumed run replaces the files that the earlier run wrote for this step.
        _remove_step_outputs(folder)
        folder.mkdir(exist_ok=True)
        self.written.add(folder.name)
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
        if self.resumed:
            # Step folders of the earlier run that this run did not write again.
            for entry in self.directory.iterdir():
                if entry.is_dir() and _STEP_FOLDER.fullmatch(entry.name) and entry.name not in self.written:
                    _remove_step_outputs(entry)
        result.alignment.write(self.directory / "alignment.tsv", self.left, self.right)
        self.run["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.run["seconds"] = result.seconds
        self.run["matches"] = len(result.alignment)
        if result.scores is not None:
            self.run["scores"] = result.scores.as_dict()
        self._write_run()

    def _write_run(self) -> None:
        _write_json(self.directory / "run.json", self.run)


_STEP_FOLDER = re.compile(r"\d{2,}-.+")
_STEP_OUTPUTS = ("alignment.tsv", "predicates.tsv", "proposals.tsv", "embeddings.npz", "step.json", "step.tmp")


def _remove_step_outputs(folder: Path) -> None:
    """Remove the files a run writes in a step folder, and the folder once it is empty. Other files stay."""
    if not folder.is_dir():
        return
    for name in _STEP_OUTPUTS:
        (folder / name).unlink(missing_ok=True)
    try:
        folder.rmdir()
    except OSError:
        pass


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
        **({"resumed": True} if record.resumed else {}),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
