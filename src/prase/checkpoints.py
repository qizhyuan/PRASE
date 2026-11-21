"""Checkpoints that let an interrupted run resume from the work it finished.

A pipeline run with an output directory saves two kinds of checkpoints in
``<output>/checkpoint/``:

- the result of every finished step, which a resumed run reuses for steps
  whose inputs, settings, files, and code are unchanged;
- the training state of an embedding model, every ``training_minutes`` and
  when training finishes, which a resumed run continues from the last saved
  epoch. A finished fit stays until its step's result is saved.

Every checkpoint carries a SHA-256 digest of its content and is read in
full and checked before anything is restored, so a damaged, incomplete, or
altered checkpoint only makes its step run again or its training start
afresh.

:class:`CheckpointPolicy` chooses which checkpoints a run saves and whether
they stay after the run finishes::

    from prase import CheckpointPolicy, methods

    pipeline = methods.prase("bootea", rounds=2)
    policy = CheckpointPolicy(training_minutes=5, keep=True)
    pipeline.run(left, right, output="runs/bootea", checkpoints=policy)
    pipeline.run(left, right, output="runs/bootea", checkpoints=policy, resume=True)  # after an interruption

A step's checkpoint is reused when the step's identity, its seed, and the
content of the state it receives are unchanged, in the same PRASE build and
library versions (:func:`environment_identity`) and, for a model, the same
PyTorch settings (:func:`torch_runtime`). The built-in steps and
models describe their settings and the files they read. A custom step, model,
or literal matcher defines ``checkpoint_key()``, which returns its settings
as JSON values with a digest of every file it reads (:func:`file_digest`);
the key also covers the source of its module. Without the method, the step
runs again on resume and the model trains without checkpoints.

:func:`training_checkpoints` gives the same protection to a model fitted
outside a pipeline.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import json
import logging
import math
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ConfigError, DataError

if TYPE_CHECKING:
    from .embedding.base import TrainingData

_LOGGER = logging.getLogger("prase")
_FORMAT = 1
MARKER = "prase-checkpoints.json"
_STEP_FILES = ("step.json", "state.npz", "state.npz.tmp", "step.json.tmp")
_OUTPUT_NAMES = re.compile(r"\d{2,}-.+|alignment\.tsv|run\.json")


@dataclass(frozen=True)
class CheckpointPolicy:
    """Which checkpoints a pipeline run saves, and whether it keeps them.

    ``steps`` saves the result of every finished step. ``training`` saves
    the state of an embedding model every ``training_minutes`` of training,
    or after every epoch when ``training_minutes`` is 0. ``keep`` leaves the
    checkpoints in place after the run finishes, so that a later run with
    ``resume`` reuses every step. ``directory`` replaces the default folder,
    ``checkpoint`` inside the output directory. It must be new, empty, or a
    checkpoint folder of an earlier run, and must not contain the output
    directory.
    """

    steps: bool = True
    training: bool = True
    training_minutes: float = 10.0
    keep: bool = False
    directory: str | Path | None = None

    def __post_init__(self) -> None:
        for name in ("steps", "training", "keep"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be true or false")
        _check_minutes("training_minutes", self.training_minutes)
        if self.directory is not None:
            if not isinstance(self.directory, (str, Path)) or not str(self.directory):
                raise ConfigError("directory must be a folder path")
            object.__setattr__(self, "directory", Path(self.directory))

    @property
    def enabled(self) -> bool:
        return self.steps or self.training


def _check_minutes(name: str, minutes: Any) -> None:
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes) \
            or minutes < 0:
        raise ConfigError(f"{name} must be a nonnegative number")


# Identities ----------------------------------------------------------------

class _Unknown(Exception):
    """A value whose content cannot be described, so a checkpoint for it is never reused."""


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file's content, read in full every time: file times cannot show every change."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path | None) -> str | None:
    """The digest of a file a step reads, or None when there is no file."""
    return None if path is None else file_digest(path)


@functools.cache
def environment_identity() -> dict[str, str | None]:
    """The PRASE build and the libraries that decide results, the same for every checkpoint of a process.

    ``source`` is a digest of the package's Python and TOML files and
    ``native`` of its PARIS extension; ``numpy`` and ``torch`` are the
    installed versions.
    """
    from . import __version__
    from .paris import native

    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in (".py", ".toml") and "__pycache__" not in path.parts:
            digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes())
    extension = getattr(native._paris, "__file__", None)
    return {
        "prase": __version__,
        "source": digest.hexdigest()[:16],
        "native": file_digest(extension)[:16] if extension else None,
        "numpy": np.__version__,
        "torch": _installed_version("torch"),
    }


def torch_runtime() -> dict[str, Any] | None:
    """The PyTorch settings of this moment that change training results, or None without PyTorch.

    They are the default floating-point type, the precision of float32 matrix
    products, the TF32 switches, and whether deterministic algorithms are enforced.
    """
    if _installed_version("torch") is None:
        return None
    import torch

    return {
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _installed_version(name: str) -> str | None:
    module = sys.modules.get(name)
    if module is not None:
        return getattr(module, "__version__", None)
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def plain_identity(value: Any) -> Any:
    """Settings as JSON values; arrays become digests. Raise :class:`_Unknown` for other objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [plain_identity(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain_identity(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return {"array": array_digest(value)}
    if isinstance(value, np.generic):
        return value.item()
    raise _Unknown(type(value).__qualname__)


def _is_prase(value: Any) -> bool:
    module = getattr(value if inspect.isclass(value) or inspect.isroutine(value) else type(value), "__module__", "")
    return module == "prase" or (module or "").startswith("prase.")


def _code(value: Any) -> dict[str, str] | None:
    """The import path of a component and, outside PRASE, a digest of its module's source."""
    target = value if inspect.isclass(value) or inspect.isroutine(value) else type(value)
    code = {"code": f"{target.__module__}:{target.__qualname__}"}
    if _is_prase(target):
        return code  # environment_identity covers the package's code
    try:
        source = inspect.getsourcefile(target)
        return None if source is None else {**code, "source": file_digest(source)}
    except (OSError, TypeError):
        return None


def component_identity(value: Any) -> Any | None:
    """What decides the results of a step, model, or function; None when a resumed run must not reuse them.

    A component defines ``checkpoint_key()``, which returns its settings as
    JSON values with a digest of every file it reads. A function of PRASE
    without captured state needs no key. Anything else always runs again.
    """
    hook = getattr(value, "checkpoint_key", None)
    if not callable(hook):
        stateless = _is_prase(value) and inspect.isfunction(value) and value.__closure__ is None
        return _code(value) if stateless else None
    code = _code(value)
    if code is None:
        return None
    try:
        key = hook()
        return None if key is None else {**code, "key": plain_identity(key)}
    except (_Unknown, OSError):
        return None


def model_identity(model: object) -> Any | None:
    """What decides a model's training, or None when unknown.

    A PRASE model is described by its settings and a digest of every file
    named in its ``path_options``. Another model needs ``checkpoint_key()``.
    Both include the PyTorch settings of :func:`torch_runtime`.
    """
    if not _is_prase(model) or callable(getattr(model, "checkpoint_key", None)):
        identity = component_identity(model)
        return None if identity is None else {**identity, "torch": torch_runtime()}
    try:
        settings = {name: plain_identity(item) for name, item in sorted(getattr(model, "__dict__", {}).items())
                    if not name.startswith("_")}
        files = {name: file_digest(getattr(model, name)) for name in getattr(type(model), "path_options", ())
                 if getattr(model, name, None)}
    except (_Unknown, OSError):
        return None
    return {**_code(model), "settings": settings, "files": files, "torch": torch_runtime()}


# Training checkpoints ------------------------------------------------------

@dataclass(frozen=True)
class _TrainingSettings:
    directory: Path
    seconds: float
    written: list[Path] = field(default_factory=list)  # the checkpoint files saved in this block


_TRAINING: ContextVar[_TrainingSettings | None] = ContextVar("prase_training_checkpoints", default=None)


@contextmanager
def training_checkpoints(directory: str | Path, *, minutes: float = 10.0) -> Iterator[list[Path]]:
    """Save the training state of models fitted in this block, and resume from a saved state.

    A model saves its state in ``directory`` every ``minutes`` of training,
    or after every epoch when ``minutes`` is 0, and once more when it
    finishes. A later fit with the same model settings, files, code, training
    data, and seed continues from the saved epoch, or returns the finished
    fit at once, with the same vectors as an uninterrupted fit. The block
    yields the list of files it saved; remove them once the result is safe::

        with training_checkpoints("checkpoints", minutes=5) as files:
            embeddings = create_model("bootea", device="cuda").fit(data, seed=0)
        embeddings.save("bootea.npz")
        for path in files:
            path.unlink()
    """
    _check_minutes("minutes", minutes)
    settings = _TrainingSettings(Path(directory), 60.0 * minutes)
    token = _TRAINING.set(settings)
    try:
        yield settings.written
    finally:
        _TRAINING.reset(token)


_HEADER = b"PRASE-TRAINING-CHECKPOINT"


class TrainingCheckpoint:
    """The saved training state of one fit, keyed by its model identity, data, and seed.

    The file starts with a SHA-256 digest of the serialized state, so that a
    truncated or altered file is detected before any part of it is read.
    """

    def __init__(self, torch, settings: _TrainingSettings, key: str, device: str) -> None:
        self.torch, self.key, self.settings = torch, key, settings
        self.path = settings.directory / f"{key}.pt"
        self.saved_at = perf_counter()
        self.payload = self._load(device)

    @classmethod
    def active(cls, torch, model: object, data: TrainingData, seed: int, device: str) -> TrainingCheckpoint | None:
        """The checkpoint of this fit when :func:`training_checkpoints` is active, else None."""
        settings = _TRAINING.get()
        if settings is None:
            return None
        key = training_key(model, data, seed)
        if key is None:
            _LOGGER.info("Training %s without checkpoints: its settings or code cannot be identified",
                         getattr(model, "name", type(model).__name__))
            return None
        return cls(torch, settings, key, device)

    def _load(self, device: str) -> dict | None:
        if not self.path.is_file():
            return None
        try:
            payload = decode_training_state(self.torch, self.path.read_bytes(), device)
        except Exception as error:  # a damaged or unreadable file: train from the start
            self.discard(f"cannot be used ({type(error).__name__}: {error})")
            return None
        if not isinstance(payload, dict) or payload.get("key") != self.key:
            self.discard("belongs to another fit")
            return None
        return payload

    def discard(self, reason: str) -> None:
        """Forget the saved state, so that training starts from the first epoch."""
        _LOGGER.warning("Training from the first epoch: the checkpoint %s %s", self.path, reason)
        self.payload = None

    def due(self) -> bool:
        return perf_counter() - self.saved_at >= self.settings.seconds

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_bytes(encode_training_state(self.torch, {"key": self.key, **payload}))
        os.replace(temporary, self.path)
        if self.path not in self.settings.written:
            self.settings.written.append(self.path)
        self.saved_at = perf_counter()
        _LOGGER.debug("Saved the training state after epoch %d to %s", payload.get("epoch", 0), self.path)


def encode_training_state(torch, payload: dict[str, Any]) -> bytes:
    """A training checkpoint file: a header with the SHA-256 digest of the serialized payload, then the payload."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    body = buffer.getvalue()
    return b" ".join((_HEADER, str(_FORMAT).encode(), hashlib.sha256(body).hexdigest().encode())) + b"\n" + body


def decode_training_state(torch, data: bytes, device: str) -> dict:
    """The payload of a training checkpoint file; an error when the file is damaged or of another format."""
    header, _, body = data.partition(b"\n")
    parts = header.split(b" ")
    if len(parts) != 3 or parts[0] != _HEADER or parts[1] != str(_FORMAT).encode():
        raise ValueError("the file is not a training checkpoint of this PRASE version")
    if parts[2] != hashlib.sha256(body).hexdigest().encode():
        raise ValueError("the file is damaged or incomplete")
    return torch.load(io.BytesIO(body), map_location=device, weights_only=True)


def training_key(model: object, data: TrainingData, seed: int) -> str | None:
    """A file name that changes with the model identity, the environment, the data, or the seed; None if unknown."""
    identity = model_identity(model)
    if identity is None:
        return None
    digest = hashlib.sha256()
    digest.update(json.dumps([environment_identity(), identity], sort_keys=True).encode("utf-8"))
    digest.update(json.dumps([seed, [[graph.name, graph.fingerprint] for graph in (data.left, data.right)]])
                  .encode("utf-8"))
    for array in (data.seeds, *data.candidates, data.rejected):
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
        digest.update(b"|")
    name = getattr(model, "name", type(model).__name__)
    return f"{name}-{digest.hexdigest()[:24]}"


# Step checkpoints ----------------------------------------------------------

class StepCheckpoints:
    """Finished step results of one pipeline run in ``directory/steps``, and training state in ``directory/training``.

    The directory holds a marker file, so that a run only ever removes
    checkpoint folders that PRASE created.
    """

    def __init__(self, directory: Path, policy: CheckpointPolicy) -> None:
        self.directory, self.policy = directory, policy
        self.steps = directory / "steps"
        self.training = directory / "training"

    @classmethod
    def open(cls, policy: CheckpointPolicy, output: str | Path | None, resume: bool) -> StepCheckpoints | None:
        """Check the checkpoint directory of a run without changing anything, and return its store."""
        directory = policy.directory if policy.directory is not None else (
            None if output is None else Path(output) / "checkpoint")
        if directory is None:
            if resume:
                raise ConfigError("resuming a run needs an output directory or a checkpoint directory")
            return None
        if output is not None:
            folder, target = Path(output).resolve(), directory.resolve()
            if target == folder or folder.is_relative_to(target):
                raise ConfigError(f"the checkpoint directory {directory} must not contain the output directory")
            if target.is_relative_to(folder) and _OUTPUT_NAMES.fullmatch(target.relative_to(folder).parts[0]):
                raise ConfigError(f"the checkpoint directory {directory} would overlap the run's own results")
        if directory.exists():
            if not directory.is_dir():
                raise DataError(f"{directory} is a file, not a checkpoint directory")
            if any(directory.iterdir()):
                if not (directory / MARKER).is_file():
                    raise DataError(f"{directory} holds files that PRASE did not write; choose a new or empty "
                                    "checkpoint directory")
                if not resume:
                    raise DataError(f"{directory} holds the checkpoints of an earlier run: resume that run, "
                                    "or choose another checkpoint directory")
        return cls(directory, policy)

    def has_steps(self) -> bool:
        return self.steps.is_dir() and any((folder / "step.json").is_file() for folder in self.steps.iterdir())

    def claim(self) -> None:
        """Create the directory and its marker."""
        self.directory.mkdir(parents=True, exist_ok=True)
        marker = self.directory / MARKER
        if not marker.is_file():
            marker.write_text(json.dumps({"format": _FORMAT, "written_by": "prase"}) + "\n", encoding="utf-8")

    def folder(self, index: int, name: str) -> Path:
        return self.steps / f"{index:02d}-{name}"

    def load(self, index: int, name: str, key: str) -> dict | None:
        """The saved arrays and record of a step whose identity matches ``key``.

        None when there is no such checkpoint or its record or arrays do not
        match the digests saved with them.
        """
        folder = self.folder(index, name)
        try:
            meta = json.loads((folder / "step.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or meta.get("format") != _FORMAT or meta.get("key") != key:
                return None
            if meta.pop("record", None) != chain_key("record", meta):
                _LOGGER.warning("Step %d (%s) runs again: its checkpoint record was altered", index, name)
                return None
            body = (folder / "state.npz").read_bytes()
            if hashlib.sha256(body).hexdigest() != meta["arrays"]:
                _LOGGER.warning("Step %d (%s) runs again: its checkpoint arrays are damaged", index, name)
                return None
            with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
                values = {name: arrays[name] for name in arrays.files}
        except Exception:  # a missing or unreadable checkpoint: run the step again
            return None
        return {"meta": meta, "arrays": values}

    def save(self, index: int, name: str, key: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        folder = self.folder(index, name)
        folder.mkdir(parents=True, exist_ok=True)
        # The record goes last, so that a step folder with a record always holds that record's arrays.
        (folder / "step.json").unlink(missing_ok=True)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        body = buffer.getvalue()
        temporary = folder / "state.npz.tmp"
        temporary.write_bytes(body)
        os.replace(temporary, folder / "state.npz")
        record = json.loads(json.dumps({"format": _FORMAT, "key": key, "arrays": hashlib.sha256(body).hexdigest(),
                                        **meta}, default=_json_default))
        record["record"] = chain_key("record", record)
        temporary = folder / "step.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, folder / "step.json")

    def finish(self) -> None:
        """Remove the checkpoints PRASE wrote, unless the policy keeps them. Other files stay."""
        if not self.policy.keep:
            remove_checkpoints(self.directory)


def remove_checkpoints(directory: str | Path) -> None:
    """Remove the checkpoints that PRASE wrote in a checkpoint directory; other files, and other folders, stay."""
    directory = Path(directory)
    if not (directory / MARKER).is_file():
        return
    steps, training = directory / "steps", directory / "training"
    if steps.is_dir():
        for folder in steps.iterdir():
            if folder.is_dir():
                for name in _STEP_FILES:
                    (folder / name).unlink(missing_ok=True)
                _remove_if_empty(folder)
        _remove_if_empty(steps)
    if training.is_dir():
        for path in training.iterdir():
            if path.is_file() and path.name.endswith((".pt", ".pt.tmp")):
                path.unlink()
        _remove_if_empty(training)
    (directory / MARKER).unlink(missing_ok=True)
    _remove_if_empty(directory)


def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass  # it holds other files, which stay


def chain_key(label: str, *parts: Any) -> str:
    """A digest of a label and JSON values, for checkpoint and run keys."""
    digest = hashlib.sha256(label.encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(json.dumps(part, sort_keys=True, default=_json_default).encode("utf-8"))
    return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    """A digest of arrays' types, shapes, and values."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def predicate_rows(predicates) -> list[dict[str, Any]]:
    return [asdict(match) for match in predicates]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
