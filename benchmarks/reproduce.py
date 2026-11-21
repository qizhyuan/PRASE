"""Run and inspect the PRASE benchmark suite.

    python benchmarks/reproduce.py list
    python benchmarks/reproduce.py check --data path/to/prase-benchmarks
    python benchmarks/reproduce.py run --method paris --dataset D-W-15K-V2
    python benchmarks/reproduce.py run --method se --model bootea --dataset D-W-15K-V2
    python benchmarks/reproduce.py run --method prase --model bootea --dataset D-W-15K-V2
    python benchmarks/reproduce.py run --all --output runs/full --word-vectors data/wiki-news-300d-1M.vec
    python benchmarks/reproduce.py run --all --output runs/full --word-vectors data/wiki-news-300d-1M.vec --resume
    python benchmarks/reproduce.py report

``ijcai2021.toml`` lists the recipes of the IJCAI 2021 paper. A recipe names a
stable method config in ``prase/`` or ``supervised/`` and the settings it
changes, or defines a baseline method in place. Runs load the pinned datasets
through ``prase.load_benchmark``. Use ``--data`` to point to a local folder
containing the named datasets in the OpenEA layout. A run writes to
``<output>/<experiment>/<dataset>/``, with ``comparison.json`` holding the
measured scores and run settings.

A run saves checkpoints in its folder while it trains. A saved run is
current when its recipe settings, evaluation protocol, PRASE build, and
NumPy and PyTorch versions are the present ones. ``--resume`` continues an
interrupted suite: it skips the current runs of the requested seed and
device, continues interrupted runs from their checkpoints, and moves every
other run in the way to ``<output>/history/<time>/`` before running it
again. ``list`` and ``report`` show runs that are not current as historical.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np

from prase import Alignment, Context, DataError, Dataset, Pipeline, PraseError, RunResult, State
from prase import available_models, load_benchmark, ranking_scores
from prase.checkpoints import (
    MARKER, CheckpointPolicy, array_digest, chain_key, environment_identity, remove_checkpoints,
)
from prase.config import build_method
from prase.datasets import BENCHMARK_NAMES, benchmark_folds, load_local_benchmark, verify_benchmark, verify_word_vectors
from prase.files import atomic_write_text
from prase.progress import Progress

SUITE = Path(__file__).resolve().with_name("ijcai2021.toml")
_LOGGER = logging.getLogger("prase")
EVALUATIONS = ("fold-ranking", "paris-seeded-ranking")
_SUITE_KEYS = {"datasets", "experiments"}
_RECIPE_KEYS = {"description", "config", "method", "dataset_options", "datasets", "rounds", "evaluation"}
_CONFIG_KEYS = {"method", "dataset_options"}


def load_suite(path: Path) -> dict:
    """Read a suite file and resolve every recipe into its method, dataset options, and runs.

    A recipe's ``config`` names a stable method config next to the suite. Its
    ``method`` and ``dataset_options`` are merged over the config's, so a recipe
    changes only the settings it lists.
    """
    suite = _read_toml(path)
    _check_keys(suite, _SUITE_KEYS, path.name)
    defaults = _datasets(suite.get("datasets", list(BENCHMARK_NAMES)), path.name)
    experiments = {}
    for name, recipe in suite.get("experiments", {}).items():
        where = f"{path.name}: experiment {name}"
        _check_keys(recipe, _RECIPE_KEYS, where)
        base = {}
        if "config" in recipe:
            config = path.parent / recipe["config"]
            if not config.is_file():
                raise SystemExit(f"{where}: config {recipe['config']} not found")
            base = _read_toml(config)
            _check_keys(base, _CONFIG_KEYS, f"{where}, config {recipe['config']}")
        method = _merged(base.get("method", {}), recipe.get("method", {}))
        if not isinstance(method.get("name"), str):
            raise SystemExit(f"{where}: needs a config or a [method] name")
        options = base.get("dataset_options", {})
        extra = recipe.get("dataset_options", {})
        datasets = _datasets(recipe.get("datasets", defaults), where)
        resolved = {"description": recipe.get("description", ""), "datasets": datasets, "method": method}
        resolved["dataset_options"] = {dataset: _merged(options.get(dataset, {}), extra.get(dataset, {}))
                                       for dataset in dict.fromkeys([*options, *extra])}
        _datasets(list(resolved["dataset_options"]), f"{where}, dataset_options")
        if "rounds" in recipe:
            rounds = recipe["rounds"]
            if method["name"] != "prase" or not all(
                dataset in datasets and isinstance(counts, list)
                and all(type(count) is int and count > 0 for count in counts)
                for dataset, counts in rounds.items()
            ):
                raise SystemExit(f"{where}: rounds lists positive PRASE round counts for the recipe's datasets")
            resolved["rounds"] = rounds
        if "evaluation" in recipe:
            evaluation = recipe["evaluation"]
            needs = {"fold-ranking": "embedding-only", "paris-seeded-ranking": "prase"}.get(evaluation)
            if needs is None:
                raise SystemExit(f"{where}: evaluation must be one of {', '.join(EVALUATIONS)}")
            if method["name"] != needs:
                raise SystemExit(f"{where}: {evaluation} evaluates an {needs} method")
            resolved["evaluation"] = evaluation
        experiments[name] = resolved
    return {"experiments": experiments}


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{path.name}: {error}") from None


def _check_keys(table: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise SystemExit(f"{where}: unknown settings {', '.join(unknown)}")


def _datasets(names, where: str) -> list[str]:
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise SystemExit(f"{where}: datasets must be a list of benchmark names")
    unknown = [name for name in names if name not in BENCHMARK_NAMES]
    if unknown:
        raise SystemExit(f"{where}: unknown datasets {', '.join(unknown)}")
    return list(names)


def _merged(base: dict, changes: dict) -> dict:
    """``base`` with the settings of ``changes``; nested tables such as ``options`` merge key by key."""
    merged = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **copy.deepcopy(value)}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


CATALOG = load_suite(SUITE)
RUNS = Path("runs")
EXPERIMENTS = tuple(CATALOG["experiments"])
METHODS = {
    "paris": "PARIS",
    "se": "SE (Embedding Only)",
    "prase": "PRASE",
}
REPORT_GROUPS = {
    **METHODS,
    "paris-seeded": "PARIS-Seeded Retrieval",
    "name-matching": "Name Matching",
}
METRIC_TITLES = {
    "precision": "Precision", "recall": "Recall", "f1": "F1", "hits@1": "Hits@1",
    "hits@5": "Hits@5", "hits@10": "Hits@10", "mr": "MR", "mrr": "MRR",
}


@dataclass(frozen=True)
class RunCase:
    experiment: str
    dataset: str
    rounds: int | None = None
    fold: int | None = None

    @property
    def workflow(self) -> str:
        recipe = CATALOG["experiments"][self.experiment]
        if recipe.get("evaluation") == "paris-seeded-ranking":
            return "paris-seeded"
        method = _method_spec(self.experiment)["method"]["name"]
        if method == "embedding-only":
            return "se"
        if method in ("str-match", "emb-match"):
            return "name-matching"
        return method

    @property
    def directory(self) -> Path:
        label = self.experiment
        if self.rounds is not None:
            label += f"-rounds-{self.rounds}"
        if self.fold is not None:
            label += f"-fold-{self.fold}"
        return Path(label) / self.dataset


def planned_runs(experiments: Iterable[str] = EXPERIMENTS, *, datasets: Sequence[str] | None = None,
                 rounds: int | None = None, fold: int | None = None,
                 include_rounds: bool = True) -> list[RunCase]:
    """Enumerate the base runs and any configured round sweeps.

    A round count equal to the recipe's own names the base run, in the base run's folder.
    """
    cases = []
    for experiment in experiments:
        spec = CATALOG["experiments"][experiment]
        for dataset in datasets or spec["datasets"]:
            cases.append(RunCase(experiment, dataset, normalized_rounds(experiment, rounds), fold))
            if include_rounds and rounds is None and fold is None:
                cases.extend(RunCase(experiment, dataset, normalized_rounds(experiment, count))
                             for count in spec.get("rounds", {}).get(dataset, ()))
    return list(dict.fromkeys(cases))


def _method_spec(experiment: str) -> dict:
    return CATALOG["experiments"][experiment]


def word_vectors(path: Path | None) -> Path:
    """The FastText file, checked against its pinned SHA-256 every time a run uses it."""
    if path is None:
        raise SystemExit("this experiment needs --word-vectors. prase download --word-vectors fetches them.")
    try:
        return verify_word_vectors(path)
    except DataError as error:
        raise SystemExit(str(error)) from None


def recipe_method(experiment: str, dataset: str, rounds: int | None = None) -> dict:
    """A recipe's method settings for one dataset, before the word vectors and the device are filled in."""
    spec = _method_spec(experiment)
    method = copy.deepcopy(spec["method"])
    extra = copy.deepcopy(spec.get("dataset_options", {}).get(dataset, {}))
    if "model" in method:
        settings = {**method.get("options", {}), **extra}
        if settings:
            method["options"] = settings
    else:
        method.update(extra)
    if rounds is not None:
        if method["name"] != "prase":
            raise SystemExit("--rounds applies to PRASE experiments")
        method["rounds"] = rounds
    return method


def method_table(experiment: str, dataset: str, *, vectors: Path | None, rounds: int | None,
                 device: str | None) -> dict:
    method = recipe_method(experiment, dataset, rounds)
    if "model" in method:
        if device is not None:
            method.setdefault("options", {})["device"] = device
        settings = method.get("options", {})
    else:
        settings = method
    for key, value in settings.items():
        if value == "@word_vectors":
            settings[key] = str(word_vectors(vectors))
    return method


def recipe_digest(experiment: str, dataset: str, rounds: int | None = None) -> str:
    """A fingerprint of the recipe settings that a run used."""
    text = json.dumps(recipe_method(experiment, dataset, normalized_rounds(experiment, rounds)), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalized_rounds(experiment: str, rounds: int | None) -> int | None:
    """None when ``rounds`` is the recipe's own round count, so that both name the same run."""
    method = _method_spec(experiment)["method"]
    if method["name"] != "prase" or rounds is None:
        return rounds
    return None if rounds == method.get("rounds", 1) else rounds


def normalized_case(case: RunCase) -> RunCase:
    return replace(case, rounds=normalized_rounds(case.experiment, case.rounds))


def evaluation_protocol(experiment: str) -> str:
    """How a recipe is scored: ``alignment`` (the final alignment), ``fold-ranking``, or ``paris-seeded-ranking``."""
    return _method_spec(experiment).get("evaluation", "alignment")


def effective_device(method: dict) -> str | None:
    """The device that an embedding method trains on, or None for a method without a model."""
    return method.get("options", {}).get("device", "cpu") if "model" in method else None


def planned_device(experiment: str, dataset: str, device: str | None) -> str | None:
    """The device that a run of the recipe trains on with ``--device``."""
    method = recipe_method(experiment, dataset)
    if "model" not in method:
        return None
    return device or method.get("options", {}).get("device", "cpu")


def run_identity(experiment: str, dataset: str, rounds: int | None, seed: int, fold: int | None,
                 device: str | None) -> dict:
    """What makes two runs interchangeable: the recipe, its protocol, rounds, fold, seed, device, and build."""
    rounds = normalized_rounds(experiment, rounds)
    return {"experiment": experiment, "dataset": dataset, "rounds": rounds, "fold": fold, "seed": seed,
            "device": device, "evaluation": evaluation_protocol(experiment),
            "recipe": recipe_digest(experiment, dataset, rounds), "engine": engine_identity()}


def run(experiment: str, dataset: str, data: Path | None, output: Path, *, seed: int, rounds: int | None,
        device: str | None, vectors: Path | None, fold: int | None = None, resume: bool = False,
        checkpoints: CheckpointPolicy | None = CheckpointPolicy()) -> dict:
    spec = CATALOG["experiments"][experiment]
    evaluation = spec.get("evaluation")
    if fold is not None and evaluation != "fold-ranking":
        raise SystemExit("--fold applies to supervised embedding evaluation")
    if rounds is not None and evaluation:
        raise SystemExit("--rounds applies to end-to-end PRASE experiments")
    method = method_table(experiment, dataset, vectors=vectors, rounds=rounds, device=device)
    graphs = load_benchmark(dataset) if data is None else load_local_benchmark(dataset, data)
    case = RunCase(experiment, dataset, rounds, fold)
    target = output / case.directory
    record = {
        "experiment": experiment,
        "dataset": dataset,
        "rounds": rounds,
        "seed": seed,
        "method": method,
        "evaluation": evaluation_protocol(experiment),
        "device": effective_device(method),
        "recipe": recipe_digest(experiment, dataset, rounds),
        "engine": engine_identity(),
        "code": code_revision(),
    }
    if resume:
        _set_aside_finished_run(target, output)
    if evaluation == "fold-ranking":
        folds = supervised_runs(dataset, graphs, target, method, seed=seed, fold=fold, resume=resume,
                                checkpoints=checkpoints, experiment=experiment, output=output)
        record["folds"] = folds
        if fold is not None:
            record["fold"] = fold
        record["seconds"] = sum(row["seconds"] for row in folds)
        measured = {key: mean(row["measured"][key] for row in folds) for key in folds[0]["measured"]}
    else:
        pipeline = build_method(method, Path.cwd())
        if evaluation == "paris-seeded-ranking":
            paris, embed, _ = pipeline.steps
            train = replace(embed, options={}, propose=False, share_embeddings=True)
            pipeline = Pipeline([paris, train, UnalignedTop1(train.seed_threshold)], name=experiment)
        reference = graphs.reference()
        result = pipeline.run(graphs.left, graphs.right, reference=reference, seed=seed, output=target,
                              checkpoints=_kept(checkpoints), resume=resume)
        record["seconds"] = result.seconds
        measured = (unaligned_hits(result, reference, train.seed_threshold)
                    if evaluation == "paris-seeded-ranking" else result.scores.as_dict())
    record["measured"] = measured
    target.mkdir(parents=True, exist_ok=True)
    write_record(target / "comparison.json", record)
    if evaluation != "fold-ranking":
        _release_checkpoints(target, checkpoints)
    return record


def _kept(checkpoints: CheckpointPolicy | None) -> CheckpointPolicy | None:
    """The policy for a pipeline whose checkpoints stay until its benchmark record is written."""
    return None if checkpoints is None else replace(checkpoints, keep=True)


def _release_checkpoints(folder: Path, checkpoints: CheckpointPolicy | None) -> None:
    """Remove a run's checkpoints once the record that needs its result is written, unless they are kept."""
    if checkpoints is not None and not checkpoints.keep:
        remove_checkpoints(folder / "checkpoint")


def engine_identity() -> dict:
    """The installed PRASE build and library versions (:func:`prase.checkpoints.environment_identity`).

    The recipes' checkout, in :func:`code_revision`, can differ from the
    installed package, so the record names both.
    """
    return dict(environment_identity())


def code_revision() -> dict:
    """The commit of the recipes' checkout, and whether the checkout has changes."""
    revision = {"commit": None, "modified": None}
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return revision
    return {**revision, "commit": commit.stdout.strip(), "modified": bool(status.stdout.strip())}


def write_record(path: Path, record: dict) -> None:
    """Write a record whole, so that list and report never read a partial one."""
    atomic_write_text(path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")


@dataclass
class UnalignedTop1:
    """Retrieve one neighbor per unaligned left entity."""

    seed_threshold: float
    name = "unaligned-top1"

    def run(self, state: State, context: Context) -> State:
        predictions = state.embeddings.nearest(*unaligned_candidates(state, self.seed_threshold))
        return replace(state, proposals=predictions)

    def checkpoint_key(self) -> dict:
        return {"seed_threshold": self.seed_threshold}


def unaligned_candidates(state: State, seed_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Entities left unaligned at the embedding step's seed threshold."""
    anchors = state.alignment.above(seed_threshold).one_to_one()
    return replace(state, alignment=anchors).unmatched()


def unaligned_hits(result: RunResult, reference: np.ndarray, seed_threshold: float) -> dict[str, float]:
    """Hits@1 of the first embedding step on reference pairs with two unaligned endpoints."""
    left, right = unaligned_candidates(result.state, seed_threshold)
    eligible = reference[np.isin(reference[:, 0], left) & np.isin(reference[:, 1], right)]
    gold = set(map(tuple, eligible.tolist()))
    correct = len(result.state.proposals.pairs() & gold)
    return {"hits@1": correct / len(gold) if gold else 0.0}


def supervised_runs(dataset: str, graphs: Dataset, target: Path, method: dict, *, seed: int, fold: int | None,
                    experiment: str, output: Path, resume: bool = False,
                    checkpoints: CheckpointPolicy | None = CheckpointPolicy()) -> list[dict]:
    """Train and rank every fold.

    With ``resume``, a fold that finished with the same run identity is kept,
    and a finished fold of another identity moves to the history folder.
    """
    splits = dict(zip(benchmark_folds(dataset), graphs.splits, strict=True))
    if not splits or (fold is not None and fold not in splits):
        available = ", ".join(map(str, sorted(splits))) or "none"
        raise SystemExit(f"{dataset}: available folds are {available}")
    rows = []
    for index in sorted(splits) if fold is None else [fold]:
        split = splits[index]
        seed_pairs = graphs.reference(split.train)
        seeds = Alignment(seed_pairs[:, 0], seed_pairs[:, 1])
        pipeline = build_method(method, Path.cwd())
        # Rank the test split after fitting the embedding-only pipeline's training step.
        train = replace(pipeline.steps[0], options={}, propose=False, share_embeddings=True)
        pipeline = Pipeline([train], name=pipeline.name)
        reference = graphs.reference(split.test)
        folder = target / f"fold-{index}"
        identity = run_identity(experiment, dataset, None, seed, index, effective_device(method))
        key = chain_key("fold", identity, graphs.left.fingerprint, graphs.right.fingerprint, array_digest(seed_pairs),
                        array_digest(reference), seed)
        saved = _finished_fold(folder, key) if resume else None
        if saved is not None:
            _LOGGER.info("Fold %d of %s finished in an earlier run: Hits@1 %.3f", index, dataset,
                         saved["measured"]["hits@1"])
            rows.append(saved)
            continue
        if resume:
            _set_aside_finished_run(folder, output)
        result = pipeline.run(graphs.left, graphs.right, seeds=seeds, seed=seed, output=folder,
                              checkpoints=_kept(checkpoints), resume=resume)
        measured = ranking_scores(result.state.embeddings, reference)
        row = {"fold": index, "train_pairs": len(seeds), "test_pairs": len(reference),
               "seconds": result.seconds, "measured": measured}
        write_record(folder / "fold.json", {"key": key, "row": row})
        _release_checkpoints(folder, checkpoints)
        rows.append(row)
    return rows


def _finished_fold(folder: Path, key: str) -> dict | None:
    try:
        saved = json.loads((folder / "fold.json").read_text(encoding="utf-8"))
        return saved["row"] if saved.get("key") == key and isinstance(saved["row"]["measured"], dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


@functools.cache
def _history_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _set_aside(folder: Path, output: Path) -> Path:
    """Move a run folder of ``output`` to ``<output>/history/<time>/``, keeping its relative path."""
    output = output.resolve()
    base = output / "history" / _history_stamp() / folder.resolve().relative_to(output)
    destination, number = base, 1
    while destination.exists():  # the same run set aside twice in one session
        number += 1
        destination = base.with_name(f"{base.name}-{number}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(folder, destination)
    _LOGGER.info("Moved %s to %s", folder.as_posix(), destination.as_posix())
    return destination


def _set_aside_finished_run(folder: Path, output: Path) -> None:
    """A finished run without kept checkpoints has nothing to resume; keep it in the history folder."""
    try:
        finished = json.loads((folder / "run.json").read_text(encoding="utf-8")).get("finished")
    except (OSError, ValueError, AttributeError):
        return
    if finished and not (folder / "checkpoint" / MARKER).is_file():
        _set_aside(folder, output)


def _saved_runs(outputs: Sequence[Path]) -> tuple[dict[RunCase, dict], list[tuple[Path, dict, list[str]]]]:
    """Current runs by case, and historical runs with the reasons they are out of date."""
    results, historical = {}, []
    for output in outputs:
        current = {}
        for path in sorted(output.glob("*/*/comparison.json")):
            row = _read_record(path)
            if row is None:
                historical.append((path, {}, ["unreadable record"]))
                continue
            reasons = outdated(row)
            if reasons:
                historical.append((path, row, reasons))
                continue
            experiment = row["experiment"]
            case = RunCase(experiment, row["dataset"], _saved_rounds(experiment, row), row.get("fold"))
            # A run that names the recipe's own round count repeats the base run, which wins.
            if row.get("rounds") is None or case not in current:
                current[case] = row
        results.update(current)
        for path in sorted(output.glob("history/*/*/*/comparison.json")):
            row = _read_record(path)
            if row is None:
                historical.append((path, {}, ["unreadable record"]))
            else:
                historical.append((path, row, outdated(row) or ["replaced by a later run"]))
    return results, historical


def _read_record(path: Path) -> dict | None:
    """A benchmark record, or None when it cannot be read or lacks the parts that every record has."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(row, dict) and isinstance(row.get("experiment"), str) and isinstance(row.get("dataset"), str)
            and isinstance(row.get("method", {}), dict) and isinstance(row.get("measured"), dict)):
        return None
    measured = row["measured"].values()
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in measured):
        return None
    return row


def outdated(row: dict, seed: int | None = None, device: str | None = None) -> list[str]:
    """Why a saved run is not current.

    A run is current when its recipe settings, evaluation protocol, PRASE
    build, and library versions are the present ones, and, when given, its
    seed and training device are ``seed`` and ``device``.
    """
    experiment = row.get("experiment")
    if experiment not in EXPERIMENTS:
        return ["recipe removed from the suite"]
    engine = row.get("engine", row.get("code", {}))
    engine = engine if isinstance(engine, dict) else {}
    current = engine_identity()
    if engine.get("prase") != current["prase"]:
        reasons = [f"PRASE {engine.get('prase') or 'version unknown'}"]
    elif any(name not in engine for name in current):
        reasons = ["PRASE build unknown"]
    elif (engine["source"], engine["native"]) != (current["source"], current["native"]):
        reasons = ["another PRASE build"]
    else:
        reasons = [f"{name} {engine[name]}" for name in ("numpy", "torch") if engine[name] != current[name]]
    if row.get("evaluation") != evaluation_protocol(experiment):
        reasons.append(f"evaluation {row.get('evaluation') or 'unknown'}")
    if seed is not None and row.get("seed") != seed:
        reasons.append(f"seed {row.get('seed')}")
    if device is not None and row.get("device") != device:
        reasons.append(f"device {row.get('device') or 'unknown'}")
    # A recipe runs on its configured datasets by default and on any other benchmark on request.
    if row.get("dataset") not in BENCHMARK_NAMES:
        return [*reasons, "unknown dataset"]
    try:
        expected = recipe_method(experiment, row["dataset"], _saved_rounds(experiment, row))
    except SystemExit:
        return [*reasons, "rounds no longer apply"]
    # The device changes the speed, not the method; a run's device is compared on its own.
    expected, saved = (_without_device(method) for method in (expected, row.get("method", {})))
    if expected["name"] == "prase":
        # A PRASE method without a round count runs one round.
        expected.setdefault("rounds", 1)
        saved.setdefault("rounds", 1)
    changes = _differences(expected, saved, "method")
    if changes:
        reasons.append("changed " + ", ".join(changes[:3]) + (" and more" if len(changes) > 3 else ""))
    return reasons


def _without_device(method: dict) -> dict:
    method = dict(method)
    if isinstance(method.get("options"), dict):
        method["options"] = {key: value for key, value in method["options"].items() if key != "device"}
        if not method["options"]:
            del method["options"]
    return method


def _differences(expected, saved, where: str) -> list[str]:
    """Paths at which saved settings differ from a recipe. ``@word_vectors`` accepts any file path."""
    if isinstance(expected, dict) and isinstance(saved, dict):
        changes = []
        for key in sorted(set(expected) | set(saved)):
            if isinstance(expected.get(key), dict) or isinstance(saved.get(key), dict):
                # Name the settings inside a table that one side leaves out.
                changes.extend(_differences(expected.get(key, {}), saved.get(key, {}), f"{where}.{key}"))
            elif key not in expected or key not in saved:
                changes.append(f"{where}.{key}")
            else:
                changes.extend(_differences(expected[key], saved[key], f"{where}.{key}"))
        return changes
    if expected == "@word_vectors":
        return [] if isinstance(saved, str) else [where]
    return [] if expected == saved else [where]


def _saved_rounds(experiment: str, row: dict) -> int | None:
    """The PRASE rounds that a saved run used, or None when they are the recipe's own."""
    method = _method_spec(experiment)["method"]
    if method["name"] != "prase":
        return row.get("rounds")
    rounds = row.get("rounds")
    if rounds is None:
        rounds = row.get("method", {}).get("rounds", 1)
    return normalized_rounds(experiment, rounds)


def _case_state(case: RunCase, outputs: Sequence[Path], seed: int | None = None,
                device: str | None = None) -> tuple[str, dict | None]:
    """The status of a planned run in the last root whose folder for it holds anything, and its record.

    A completed record counts when it is current and belongs to this run,
    after the recipe's own round count is treated as no round count.
    """
    for output in reversed(outputs):
        target = output / case.directory
        if not target.is_dir() or not any(target.iterdir()):
            continue
        comparison = target / "comparison.json"
        if comparison.is_file():
            row = _read_record(comparison)
            if row is None:
                return "Historical", None
            if outdated(row, seed, device):
                return "Historical", row
            saved = RunCase(row["experiment"], row["dataset"], _saved_rounds(row["experiment"], row), row.get("fold"))
            if saved != normalized_case(case):
                return "Historical", row
            return "Completed", row
        for record in (target / "run.json", *target.glob("fold-*/run.json")):
            try:
                if record.is_file() and json.loads(record.read_text(encoding="utf-8")).get("failed"):
                    return "Failed", None
            except (OSError, ValueError, AttributeError):
                continue
        return "Incomplete", None
    return "Pending", None


def _status(case: RunCase, outputs: Sequence[Path], seed: int | None = None, device: str | None = None) -> str:
    return _case_state(case, outputs, seed, device)[0]


def coverage(outputs: Sequence[Path]) -> dict[str, dict[str, int]]:
    """Count planned runs by workflow and status across the selected output roots."""
    counts = {workflow: dict.fromkeys(("Completed", "Historical", "Failed", "Incomplete", "Pending"), 0)
              for workflow in REPORT_GROUPS}
    for case in planned_runs():
        counts[case.workflow][_status(case, outputs)] += 1
    return counts


def _table(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    columns = [i for i, header in enumerate(headers)
               if header not in ("Rounds", "Fold") or any(row[i] for row in rows)]
    print(f"## {title}\n")
    print("| " + " | ".join(headers[i] for i in columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for cells in rows:
        print("| " + " | ".join(cells[i] for i in columns) + " |")
    print()


def _case_cells(case: RunCase) -> tuple[str, str, str, str]:
    rounds = case.rounds
    if rounds is None and case.workflow == "prase":
        rounds = _method_spec(case.experiment)["method"].get("rounds", 1)
    fold = "" if case.fold is None else str(case.fold)
    return case.experiment, case.dataset, "" if rounds is None else str(rounds), fold


def list_runs(outputs: Sequence[Path], *, experiments: Iterable[str] = EXPERIMENTS,
              datasets: Sequence[str] | None = None, method: str | None = None) -> None:
    cases = planned_runs(experiments)
    for name, title in REPORT_GROUPS.items():
        if method is not None and method != name:
            continue
        rows = []
        for case in cases:
            if case.workflow != name or (datasets is not None and case.dataset not in datasets):
                continue
            status, row = _case_state(case, outputs)
            measured = (row or {}).get("measured", {}) if status == "Completed" else {}
            key = "f1" if "f1" in measured else "hits@1"
            score = f"{METRIC_TITLES[key]} {measured[key]:.3f}" if key in measured else ""
            rows.append((*_case_cells(case)[:3], status, score))
        if rows:
            _table(title, ("Experiment", "Dataset", "Rounds", "Status", "Measured"), rows)


def report(outputs: Sequence[Path]) -> None:
    results, historical = _saved_runs(outputs)
    cases = sorted(results, key=lambda case: (EXPERIMENTS.index(case.experiment), case.dataset,
                                              case.rounds or 0, case.fold or 0))
    for name, title in REPORT_GROUPS.items():
        group = [case for case in cases if case.workflow == name]
        if not group:
            continue
        metrics = [key for key in METRIC_TITLES if any(key in results[case]["measured"] for case in group)]
        headers = ("Experiment", "Dataset", "Rounds", "Fold", "Seed", *(METRIC_TITLES[key] for key in metrics))
        rows = []
        for case in group:
            row = results[case]
            measured = row["measured"]
            scores = [f"{measured[key]:.3f}" if key in measured else "" for key in metrics]
            rows.append((*_case_cells(case), str(row.get("seed", "")), *scores))
        _table(title, headers, rows)
    if historical:
        rows = []
        for path, row, reasons in historical:
            measured = row.get("measured", {})
            key = "f1" if "f1" in measured else "hits@1"
            score = f"{METRIC_TITLES[key]} {measured[key]:.3f}" if key in measured else ""
            version = row.get("engine", row.get("code", {})).get("prase") or ""
            rows.append((str(row.get("experiment", "")), str(row.get("dataset", "")), score, version,
                         "; ".join(reasons), path.parent.as_posix()))
        _table("Historical Runs", ("Experiment", "Dataset", "Measured", "PRASE", "Why Historical", "Folder"), rows)
    if not cases and not historical:
        print("No completed benchmark runs found.")


def _needs_vectors(case: RunCase) -> bool:
    spec = _method_spec(case.experiment)
    method = spec["method"]
    options = spec.get("dataset_options", {}).get(case.dataset, {})
    return "@word_vectors" in (*method.values(), *method.get("options", {}).values(), *options.values())


def _run_cases(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[RunCase]:
    if args.entire_suite and (args.dataset or args.rounds is not None or args.fold is not None):
        parser.error("--all runs the configured suite without dataset, round, or fold overrides")
    if not args.entire_suite and not (args.dataset or args.all_datasets):
        parser.error("select --dataset or --all-datasets")
    if args.method in ("se", "prase"):
        if args.model is None:
            parser.error(f"--method {args.method} requires --model")
        prefix = "supervised" if args.method == "se" else "prase"
        experiments = (f"{prefix}-{args.model}",)
        if experiments[0] not in EXPERIMENTS:
            parser.error(f"no {args.method.upper()} benchmark recipe for {args.model}")
    else:
        if args.model is not None:
            parser.error("--model applies to --method se and --method prase")
        experiments = ("paris",) if args.method == "paris" else args.experiment or EXPERIMENTS
    if args.include_rounds and (args.rounds is not None or args.fold is not None):
        parser.error("--include-rounds uses the configured sweeps without round or fold overrides")
    sweeps = args.entire_suite or args.include_rounds
    cases = planned_runs(experiments, datasets=args.dataset, rounds=args.rounds,
                         fold=args.fold, include_rounds=sweeps)
    if not cases:
        parser.error("no benchmark runs match the selection")
    for case in cases:
        target = args.output / case.directory
        if target.exists() and not target.is_dir():
            parser.error(f"{target} is a file, not a run folder")
        if not args.resume and target.is_dir() and any(target.iterdir()):
            parser.error(f"{target} already contains a run. Choose a fresh --output directory, or add --resume.")
    if any(_needs_vectors(case) for case in cases):
        word_vectors(args.word_vectors)
    return cases


def _resumable_cases(cases: list[RunCase], output: Path, seed: int, device: str | None) -> list[RunCase]:
    """Leave out current runs of ``seed`` on ``device``, and move other completed runs to the history folder."""
    remaining = []
    for case in cases:
        status = _status(case, [output], seed, planned_device(case.experiment, case.dataset, device))
        if status == "Completed":
            _LOGGER.info("Skipped %s: it finished with the current recipe, PRASE build, seed, and device",
                         case.directory.as_posix())
            continue
        if status == "Historical":
            _set_aside(output / case.directory, output)
        remaining.append(case)
    return remaining


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="show configured runs and their recorded status")
    list_parser.add_argument("--output", type=Path, action="append", help=f"inspect a run root (default {RUNS})")
    list_parser.add_argument("--method", choices=METHODS, help="show one workflow")
    list_parser.add_argument("--experiment", action="append", choices=EXPERIMENTS, metavar="RECIPE")
    list_parser.add_argument("--dataset", action="append", choices=BENCHMARK_NAMES)

    check_parser = commands.add_parser("check", help="verify local copies of the benchmark datasets")
    check_parser.add_argument("--data", type=Path, required=True,
                              help="directory containing the local benchmark copies")
    check_parser.add_argument("--word-vectors", type=Path, help="also check the word vector file")

    run_parser = commands.add_parser("run", help="run benchmarks and record measured results")
    recipes = run_parser.add_mutually_exclusive_group(required=True)
    recipes.add_argument("--method", choices=METHODS, help="PARIS, SE (embedding-only), or PRASE")
    recipes.add_argument("--experiment", action="append", choices=EXPERIMENTS, metavar="RECIPE",
                         help="select a recipe shown by list")
    recipes.add_argument("--all", dest="entire_suite", action="store_true", help="run the entire configured suite")
    run_parser.add_argument("--model", choices=available_models(), help="embedding model for SE or PRASE")
    selection = run_parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", action="append", choices=BENCHMARK_NAMES)
    selection.add_argument("--all-datasets", action="store_true", help="use each recipe's configured datasets")
    run_parser.add_argument("--data", type=Path, help="use verified local benchmark copies")
    run_parser.add_argument("--output", type=Path, default=RUNS, help=f"where to write runs (default {RUNS})")
    run_parser.add_argument("--seed", type=int, default=0, help="random seed of every run (default 0)")
    run_parser.add_argument("--rounds", type=int, help="override the number of PRASE rounds")
    run_parser.add_argument("--include-rounds", action="store_true",
                            help="include each recipe's configured round sweeps")
    run_parser.add_argument("--fold", type=int, help="select one SE fold (default all)")
    run_parser.add_argument("--device", help="embedding model device, such as cuda")
    run_parser.add_argument("--word-vectors", type=Path, help="FastText vectors, for MultiKE and EMB-Match")
    run_parser.add_argument("--verbose", action="store_true", help="log each embedding validation check")
    run_parser.add_argument("--resume", action="store_true",
                            help="skip finished runs, continue interrupted ones, and rerun outdated ones")
    run_parser.add_argument("--checkpoint-minutes", type=float, default=10.0, metavar="MINUTES",
                            help="minutes between training checkpoints (default 10; 0 saves after every epoch)")
    run_parser.add_argument("--keep-checkpoints", action="store_true",
                            help="keep each run's checkpoint folder after it finishes")
    run_parser.add_argument("--no-checkpoints", action="store_true", help="save no checkpoints")

    report_parser = commands.add_parser("report", help="tabulate measured results by workflow")
    report_parser.add_argument("output", type=Path, nargs="*", help=f"run roots (default {RUNS}; later roots win)")

    args = parser.parse_args(argv)
    if args.command == "check":
        for name in BENCHMARK_NAMES:
            print(f"{name}: {verify_benchmark(name, args.data)}")
        if args.word_vectors is not None:
            print(f"word vectors: {word_vectors(args.word_vectors)}")
    elif args.command == "list":
        list_runs(args.output or [RUNS], experiments=args.experiment or EXPERIMENTS,
                  datasets=args.dataset, method=args.method)
    elif args.command == "run":
        if args.no_checkpoints and args.resume:
            run_parser.error("--resume needs checkpoints")
        cases = _run_cases(args, run_parser)
        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s %(message)s")
        checkpoints = None if args.no_checkpoints else CheckpointPolicy(
            training_minutes=args.checkpoint_minutes, keep=args.keep_checkpoints)
        if args.resume:
            cases = _resumable_cases(cases, args.output, args.seed, args.device)
        progress = Progress("Benchmark suite", len(cases), unit="run")
        for number, case in enumerate(cases, 1):
            _LOGGER.info("Run %d of %d: %s", number, len(cases), case.directory.as_posix())
            row = run(case.experiment, case.dataset, args.data, args.output, seed=args.seed, rounds=case.rounds,
                      device=args.device, vectors=args.word_vectors, fold=case.fold, resume=args.resume,
                      checkpoints=checkpoints)
            metric = "f1" if "f1" in row["measured"] else "hits@1"
            print(f"{case.directory.as_posix()}  {METRIC_TITLES[metric]} {row['measured'][metric]:.3f}", flush=True)
            progress.update(number, force=True)
    else:
        report(args.output or [RUNS])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PraseError as error:
        sys.exit(str(error))
