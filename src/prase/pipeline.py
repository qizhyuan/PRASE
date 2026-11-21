"""Composable alignment pipelines.

A pipeline is a sequence of steps. Every step receives a :class:`State` and
returns a new one: the two graphs, the current alignment, and what later
steps may use, such as proposals and embeddings. PARIS, PRASE, and the
baselines are all pipelines (see :mod:`prase.methods`), and any object with a
``name`` and a ``run(state, context)`` method can be a step.

The runner times every step, evaluates the alignment after each step when a
reference is given, and can save every intermediate result to a directory.
With an output directory it also saves checkpoints, so that a run resumed
after an interruption reuses the steps it finished (see
:mod:`prase.checkpoints`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .alignment import Alignment, PredicateMatch
from .checkpoints import (
    CheckpointPolicy, StepCheckpoints, array_digest, chain_key, component_identity, environment_identity,
    file_identity, model_identity, predicate_rows, training_checkpoints,
)
from .decisions import Constraints, Decisions
from .embedding import Embeddings, EmbeddingModel, TrainingData, create_model
from .embedding.common import check_device
from .errors import ConfigError, DataError
from .kg import KG
from .metrics import Scores, evaluate
from .paris import LiteralMatcher, Paris, ParisConfig, exact_literals

_LOGGER = logging.getLogger("prase")


@dataclass(frozen=True, eq=False)
class State:
    """Graphs and alignment data passed from one step to the next.

    ``alignment`` holds the current matches. ``proposals`` and ``embeddings``
    hold candidate pairs and vectors for later steps. ``seeds`` holds known
    pairs supplied by the caller, and ``constraints`` contains resolved manual
    decisions. To change the state, return an updated copy with
    ``dataclasses.replace``.
    """

    left: KG
    right: KG
    constraints: Constraints = field(default_factory=Constraints)
    seeds: Alignment = field(default_factory=Alignment)
    alignment: Alignment = field(default_factory=Alignment)
    proposals: Alignment | None = None
    embeddings: Embeddings | None = None
    predicates: tuple[PredicateMatch, ...] = ()

    def unmatched(self) -> tuple[np.ndarray, np.ndarray]:
        """Entities of each graph without a counterpart in the alignment."""
        left = np.setdiff1d(np.arange(self.left.num_entities), self.alignment.left)
        right = np.setdiff1d(np.arange(self.right.num_entities), self.alignment.right)
        return left, right


class Context:
    """A step's random seed, result metadata, and run-scoped cache."""

    def __init__(self, seed: int, cache: dict[Any, Any]) -> None:
        self.seed = seed
        self.values: dict[str, Any] = {}
        self.observations: list[tuple[str, Alignment, dict[str, Any]]] = []
        self._cache = cache

    def report(self, **values: Any) -> None:
        """Record counts or settings for this step's record."""
        self.values.update(values)

    def observe(self, label: str, alignment: Alignment, **values: Any) -> None:
        """Record an intermediate alignment, such as one PARIS iteration, for evaluation."""
        self.observations.append((label, alignment, values))

    def cached(self, key: Any, build: Callable[[], Any]) -> Any:
        """Reuse an object across steps of one run, such as a PARIS index."""
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]


@runtime_checkable
class Step(Protocol):
    """A named operation that returns the state for the next pipeline step."""

    name: str

    def run(self, state: State, context: Context) -> State: ...


@dataclass
class StepRecord:
    index: int
    name: str
    seconds: float
    values: dict[str, Any]
    scores: Scores | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    proposal_precision: float | None = None
    resumed: bool = False


@dataclass
class RunResult:
    """Final state, step records, and elapsed time for a pipeline run."""

    state: State
    steps: list[StepRecord]
    seconds: float

    @property
    def alignment(self) -> Alignment:
        return self.state.alignment

    @property
    def scores(self) -> Scores | None:
        return self.steps[-1].scores if self.steps else None


class Pipeline:
    """An ordered list of steps with one entry point, :meth:`run`."""

    def __init__(self, steps: Sequence[Step], *, name: str = "pipeline") -> None:
        steps = list(steps)
        for step in steps:
            if not isinstance(step, Step):
                raise ConfigError(f"{step!r} is not a pipeline step: it needs a name and run(state, context)")
            # A step's name names its folders in the output and checkpoint directories.
            if not isinstance(step.name, str) or step.name in ("", ".", "..") or any(c in step.name for c in "/\\:"):
                raise ConfigError(f"step name {step.name!r} must be a plain folder name")
        self.steps, self.name = steps, name

    def __repr__(self) -> str:
        return f"Pipeline({self.name!r}, {[step.name for step in self.steps]})"

    def run(
        self,
        left: KG,
        right: KG,
        *,
        decisions: Decisions | None = None,
        seeds: Alignment | None = None,
        seed: int = 0,
        reference: np.ndarray | None = None,
        output: str | Path | None = None,
        save_embeddings: bool = False,
        checkpoints: CheckpointPolicy | None = CheckpointPolicy(),
        resume: bool = False,
    ) -> RunResult:
        """Run every step in order.

        ``decisions`` supplies manual approvals and rejections. ``seeds`` is
        an alignment of known pairs available to steps. ``seed`` determines
        their random seeds. ``reference`` is an array of index pairs used to
        score step alignments and observations and measure proposal precision.
        An ``output`` directory stores the final alignment and per-step records.

        ``checkpoints`` chooses the checkpoints saved in ``<output>/checkpoint``
        (see :class:`~prase.CheckpointPolicy`), and None saves none. With
        ``resume``, the run continues an unfinished run in ``output``, or a
        finished run whose checkpoints were kept: it reuses every step whose
        identity, seed, and input state are unchanged, and continues an
        interrupted model fit from its last saved epoch. Every argument is
        checked before the output directory changes.
        """
        from .records import RunWriter

        _check_arguments(left, right, seeds, seed, reference, output, save_embeddings, checkpoints, resume)
        constraints = decisions.resolve(left, right) if decisions else Constraints()
        state = State(left, right, constraints, seeds if seeds is not None else Alignment())
        if len(constraints.approved):
            state = replace(state, alignment=_approved(constraints))
        gold = None if reference is None else set(map(tuple, np.asarray(reference).reshape(-1, 2).tolist()))
        store = _checkpoint_store(checkpoints, output, resume)
        if resume and output is not None:
            _check_resumable(Path(output), store)
        writer = RunWriter(output, self, left, right, seed, save_embeddings, resume=resume) \
            if output is not None else None
        if store is not None:
            store.claim()
        saving = store is not None and store.policy.steps
        reusing = resume and saving
        cache: dict[Any, Any] = {}
        records = []
        start = perf_counter()
        earlier = 0.0  # time that reused steps took in the run that saved them
        for index, step in enumerate(self.steps, 1):
            context = Context(_derived_seed(seed, index), cache)
            key = _step_key(step, context.seed, state) if saving else None
            previous = state
            saved = store.load(index, step.name, key) if reusing and key is not None else None
            if saved is not None:
                try:
                    state, seconds, values, observations = _restore(previous, saved)
                except Exception as error:  # an incomplete or inconsistent checkpoint: run the step again
                    _LOGGER.warning("Step %d (%s) runs again: its checkpoint is unusable (%s: %s)",
                                    index, step.name, type(error).__name__, error)
                    saved = None
                else:
                    context.values.update(values)
                    context.observations.extend(observations)
                    earlier += seconds
            if saved is None:
                _LOGGER.info("Step %d of %d (%s) started", index, len(self.steps), step.name)
                began = perf_counter()
                scope = training_checkpoints(store.training, minutes=store.policy.training_minutes) \
                    if store is not None and store.policy.training else nullcontext([])
                try:
                    with scope as trained:
                        state = step.run(state, context)
                except Exception as error:
                    if writer is not None:
                        writer.fail(index, step.name, error)
                    raise
                seconds = perf_counter() - began
                if key is not None and _save(store, index, step.name, key, previous, state, context, seconds):
                    # The step's result is saved, so the finished fits that produced it are no longer needed.
                    for path in trained:
                        path.unlink(missing_ok=True)
            record = StepRecord(index, step.name, seconds, context.values, resumed=saved is not None)
            if gold is not None:
                record.scores = evaluate(state.alignment, gold)
                if state.proposals is not None:
                    record.proposal_precision = evaluate(state.proposals, gold).precision
            for label, alignment, values in context.observations:
                entry = {"label": label, "pairs": len(alignment), **values}
                if gold is not None:
                    entry["scores"] = evaluate(alignment, gold).as_dict()
                record.observations.append(entry)
            records.append(record)
            scores = f", F1 {record.scores.f1:.4f}" if record.scores else ""
            if record.resumed:
                _LOGGER.info("Step %d of %d (%s) reused from its checkpoint, which took %.1f s%s",
                             index, len(self.steps), step.name, seconds, scores)
            else:
                _LOGGER.info("Step %d of %d (%s) took %.1f s%s", index, len(self.steps), step.name, seconds, scores)
            if writer is not None:
                writer.step(record, previous, state)
        result = RunResult(state, records, perf_counter() - start + earlier)
        if writer is not None:
            writer.finish(result)
        if store is not None:
            store.finish()
        return result


def _check_arguments(left, right, seeds, seed, reference, output, save_embeddings, checkpoints, resume) -> None:
    """Reject invalid arguments of :meth:`Pipeline.run` before anything is written."""
    if not (isinstance(left, KG) and isinstance(right, KG)):
        raise ConfigError("a pipeline runs on two KG graphs")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("seed must be a nonnegative integer")
    for name, value in (("save_embeddings", save_embeddings), ("resume", resume)):
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be true or false")
    if output is not None and not isinstance(output, (str, Path)):
        raise ConfigError("output must be a directory path")
    if checkpoints is not None and not isinstance(checkpoints, CheckpointPolicy):
        raise ConfigError("checkpoints must be a CheckpointPolicy or None")
    if seeds is not None:
        if not isinstance(seeds, Alignment):
            raise ConfigError("seeds must be an Alignment")
        _check_pairs("seeds", seeds.left, seeds.right, left, right)
        if not np.isfinite(seeds.score).all():
            raise DataError("seed scores must be finite")
    if reference is not None and np.asarray(reference).size:
        pairs = np.asarray(reference)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise DataError("reference must be an (n, 2) array of entity index pairs")
        _check_pairs("reference pairs", pairs[:, 0], pairs[:, 1], left, right)


def _check_pairs(what: str, left_ids: np.ndarray, right_ids: np.ndarray, left: KG, right: KG) -> None:
    if len(left_ids) and (left_ids.min() < 0 or left_ids.max() >= left.num_entities
                          or right_ids.min() < 0 or right_ids.max() >= right.num_entities):
        raise DataError(f"{what} need entity indices below {left.num_entities} on the left and "
                        f"{right.num_entities} on the right")


def _checkpoint_store(policy: CheckpointPolicy | None, output: str | Path | None,
                      resume: bool) -> StepCheckpoints | None:
    """Check the checkpoint settings of a run, before anything is written."""
    if policy is None or not policy.enabled:
        if resume:
            raise ConfigError("resuming a run needs a checkpoint policy")
        return None
    return StepCheckpoints.open(policy, output, resume)


def _check_resumable(output: Path, store: StepCheckpoints | None) -> None:
    """Refuse to resume a finished run that kept no step checkpoints, which would only repeat it."""
    try:
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(record, dict) and record.get("finished") and (store is None or not store.has_steps()):
        raise DataError(f"{output} holds a finished run without checkpoints, so there is nothing to resume. "
                        "Choose a new output directory to run it again.")


def _step_key(step: Step, seed: int, state: State) -> str | None:
    """The key of a step's checkpoint: the environment, the step, its seed, and the state it receives.

    None when the step cannot be identified, so that it always runs.
    """
    identity = component_identity(step)
    if identity is None:
        return None
    return chain_key("step", environment_identity(), identity, step.name, seed, _state_digest(state))


def _state_digest(state: State) -> str:
    """A digest of everything a step can read from its state, including the graphs' names."""
    from .records import _plain

    def pairs(alignment: Alignment | None) -> str | None:
        return None if alignment is None else array_digest(alignment.left, alignment.right, alignment.score)

    embeddings = state.embeddings
    vectors = None if embeddings is None else [
        array_digest(embeddings.left, embeddings.right), embeddings.metric, embeddings.csls_k, embeddings.device,
        _plain(embeddings.info),
    ]
    rejected = np.array(sorted(state.constraints.rejected), dtype=np.int64).reshape(-1, 2)
    graphs = [[graph.name, graph.fingerprint] for graph in (state.left, state.right)]
    return chain_key("state", graphs, pairs(state.seeds), array_digest(state.constraints.approved, rejected),
                     pairs(state.alignment), pairs(state.proposals), vectors, predicate_rows(state.predicates))


# The parts of a state that a step's checkpoint saves when the step changes them.
_FIELDS = ("alignment", "proposals", "embeddings", "predicates")


def _save(store: StepCheckpoints, index: int, name: str, key: str, before: State, after: State,
          context: Context, seconds: float) -> bool:
    """Save what the step changed, so that a resumed run can skip it, and return whether it was saved."""
    from .records import _plain

    if any(getattr(after, field) is not getattr(before, field) for field in ("left", "right", "constraints", "seeds")):
        _LOGGER.debug("Step %d (%s) changed the graphs, seeds, or constraints, so it has no checkpoint", index, name)
        return False
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"seconds": seconds, "values": _plain(context.values), "changed": [], "observations": []}
    for field in _FIELDS:
        value = getattr(after, field)
        if value is getattr(before, field):
            continue
        meta["changed"].append(field)
        if value is None:
            continue
        if field == "embeddings":
            arrays.update(embeddings_left=value.left, embeddings_right=value.right)
            meta["embeddings"] = {"metric": value.metric, "csls_k": value.csls_k, "device": value.device,
                                  "info": _plain(value.info)}
        elif field == "predicates":
            meta["predicates"] = predicate_rows(value)
        else:
            arrays.update(_alignment_arrays(field, value))
    for number, (label, alignment, values) in enumerate(context.observations):
        arrays.update(_alignment_arrays(f"observation{number}", alignment))
        meta["observations"].append({"label": label, "values": _plain(values)})
    store.save(index, name, key, meta, arrays)
    return True


def _restore(before: State, saved: dict) -> tuple[State, float, dict, list]:
    """The state after a saved step, the seconds it took, its values, and its observations.

    Raise an error when the checkpoint misses a part or does not fit the graphs.
    """
    meta, arrays = saved["meta"], saved["arrays"]
    changed, seconds, values, observations = meta["changed"], meta["seconds"], meta["values"], meta["observations"]
    if not (isinstance(changed, list) and set(changed) <= set(_FIELDS) and isinstance(values, dict)
            and isinstance(observations, list) and isinstance(seconds, (int, float)) and seconds >= 0):
        raise ValueError("the record lacks its changes, values, observations, or time")
    if len(set(changed)) != len(changed):
        raise ValueError("the record repeats a change")
    # Every array belongs to a change or an observation that the record lists, and the other way around.
    expected = {f"{field}_{part}" for field in ("alignment", "proposals") if f"{field}_left" in arrays
                for part in ("left", "right", "score")}
    expected |= {f"observation{number}_{part}" for number in range(len(observations))
                 for part in ("left", "right", "score")}
    if "embeddings" in changed and meta.get("embeddings") is not None:
        expected |= {"embeddings_left", "embeddings_right"}
    present = {field for field in ("alignment", "proposals", "embeddings") if f"{field}_left" in arrays}
    if set(arrays) != expected or not present <= set(changed) or ("alignment" in changed) != (
            "alignment" in present):
        raise ValueError("the record and its arrays do not match")
    if "predicates" in changed and not isinstance(meta.get("predicates"), list):
        raise ValueError("the record lacks its predicate matches")
    sizes = (before.left.num_entities, before.right.num_entities)
    updates: dict[str, Any] = {}
    for field in changed:
        if field == "embeddings":
            options = meta.get("embeddings")
            if options is None:
                updates[field] = None
                continue
            vectors = arrays["embeddings_left"], arrays["embeddings_right"]
            if any(table.dtype != np.float32 for table in vectors) or tuple(map(len, vectors)) != sizes:
                raise ValueError("the embeddings do not match the graphs")
            if type(options["csls_k"]) is not int or options["csls_k"] < 1 or not isinstance(options["info"], dict):
                raise ValueError("the embedding settings are invalid")
            check_device(options["device"])
            updates[field] = Embeddings(*vectors, options["metric"], csls_k=options["csls_k"],
                                        device=options["device"], info=options["info"])
        elif field == "predicates":
            matches = tuple(PredicateMatch(**row) for row in meta["predicates"])
            for match in matches:
                probabilities = (match.left_in_right, match.right_in_left)
                if not (isinstance(match.left, str) and isinstance(match.right, str) and all(
                        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1
                        for value in probabilities)):
                    raise ValueError("a predicate match is invalid")
            updates[field] = matches
        else:
            alignment = _alignment_from(arrays, field, sizes)
            if alignment is None and field == "alignment":
                raise KeyError("alignment_left")
            updates[field] = alignment
    restored = []
    for number, entry in enumerate(observations):
        alignment = _alignment_from(arrays, f"observation{number}", sizes)
        if alignment is None or not isinstance(entry["values"], dict):
            raise ValueError(f"observation {number} is incomplete")
        restored.append((str(entry["label"]), alignment, entry["values"]))
    return replace(before, **updates), float(seconds), values, restored


def _alignment_arrays(prefix: str, alignment: Alignment) -> dict[str, np.ndarray]:
    return {f"{prefix}_left": alignment.left, f"{prefix}_right": alignment.right, f"{prefix}_score": alignment.score}


def _alignment_from(arrays: dict[str, np.ndarray], prefix: str, sizes: tuple[int, int]) -> Alignment | None:
    if f"{prefix}_left" not in arrays:
        return None
    left, right, score = (arrays[f"{prefix}_{part}"] for part in ("left", "right", "score"))
    if left.dtype != np.int64 or right.dtype != np.int64 or score.dtype != np.float64:
        raise ValueError(f"{prefix} has arrays of another type")
    alignment = Alignment(left, right, score)
    for side, size in zip((alignment.left, alignment.right), sizes, strict=True):
        if len(side) and (side.min() < 0 or side.max() >= size):
            raise ValueError(f"{prefix} names entities outside the graphs")
    if not np.isfinite(alignment.score).all():
        raise ValueError(f"{prefix} has non-finite scores")
    return alignment


def _derived_seed(seed: int, index: int) -> int:
    return int(np.random.SeedSequence(seed, spawn_key=(index,)).generate_state(1, dtype=np.uint32)[0])


def _approved(constraints: Constraints) -> Alignment:
    return Alignment(constraints.approved[:, 0], constraints.approved[:, 1])


def _user_seeds(state: State) -> Alignment:
    """The caller's seeds without rejected pairs and without pairs that touch an approved entity."""
    seeds = state.seeds.without(state.constraints.rejected)
    approved = state.constraints.approved
    if not len(approved):
        return seeds
    return seeds.subset(~np.isin(seeds.left, approved[:, 0]) & ~np.isin(seeds.right, approved[:, 1]))


def _merge(*alignments: Alignment | None, one_to_one: bool = False) -> Alignment:
    """Concatenate alignments; earlier ones win for a left entity they already cover.

    With ``one_to_one``, earlier ones also win for right entities, and each
    alignment is reduced to its best one-to-one subset.
    """
    result = Alignment()
    for alignment in alignments:
        if alignment is None or not len(alignment):
            continue
        fresh = ~np.isin(alignment.left, result.left)
        if one_to_one:
            fresh &= ~np.isin(alignment.right, result.right)
        alignment = alignment.subset(fresh)
        result = result.concat(alignment.one_to_one() if one_to_one else alignment)
    return result


# Built-in steps -------------------------------------------------------------

@dataclass
class ParisStep:
    """Infer matches using graph facts and optional feedback from earlier steps.

    User seeds and proposals initialize entity counterparts. Seeds that are
    rejected or touch an approved entity are left out, so they never displace
    a proposal. Proposal scores are scaled by ``proposal_weight``. After the configured warmup, PARIS
    can combine structural scores with embedding cosine similarity. This step
    clears the proposals and embeddings from the returned state.

    ``literal_matcher(left, right)`` returns the scored literal pairs that start
    the reasoning. The default pairs equal values. A resumed run reuses this
    step with another matcher only when the matcher has a ``checkpoint_key``.
    """

    config: ParisConfig = field(default_factory=ParisConfig)
    proposal_weight: float = 1.0
    use_proposals: bool = True
    use_embeddings: bool = True
    literal_matcher: LiteralMatcher = exact_literals
    name: str = "paris"

    def run(self, state: State, context: Context) -> State:
        reasoner = context.cached(
            ("paris", id(state.left), id(state.right), self.config, self.literal_matcher),
            lambda: Paris(state.left, state.right, self.config, literal_matcher=self.literal_matcher),
        )
        proposals = state.proposals.scaled(self.proposal_weight) if self.use_proposals and state.proposals else None
        seeds = _merge(_user_seeds(state), proposals)
        embeddings = state.embeddings if self.use_embeddings else None

        def observe(iteration: int, alignment: Alignment, stats: dict[str, int]) -> None:
            context.observe(f"iteration {iteration}", alignment, **stats)

        result = reasoner.run(seeds=seeds, embeddings=embeddings, constraints=state.constraints, on_iteration=observe)
        context.report(
            engine=result.engine, literal_pairs=len(reasoner.literal_pairs), seeds=len(seeds),
            embeddings=embeddings is not None, matches=len(result.alignment),
        )
        return replace(state, alignment=result.alignment, predicates=result.predicates, proposals=None,
                       embeddings=None)

    def checkpoint_key(self) -> dict[str, Any] | None:
        """The settings and the literal matcher's identity, or None for a matcher without a key."""
        matcher = component_identity(self.literal_matcher)
        if matcher is None:
            return None
        return {"config": self.config.as_dict(), "proposal_weight": self.proposal_weight,
                "use_proposals": self.use_proposals, "use_embeddings": self.use_embeddings,
                "literal_matcher": matcher}


@dataclass
class Embed:
    """Train a model and pass proposals, entity vectors, or both to later steps.

    Training pairs come from approvals, user seeds, and current matches that
    meet ``seed_threshold``, without rejected pairs. The first source takes
    precedence when pairs conflict. Proposal candidates exclude their endpoints. With ``propose``,
    the model can choose pairs, with nearest neighbors as the default.
    ``share_embeddings`` retains the vectors in the returned state.
    """

    model: EmbeddingModel | str
    options: dict[str, Any] = field(default_factory=dict)
    seed_threshold: float = 0.1
    propose: bool = True
    share_embeddings: bool = True
    name: str = "embed"

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = create_model(self.model, **self.options)
        elif self.options:
            raise ConfigError("options apply only when the model is given by name")
        if not self.propose and not self.share_embeddings:
            raise ConfigError("an embed step must propose pairs, share embeddings, or both")
        if self.name == "embed":
            self.name = f"embed-{getattr(self.model, 'name', type(self.model).__name__)}"

    def run(self, state: State, context: Context) -> State:
        rejected = state.constraints.rejected
        seeds = _merge(
            _approved(state.constraints), _user_seeds(state),
            state.alignment.above(self.seed_threshold).without(rejected), one_to_one=True,
        )
        left = np.setdiff1d(np.arange(state.left.num_entities), seeds.left)
        right = np.setdiff1d(np.arange(state.right.num_entities), seeds.right)
        data = TrainingData(state.left, state.right, np.stack((seeds.left, seeds.right), axis=1), (left, right),
                            np.array(sorted(rejected), dtype=np.int64).reshape(-1, 2))
        embeddings = self.model.fit(data, seed=context.seed)
        proposals = None
        if self.propose:
            propose = getattr(self.model, "propose", None)
            proposals = (propose(embeddings, data) if callable(propose)
                         else embeddings.nearest(left, right, excluded=data.rejected))
            proposals = proposals.without(state.constraints.rejected)
        context.report(
            seeds=len(seeds), candidates=[len(left), len(right)],
            proposals=0 if proposals is None else len(proposals), metric=embeddings.metric, dim=embeddings.dim,
            **({"training": embeddings.info} if embeddings.info else {}),
        )
        return replace(
            state, proposals=proposals, embeddings=embeddings if self.share_embeddings else None,
        )

    def checkpoint_key(self) -> dict[str, Any] | None:
        """The step settings and the model's settings, files, and code."""
        model = model_identity(self.model)
        if model is None:
            return None
        return {"model": model, "seed_threshold": self.seed_threshold, "propose": self.propose,
                "share_embeddings": self.share_embeddings}


@dataclass
class Predict:
    """Add proposed pairs to the alignment, as used by the embedding-only preset.

    The result keeps the current alignment and adds the best one-to-one
    subset of the proposals between entities it leaves unmatched. Rejected
    proposals are left out.
    """

    name: str = "predict"

    def run(self, state: State, context: Context) -> State:
        if state.proposals is None:
            raise ConfigError("predict needs proposals from an earlier step")
        proposals = state.proposals.without(state.constraints.rejected)
        free = ~np.isin(proposals.left, state.alignment.left) & ~np.isin(proposals.right, state.alignment.right)
        alignment = state.alignment.concat(proposals.subset(free).one_to_one())
        context.report(added=len(alignment) - len(state.alignment))
        return replace(state, alignment=alignment, proposals=None)

    def checkpoint_key(self) -> dict[str, Any]:
        return {}


@dataclass
class Load:
    """Load saved proposals (TSV) and embeddings (``.npz``) instead of training a model."""

    proposals: str | Path | None = None
    embeddings: str | Path | None = None
    name: str = "load"

    def run(self, state: State, context: Context) -> State:
        proposals = Alignment.read(self.proposals, state.left, state.right) if self.proposals else None
        embeddings = Embeddings.load(self.embeddings) if self.embeddings else None
        if embeddings is not None and (
            len(embeddings.left) != state.left.num_entities or len(embeddings.right) != state.right.num_entities
        ):
            raise ConfigError(f"{self.embeddings}: embeddings do not match the graphs' entity counts")
        if proposals is not None:
            proposals = proposals.without(state.constraints.rejected)
        context.report(proposals=0 if proposals is None else len(proposals), embeddings=embeddings is not None)
        return replace(state, proposals=proposals, embeddings=embeddings)

    def checkpoint_key(self) -> dict[str, Any]:
        """The digests of the two files, so that an edited file is read again."""
        return {"proposals": file_identity(self.proposals), "embeddings": file_identity(self.embeddings)}
