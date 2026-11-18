"""Composable alignment pipelines.

A pipeline is a sequence of steps. Every step receives a :class:`State` and
returns a new one: the two graphs, the current alignment, and what later
steps may use, such as proposals and embeddings. PARIS, PRASE, and the
baselines are all pipelines (see :mod:`prase.methods`), and any object with a
``name`` and a ``run(state, context)`` method can be a step.

The runner times every step, evaluates the alignment after each step when a
reference is given, and can save every intermediate result to a directory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .alignment import Alignment, PredicateMatch
from .decisions import Constraints, Decisions
from .embedding import Embeddings, EmbeddingModel, TrainingData, create_model
from .errors import ConfigError
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
    ) -> RunResult:
        """Run every step in order.

        ``decisions`` supplies manual approvals and rejections. ``seeds`` is
        an alignment of known pairs available to steps. ``seed`` determines
        their random seeds. ``reference`` is an array of index pairs used to
        score step alignments and observations and measure proposal precision.
        An ``output`` directory stores the final alignment and per-step records.
        """
        from .records import RunWriter

        constraints = decisions.resolve(left, right) if decisions else Constraints()
        state = State(left, right, constraints, seeds if seeds is not None else Alignment())
        if len(constraints.approved):
            state = replace(state, alignment=_approved(constraints))
        gold = None if reference is None else set(map(tuple, np.asarray(reference).reshape(-1, 2).tolist()))
        writer = RunWriter(output, self, left, right, seed, save_embeddings) if output is not None else None
        cache: dict[Any, Any] = {}
        records = []
        start = perf_counter()
        for index, step in enumerate(self.steps, 1):
            context = Context(_derived_seed(seed, index), cache)
            began = perf_counter()
            previous = state
            try:
                state = step.run(state, context)
            except Exception as error:
                if writer is not None:
                    writer.fail(index, step.name, error)
                raise
            record = StepRecord(index, step.name, perf_counter() - began, context.values)
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
            _LOGGER.info("Step %d (%s) took %.1f s%s", index, step.name, record.seconds,
                         f", F1 {record.scores.f1:.4f}" if record.scores else "")
            if writer is not None:
                writer.step(record, previous, state)
        result = RunResult(state, records, perf_counter() - start)
        if writer is not None:
            writer.finish(result)
        return result


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
    the reasoning. The default pairs equal values.
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
