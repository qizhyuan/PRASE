"""Run and inspect the PRASE benchmark suite.

    python benchmarks/reproduce.py list
    python benchmarks/reproduce.py check --data path/to/prase-benchmarks
    python benchmarks/reproduce.py run --method paris --dataset D-W-15K-V2
    python benchmarks/reproduce.py run --method se --model bootea --dataset D-W-15K-V2
    python benchmarks/reproduce.py run --method prase --model bootea --dataset D-W-15K-V2
    python benchmarks/reproduce.py run --all --output runs/full --word-vectors data/wiki-news-300d-1M.vec
    python benchmarks/reproduce.py report

``experiments.toml`` lists pipeline recipes and their dataset coverage.
Runs load the pinned datasets through ``prase.load_benchmark``. Use
``--data`` to point to a local folder containing the named datasets in the
OpenEA layout. A run writes to ``<output>/<experiment>/<dataset>/``,
with ``comparison.json`` holding the measured scores and run settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

import numpy as np

from prase import Alignment, Context, DataError, Dataset, Pipeline, PraseError, RunResult, State
from prase import available_models, load_benchmark, ranking_scores
from prase.config import build_method
from prase.datasets import BENCHMARK_NAMES, benchmark_folds, load_local_benchmark, verify_benchmark, verify_word_vectors

CATALOG = tomllib.loads(Path(__file__).with_name("experiments.toml").read_text(encoding="utf-8"))
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
    """Enumerate the base runs and any configured round sweeps."""
    cases = []
    for experiment in experiments:
        spec = CATALOG["experiments"][experiment]
        for dataset in datasets or spec["datasets"]:
            cases.append(RunCase(experiment, dataset, rounds, fold))
            if include_rounds and rounds is None and fold is None:
                cases.extend(RunCase(experiment, dataset, count) for count in spec.get("rounds", {}).get(dataset, ()))
    return list(dict.fromkeys(cases))


def _method_spec(experiment: str) -> dict:
    spec = CATALOG["experiments"][experiment]
    return CATALOG["experiments"][spec["method"]] if isinstance(spec["method"], str) else spec


def word_vectors(path: Path | None) -> Path:
    if path is None:
        raise SystemExit("this experiment needs --word-vectors. prase download --word-vectors fetches them.")
    try:
        return verify_word_vectors(path)
    except DataError as error:
        raise SystemExit(str(error)) from None


def method_table(experiment: str, dataset: str, *, vectors: Path | None, rounds: int | None,
                 device: str | None) -> dict:
    spec = _method_spec(experiment)
    method = copy.deepcopy(spec["method"])
    extra = spec.get("dataset_options", {}).get(dataset, {})
    if "model" in method:
        settings = method.get("options", {})
        settings.update(extra)
        if device is not None:
            settings["device"] = device
        if settings:
            method["options"] = settings
    else:
        method.update(extra)
        settings = method
    for key, value in settings.items():
        if value == "@word_vectors":
            settings[key] = str(word_vectors(vectors))
    if rounds is not None:
        if method["name"] != "prase":
            raise SystemExit("--rounds applies to PRASE experiments")
        method["rounds"] = rounds
    return method


def run(experiment: str, dataset: str, data: Path | None, output: Path, *, seed: int, rounds: int | None,
        device: str | None, vectors: Path | None, fold: int | None = None) -> dict:
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
    }
    if evaluation == "fold-ranking":
        folds = supervised_runs(dataset, graphs, target, method, seed=seed, fold=fold)
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
        result = pipeline.run(graphs.left, graphs.right, reference=reference, seed=seed, output=target)
        record["seconds"] = result.seconds
        measured = (unaligned_hits(result, reference, train.seed_threshold)
                    if evaluation == "paris-seeded-ranking" else result.scores.as_dict())
    record["measured"] = measured
    target.mkdir(parents=True, exist_ok=True)
    write_record(target / "comparison.json", record)
    return record


def write_record(path: Path, record: dict) -> None:
    """Write through a temporary file, so that list and report never read a partial record."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class UnalignedTop1:
    """Retrieve one neighbor per unaligned left entity."""

    seed_threshold: float
    name = "unaligned-top1"

    def run(self, state: State, context: Context) -> State:
        predictions = state.embeddings.nearest(*unaligned_candidates(state, self.seed_threshold))
        return replace(state, proposals=predictions)


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


def supervised_runs(dataset: str, graphs: Dataset, target: Path, method: dict, *,
                    seed: int, fold: int | None) -> list[dict]:
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
        result = pipeline.run(graphs.left, graphs.right, seeds=seeds, seed=seed, output=target / f"fold-{index}")
        reference = graphs.reference(split.test)
        measured = ranking_scores(result.state.embeddings, reference)
        rows.append({"fold": index, "train_pairs": len(seeds), "test_pairs": len(reference),
                     "seconds": result.seconds, "measured": measured})
    return rows


def _completed_runs(outputs: Sequence[Path]) -> dict[RunCase, dict]:
    results = {}
    for output in outputs:
        current = {}
        for path in sorted(output.glob("*/*/comparison.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            experiment = row["experiment"]
            if experiment not in EXPERIMENTS:
                continue
            rounds = row.get("rounds")
            case = RunCase(experiment, row["dataset"], None if rounds == 1 else rounds, row.get("fold"))
            if rounds != 1 or case not in current:
                current[case] = row
        results.update(current)
    return results


def _status(case: RunCase, outputs: Sequence[Path], results: dict[RunCase, dict]) -> str:
    if case in results:
        return "Completed"
    for output in reversed(outputs):
        target = output / case.directory
        if not target.is_dir() or not any(target.iterdir()):
            continue
        for record in (target / "run.json", *target.glob("fold-*/run.json")):
            if record.is_file() and json.loads(record.read_text(encoding="utf-8")).get("failed"):
                return "Failed"
        return "Incomplete"
    return "Pending"


def coverage(outputs: Sequence[Path]) -> dict[str, dict[str, int]]:
    """Count planned runs by workflow and status across the selected output roots."""
    results = _completed_runs(outputs)
    counts = {workflow: dict.fromkeys(("Completed", "Failed", "Incomplete", "Pending"), 0)
              for workflow in REPORT_GROUPS}
    for case in planned_runs():
        counts[case.workflow][_status(case, outputs, results)] += 1
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
    results = _completed_runs(outputs)
    cases = planned_runs(experiments)
    for name, title in REPORT_GROUPS.items():
        if method is not None and method != name:
            continue
        rows = []
        for case in cases:
            if case.workflow != name or (datasets is not None and case.dataset not in datasets):
                continue
            measured = results.get(case, {}).get("measured", {})
            key = "f1" if "f1" in measured else "hits@1"
            score = f"{METRIC_TITLES[key]} {measured[key]:.3f}" if key in measured else ""
            rows.append((*_case_cells(case)[:3], _status(case, outputs, results), score))
        if rows:
            _table(title, ("Experiment", "Dataset", "Rounds", "Status", "Measured"), rows)


def report(outputs: Sequence[Path]) -> None:
    results = _completed_runs(outputs)
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
    if not cases:
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
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            parser.error(f"{target} already contains a run. Choose a fresh --output directory.")
    if any(_needs_vectors(case) for case in cases):
        word_vectors(args.word_vectors)
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="show configured runs and their recorded status")
    list_parser.add_argument("--output", type=Path, action="append", help=f"inspect a run root (default {RUNS})")
    list_parser.add_argument("--method", choices=METHODS, help="show one workflow")
    list_parser.add_argument("--experiment", action="append", choices=EXPERIMENTS, metavar="RECIPE")
    list_parser.add_argument("--dataset", action="append", choices=BENCHMARK_NAMES)

    check_parser = commands.add_parser("check", help="verify local copies of the benchmark datasets")
    check_parser.add_argument("--data", type=Path, required=True, help="directory containing the local benchmark copies")
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
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--rounds", type=int, help="override the number of PRASE rounds")
    run_parser.add_argument("--include-rounds", action="store_true", help="include each recipe's configured round sweeps")
    run_parser.add_argument("--fold", type=int, help="select one SE fold (default all)")
    run_parser.add_argument("--device", help="embedding model device, such as cuda")
    run_parser.add_argument("--word-vectors", type=Path, help="FastText vectors, for MultiKE and EMB-Match")
    run_parser.add_argument("--verbose", action="store_true", help="log each embedding validation check")

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
        cases = _run_cases(args, run_parser)
        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s %(message)s")
        for case in cases:
            row = run(case.experiment, case.dataset, args.data, args.output, seed=args.seed, rounds=case.rounds,
                      device=args.device, vectors=args.word_vectors, fold=case.fold)
            metric = "f1" if "f1" in row["measured"] else "hits@1"
            print(f"{case.directory.as_posix()}  {METRIC_TITLES[metric]} {row['measured'][metric]:.3f}", flush=True)
    else:
        report(args.output or [RUNS])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PraseError as error:
        sys.exit(str(error))
