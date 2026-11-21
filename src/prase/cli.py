"""Command line for running PRASE, scoring alignments, and downloading benchmarks and word vectors."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .alignment import Alignment
from .checkpoints import CheckpointPolicy
from .config import load_config
from .datasets import BENCHMARK_NAMES, download_benchmark, download_word_vectors, load_dataset, read_links
from .decisions import Decisions
from .embedding import available_models
from .errors import ConfigError, DataError, PraseError
from .metrics import threshold_scores


def _benchmark_name(value: str) -> str:
    if value not in BENCHMARK_NAMES:
        raise argparse.ArgumentTypeError(f"unknown benchmark {value!r}. Choose from {', '.join(BENCHMARK_NAMES)}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prase", description="Knowledge graph alignment with PARIS and PRASE.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the pipeline described by a TOML file")
    run.add_argument("config", type=Path, help="the TOML run file")
    run.add_argument("--output", type=Path, help="write the results here instead of the run file's [output] directory")
    run.add_argument("--decisions", type=Path, help="approve/reject decisions from a review")
    run.add_argument("--seed", type=int, help="replace the run file's seed")
    run.add_argument("--resume", action="store_true",
                     help="continue the interrupted run in the output directory from its checkpoints")
    run.add_argument("--checkpoint-minutes", type=float, metavar="MINUTES",
                     help="minutes between training checkpoints (default 10; 0 saves after every epoch)")
    run.add_argument("--keep-checkpoints", action="store_true",
                     help="keep the checkpoint folder after the run finishes")
    run.add_argument("--no-checkpoints", action="store_true", help="save no checkpoints")
    run.add_argument("--quiet", action="store_true", help="only print the final scores")

    evaluate = commands.add_parser("evaluate", help="score an alignment TSV against reference links")
    evaluate.add_argument("alignment", type=Path, help="an alignment TSV, such as a run's alignment.tsv")
    evaluate.add_argument("--dataset", type=Path, required=True, help="OpenEA-layout folder of the two graphs")
    evaluate.add_argument("--reference", type=Path, help="links file; defaults to the dataset's ent_links")

    download = commands.add_parser("download", help="download and verify PRASE benchmarks and word vectors")
    download.add_argument("names", nargs="*", type=_benchmark_name, metavar="NAME",
                          help="benchmark IDs (default all six, or none with --word-vectors). "
                               f"Choose from {', '.join(BENCHMARK_NAMES)}")
    download.add_argument("--word-vectors", action="store_true",
                          help="download the English FastText vectors for MultiKE and EMB-Match")
    download.add_argument("--vectors-dir", type=Path, default=Path("data"),
                          help="where to put the word vectors (default data)")

    commands.add_parser("models", help="list the built-in embedding models")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "download":
            return _download(args)
        print("\n".join(available_models()))
        return 0
    except PraseError as error:
        print(f"prase: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    config = load_config(args.config)
    if args.output is not None:
        config = replace(config, output=args.output)
    if args.decisions is not None:
        decisions = Decisions.read(args.decisions)
        config = replace(config, decisions=config.decisions.merged(decisions) if config.decisions else decisions)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.no_checkpoints:
        if args.resume or args.keep_checkpoints or args.checkpoint_minutes is not None:
            raise ConfigError("--no-checkpoints cannot be combined with other checkpoint options")
        config = replace(config, checkpoints=None)
    elif args.keep_checkpoints or args.checkpoint_minutes is not None:
        policy = config.checkpoints or CheckpointPolicy()
        if args.keep_checkpoints:
            policy = replace(policy, keep=True)
        if args.checkpoint_minutes is not None:
            policy = replace(policy, training=True, training_minutes=args.checkpoint_minutes)
        config = replace(config, checkpoints=policy)
    result = config.run(resume=args.resume)
    summary = {"pipeline": config.pipeline.name, "matches": len(result.alignment), "seconds": round(result.seconds, 2)}
    if result.scores is not None:
        summary.update({key: round(value, 4) if isinstance(value, float) else value
                        for key, value in result.scores.as_dict().items()})
    if config.output is not None:
        summary["output"] = str(config.output)
    print(json.dumps(summary))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    links = read_links(args.reference) if args.reference else dataset.links
    if not links:
        raise DataError(f"{args.dataset}: evaluation needs reference links, from an ent_links file or --reference")
    reference = dataset.reference(links)
    alignment = Alignment.read(args.alignment, dataset.left, dataset.right)
    for threshold, scores in threshold_scores(alignment, reference).items():
        print(f"{threshold:.1f}\tP={scores.precision:.4f}\tR={scores.recall:.4f}\tF1={scores.f1:.4f}"
              f"\t{scores.predicted} pairs")
    return 0


def _download(args: argparse.Namespace) -> int:
    if args.names:
        for name in dict.fromkeys(args.names):
            print(download_benchmark(name) / name)
    elif not args.word_vectors:
        print(download_benchmark())
    if args.word_vectors:
        print(download_word_vectors(args.vectors_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
