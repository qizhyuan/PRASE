"""Load graph pairs from local files or the pinned benchmark collection, and fetch and read word vectors."""

from __future__ import annotations

import hashlib
import tempfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from .errors import DataError
from .kg import KG, Normalizer, clean_literal
from .tsv import PathLike

Link = tuple[str, str]

_BENCHMARKS = tomllib.loads(files(__package__).joinpath("_benchmark_manifest.toml").read_text(encoding="utf-8"))
BENCHMARK_NAMES = tuple(_BENCHMARKS["datasets"])


@dataclass(frozen=True)
class Split:
    """One train, validation, and test partition of the reference links."""

    train: tuple[Link, ...]
    valid: tuple[Link, ...]
    test: tuple[Link, ...]


@dataclass(frozen=True, eq=False)
class Dataset:
    """A graph pair with optional reference links and evaluation splits."""

    name: str
    left: KG
    right: KG
    links: tuple[Link, ...]
    splits: tuple[Split, ...] = ()

    def reference(self, links: Iterable[Link] | None = None) -> np.ndarray:
        """Resolve the supplied links or this dataset's reference to index pairs.

        The result has shape ``(n, 2)``. A link that names an entity missing
        from either graph is an error.
        """
        return link_indices(self.links if links is None else links, self.left, self.right)


def read_links(path: PathLike) -> tuple[Link, ...]:
    """Read ``left_id, right_id`` rows. A third score column is ignored."""
    links = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = [field.strip() for field in line.strip("\r\n").split("\t")]
            if not line.strip() or (number == 1 and fields[:2] == ["left", "right"]):
                continue
            if len(fields) < 2:
                raise DataError(f"{path}:{number}: expected a left and a right entity")
            links.append((fields[0], fields[1]))
    return tuple(links)


def link_indices(links: Iterable[Link], left: KG, right: KG) -> np.ndarray:
    """Resolve ``(left_id, right_id)`` links to an ``(n, 2)`` array of entity indices."""
    left_index, right_index = left.entity_index, right.entity_index
    links = list(links)
    unknown = [(a, b) for a, b in links if a not in left_index or b not in right_index]
    if unknown:
        a, b = unknown[0]
        raise DataError(f"{len(unknown)} of {len(links)} links name entities missing from the graphs, "
                        f"such as {a} -> {b}")
    return np.array([(left_index[a], right_index[b]) for a, b in links], dtype=np.int64).reshape(-1, 2)


def load_dataset(path: PathLike, *, normalize: Normalizer | None = clean_literal, name: str | None = None) -> Dataset:
    """Load a folder in the OpenEA layout.

    The folder holds ``rel_triples_1``, ``rel_triples_2``, optional
    ``attr_triples_1`` and ``attr_triples_2``, an optional ``ent_links``
    reference, and optional ``721_5fold/<k>/{train,valid,test}_links`` splits.
    """
    return _load_dataset(Path(path), normalize, name)


def _load_dataset(root: Path, normalize: Normalizer | None, name: str | None,
                  fold_numbers: tuple[int, ...] | None = None) -> Dataset:
    if not (root / "rel_triples_1").is_file() or not (root / "rel_triples_2").is_file():
        raise DataError(f"{root}: expected rel_triples_1 and rel_triples_2 (OpenEA layout)")

    def graph(side: int) -> KG:
        attributes = root / f"attr_triples_{side}"
        return KG.from_tsv(
            root / f"rel_triples_{side}", attributes if attributes.is_file() else None,
            name=f"{root.name}/{side}", normalize=normalize,
        )

    links = read_links(root / "ent_links") if (root / "ent_links").is_file() else ()
    fold_root = root / "721_5fold"
    if fold_numbers is not None:
        folds = [fold_root / str(number) for number in fold_numbers]
    elif fold_root.is_dir():
        folds = sorted((p for p in fold_root.iterdir() if p.is_dir() and p.name.isdigit()),
                       key=lambda p: int(p.name))
    else:
        folds = []
    splits = tuple(Split(*(
        read_links(fold / f"{part}_links") if (fold / f"{part}_links").is_file() else ()
        for part in ("train", "valid", "test")
    )) for fold in folds)
    return Dataset(name or root.name, graph(1), graph(2), links, splits)


def _benchmark_files(name: str) -> dict[str, str]:
    try:
        return _BENCHMARKS["datasets"][name]["sha256"]
    except KeyError:
        raise DataError(f"unknown benchmark {name!r}. Choose from {', '.join(BENCHMARK_NAMES)}.") from None


def benchmark_folds(name: str) -> tuple[int, ...]:
    """Return the split numbers pinned in the manifest, in loading order."""
    return tuple(sorted({int(path.split("/")[1]) for path in _benchmark_files(name) if path.startswith("721_5fold/")}))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_files(folder: Path, checksums: dict[str, str]) -> None:
    for file, expected in checksums.items():
        path = folder / file
        if not path.is_file():
            raise DataError(f"{path}: missing benchmark file")
        if _sha256(path) != expected:
            raise DataError(f"{path}: SHA-256 does not match the pinned benchmark")


def verify_benchmark(name: str, directory: PathLike) -> Path:
    """Check the pinned files in ``directory/name`` and return that folder."""
    checksums = _benchmark_files(name)
    folder = Path(directory) / name
    _verify_files(folder, checksums)
    return folder


def download_benchmark(name: str | None = None, directory: PathLike | None = None) -> Path:
    """Download and verify one benchmark or all six when ``name`` is omitted.

    Return the snapshot root containing the dataset folders. With no
    ``directory``, Hugging Face stores the snapshot in its standard cache.
    """
    from huggingface_hub import snapshot_download

    names = BENCHMARK_NAMES if name is None else (name,)
    paths = [f"{dataset}/{file}" for dataset in names for file in _benchmark_files(dataset)]
    source = _BENCHMARKS["source"]
    try:
        root = Path(snapshot_download(
            source["repository"], repo_type="dataset", revision=source["revision"], local_dir=directory,
            allow_patterns=[*source["sha256"], *paths],
        ))
    except OSError as error:
        raise DataError(f"could not download benchmarks from {source['repository']}: {error}") from error
    _verify_files(root, source["sha256"])
    for dataset in names:
        verify_benchmark(dataset, root)
    return root


def verify_word_vectors(path: PathLike) -> Path:
    """Check that ``path`` holds the pinned FastText vectors and return it."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{path}: missing word vector file")
    if _sha256(path) != _BENCHMARKS["word_vectors"]["sha256"]:
        raise DataError(f"{path}: SHA-256 does not match the pinned word vectors")
    return path


def download_word_vectors(directory: PathLike = "data") -> Path:
    """Download and verify the English FastText vectors that MultiKE and EMB-Match read.

    Return the path of ``wiki-news-300d-1M.vec`` in ``directory``. A verified
    copy already there is kept, and the zip archive is removed after extraction.
    """
    from huggingface_hub.utils import tqdm

    pin = _BENCHMARKS["word_vectors"]
    folder = Path(directory)
    target = folder / pin["file"]
    if target.is_file() and _sha256(target) == pin["sha256"]:
        return target
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder, prefix=".download-") as scratch:
        archive, extracted = Path(scratch) / f"{pin['file']}.zip", Path(scratch) / pin["file"]
        try:
            request = urllib.request.Request(pin["url"], headers={"User-Agent": "prase"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
                size = int(response.headers.get("Content-Length") or 0) or None
                with tqdm(total=size, unit="B", unit_scale=True, desc=archive.name) as progress:
                    while chunk := response.read(1 << 20):
                        stream.write(chunk)
                        progress.update(len(chunk))
            with zipfile.ZipFile(archive) as bundle:
                bundle.extract(pin["file"], scratch)
        except (OSError, zipfile.BadZipFile, KeyError) as error:
            raise DataError(f"could not download word vectors from {pin['url']}: {error}") from error
        verify_word_vectors(extracted)
        extracted.replace(target)
    return target


def read_word_vectors(path: PathLike, vocabulary: set[str]) -> tuple[dict[str, np.ndarray], int]:
    """Read the vectors of ``vocabulary`` from a text file of ``word v1 v2 ...`` lines.

    Return the vectors found and the file's vector width. A first line that
    holds two numbers, the vector count and width, is skipped.
    """
    vectors, width = {}, None
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            word, _, rest = line.rstrip("\n").partition(" ")
            if width is None:
                width = len(rest.split())
                if number == 1 and width == 1:
                    width = None
                    continue
            if word not in vocabulary or word in vectors:
                continue
            try:
                vector = np.array(rest.split(), dtype=np.float32)
            except ValueError as error:
                raise DataError(f"{path}:{number}: invalid word vector") from error
            if len(vector) != width or not np.isfinite(vector).all():
                raise DataError(f"{path}:{number}: word vectors must be finite and of equal width")
            vectors[word] = vector
    if not width:
        raise DataError(f"{path}: no word vectors found")
    return vectors, width


def load_benchmark(name: str) -> Dataset:
    """Load a verified benchmark by name, downloading it into the Hugging Face cache if needed."""
    root = download_benchmark(name)
    return _load_dataset(root / name, clean_literal, name, benchmark_folds(name))


def load_local_benchmark(name: str, directory: PathLike) -> Dataset:
    """Load a benchmark from a verified local copy in ``directory/name``."""
    folder = verify_benchmark(name, directory)
    return _load_dataset(folder, clean_literal, name, benchmark_folds(name))


def read_pairs_with_scores(path: PathLike) -> list[tuple[str, str, float]]:
    """Read ``left_id, right_id[, score]`` rows; a missing score is 1."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.strip("\r\n").split("\t")]
            if number == 1 and fields[:2] == ["left", "right"]:
                continue  # The header that Alignment.write adds.
            if len(fields) not in (2, 3):
                raise DataError(f"{path}:{number}: expected left_id, right_id, and an optional score")
            try:
                score = float(fields[2]) if len(fields) == 3 else 1.0
            except ValueError as error:
                raise DataError(f"{path}:{number}: invalid score {fields[2]!r}") from error
            rows.append((fields[0], fields[1], score))
    return rows


__all__ = [
    "BENCHMARK_NAMES", "Dataset", "Link", "Split", "benchmark_folds", "download_benchmark", "download_word_vectors",
    "link_indices", "load_benchmark", "load_dataset", "load_local_benchmark", "read_links", "read_pairs_with_scores",
    "read_word_vectors", "verify_benchmark", "verify_word_vectors",
]
