"""Read and write tab-separated files."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .errors import DataError

PathLike = str | os.PathLike


def read_rows(path: PathLike, width: int, *, rest: bool = False) -> Iterator[list[str]]:
    """Yield the stripped fields of each nonblank line.

    With ``rest=True`` the last field keeps any further tabs, which attribute
    values sometimes contain.
    """
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t", width - 1) if rest else line.split("\t")
            if len(fields) != width:
                raise DataError(f"{path}:{number}: expected {width} tab-separated fields, found {len(fields)}")
            yield [field.strip() for field in fields]


def write_rows(path: PathLike, header: Iterable[str] | None, rows: Iterable[Iterable[object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        if header is not None:
            stream.write("\t".join(header) + "\n")
        for row in rows:
            stream.write("\t".join(_format(value) for value in row) + "\n")


def _format(value: object) -> str:
    if isinstance(value, float):
        return repr(float(value))
    text = str(value)
    if "\t" in text or "\n" in text:
        raise DataError(f"cannot write a field containing a tab or newline: {text!r}")
    return text
