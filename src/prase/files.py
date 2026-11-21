"""Files written whole or not at all, and sealed files that carry the digest of their content.

Every write goes to a temporary file with a unique name in the target's
folder and then replaces the target in one step, so a reader never sees a
partial file and concurrent writers never share a temporary file. A sealed
file starts with a fixed-size header that names its kind and holds the
SHA-256 digest of the rest; it is written and checked as a stream, without
holding its content in memory, and its header shows that PRASE wrote it.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_FORMAT = b"1"
_DIGEST_SIZE = 64
_ATTEMPTS = 5


def atomic_write(path: str | Path, write: Callable[[BinaryIO], None]) -> None:
    """Write ``path`` through ``write(stream)`` into a unique temporary file, then replace it in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".prase-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write(path, lambda stream: stream.write(text.encode("utf-8")))


def _replace(temporary: str, path: Path) -> None:
    # On Windows a reader that holds the target open blocks the rename for a moment.
    for attempt in range(_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _header(magic: bytes, digest: bytes) -> bytes:
    return b" ".join((magic, _FORMAT, digest)) + b"\n"


def header_size(magic: bytes) -> int:
    return len(_header(magic, b"0" * _DIGEST_SIZE))


def write_sealed(path: str | Path, magic: bytes, write_body: Callable[[BinaryIO], None]) -> None:
    """Write a sealed file: the header with the digest of the body that ``write_body(stream)`` streams."""
    def write(stream: BinaryIO) -> None:
        stream.write(_header(magic, b"0" * _DIGEST_SIZE))
        writer = _HashingWriter(stream)
        write_body(writer)
        stream.seek(0)
        stream.write(_header(magic, writer.digest.hexdigest().encode()))

    atomic_write(path, write)


@contextmanager
def open_sealed(path: str | Path, magic: bytes) -> Iterator[BinaryIO]:
    """The body of a sealed file of this kind, after checking its digest; an error for any other file."""
    with Path(path).open("rb") as stream:
        header = stream.read(header_size(magic))
        parts = header.rstrip(b"\n").split(b" ")
        if len(parts) != 3 or parts[0] != magic or parts[1] != _FORMAT or not header.endswith(b"\n"):
            raise ValueError("the file is not a checkpoint of this kind and version")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        if digest.hexdigest().encode() != parts[2]:
            raise ValueError("the file is damaged or incomplete")
        yield Window(stream, len(header))


def is_sealed(path: str | Path, magics: tuple[bytes, ...]) -> bool:
    """Whether a file starts with the header of one of these kinds, as every file PRASE seals does."""
    try:
        with Path(path).open("rb") as stream:
            start = stream.read(max(map(len, magics)) + 1)
    except OSError:
        return False
    return any(start.startswith(magic + b" ") for magic in magics)


class _HashingWriter(io.RawIOBase):
    """A write-only stream that passes data on and keeps its SHA-256 digest."""

    def __init__(self, stream: BinaryIO) -> None:
        super().__init__()
        self.stream, self.digest = stream, hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        view = memoryview(data)
        self.digest.update(view)
        self.stream.write(view)
        return len(view)


class Window(io.RawIOBase):
    """A read-only view of a file from ``offset`` to its end, for readers that expect a whole file."""

    def __init__(self, stream: BinaryIO, offset: int) -> None:
        super().__init__()
        self.stream, self.offset = stream, offset
        self.stream.seek(offset)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, position: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position += self.offset
        return self.stream.seek(position, whence) - self.offset

    def tell(self) -> int:
        return self.stream.tell() - self.offset

    def readinto(self, buffer) -> int:
        return self.stream.readinto(buffer)
