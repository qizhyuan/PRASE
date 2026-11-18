"""Manual review decisions that every pipeline step respects.

An approved pair is a fixed match: reasoning keeps it at probability 1 and
embedding models train on it. A rejected pair is never proposed or matched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import DataError
from .kg import KG
from .tsv import PathLike, write_rows

APPROVE = "approve"
REJECT = "reject"


@dataclass(frozen=True)
class Decisions:
    """Approved and rejected ``(left_id, right_id)`` pairs."""

    approved: frozenset[tuple[str, str]] = frozenset()
    rejected: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "approved", frozenset(map(tuple, self.approved)))
        object.__setattr__(self, "rejected", frozenset(map(tuple, self.rejected)))
        conflicts = self.approved & self.rejected
        if conflicts:
            raise DataError(f"pair both approved and rejected: {sorted(conflicts)[0]}")
        for side in (0, 1):
            values = [pair[side] for pair in self.approved]
            if len(values) != len(set(values)):
                raise DataError("approved pairs must be one-to-one")

    def __bool__(self) -> bool:
        return bool(self.approved or self.rejected)

    @classmethod
    def read(cls, path: PathLike) -> Decisions:
        """Read ``left_id, right_id, decision`` rows, where decision is approve or reject.

        A header row and extra columns, such as a copied score, are ignored.
        Rows with an empty decision are skipped.
        """
        approved, rejected = set(), set()
        with Path(path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                fields = [value.strip() for value in line.strip("\r\n").split("\t")]
                if not line.strip() or (number == 1 and fields[:2] == ["left", "right"]):
                    continue
                if len(fields) < 3:
                    raise DataError(f"{path}:{number}: expected left_id, right_id, and a decision")
                decision = fields[-1].lower()
                if not decision:
                    continue
                if decision not in (APPROVE, REJECT):
                    raise DataError(f"{path}:{number}: decision must be {APPROVE!r} or {REJECT!r}")
                (approved if decision == APPROVE else rejected).add((fields[0], fields[1]))
        return cls(frozenset(approved), frozenset(rejected))

    def write(self, path: PathLike) -> None:
        rows = [(a, b, APPROVE) for a, b in sorted(self.approved)]
        rows += [(a, b, REJECT) for a, b in sorted(self.rejected)]
        write_rows(path, ("left", "right", "decision"), rows)

    def merged(self, other: Decisions) -> Decisions:
        """Combine two sets of decisions; ``other`` wins where they disagree."""
        taken_left = {a for a, _ in other.approved}
        taken_right = {b for _, b in other.approved}
        kept = {
            (a, b) for a, b in self.approved
            if (a, b) not in other.rejected and a not in taken_left and b not in taken_right
        }
        return Decisions(
            frozenset(kept | other.approved), frozenset((self.rejected - other.approved) | other.rejected),
        )

    def resolve(self, left: KG, right: KG) -> Constraints:
        def indices(pairs: Iterable[tuple[str, str]]) -> list[tuple[int, int]]:
            rows = []
            for a, b in sorted(pairs):
                if a not in left.entity_index or b not in right.entity_index:
                    raise DataError(f"decision refers to an unknown entity pair ({a!r}, {b!r})")
                rows.append((left.entity_index[a], right.entity_index[b]))
            return rows

        return Constraints(
            np.array(indices(self.approved), dtype=np.int64).reshape(-1, 2),
            frozenset(indices(self.rejected)),
        )


@dataclass(frozen=True)
class Constraints:
    """Decisions resolved to entity indices of one graph pair."""

    approved: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    rejected: frozenset[tuple[int, int]] = frozenset()

    def __bool__(self) -> bool:
        return bool(len(self.approved) or self.rejected)
