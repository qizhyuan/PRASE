"""Knowledge graphs stored as integer triple tables."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .errors import DataError
from .tsv import PathLike, read_rows

RELATION = "relation"
ATTRIBUTE = "attribute"
Normalizer = Callable[[str], str]
_LANGUAGE_TAG = re.compile(r"@[A-Za-z]+(?:-[A-Za-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class Predicate:
    name: str
    kind: str  # RELATION links two entities; ATTRIBUTE links an entity to a literal.


def clean_literal(value: str) -> str:
    """Reduce a literal in RDF notation to its lexical form.

    ``"1990"^^<http://www.w3.org/2001/XMLSchema#gYear>``, ``"Berlin"@en``, and
    ``Berlin`` become ``1990``, ``Berlin``, and ``Berlin``, so values from
    different graphs compare by their text. Angle brackets around an IRI are
    removed.
    """
    value = value.strip()
    if value.startswith('"'):
        end = value.rfind('"')
        suffix = value[end + 1:]
        if end > 0 and (not suffix or suffix.startswith("^^") or _LANGUAGE_TAG.fullmatch(suffix)):
            return value[1:end].strip()
    value = value.split("^^", 1)[0].strip()
    if value.startswith("<"):
        value = value[1:]
    if value.endswith(">"):
        value = value[:-1]
    if value.startswith('"'):
        value = value[1:]
    if value.endswith('"'):
        value = value[:-1]
    return value.strip()


class KG:
    """A knowledge graph with integer indices for entities, predicates, and literals.

    ``relation_triples`` is an ``(n, 3)`` array of ``(head, predicate, tail)``
    entity triples, and ``attribute_triples`` an ``(m, 3)`` array of
    ``(entity, predicate, literal)``. A repeated triple is kept once, at its
    first position. Predicates of both kinds share one index space. All
    arrays are read-only.
    """

    def __init__(
        self,
        entities: Iterable[str],
        predicates: Iterable[Predicate],
        literals: Iterable[str],
        relation_triples: object,
        attribute_triples: object,
        *,
        name: str = "",
    ) -> None:
        self.entities = tuple(entities)
        self.predicates = tuple(predicates)
        self.literals = tuple(literals)
        self.relation_triples = _triple_array(relation_triples)
        self.attribute_triples = _triple_array(attribute_triples)
        self.name = name
        self._validate()

    def _validate(self) -> None:
        if len(set(self.entities)) != len(self.entities):
            raise DataError(f"{self.name or 'graph'}: duplicate entity identifiers")
        kinds = np.array([predicate.kind == RELATION for predicate in self.predicates], dtype=bool)
        for array, kind, objects in (
            (self.relation_triples, RELATION, len(self.entities)),
            (self.attribute_triples, ATTRIBUTE, len(self.literals)),
        ):
            if not len(array):
                continue
            if array.min() < 0 or array[:, 0].max() >= len(self.entities) or array[:, 2].max() >= objects:
                raise DataError(f"{self.name or 'graph'}: {kind} triple index out of range")
            if array[:, 1].max() >= len(self.predicates) or (kinds[array[:, 1]] != (kind == RELATION)).any():
                raise DataError(f"{self.name or 'graph'}: {kind} triples must use {kind} predicates")

    @classmethod
    def from_triples(
        cls,
        relations: Iterable[tuple[str, str, str]] = (),
        attributes: Iterable[tuple[str, str, str]] = (),
        *,
        entities: Iterable[str] = (),
        name: str = "",
        normalize: Normalizer | None = clean_literal,
    ) -> KG:
        """Build a graph from string triples. Indices follow first appearance.

        ``entities`` fixes the order of known entities, including entities
        without triples. Duplicate triples are removed, and ``normalize``
        is applied to every literal value.
        """
        entity_index: dict[str, int] = {entity: index for index, entity in enumerate(dict.fromkeys(entities))}
        predicate_index: dict[tuple[str, str], int] = {}
        literal_index: dict[str, int] = {}

        def entity(identifier: str) -> int:
            return entity_index.setdefault(identifier, len(entity_index))

        def predicate(identifier: str, kind: str) -> int:
            return predicate_index.setdefault((identifier, kind), len(predicate_index))

        relation_rows = dict.fromkeys(
            (entity(head), predicate(relation, RELATION), entity(tail)) for head, relation, tail in relations
        )
        attribute_rows = dict.fromkeys(
            (
                entity(subject),
                predicate(attribute, ATTRIBUTE),
                literal_index.setdefault(normalize(value) if normalize else value, len(literal_index)),
            )
            for subject, attribute, value in attributes
        )
        return cls(
            entity_index,
            (Predicate(identifier, kind) for identifier, kind in predicate_index),
            literal_index,
            list(relation_rows),
            list(attribute_rows),
            name=name,
        )

    @classmethod
    def from_tsv(
        cls,
        relations: PathLike,
        attributes: PathLike | None = None,
        *,
        name: str | None = None,
        normalize: Normalizer | None = clean_literal,
    ) -> KG:
        """Read ``head, relation, tail`` and ``entity, attribute, value`` TSV files."""
        attribute_rows = read_rows(attributes, 3, rest=True) if attributes is not None else ()
        return cls.from_triples(
            read_rows(relations, 3), attribute_rows,
            name=Path(relations).name if name is None else name, normalize=normalize,
        )

    @classmethod
    def from_rdf(
        cls, path: PathLike, *, format: str | None = None, name: str | None = None,
        normalize: Normalizer | None = clean_literal,
    ) -> KG:
        """Read an RDF file with RDFLib. Literal objects become attribute triples."""
        try:
            import rdflib
        except ModuleNotFoundError as error:
            raise DataError("RDF input requires RDFLib: pip install 'prase[rdf]'") from error
        graph = rdflib.Graph()
        try:
            graph.parse(Path(path), format=format)
        except Exception as error:  # RDFLib raises parser-specific exceptions.
            raise DataError(f"{path}: {error}") from error
        blank_nodes: dict[object, str] = {}

        def resource(node: object) -> str:
            if isinstance(node, rdflib.BNode):
                return blank_nodes.setdefault(node, f"_:b{len(blank_nodes)}")
            return str(node)

        relations, attributes = [], []
        for subject, predicate, value in graph:
            if isinstance(value, rdflib.Literal):
                attributes.append((resource(subject), str(predicate), str(value)))
            else:
                relations.append((resource(subject), str(predicate), resource(value)))
        return cls.from_triples(
            relations, attributes, name=Path(path).name if name is None else name, normalize=normalize,
        )

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_literals(self) -> int:
        return len(self.literals)

    @cached_property
    def entity_index(self) -> dict[str, int]:
        return {identifier: index for index, identifier in enumerate(self.entities)}

    @cached_property
    def fingerprint(self) -> str:
        """SHA-256 of the graph content, independent of file paths."""
        digest = hashlib.sha256()
        for values in (self.entities, self.literals):
            digest.update("\n".join(values).encode("utf-8"))
            digest.update(b"\0")
        digest.update("\n".join(f"{p.kind}:{p.name}" for p in self.predicates).encode("utf-8"))
        digest.update(self.relation_triples.tobytes())
        digest.update(self.attribute_triples.tobytes())
        return digest.hexdigest()

    def summary(self) -> dict[str, int]:
        relations = sum(predicate.kind == RELATION for predicate in self.predicates)
        return {
            "entities": self.num_entities,
            "relations": relations,
            "attributes": len(self.predicates) - relations,
            "literals": self.num_literals,
            "relation_triples": len(self.relation_triples),
            "attribute_triples": len(self.attribute_triples),
        }

    def __repr__(self) -> str:
        counts = ", ".join(f"{key}={value}" for key, value in self.summary().items())
        return f"KG({self.name!r}, {counts})"


def local_name(identifier: str) -> str:
    """The last path or fragment segment of an IRI, unquoted, with underscores as spaces."""
    return unquote(re.split(r"[/#]", identifier.rstrip("/#"))[-1]).replace("_", " ").strip()


def entity_names(kg: KG, predicates: Sequence[str] = (), *, local_names: bool = True) -> list[str | None]:
    """Name each entity by the first of ``predicates`` it has a value for.

    When an entity has several values for that predicate, the
    lexicographically first one is used. Entities without any of the
    predicates get their identifier's local name, or ``None`` when
    ``local_names`` is false.
    """
    names: list[str | None] = [local_name(entity) if local_names else None for entity in kg.entities]
    if predicates:
        rank = {name: position for position, name in enumerate(predicates)}
        best: dict[int, tuple[int, str]] = {}
        for entity, predicate, literal in kg.attribute_triples.tolist():
            position = rank.get(kg.predicates[predicate].name)
            if position is not None and kg.literals[literal]:
                candidate = (position, kg.literals[literal])
                if entity not in best or candidate < best[entity]:
                    best[entity] = candidate
        for entity, (_, value) in best.items():
            names[entity] = value
    return names


def _triple_array(rows: object) -> np.ndarray:
    array = np.array(rows, dtype=np.int64)
    if array.size == 0:
        array = np.zeros((0, 3), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DataError("triples must be an (n, 3) table of indices")
    # PARIS counts each fact once, so a repeated triple is kept at its first position only.
    _, first = np.unique(array, axis=0, return_index=True)
    if len(first) < len(array):
        array = array[np.sort(first)]
    array.setflags(write=False)
    return array
