"""Synthetic graph pairs with a known alignment."""

from __future__ import annotations

import random

import numpy as np

from prase import KG


def graph_pair(size: int = 60, *, seed: int = 0, drop: float = 0.1, named: float = 0.6):
    """Two noisy copies of one random graph, with renamed entities and predicates.

    About ``named`` of the entities have a unique name literal shared by both
    copies, every entity has a shared year literal, and each copy drops about
    ``drop`` of the relation triples. Returns ``(left, right, reference)``.
    """
    rng = random.Random(seed)
    edges = {(rng.randrange(size), rng.randrange(4), rng.randrange(size)) for _ in range(4 * size)}
    edges = sorted((h, r, t) for h, r, t in edges if h != t)
    permutation = list(range(size))
    rng.shuffle(permutation)

    def copy(prefix: str, rename) -> KG:
        relations = [
            (f"{prefix}:e{rename(h)}", f"{prefix}:rel{r}", f"{prefix}:e{rename(t)}")
            for h, r, t in edges if rng.random() > drop
        ]
        attributes = []
        for entity in range(size):
            if entity < named * size:
                attributes.append((f"{prefix}:e{rename(entity)}", f"{prefix}:name", f'"Entity {entity}"'))
            attributes.append((f"{prefix}:e{rename(entity)}", f"{prefix}:year", f"{1900 + entity % 7}"))
        # Index entities by their own name, so the right graph's order differs.
        return KG.from_triples(relations, attributes, entities=[f"{prefix}:e{k}" for k in range(size)], name=prefix)

    left = copy("L", lambda entity: entity)
    right = copy("R", lambda entity: permutation[entity])
    reference = np.array(
        [(left.entity_index[f"L:e{e}"], right.entity_index[f"R:e{permutation[e]}"]) for e in range(size)],
        dtype=np.int64,
    )
    return left, right, reference


def write_openea(folder, left: KG, right: KG, reference) -> None:
    """Write a graph pair in the OpenEA layout."""
    folder.mkdir(parents=True, exist_ok=True)
    for side, kg in ((1, left), (2, right)):
        with (folder / f"rel_triples_{side}").open("w", encoding="utf-8") as stream:
            for h, p, t in kg.relation_triples.tolist():
                stream.write(f"{kg.entities[h]}\t{kg.predicates[p].name}\t{kg.entities[t]}\n")
        with (folder / f"attr_triples_{side}").open("w", encoding="utf-8") as stream:
            for e, p, v in kg.attribute_triples.tolist():
                stream.write(f"{kg.entities[e]}\t{kg.predicates[p].name}\t{kg.literals[v]}\n")
    with (folder / "ent_links").open("w", encoding="utf-8") as stream:
        for i, j in np.asarray(reference).tolist():
            stream.write(f"{left.entities[i]}\t{right.entities[j]}\n")
