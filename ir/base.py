"""Core data model for ``ir``.

Retrieval in ``ir`` flows through four small, explicit types:

- :class:`Artifact` — a logical item in a corpus (a file, a skill, a package).
  Opaque ``raw`` payload plus ``metadata``.
- :class:`Surface` — one *embeddable unit* derived from an artifact. A single
  artifact may yield several heterogeneous surfaces (a short description, an
  AI-authored synopsis, a list of problem classes, body chunks). The
  artifact→surfaces decomposition is the job of an
  :class:`~ir.strategy.IndexingStrategy`.
- :class:`IndexPlan` — a strategy's output for one artifact: the
  ``filter_fields`` (hard-filterable metadata, *not* embedded) and the list of
  surfaces (embedded).
- :class:`Record` — a stored, embedded surface (one row in the index; maps
  directly to a ``vd`` ``Document``).
- :class:`SearchHit` — a scored record returned by retrieval, with a helper to
  collapse multiple surface-hits of the same artifact.

The split between **filter_fields** (metadata you filter on) and **surfaces**
(text you embed) is deliberate and central: good retrieval is hard metadata
filtering *and* semantic ranking, not only embeddings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FilterFields = Mapping[str, Any]
"""Non-embedded, hard-filterable metadata for an artifact (name, owner, tags)."""


def storage_key(*parts: str) -> str:
    """Stable, filesystem-safe id from arbitrary string parts (truncated SHA-256)."""
    h = hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()
    return h[:24]


@dataclass
class Artifact:
    """A logical corpus item before decomposition into surfaces."""

    id: str
    raw: Any
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Surface:
    """One embeddable unit derived from an artifact.

    ``kind`` names the surface type (e.g. ``"description"``, ``"synopsis"``,
    ``"problem_class"``, ``"chunk"``) so a query can match the *right part* of
    an artifact. ``granularity`` is a coarse hint (``"document"`` / ``"chunk"``
    / ``"field"``). ``metadata`` is surface-local (e.g. chunk offsets).
    """

    artifact_id: str
    kind: str
    text: str
    granularity: str = "document"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class IndexPlan:
    """An :class:`~ir.strategy.IndexingStrategy`'s output for one artifact."""

    filter_fields: dict = field(default_factory=dict)
    surfaces: list[Surface] = field(default_factory=list)


@dataclass(frozen=True)
class Record:
    """A stored, embedded surface — one row of the index."""

    id: str
    artifact_id: str
    surface_kind: str
    surface_index: int
    text: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(artifact_id: str, surface_kind: str, surface_index: int) -> str:
        """Deterministic storage id for a surface of an artifact."""
        return storage_key(artifact_id, surface_kind, str(surface_index))


@dataclass(frozen=True)
class SearchHit:
    """A scored record returned by retrieval (higher score = closer)."""

    artifact_id: str
    surface_kind: str
    score: float
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def best_per_artifact(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Collapse hits to the highest-scoring surface per artifact.

    Returns the surviving hits sorted by score (descending).
    """
    seen: dict[str, SearchHit] = {}
    for h in hits:
        cur = seen.get(h.artifact_id)
        if cur is None or h.score > cur.score:
            seen[h.artifact_id] = h
    return sorted(seen.values(), key=lambda h: h.score, reverse=True)
