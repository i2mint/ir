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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

FilterFields = Mapping[str, Any]
"""Non-embedded, hard-filterable metadata for an artifact (name, owner, tags)."""

#: Metadata keys checked, in order, for a disclosure *pointer* — the file/dir
#: whose contents are an artifact's body. A :class:`SearchHit` is a
#: *pointer + snippet* (ir_09 §5): ``text`` is the snippet, the pointer is the
#: key a resource store dereferences to the full payload. Skills stamp
#: ``skill_path``; packages / reports / files stamp ``path`` (see
#: :mod:`ir.sources`). :mod:`ir.select` re-exports this for disclosure.
POINTER_KEYS = ("skill_path", "path")


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
    """A scored record returned by retrieval (higher score = closer).

    Maps onto ir_09's ``Result``: ``text`` is the snippet, ``score`` the rank
    score, ``metadata`` the meta, and :attr:`pointer` the key into a resource
    store (ir_09 §5). :meth:`to_dict` is the serialization-clean form for a
    cross-process / subagent boundary (no numpy scalars leak).

    ``source`` is the corpus/source name the hit came from (``None`` when
    unattributed — e.g. an ad-hoc corpus without a name). It is a first-class
    field, not a metadata key, because ``metadata`` is the strategy-owned
    hard-filter namespace and provenance is structural: artifact identity is
    only unique *within* a source, so any cross-source operation keys on
    ``(source, artifact_id)`` (see :func:`best_per_artifact`).
    """

    artifact_id: str
    surface_kind: str
    score: float
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def pointer(self) -> str | None:
        """The disclosure pointer on this hit, if any (see :data:`POINTER_KEYS`)."""
        for key in POINTER_KEYS:
            p = self.metadata.get(key)
            if p:
                return p
        return None

    def to_dict(self) -> dict:
        """JSON-serializable form (``score`` cast to a Python ``float``)."""
        return {
            "artifact_id": self.artifact_id,
            "surface_kind": self.surface_kind,
            "score": float(self.score),
            "text": self.text,
            "metadata": dict(self.metadata),
            "source": self.source,
        }


def best_per_artifact(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Collapse hits to the highest-scoring surface per artifact.

    Returns the surviving hits sorted by score (descending). Identity is
    ``(source, artifact_id)``: the same id in two different sources names two
    different artifacts (the skills-corpus "dol" is not the packages-corpus
    "dol"), so cross-source input never collapses them. Single-source input
    (all hits sharing one ``source``, or all ``None``) behaves exactly as an
    id-keyed collapse. Note the raw-score comparison and the final sort assume
    one score scale — sound within a source; for mixed-source hits use
    :func:`ir.retrieve.fuse_hits`, which only compares scores within a source.
    """
    seen: dict[tuple[str | None, str], SearchHit] = {}
    for h in hits:
        key = (h.source, h.artifact_id)
        cur = seen.get(key)
        if cur is None or h.score > cur.score:
            seen[key] = h
    return sorted(seen.values(), key=lambda h: h.score, reverse=True)


def tag_source(hits: Sequence[SearchHit], source: str | None) -> list[SearchHit]:
    """Stamp *source* on every hit that doesn't already carry one.

    Existing tags win: a hit already attributed to a corpus keeps that
    attribution (so re-tagging under a different registry key cannot
    double-count one corpus as two sources). A ``None`` source is the
    untagged pseudo-source — hits pass through unattributed.
    """
    return [h if h.source is not None else replace(h, source=source) for h in hits]
