"""Indexing strategies — the "what do we index?" seam.

An :class:`IndexingStrategy` decomposes one artifact into an
:class:`~ir.base.IndexPlan`: the ``filter_fields`` (hard-filterable metadata,
*not* embedded) and a list of :class:`~ir.base.Surface` (embeddable units). This
is the central extensibility point of ``ir``: a naive corpus uses
:class:`WholeText`; a structured corpus (a package) decomposes into several
heterogeneous surfaces so a query can match the *right part* of an artifact,
and constrains candidates by metadata *before* semantic ranking.

Shipped strategies:

- :class:`WholeText` — one surface = the whole text. The out-of-the-box default.
- :class:`Chunked` — split the text into overlapping chunks (one surface each).
- :class:`Skill` — embed ``name + description`` only (the body stays on disk,
  per the capability-discovery research); name/parent become filter fields.
- :class:`Package` — ``name + description`` plus README chunks as surfaces;
  name/owner/deps become filter fields (AI synopsis / problem-class surfaces
  are a documented extension).

Every strategy is a plain callable-ish object with a ``decompose`` method, so
custom strategies need only match the :class:`IndexingStrategy` protocol.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .base import IndexPlan, Surface


@runtime_checkable
class IndexingStrategy(Protocol):
    """Decompose one artifact into filter fields + embeddable surfaces."""

    def decompose(
        self, artifact_id: str, raw: Any, metadata: Mapping[str, Any] | None = None
    ) -> IndexPlan: ...


def _text_of(raw: Any, text_key: str | None = None) -> str:
    """Best-effort text extraction from a raw artifact payload."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        if text_key is not None:
            return str(raw.get(text_key, "") or "")
        if "text" in raw:
            return str(raw.get("text", "") or "")
        # join string-valued fields as a fallback
        return "\n".join(str(v) for v in raw.values() if isinstance(v, str))
    return str(raw)


def _split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-packing chunker: greedily fill ~``chunk_size`` chunks.

    Splits on blank lines, then packs whole paragraphs up to ``chunk_size``
    (carrying an ``overlap`` tail between chunks). Paragraphs longer than
    ``chunk_size`` are hard-split. This packs to a target size rather than
    emitting one chunk per paragraph.

    Never emits blank chunks: hard-splitting a paragraph that contains a long
    whitespace run (no blank line, so the paragraph regex doesn't split it)
    can slice out whitespace-only pieces — those are skipped, so every chunk
    is embeddable and per-chunk metadata (``chunk_index`` / ``n_chunks``)
    counts only real chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        if len(para) > chunk_size:  # hard-split an oversized paragraph
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(1, chunk_size - overlap)
            chunks.extend(
                piece
                for i in range(0, len(para), step)
                if (piece := para[i : i + chunk_size]).strip()
            )
            continue
        if cur and len(cur) + 2 + len(para) > chunk_size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail}\n\n{para}".strip() if tail else para
        else:
            cur = f"{cur}\n\n{para}".strip() if cur else para
    if cur:
        chunks.append(cur)
    return chunks


class WholeText:
    """One surface = the entire text. Sensible default for a naive corpus."""

    def __init__(self, *, text_key: str | None = None, kind: str = "document"):
        self.text_key = text_key
        self.kind = kind

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        text = _text_of(raw, self.text_key)
        surfaces = (
            [Surface(artifact_id, self.kind, text, granularity="document")]
            if text.strip()
            else []
        )
        return IndexPlan(filter_fields=meta, surfaces=surfaces)


class Chunked:
    """Split the artifact's text into overlapping chunk surfaces."""

    def __init__(
        self,
        *,
        chunk_size: int = 1200,
        overlap: int = 200,
        text_key: str | None = None,
        kind: str = "chunk",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_key = text_key
        self.kind = kind

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        text = _text_of(raw, self.text_key)
        chunks = _split(text, chunk_size=self.chunk_size, overlap=self.overlap)
        surfaces = [
            Surface(
                artifact_id,
                self.kind,
                chunk,
                granularity="chunk",
                metadata={"chunk_index": i, "n_chunks": len(chunks)},
            )
            for i, chunk in enumerate(chunks)
        ]
        return IndexPlan(filter_fields=meta, surfaces=surfaces)


class Skill:
    """Capability strategy: embed ``name + description`` only.

    The body (SKILL.md) is loaded post-selection and is *not* indexed; name and
    parent are filter fields.
    """

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        raw = raw if isinstance(raw, Mapping) else {"name": str(raw)}
        name = str(raw.get("name", artifact_id))
        description = str(raw.get("description", "") or "")
        filter_fields = {
            "name": name,
            "parent": raw.get("parent"),
            **meta,
        }
        text = f"{name}\n\n{description}".strip()
        surfaces = [Surface(artifact_id, "capability", text, granularity="field")]
        return IndexPlan(filter_fields=filter_fields, surfaces=surfaces)


class Package:
    """Package strategy: ``name + description`` surface plus README chunks.

    Filter fields capture ownership (ours vs third-party), name, deps. AI
    synopsis / problem-class surfaces are a documented extension point.

    Surface indexing: the ``description`` surface (kept whenever *name* or
    *description* is non-empty) occupies plan position 0, so ``readme_chunk``
    *j* is stored with
    ``Record.surface_index == j + 1`` while its surface metadata says
    ``chunk_index == j`` — ``surface_index`` is plan-global, ``chunk_index``
    per-kind (see :meth:`ir.base.Record.make_id`). Never derive sibling record
    ids from ``chunk_index``; use the ledger
    (:func:`ir.retrieve.records_for_artifact`). ``n_chunks`` is stamped on
    readme chunks at decompose time, but corpora built before the stamp keep
    records without it until the artifact re-indexes (content / embedder /
    strategy change) — read it with ``metadata.get("n_chunks")``.
    """

    def __init__(self, *, chunk_size: int = 1500, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        raw = raw if isinstance(raw, Mapping) else {"name": str(raw)}
        name = str(raw.get("name", artifact_id))
        description = str(raw.get("description", "") or "")
        readme = str(raw.get("readme", "") or "")
        filter_fields = {
            "name": name,
            "owner": raw.get("owner"),
            "has_readme": bool(readme.strip()),
            "deps": list(raw.get("deps", []) or []),
            **meta,
        }
        surfaces = [
            Surface(
                artifact_id,
                "description",
                f"{name}: {description}".strip(": ").strip(),
                granularity="field",
            )
        ]
        chunks = _split(readme, chunk_size=self.chunk_size, overlap=self.overlap)
        for i, chunk in enumerate(chunks):
            surfaces.append(
                Surface(
                    artifact_id,
                    "readme_chunk",
                    chunk,
                    granularity="chunk",
                    metadata={"chunk_index": i, "n_chunks": len(chunks)},
                )
            )
        # Drop empty surfaces (e.g. no description and no README).
        surfaces = [s for s in surfaces if s.text.strip()]
        return IndexPlan(filter_fields=filter_fields, surfaces=surfaces)
