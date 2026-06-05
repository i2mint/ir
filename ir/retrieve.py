"""Retrieval — hard metadata filtering + dense brute-force ranking.

:func:`search` embeds the query, applies a hard metadata filter to narrow the
candidate set (the ``vd`` Mongo-style filter language — ownership, name, tags),
ranks the survivors by cosine similarity (exact brute force; correct and
instant at ``ir``'s corpus sizes), and collapses surface-hits to the best
surface per artifact.

Hybrid lexical fusion (BM25 + ``vd.reciprocal_rank_fusion``) and cross-encoder
reranking are deliberate next-step seams (see the retrieval issue): dense +
filter is the shippable baseline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .base import SearchHit, best_per_artifact


def _matches(metadata: Mapping[str, Any], filter: Mapping[str, Any]) -> bool:
    """Mongo-style metadata match, via ``vd`` when available."""
    try:
        from vd.filters import matches_filter

        return matches_filter(dict(metadata), dict(filter))
    except Exception:
        # Minimal equality fallback if vd is unavailable.
        return all(metadata.get(k) == v for k, v in filter.items())


def _embed_query(embedder, query: str) -> np.ndarray:
    from .index import _embed

    vec = _embed(embedder, [query], "query")[0]
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def search(
    corpus,
    query: str,
    *,
    k: int = 10,
    filter: Mapping[str, Any] | None = None,
    surfaces: Iterable[str] | None = None,
    per_artifact: bool = True,
) -> list[SearchHit]:
    """Return the top-*k* :class:`~ir.base.SearchHit` for *query*.

    Parameters
    ----------
    k : number of results.
    filter : a ``vd`` Mongo-style filter over record metadata (hard filter).
    surfaces : restrict to these surface kinds (e.g. ``{"description"}``).
    per_artifact : collapse to the best surface per artifact (default True).
    """
    ids, mat, metas = corpus.store.matrix()
    if not ids:
        return []

    surface_set = set(surfaces) if surfaces is not None else None
    candidates = []
    for j, m in enumerate(metas):
        if surface_set is not None and m["surface_kind"] not in surface_set:
            continue
        if filter is not None and not _matches(m.get("metadata", {}), filter):
            continue
        candidates.append(j)
    if not candidates:
        return []

    qv = _embed_query(corpus.embedder, query)
    sub = mat[candidates]
    scores = sub @ qv

    # Over-fetch before collapsing to artifacts so dedupe doesn't starve top-k.
    take = min(len(candidates), max(k * 5, 50) if per_artifact else k)
    order = np.argsort(-scores)[:take]

    hits = [
        SearchHit(
            artifact_id=metas[candidates[o]]["artifact_id"],
            surface_kind=metas[candidates[o]]["surface_kind"],
            score=float(scores[o]),
            text=metas[candidates[o]]["text"],
            metadata=metas[candidates[o]].get("metadata", {}),
        )
        for o in order
    ]
    if per_artifact:
        hits = best_per_artifact(hits)
    return hits[:k]
