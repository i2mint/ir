"""Retrieval — hard metadata filtering + dense / lexical / hybrid ranking.

:func:`search` embeds the query, applies a hard metadata filter to narrow the
candidate set (the ``vd`` Mongo-style filter language — ownership, name, tags),
then ranks the survivors. Three ranking modes share that one filtered candidate
set:

- ``"dense"`` (default) — exact brute-force cosine over the embedding matrix.
- ``"lexical"`` — Okapi BM25 over the candidates' text (``vd.bm25_lexical_search``).
- ``"hybrid"`` — dense and lexical fused by Reciprocal Rank Fusion
  (``vd.reciprocal_rank_fusion``), the rank-based fuse that sidesteps the
  cosine/BM25 score-scale mismatch.

Hybrid matters for short, identifier-heavy capability text (skill / package /
tool names), the regime where dense-only retrieval fails silently on exact
identifiers and rare terms. An optional ``rerank`` hook (any
:class:`ef.Reranker`) re-scores the top fused candidates with a cross-encoder;
it defaults to ``None`` so retrieval stays offline and API-free out of the box.

All ranking is exact brute force — correct and instant at ``ir``'s corpus
sizes — and surface-hits collapse to the best surface per artifact. Lexical and
fusion reuse ``vd``; if ``vd`` is unavailable, hybrid degrades to dense and
lexical returns no results (both with a warning), so a missing optional dep
never hard-fails a search.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .base import SearchHit, best_per_artifact

#: Ranking modes accepted by :func:`search`.
MODES = ("dense", "lexical", "hybrid")

#: RRF rank constant — the ``k`` of ``1 / (k + rank)`` (standard default 60).
DFLT_RRF_K = 60


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


def _filter_candidates(
    metas: list[dict],
    surfaces: Iterable[str] | None,
    filter: Mapping[str, Any] | None,
) -> list[int]:
    """Row indices surviving the surface restriction and hard metadata filter."""
    surface_set = set(surfaces) if surfaces is not None else None
    candidates = []
    for j, m in enumerate(metas):
        if surface_set is not None and m["surface_kind"] not in surface_set:
            continue
        if filter is not None and not _matches(m.get("metadata", {}), filter):
            continue
        candidates.append(j)
    return candidates


def _dense_ranked(
    corpus,
    mat: np.ndarray,
    ids: list[str],
    candidates: list[int],
    query: str,
    fetch: int,
) -> list[tuple[str, float]]:
    """Top-*fetch* ``(record_id, cosine)`` pairs by exact brute-force cosine."""
    qv = _embed_query(corpus.embedder, query)
    if qv.shape[0] != mat.shape[1]:
        raise ValueError(
            f"Query embedding dim {qv.shape[0]} != index dim {mat.shape[1]}. "
            f"The corpus was built with a different embedder than the one "
            f"querying it; rebuild the corpus or use its original embedder."
        )
    sub = mat[candidates]
    scores = sub @ qv
    order = np.argsort(-scores)[:fetch]
    return [(ids[candidates[o]], float(scores[o])) for o in order]


def _lexical_ranked(
    ids: list[str],
    metas: list[dict],
    candidates: list[int],
    query: str,
    fetch: int,
    bm25: Mapping[str, Any] | None,
) -> list[tuple[str, float]]:
    """Top-*fetch* ``(record_id, bm25)`` pairs via ``vd.bm25_lexical_search``.

    The candidate texts are exposed to ``vd`` as a zero-copy mapping view
    (``record_id -> obj`` with ``.text`` / ``.metadata``) so no vectors are
    duplicated and BM25 runs only over the already-filtered candidates. Returns
    ``[]`` (with a warning) if ``vd``'s lexical search is unavailable, letting
    hybrid degrade to dense.
    """
    try:
        from vd import bm25_lexical_search
    except Exception:
        warnings.warn(
            "vd.bm25_lexical_search unavailable; BM25 lexical ranking is "
            "skipped. Hybrid falls back to dense; lexical mode returns no "
            "results. Install `vd` for lexical/hybrid retrieval.",
            stacklevel=3,
        )
        return []

    collection = {
        ids[j]: SimpleNamespace(
            text=metas[j]["text"], metadata=metas[j].get("metadata", {})
        )
        for j in candidates
    }
    results = bm25_lexical_search(collection, query, limit=fetch, **(bm25 or {}))
    return [(r["id"], float(r["score"])) for r in results]


def _rrf_fuse(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    rrf_k: int,
    fetch: int,
) -> list[tuple[str, float]]:
    """Fuse dense + lexical rankings via ``vd.reciprocal_rank_fusion``.

    Falls back to the dense ranking if ``vd`` is unavailable or lexical is
    empty (RRF of a single non-empty list is just that list's order).
    """
    if not lexical:
        return dense[:fetch]
    try:
        from vd import reciprocal_rank_fusion
    except Exception:
        warnings.warn(
            "vd.reciprocal_rank_fusion unavailable; falling back to dense "
            "ranking for hybrid search.",
            stacklevel=3,
        )
        return dense[:fetch]

    fused = reciprocal_rank_fusion(
        [
            [{"id": rid, "score": s} for rid, s in dense],
            [{"id": rid, "score": s} for rid, s in lexical],
        ],
        k=rrf_k,
    )
    return [(item["id"], float(item["rrf_score"])) for item in fused[:fetch]]


def _apply_rerank(
    query: str,
    ranked: list[tuple[str, float]],
    meta_by_id: Mapping[str, dict],
    rerank: Callable,
    limit: int,
) -> list[tuple[str, float]]:
    """Re-score the ranked candidates with an :class:`ef.Reranker` via ``ef.rerank``.

    The full candidate list is handed to the reranker; it may reorder freely
    and the top ``limit`` are kept. The reranker score replaces the candidate
    score, and ``_rid`` rides along on each segment so the reordered output
    maps back to records.
    """
    from ef import rerank as ef_rerank

    segments = [{"text": meta_by_id[rid]["text"], "_rid": rid} for rid, _ in ranked]
    reordered = ef_rerank(query, segments, rerank, limit=limit)
    return [
        (s["_rid"], float(s.get("metadata", {}).get("rerank_score", 0.0)))
        for s in reordered
    ]


def search(
    corpus,
    query: str,
    *,
    k: int = 10,
    filter: Mapping[str, Any] | None = None,
    surfaces: Iterable[str] | None = None,
    per_artifact: bool = True,
    mode: str = "dense",
    rrf_k: int = DFLT_RRF_K,
    fetch_k: int | None = None,
    rerank: Callable | None = None,
    bm25: Mapping[str, Any] | None = None,
) -> list[SearchHit]:
    """Return the top-*k* :class:`~ir.base.SearchHit` for *query*.

    Parameters
    ----------
    k : number of results.
    filter : a ``vd`` Mongo-style filter over record metadata (hard filter).
    surfaces : restrict to these surface kinds (e.g. ``{"description"}``).
    per_artifact : collapse to the best surface per artifact (default True).
    mode : ``"dense"`` (default, cosine), ``"lexical"`` (BM25), or ``"hybrid"``
        (dense + BM25 fused by Reciprocal Rank Fusion). Hybrid is the strongest
        default for short, identifier-heavy text; ``"dense"`` is the historical
        behavior and is kept as the default for backward compatibility.
    rrf_k : the RRF rank constant for ``"hybrid"`` (standard default 60).
    fetch_k : candidate depth before fusion / reranking / dedupe
        (default ``max(k*5, 50)`` when collapsing per artifact, else ``k``).
    rerank : an optional :class:`ef.Reranker` (``(query, segments) -> scores``)
        that re-scores the ranked candidates after fusion. The reranker sees
        the full candidate list (up to ``fetch_k`` items) and may reorder it.
        Default ``None`` keeps retrieval offline (no model download / API call).
    bm25 : optional Okapi params forwarded to ``vd.bm25_lexical_search``
        (e.g. ``{"k1": 1.5, "b": 0.75}``).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    ids, mat, metas = corpus.store.matrix()
    if not ids:
        return []

    candidates = _filter_candidates(metas, surfaces, filter)
    if not candidates:
        return []

    fetch = fetch_k if fetch_k is not None else (max(k * 5, 50) if per_artifact else k)

    if mode == "lexical":
        ranked = _lexical_ranked(ids, metas, candidates, query, fetch, bm25)
    else:
        dense = _dense_ranked(corpus, mat, ids, candidates, query, fetch)
        if mode == "dense":
            ranked = dense
        else:  # hybrid
            lexical = _lexical_ranked(ids, metas, candidates, query, fetch, bm25)
            ranked = _rrf_fuse(dense, lexical, rrf_k, fetch)

    meta_by_id = {ids[j]: metas[j] for j in candidates}

    if rerank is not None and ranked:
        ranked = _apply_rerank(query, ranked, meta_by_id, rerank, fetch)

    hits = [
        SearchHit(
            artifact_id=meta_by_id[rid]["artifact_id"],
            surface_kind=meta_by_id[rid]["surface_kind"],
            score=score,
            text=meta_by_id[rid]["text"],
            metadata=meta_by_id[rid].get("metadata", {}),
        )
        for rid, score in ranked
    ]
    if per_artifact:
        hits = best_per_artifact(hits)
    return hits[:k]
