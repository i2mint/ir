"""Retrieval — hard metadata filtering + dense / lexical / hybrid ranking.

:func:`search` embeds the query, applies a hard metadata filter to narrow the
candidate set (the ``vd`` Mongo-style filter language — ownership, name, tags),
then ranks the survivors. Three ranking modes share that one filtered candidate
set:

- ``"dense"`` (default) — exact brute-force cosine over the embedding matrix.
- ``"lexical"`` — Okapi BM25 over the candidates' text (``vd.bm25_lexical_search``).
- ``"hybrid"`` — dense and lexical fused, either by Reciprocal Rank Fusion
  (``fusion="rrf"``, the default ``vd.reciprocal_rank_fusion`` rank-based fuse
  that sidesteps the cosine/BM25 score-scale mismatch) or by a
  magnitude-preserving convex blend (``fusion="blend"``) that keeps the dense
  cosine's absolute scale for better abstention separability (see ir_08).

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

import json
import warnings
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .base import SearchHit, best_per_artifact
from .formulate import Formulator

#: Ranking modes accepted by :func:`search`.
MODES = ("dense", "lexical", "hybrid")

#: Hybrid fusion methods accepted by :func:`search` (``mode="hybrid"``).
#: ``"rrf"`` is the rank-based default; ``"blend"`` preserves score magnitude
#: (see :func:`_blend_fuse` and ir_08) for better abstention separability.
FUSIONS = ("rrf", "blend")

#: RRF rank constant — the ``k`` of ``1 / (k + rank)`` (standard default 60).
DFLT_RRF_K = 60

#: Dense weight in the ``"blend"`` fusion convex combination (``1-alpha`` on the
#: bounded lexical term). ``0.5`` weighs the two equally.
DFLT_BLEND_ALPHA = 0.5

#: BM25 saturation constant for ``"blend"`` fusion: ``bm25 -> bm25/(bm25+k)``, a
#: bounded squash to ``[0, 1)`` that needs no *per-query* normalization (which
#: would erase the absolute-magnitude signal abstention calibration depends on).
DFLT_BM25_SAT_K = 8.0


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


def _candidate_key(
    surfaces: tuple[str, ...] | None, filter: Mapping[str, Any] | None
) -> tuple:
    """A hashable key identifying a candidate set (surface restriction + filter).

    For a fixed :class:`~ir.index.Corpus` instance the candidate set is fully
    determined by the surface restriction and the hard metadata filter, so two
    searches sharing both can share one BM25 index. The filter is canonicalized
    with sorted keys so logically-equal filters land on the same cache entry.
    """
    f = json.dumps(filter, sort_keys=True, default=str) if filter is not None else None
    return (surfaces, f)


def _bm25_index_for(corpus, ids, metas, candidates, cache_key):
    """A ``vd.BM25Index`` over the candidate texts, cached on the corpus instance.

    BM25's term statistics (document frequencies, lengths) are
    *query-independent*, so the index is built once per candidate set and reused
    across queries. Rebuilding it on every query — re-tokenizing every document
    each time — is the lexical/hybrid bottleneck for batch evaluation and the
    reason large corpora did not scale. The cache lives on the corpus instance,
    which is immutable during its lifetime and freshly created whenever the
    corpus is (re)built or reopened, so the cache never goes stale.

    The candidate texts are exposed to ``vd`` as a zero-copy mapping view
    (``record_id -> obj`` with ``.text`` / ``.metadata``) so no vectors are
    duplicated and BM25 covers only the already-filtered candidates. Returns
    ``None`` (with a warning) if ``vd`` is unavailable, letting lexical mode
    return nothing and hybrid degrade to dense.
    """
    try:
        from vd import BM25Index
    except Exception:
        warnings.warn(
            "vd.BM25Index unavailable; BM25 lexical ranking is skipped. Hybrid "
            "falls back to dense; lexical mode returns no results. Install `vd` "
            "for lexical/hybrid retrieval.",
            stacklevel=3,
        )
        return None

    cache = getattr(corpus, "_bm25_index_cache", None)
    if cache is None:
        cache = {}
        try:
            corpus._bm25_index_cache = cache
        except Exception:
            pass  # corpus doesn't admit attribute caching → build fresh each call
    if cache_key in cache:
        return cache[cache_key]

    collection = {
        ids[j]: SimpleNamespace(
            text=metas[j]["text"], metadata=metas[j].get("metadata", {})
        )
        for j in candidates
    }
    index = BM25Index(collection)
    cache[cache_key] = index
    return index


def _lexical_ranked(
    index,
    query: str,
    fetch: int,
    bm25: Mapping[str, Any] | None,
) -> list[tuple[str, float]]:
    """Top-*fetch* ``(record_id, bm25_score)`` pairs from a prebuilt BM25 index.

    ``index`` is a ``vd.BM25Index`` (see :func:`_bm25_index_for`) or ``None``
    when ``vd`` is unavailable, in which case lexical ranking is empty.
    """
    if index is None:
        return []
    results = index.search(query, limit=fetch, **(bm25 or {}))
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


def _blend_fuse(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    alpha: float,
    bm25_sat_k: float,
    fetch: int,
) -> list[tuple[str, float]]:
    """Fuse dense + lexical by a magnitude-preserving convex blend.

    ``fused = alpha * cosine + (1 - alpha) * bm25 / (bm25 + bm25_sat_k)``.

    Unlike :func:`_rrf_fuse` (rank-only — every query's top hit collapses to ~the
    same fused score, which destroys the score-distribution separability that
    abstention calibration relies on; see ir_07/ir_08), this keeps the dense
    cosine's **absolute** magnitude. The dense term carries the in-scope /
    out-of-scope signal (it is low across the board for an irrelevant query),
    while the BM25 term is squashed into a bounded ``[0, 1)`` range with a *fixed*
    constant — deliberately **not** per-query min-max normalized, which would
    rescale every query's best hit to 1.0 and wash that signal out. Cosine is
    already on a fixed ``[-1, 1]`` scale, so the two terms are commensurable.

    Falls back to the single non-empty ranking (dense, when ``vd`` is missing) so
    it degrades exactly like :func:`_rrf_fuse`.
    """
    if not lexical:
        return dense[:fetch]
    if not dense:
        return lexical[:fetch]
    dense_score = dict(dense)
    lex_score = dict(lexical)
    # Preserve first-seen order for deterministic tie-breaking (dense first).
    ids = list(dict.fromkeys([rid for rid, _ in dense] + [rid for rid, _ in lexical]))
    fused = []
    for rid in ids:
        d = dense_score.get(rid, 0.0)
        b = lex_score.get(rid, 0.0)
        b_sat = b / (b + bm25_sat_k) if b > 0 else 0.0
        fused.append((rid, alpha * d + (1.0 - alpha) * b_sat))
    fused.sort(key=lambda kv: -kv[1])
    return fused[:fetch]


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
    fusion: str = "rrf",
    rrf_k: int = DFLT_RRF_K,
    alpha: float = DFLT_BLEND_ALPHA,
    bm25_sat_k: float = DFLT_BM25_SAT_K,
    fetch_k: int | None = None,
    rerank: Callable | None = None,
    bm25: Mapping[str, Any] | None = None,
    formulate: Formulator | None = None,
) -> list[SearchHit]:
    """Return the top-*k* :class:`~ir.base.SearchHit` for *query*.

    Parameters
    ----------
    k : number of results.
    filter : a ``vd`` Mongo-style filter over record metadata (hard filter).
    surfaces : restrict to these surface kinds (e.g. ``{"description"}``).
    per_artifact : collapse to the best surface per artifact (default True).
    mode : ``"dense"`` (default, cosine), ``"lexical"`` (BM25), or ``"hybrid"``
        (dense + BM25 fused). Hybrid is the strongest default for short,
        identifier-heavy text; ``"dense"`` is the historical behavior and is
        kept as the default for backward compatibility.
    fusion : how ``"hybrid"`` fuses dense + lexical — ``"rrf"`` (default,
        rank-based Reciprocal Rank Fusion) or ``"blend"`` (magnitude-preserving
        convex blend; better abstention separability — see ir_08). Ignored for
        non-hybrid modes.
    rrf_k : the RRF rank constant for ``fusion="rrf"`` (standard default 60).
    alpha : dense weight for ``fusion="blend"`` (``1-alpha`` on lexical).
    bm25_sat_k : BM25 saturation constant for ``fusion="blend"``.
    fetch_k : candidate depth before fusion / reranking / dedupe
        (default ``max(k*5, 50)`` when collapsing per artifact, else ``k``).
    rerank : an optional :class:`ef.Reranker` (``(query, segments) -> scores``)
        that re-scores the ranked candidates after fusion. The reranker sees
        the full candidate list (up to ``fetch_k`` items) and may reorder it.
        Default ``None`` keeps retrieval offline (no model download / API call).
    bm25 : optional Okapi params forwarded to ``vd.bm25_lexical_search``
        (e.g. ``{"k1": 1.5, "b": 0.75}``).
    formulate : an optional :data:`~ir.formulate.Formulator` (``query -> str |
        [str, ...]``) applied *before* retrieval — rewrite / expand / HyDE
        (ir_09 §3). Identity by default (embed the query verbatim). When it
        returns several queries, ir runs each and fuses the results (best surface
        per artifact). See :func:`ir.make_llm_formulator`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if fusion not in FUSIONS:
        raise ValueError(f"unknown fusion {fusion!r}; expected one of {FUSIONS}")

    # Materialize once: surfaces may be a one-shot iterable, and it is both the
    # filter input and part of the BM25 cache key.
    surfaces = tuple(surfaces) if surfaces is not None else None

    # Query formulation (ir_09 §3): expand/rewrite before retrieval. A single
    # query continues the normal path; multiple queries fan out and fuse. The
    # sub-searches pass formulate=None (no re-formulation) and per_artifact=False
    # so the cross-query merge collapses to the best surface per artifact once.
    if formulate is not None:
        produced = formulate(query)
        queries = (
            [produced]
            if isinstance(produced, str)
            else [q for q in produced if isinstance(q, str) and q.strip()]
        )
        if not queries:
            queries = [query]
        if len(queries) > 1:
            merged: list[SearchHit] = []
            for q in queries:
                merged.extend(
                    search(
                        corpus,
                        q,
                        k=k,
                        filter=filter,
                        surfaces=surfaces,
                        per_artifact=False,
                        mode=mode,
                        fusion=fusion,
                        rrf_k=rrf_k,
                        alpha=alpha,
                        bm25_sat_k=bm25_sat_k,
                        fetch_k=fetch_k,
                        rerank=rerank,
                        bm25=bm25,
                    )
                )
            merged = (
                best_per_artifact(merged)
                if per_artifact
                else sorted(merged, key=lambda h: h.score, reverse=True)
            )
            return merged[:k]
        query = queries[0]

    ids, mat, metas = corpus.store.matrix()
    if not ids:
        return []

    candidates = _filter_candidates(metas, surfaces, filter)
    if not candidates:
        return []

    fetch = fetch_k if fetch_k is not None else (max(k * 5, 50) if per_artifact else k)

    if mode == "lexical":
        index = _bm25_index_for(
            corpus, ids, metas, candidates, _candidate_key(surfaces, filter)
        )
        ranked = _lexical_ranked(index, query, fetch, bm25)
    else:
        dense = _dense_ranked(corpus, mat, ids, candidates, query, fetch)
        if mode == "dense":
            ranked = dense
        else:  # hybrid
            index = _bm25_index_for(
                corpus, ids, metas, candidates, _candidate_key(surfaces, filter)
            )
            lexical = _lexical_ranked(index, query, fetch, bm25)
            if fusion == "blend":
                ranked = _blend_fuse(dense, lexical, alpha, bm25_sat_k, fetch)
            else:
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


# =========================================================================== #
# Retriever adapter — ir's leaf in a Composable Search Agent (ir_09 §3)
# =========================================================================== #

#: A retriever: ``(query, **overrides) -> list[SearchHit]`` — ir_09's Retriever
#: leaf (one query, one corpus) as a swappable callable. :func:`as_retriever`
#: binds one corpus to this contract so an orchestration layer (e.g. ``raglab``)
#: can register an ir corpus as one source key without importing ir internals.
Retriever = Callable[..., "list[SearchHit]"]


def as_retriever(corpus_or_name, **search_defaults) -> Retriever:
    """Bind ONE corpus to the uniform :data:`Retriever` contract.

    Returns ``retrieve(query, **overrides) -> list[SearchHit]`` that calls
    :func:`search` with ``search_defaults`` (a per-call kwarg overrides a bound
    default). A corpus *name* is resolved once via :func:`ir.open_corpus`; pass
    an open :class:`~ir.index.Corpus` to skip that. The returned callable carries
    the bound corpus on ``.corpus`` for introspection.

    >>> retr = as_retriever(corpus, mode="hybrid", k=20)   # doctest: +SKIP
    >>> hits = retr("how do I deploy the app")             # doctest: +SKIP
    >>> hits = retr("deploy", filter={"owner": "me"})      # doctest: +SKIP
    """
    from .index import open_corpus

    corpus = (
        open_corpus(corpus_or_name)
        if isinstance(corpus_or_name, str)
        else corpus_or_name
    )

    def retrieve(query: str, **overrides) -> list[SearchHit]:
        return search(corpus, query, **{**search_defaults, **overrides})

    retrieve.corpus = corpus
    return retrieve
