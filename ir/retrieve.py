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

Beyond the within-corpus channel fusion above, :func:`fuse_hits` merges ranked
hit lists from **different sources** (corpora / embedders / modes) by weighted
Reciprocal Rank Fusion — raw scores never cross a source boundary, only ranks
do. It is the shared cross-source merge primitive consumed by federated
:func:`ir.discover` and by an orchestration layer's fan-in reranker (ir_09 §3).
Every search hit carries its corpus name as :attr:`~ir.base.SearchHit.source`
and its surface's plan position as :attr:`~ir.base.SearchHit.surface_index`.

Two optional ranking signals act inside :func:`search`: ``surface_weights``
multiplies the first-stage score per surface kind, and ``dep_boost`` adds a
structural score per candidate after the reranker (the shipped
:func:`dependency_evidence_boost` rewards dependencies on a caller-supplied
library set). Both default to off.

:func:`records_for_artifact` is the hit-operation beneath retrieval-time
context expansion: given a hit's ``artifact_id``, it returns *all* of that
artifact's stored records (its sibling surfaces), ordered — resolved through
the ledger, never by re-deriving record ids. The expansion operator itself
(:func:`ir.expand.expand` — retrieve → expand → rerank) builds on it.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .base import Record, SearchHit, best_per_artifact, ledger_key, tag_source
from .embed import EmbedderMismatch
from .formulate import Formulator
from .rerank import DFLT_RERANK_MAX_CHARS, passage_texts, resolve_reranker

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
        name = getattr(corpus, "name", "<name>")
        built_with = getattr(corpus, "embedder_id", "") or "unknown"
        raise EmbedderMismatch(
            f"Query embedding dim {qv.shape[0]} != index dim {mat.shape[1]}: "
            f"corpus {name!r} was built with embedder {built_with!r} but is "
            f"being queried with a different embedder. Rebuild it with the embedder "
            f"you query with (registered corpora: `ir build {name}`; corpora built "
            f"from Python: `ir.build(...)` again); or query without embeddings: "
            f"`ir search {name} <query> --mode lexical`."
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
    fetch: int | None,
) -> list[tuple[str, float]]:
    """Fuse dense + lexical rankings via ``vd.reciprocal_rank_fusion``.

    Falls back to the dense ranking if ``vd`` is unavailable or lexical is
    empty (RRF of a single non-empty list is just that list's order). A
    ``fetch`` of ``None`` keeps the whole fused union.
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
    fetch: int | None,
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


def _check_surface_weights(
    surface_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate ``surface_weights``: finite, non-negative numbers keyed by kind."""
    if not surface_weights:
        return {}
    checked = {}
    for kind, weight in surface_weights.items():
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                f"surface_weights[{kind!r}] = {weight!r}; a surface weight must be a "
                f"finite, non-negative number (use surfaces= to exclude a kind)."
            )
        checked[str(kind)] = float(weight)
    return checked


def _weight_surfaces(
    ranked: list[tuple[str, float]],
    meta_by_id: Mapping[str, dict],
    weights: Mapping[str, float],
) -> list[tuple[str, float]]:
    """Multiply each score by its surface kind's weight (default 1), best first.

    The sort is stable, so equally scored candidates keep their first-stage order.
    """
    weighted = [
        (rid, score * weights.get(meta_by_id[rid]["surface_kind"], 1.0))
        for rid, score in ranked
    ]
    weighted.sort(key=lambda kv: -kv[1])
    return weighted


#: A ranking-time score boost: ``hit -> float``, added to each candidate's score
#: after fusion and the reranker, before the per-artifact collapse. See
#: :func:`dependency_evidence_boost`.
ScoreBoost = Callable[[SearchHit], float]


def _boosted(hits: list[SearchHit], boost: ScoreBoost) -> list[SearchHit]:
    """``hits`` with ``boost(hit)`` added to each score, best first (stable)."""
    out = []
    for hit in hits:
        extra = boost(hit)
        if (
            isinstance(extra, bool)
            or not isinstance(extra, (int, float))
            or not math.isfinite(extra)
        ):
            raise ValueError(
                f"dep_boost returned {extra!r} for {hit.artifact_id!r}; "
                f"expected a finite number."
            )
        out.append(replace(hit, score=hit.score + float(extra)) if extra else hit)
    return sorted(out, key=lambda h: h.score, reverse=True)


def _norm_dist_name(name: str) -> str:
    """A bare distribution name, lower-cased and PEP 503-normalized."""
    from .graph import _dep_name

    return re.sub(r"[-_.]+", "-", _dep_name(str(name)))


def dependency_evidence_boost(
    relevant_libs: Iterable[str],
    *,
    weight: float,
    saturate: int = 1,
    deps_field: str = "deps",
) -> ScoreBoost:
    """A ``dep_boost`` for :func:`search`: reward dependencies on *relevant_libs*.

    Reads the candidate's ``deps`` filter field (as :class:`~ir.strategy.Package`
    stores it), reduces each spec to its bare, PEP 503-normalized name, and
    returns ``weight * min(n, saturate) / saturate`` where ``n`` counts the
    dependencies found in *relevant_libs*. With the default ``saturate=1`` a
    single relevant dependency earns the full ``weight``. A structural signal,
    cheap and model-free, that a prose reranker cannot be fooled out of: an
    async-task package whose README says "DAG" does not depend on graph
    libraries.

    *relevant_libs* is the **caller's** call: which libraries matter for a goal is
    a planning decision that belongs to the agent layer, never to ``ir``.
    ``weight`` is added to the score on the search's scale (an RRF fused score is
    near ``0.03``, a cosine near ``0.5``, a sigmoid reranker in ``(0, 1)``), so
    tune it per configuration.

    >>> from ir.base import SearchHit
    >>> boost = dependency_evidence_boost({"networkx"}, weight=0.1)
    >>> boost(SearchHit("unbox", "description", 0.4, "", {"deps": ["networkx>=3"]}))
    0.1
    >>> boost(SearchHit("au", "description", 0.5, "", {"deps": ["anyio"]}))
    0.0
    """
    if not math.isfinite(weight):
        raise ValueError(f"weight must be a finite number, got {weight!r}")
    if not isinstance(saturate, int) or saturate < 1:
        raise ValueError(f"saturate must be a positive int, got {saturate!r}")
    libs = frozenset(_norm_dist_name(lib) for lib in relevant_libs)

    def boost(hit: SearchHit) -> float:
        deps = hit.metadata.get(deps_field) or ()
        if isinstance(deps, str):
            deps = [deps]
        names = {_norm_dist_name(d) for d in deps}
        n = len(names & libs)
        return weight * min(n, saturate) / saturate if n else 0.0

    return boost


def _apply_rerank(
    corpus,
    query: str,
    ranked: list[tuple[str, float]],
    meta_by_id: Mapping[str, dict],
    rerank: Callable,
    limit: int,
    *,
    rerank_text: Any = "surface",
    max_chars: int = DFLT_RERANK_MAX_CHARS,
) -> list[tuple[str, float]]:
    """Re-score the ranked candidates with an :class:`ef.Reranker` via ``ef.rerank``.

    The reranker reads :func:`ir.rerank.passage_texts` of each candidate (the matched
    surface by default) and may reorder freely; the top ``limit`` are kept and its score
    replaces the candidate score. Candidates that share a passage (every surface of one
    artifact under ``rerank_text="artifact"``) are scored once and share the score, in
    their first-stage order, so a cross-encoder is never run twice on the same text.
    """
    from ef import rerank as ef_rerank

    texts = passage_texts(
        corpus, [meta_by_id[rid] for rid, _ in ranked], rerank_text, max_chars=max_chars
    )
    groups: dict[str, list[str]] = {}
    for text, (rid, _) in zip(texts, ranked, strict=True):
        groups.setdefault(text, []).append(rid)
    segments = [{"text": text, "_group": text} for text in groups]
    reordered = ef_rerank(query, segments, rerank)
    out: list[tuple[str, float]] = []
    for segment in reordered:
        score = float(segment.get("metadata", {}).get("rerank_score", 0.0))
        out.extend((rid, score) for rid in groups[segment["_group"]])
    return out[:limit]


def _distinct_surfaces(hits: list[SearchHit]) -> list[SearchHit]:
    """``hits`` with each ``(artifact, surface)`` once, at its best score, best first."""
    best: dict[tuple, SearchHit] = {}
    for hit in hits:
        key = (hit.artifact_id, hit.surface_kind, hit.surface_index)
        if key not in best or hit.score > best[key].score:
            best[key] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def _rerank_hits(
    corpus,
    query: str,
    hits: list[SearchHit],
    rerank: Callable,
    *,
    rerank_text: Any,
    max_chars: int,
) -> list[SearchHit]:
    """Rerank already-built hits (the multi-query merge) against ``query``.

    Hits are keyed by position; the reranker sees the same passages
    :func:`_apply_rerank` would give it, and each hit keeps its surface but takes the
    reranker's score.
    """
    metas = {
        str(i): {
            "artifact_id": h.artifact_id,
            "surface_kind": h.surface_kind,
            "surface_index": h.surface_index,
            "text": h.text,
            "metadata": h.metadata,
        }
        for i, h in enumerate(hits)
    }
    ranked = _apply_rerank(
        corpus,
        query,
        [(rid, h.score) for rid, h in zip(metas, hits, strict=True)],
        metas,
        rerank,
        len(hits),
        rerank_text=rerank_text,
        max_chars=max_chars,
    )
    return [replace(hits[int(rid)], score=score) for rid, score in ranked]


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
    rerank: Callable | str | None = None,
    rerank_text: str | Callable = "surface",
    rerank_max_chars: int = DFLT_RERANK_MAX_CHARS,
    bm25: Mapping[str, Any] | None = None,
    formulate: Formulator | None = None,
    surface_weights: Mapping[str, float] | None = None,
    dep_boost: ScoreBoost | None = None,
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
        (default ``max(k*5, 50)`` when collapsing per artifact or reranking, else ``k``).
    rerank : an optional :class:`ef.Reranker` (``(query, segments) -> scores``)
        that re-scores the ranked candidates after fusion -- the precision stage.
        The reranker sees the full candidate list (up to ``fetch_k`` items) and may
        reorder it; reranked hits carry the *reranker's* score, which is on its own
        scale (see :mod:`ir.rerank`, and :func:`ir.rerank.calibration_key` for
        abstention floors). Pass ``"cross-encoder"`` (or ``"cross-encoder:<model>"``)
        for ef's sentence-transformers cross-encoder, imported lazily; when that
        extra (``ir[rerank]``) is missing the search warns once and keeps the
        first-stage ranking. Default ``None`` keeps retrieval offline (no model
        download / API call).
    rerank_text : what the reranker reads for each candidate -- ``"surface"``
        (default; the matched surface's text), ``"artifact"`` (the artifact's
        sibling surfaces joined in plan order, so a thin description is judged
        against its full README), or a callable ``(corpus, meta) -> str``.
    rerank_max_chars : cap on one ``"artifact"`` passage's length.
    bm25 : optional Okapi params forwarded to ``vd.bm25_lexical_search``
        (e.g. ``{"k1": 1.5, "b": 0.75}``).
    formulate : an optional :data:`~ir.formulate.Formulator` (``query -> str |
        [str, ...]``) applied *before* retrieval — rewrite / expand / HyDE
        (ir_09 §3). Identity by default (embed the query verbatim). When it
        returns several queries, ir runs each and fuses the results (best surface
        per artifact). See :func:`ir.make_llm_formulator`.
    surface_weights : optional ``{surface_kind: weight}`` multipliers, e.g.
        ``{"deps": 2.0}`` to let a high-precision ``deps`` surface outrank prose
        chunks. Each surface-hit's first-stage score (the fused score in hybrid
        mode) is multiplied by its kind's weight (default 1.0) before the
        ``fetch_k`` cut, the reranker and the per-artifact collapse; kinds the
        corpus lacks are ignored. Weights must be finite and non-negative. Meant
        for the non-negative scores of hybrid fusion and BM25 (a negative dense
        cosine moves *down* when multiplied by a weight above 1). Weighted scores
        are on a different scale from unweighted ones, so an abstention floor
        calibrated without the weights does not apply to them. Default ``None``
        weighs every surface equally, as before.
    dep_boost : an optional :data:`ScoreBoost` (``hit -> float``) whose value is
        added to each fetched candidate's score after fusion and the reranker,
        before the per-artifact collapse -- a structural signal next to the
        reranker's textual one. :func:`dependency_evidence_boost` builds one from
        a caller-supplied library set. It reorders the fetched candidates (raise
        ``fetch_k`` to show it more); boosted scores need their own abstention
        floor. Default ``None``: no boost.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    rerank = resolve_reranker(rerank)  # str -> callable | None (graceful fallback)
    asked = query  # the reranker judges against what the caller asked, not a rewrite
    if fusion not in FUSIONS:
        raise ValueError(f"unknown fusion {fusion!r}; expected one of {FUSIONS}")
    weights = _check_surface_weights(surface_weights)

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
            # A reranker needs its candidate pool to over-fetch, and must judge the
            # merged pool against the original query: the sub-searches run without it.
            depth = (
                (fetch_k if fetch_k is not None else max(k * 5, 50))
                if rerank is not None or dep_boost is not None
                else k
            )
            merged: list[SearchHit] = []
            for q in queries:
                merged.extend(
                    search(
                        corpus,
                        q,
                        k=depth,
                        filter=filter,
                        surfaces=surfaces,
                        per_artifact=False,
                        mode=mode,
                        fusion=fusion,
                        rrf_k=rrf_k,
                        alpha=alpha,
                        bm25_sat_k=bm25_sat_k,
                        fetch_k=fetch_k,
                        bm25=bm25,
                        surface_weights=weights,
                    )
                )
            if rerank is not None and merged:
                merged = _rerank_hits(
                    corpus,
                    asked,
                    _distinct_surfaces(merged),
                    rerank,
                    rerank_text=rerank_text,
                    max_chars=rerank_max_chars,
                )
            if dep_boost is not None:
                merged = _boosted(merged, dep_boost)
            merged = (
                best_per_artifact(merged)
                if per_artifact
                else sorted(merged, key=lambda h: h.score, reverse=True)
            )
            return merged[:k]
        query = queries[0]

    # Lexical ranking scores on text alone, so it must not pay the embedding
    # matrix's I/O; dense/hybrid need the matrix. (Both reuse the same cache.)
    if mode == "lexical":
        ids, metas = corpus.store.metas()
        mat = None
    else:
        ids, mat, metas = corpus.store.matrix()
    if not ids:
        return []

    candidates = _filter_candidates(metas, surfaces, filter)
    if not candidates:
        return []

    # A reranker can only promote what it is shown, so it over-fetches even when
    # the surfaces are not collapsed per artifact (which alone would fetch just k).
    over_fetch = per_artifact or rerank is not None or dep_boost is not None
    fetch = fetch_k if fetch_k is not None else (max(k * 5, 50) if over_fetch else k)
    # With surface weights, fusion keeps its whole union so the weights act
    # before the fetch cut: a boosted surface fused just below it still counts.
    cut = None if weights else fetch

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
                ranked = _blend_fuse(dense, lexical, alpha, bm25_sat_k, cut)
            else:
                ranked = _rrf_fuse(dense, lexical, rrf_k, cut)

    meta_by_id = {ids[j]: metas[j] for j in candidates}

    if weights:
        ranked = _weight_surfaces(ranked, meta_by_id, weights)[:fetch]

    if rerank is not None and ranked:
        ranked = _apply_rerank(
            corpus,
            asked,
            ranked,
            meta_by_id,
            rerank,
            fetch,
            rerank_text=rerank_text,
            max_chars=rerank_max_chars,
        )

    source = getattr(corpus, "name", None)
    hits = [
        SearchHit(
            artifact_id=meta_by_id[rid]["artifact_id"],
            surface_kind=meta_by_id[rid]["surface_kind"],
            score=score,
            text=meta_by_id[rid]["text"],
            metadata=meta_by_id[rid].get("metadata", {}),
            source=source,
            surface_index=meta_by_id[rid].get("surface_index"),
        )
        for rid, score in ranked
    ]
    if dep_boost is not None:
        hits = _boosted(hits, dep_boost)
    if per_artifact:
        hits = best_per_artifact(hits)
    return hits[:k]


# =========================================================================== #
# Cross-source fusion — merging ranked hit lists from heterogeneous sources
# =========================================================================== #

#: How cross-source duplicates are detected in :func:`fuse_hits`: ``None``
#: (default — hits from different sources never merge; same id, different
#: corpus = different artifact), the string ``"pointer"`` (merge hits whose
#: :attr:`~ir.base.SearchHit.pointer` match — opt-in, for corpora that
#: genuinely index the same files), or any ``hit -> hashable`` callable (a
#: falsy key falls back to per-source identity).
Identity = Callable[[SearchHit], Any] | str | None


def _resolve_identity(identity) -> Callable[[SearchHit], Any] | None:
    """Resolve an :data:`Identity` spec to a key function (or ``None``)."""
    if identity is None:
        return None
    if identity == "pointer":
        return lambda h: h.pointer
    if callable(identity):
        return identity
    raise ValueError(
        f"unknown identity {identity!r}; expected None, 'pointer', or a callable"
    )


def fuse_hits(
    hits_by_source: Mapping[str | None, Sequence[SearchHit]],
    *,
    rrf_k: int = DFLT_RRF_K,
    weights: Mapping[str, float] | None = None,
    identity: Identity = None,
    k: int | None = None,
) -> list[SearchHit]:
    """Merge per-source ranked hit lists into one ranking — by rank, not score.

    The cross-source counterpart of the within-corpus hybrid fusion: scores
    from different (corpus, mode, embedder) tuples live on incommensurable
    scales (ir_07: "a different model re-scales everything"), so **raw scores
    never cross the source boundary** — within each source they order and
    dedup that source's hits (one scale, sound), and across sources only ranks
    interact, via weighted Reciprocal Rank Fusion: each hit contributes
    ``weights[source] / (rrf_k + rank)``.

    Args:
        hits_by_source: ``{source_name: ranked hits}``. Hits without a
            ``source`` are stamped with their mapping key (existing tags win,
            so one corpus bound under two keys still counts as one source).
            A ``None`` key is the *untagged pseudo-source*: its hits fuse as
            one rank group and stay unattributed (``source=None``). Within
            each list, duplicate artifacts — and, when ``identity`` is given,
            identity-duplicates — collapse to their best raw score before
            ranking, so a multi-query / multi-round pool can never
            double-count one artifact's RRF mass.
        rrf_k: the RRF rank constant (standard default 60).
        weights: optional per-source trust dial (default 1.0 each) — a
            source's contribution scales linearly, no score comparability
            needed. Keys naming sources absent from ``hits_by_source`` are
            ignored (a per-round pool may legitimately lack a configured
            source); callers with a closed source set should validate keys
            upfront, as federated :func:`ir.discover` does.
        identity: how cross-source duplicates merge — see :data:`Identity`.
            Default ``None``: never; each ``(source, artifact_id)`` stays a
            distinct result.
        k: truncate the fused ranking to this many hits.

    Returns:
        the fused hits, best-first. Each carries the fused score in ``score``
        and keeps its pre-fusion magnitude as ``metadata["source_score"]`` (+
        ``"source_rank"``), so downstream consumers (abstention gates, LLM
        judges) never lose the per-source signal. When an ``identity`` merge
        combined several sources' hits, ``metadata["fused_sources"]`` lists
        them and the representative hit is the one with the best rank.
        **Single-source input passes through with raw scores** (RRF of one
        list is that list's order — same convention as the hybrid fusion's
        single-channel fallback), so the fused-score rescaling only happens
        when there is genuinely something to fuse. The post-fusion ``score``
        is ordinal: valid for ordering and relative cuts, meaningless against
        absolute floors — apply calibrated ``min_score`` floors per source,
        *before* fusing (see ir_07/ir_08 and ``ir.discover``'s federated form).
    """
    ident = _resolve_identity(identity)

    # Per source: stamp provenance, collapse duplicates (keep-max raw score),
    # rank best-first by raw score — the only place raw scores are compared,
    # and only ever within one source.
    ranked_by_source: list[tuple[str | None, list[SearchHit]]] = []
    for name, hits in hits_by_source.items():
        deduped = best_per_artifact(tag_source(hits, name))
        if ident is not None:
            # Identity-duplicates within ONE source must also collapse before
            # ranking — otherwise two same-identity artifacts in one list both
            # contribute mass to one fused key (intra-source double counting).
            # ``deduped`` is best-first, so first-wins keeps the best-scored
            # representative; ranks compact naturally.
            by_key: dict[Any, SearchHit] = {}
            for h in deduped:
                by_key.setdefault(ident(h) or (h.source, h.artifact_id), h)
            deduped = list(by_key.values())
        if deduped:
            ranked_by_source.append((name, deduped))

    if not ranked_by_source:
        return []
    if len(ranked_by_source) == 1:
        only = ranked_by_source[0][1]
        return only[:k] if k is not None else only

    w = dict(weights) if weights else {}
    fused: dict[Any, float] = {}
    # key -> (best_rank, source_position, representative hit)
    rep: dict[Any, tuple[int, int, SearchHit]] = {}
    members: dict[Any, list[str]] = {}
    for pos, (name, ranked) in enumerate(ranked_by_source):
        weight = float(w.get(name, 1.0))
        for rank, h in enumerate(ranked, start=1):
            key = ident(h) if ident is not None else None
            if not key:
                key = (h.source, h.artifact_id)
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank)
            members.setdefault(key, []).append(name)
            cur = rep.get(key)
            if cur is None or (rank, pos) < (cur[0], cur[1]):
                rep[key] = (rank, pos, h)

    scored = []
    for key, score in fused.items():
        rank, pos, h = rep[key]
        meta = {
            **dict(h.metadata),
            "source_score": float(h.score),
            "source_rank": int(rank),
        }
        sources = list(dict.fromkeys(members[key]))
        if len(sources) > 1:
            meta["fused_sources"] = sources
        scored.append((float(score), rank, pos, h, meta))
    # Deterministic order: fused score, then best pre-fusion rank, then the
    # caller's source order (documented as the priority order for rank ties —
    # raw scores never break cross-source ties), then artifact_id.
    scored.sort(key=lambda t: (-t[0], t[1], t[2], t[3].artifact_id))
    out = [replace(h, score=score, metadata=meta) for score, _, _, h, meta in scored]
    return out[:k] if k is not None else out


# =========================================================================== #
# Sibling addressing — ledger-backed artifact -> ordered-records lookup
# =========================================================================== #


class NoLedgerEntry(KeyError):
    """The corpus has no ledger entry for the requested artifact.

    The *benign* miss (unknown artifact, or a corpus built without ledger
    bookkeeping) — distinct from a stale/torn ledger, which raises a plain
    ``KeyError`` describing the corruption. A ``KeyError`` subclass so
    existing ``except KeyError`` callers keep working; the disclosure seam
    catches this one specifically and tolerates it per hit.
    """


def records_for_artifact(
    store_or_corpus, artifact_id: str, *, surface_kind: str | None = None
) -> list[Record]:
    """All stored records of *artifact_id*, ordered by ``surface_index``.

    The sibling-addressing primitive beneath retrieval-time context expansion:
    a :class:`~ir.base.SearchHit` names its artifact (and, via
    :attr:`~ir.base.SearchHit.surface_index`, which surface of it matched);
    this returns every surface of that artifact, in plan order, so an expansion
    policy can stitch neighbors / parents around the hit.

    Resolution is **ledger-backed only**: the artifact's ledger entry lists its
    ``record_ids``. Record ids are never re-derived from a per-kind index like
    ``metadata["chunk_index"]`` — on multi-kind strategies that index differs
    from the plan-global ``surface_index`` baked into the id (see
    :meth:`ir.base.Record.make_id`), so derivation would fetch wrong or missing
    siblings.

    Args:
        store_or_corpus: a :class:`~ir.store.CorpusStore`, anything carrying
            one as ``.store`` (e.g. a :class:`~ir.index.Corpus`), or a corpus
            *name* — resolved straight to its local store: sibling lookup
            never embeds, so unlike :func:`ir.open_corpus` no embedder is
            loaded.
        artifact_id: the artifact whose surfaces to fetch.
        surface_kind: restrict to one surface kind (e.g. ``"readme_chunk"``);
            a known artifact with no surfaces of that kind yields ``[]``.

    Raises:
        NoLedgerEntry: the ledger has no entry for *artifact_id* (an unknown
            artifact, or a corpus built without :func:`ir.index.build`'s
            ledger bookkeeping). A ``KeyError`` subclass.
        KeyError: an entry exists but lists a record missing from the store
            (a stale ledger: interrupted build or out-of-band
            ``delete_record``) — data corruption, named in the message.
    """
    if isinstance(store_or_corpus, str):
        from .store import CorpusStore

        store_or_corpus = CorpusStore.local(store_or_corpus)
    store = getattr(store_or_corpus, "store", store_or_corpus)
    entry = store.get_ledger_entry(ledger_key(artifact_id))
    if entry is None:
        raise NoLedgerEntry(
            f"no ledger entry for artifact {artifact_id!r}: unknown artifact, "
            f"or a corpus built without ledger bookkeeping"
        )
    records = []
    for rid in entry.get("record_ids", []):
        try:
            records.append(store.get_record(rid))
        except KeyError:
            raise KeyError(
                f"ledger entry for artifact {artifact_id!r} lists record "
                f"{rid!r}, which is missing from the store (stale ledger — "
                f"interrupted build or out-of-band delete?); rebuild the corpus"
            ) from None
    if surface_kind is not None:
        records = [r for r in records if r.surface_kind == surface_kind]
    return sorted(records, key=lambda r: r.surface_index)


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
