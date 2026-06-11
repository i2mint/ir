"""Capability-discovery evaluation — measuring how well a corpus is *found*.

``ir.eval`` answers the question that the rest of ``ir`` raises: *does retrieval
actually surface the right capability?* It is a thin, deterministic scoring layer
over the substrate already in place — ``ir.retrieve.search`` for ranking and
``ef.evaluation`` for the metric mathematics — plus the genuinely new pieces a
*capability-discovery* eval needs that generic retrieval evaluation does not.

Design stance (retrieval-centric, reuse ``ef``)
-----------------------------------------------
``ir``'s corpora are **documents** — skills, packages, reports — each a
name + description (+ body), joined on a stable ``artifact_id``. They are *not*
a typed function-calling registry with argument signatures. So the eval that
fits is **retrieval of the right artifact**, not argument-dict scoring:

- **recall@k / NDCG@k / MRR / MAP** — computed by ``ef.evaluation``'s pure,
  tested primitives (:func:`ef.evaluation.recall_at_k`, … ) and driver
  (:func:`ef.evaluation.evaluate_retrieval`). ``ir.eval`` does not reimplement
  them; it adapts an ``ir`` corpus to ``ef``'s retriever contract and reads back
  a :class:`ef.evaluation.RetrievalEvalReport`.
- **distractor-robustness curve** — retrieval accuracy as the catalog grows
  (1 gold + N−1 distractors). The single most diagnostic capability-discovery
  metric, and one ``ef`` does not provide.
- **failure-mode taxonomy** — every gold case is classed ``hit_rank_1`` /
  ``surfaced_low_rank`` / ``retrieval_miss`` so a headline number decomposes
  into *why* it is what it is.
- **abstention** — an optional score-threshold proxy for "no artifact applies"
  cases. True abstention is a *selection* concern (a selector commits or
  refuses); here it is a retrieval-side diagnostic, clearly labelled as such.

The unit of evaluation is a :class:`DiscoveryCase`: an intent (``query``) and
the ``gold`` ``artifact_id``\\ s that should answer it (empty ``gold`` = an
abstention case). Cases are plain data — :func:`save_cases` / :func:`load_cases`
round-trip them as JSONL so an evaluation set is a committable, reproducible
fixture (the corpora themselves are machine-specific and live, so freezing the
cases is what makes a run repeatable; :func:`validate_cases` flags gold ids that
have since drifted out of the corpus).

Case *generation* (back-translation from a corpus with name-masking) needs an
LLM and is deliberately **out of scope here** — this module scores a given set
of cases offline, with no network and no model download. (Dense scoring needs
only numpy; ``lexical`` / ``hybrid`` ranking additionally need the optional
``vd`` dependency — a hybrid run that silently fell back to dense because ``vd``
was absent is flagged in the report.)

Quick start::

    import ir
    from ir import eval as ev

    corpus = ir.build(ir.CorpusSource.from_skills())     # or ir.open_corpus("skills")
    cases = ev.load_cases("skills_eval.jsonl")
    report = ev.evaluate_discovery(corpus, cases, mode="hybrid")
    print(report)                                        # NDCG@10 + taxonomy

    # Just the BEIR-shaped metrics (pure ef), e.g. to A/B dense vs hybrid:
    ev.retrieval_report(corpus, cases, mode="dense").primary    # mean NDCG@10
    ev.retrieval_report(corpus, cases, mode="hybrid").primary
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import SearchHit
from .retrieve import search as _search
from .select import DFLT_GAP_RATIO, DFLT_MAX_K, DFLT_REL_THRESHOLD

#: Rank cutoffs reported by default.
DFLT_K_VALUES = (1, 5, 10)

#: Metrics reported by default — the ``ef.evaluation`` retrieval metric names.
DFLT_METRICS = ("ndcg", "recall", "precision", "mrr", "map")

#: Default ranking mode for the eval adapter (hybrid is ``ir``'s strongest).
DFLT_MODE = "hybrid"

#: Catalog sizes swept by :func:`distractor_robustness_curve` (RAG-MCP-style).
DFLT_DISTRACTOR_SIZES = (1, 4, 8, 16, 32, 64, 128)

#: Defaults for :func:`distractor_robustness_curve` — deliberately *not*
#: ``DFLT_MODE``: the curve builds throwaway in-memory sub-corpora, so it favours
#: the fast, always-offline dense + hashing path. Pass ``mode="hybrid"`` /
#: ``embedder="default"`` to measure the production configuration instead.
DFLT_CURVE_MODE = "dense"
DFLT_CURVE_EMBEDDER = "light"

#: ``max_k`` values swept by :func:`sweep_selector` (the commit-size cap).
DFLT_SWEEP_MAX_K = (1, 2, 3, 5, 8)

#: ``rel`` values swept by :func:`sweep_selector` (relative-to-top keep band).
DFLT_SWEEP_REL = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: ``min_score`` floors swept by :func:`sweep_selector`. Default ``(None,)`` —
#: the absolute floor is mode-specific (cosine / RRF / BM25 scales differ), so
#: it is left off unless the caller supplies calibrated values to sweep.
DFLT_SWEEP_MIN_SCORE = (None,)

#: Default objective :func:`SelectionSweep.best` optimizes — the standard
#: precision/recall balance. ``mean_selected_size`` is the only metric minimized;
#: every other is maximized (see :data:`_SWEEP_MINIMIZE`).
DFLT_SWEEP_OBJECTIVE = "selection_f1"

#: Selection metrics where *smaller is better* (all others: larger is better).
_SWEEP_MINIMIZE = frozenset({"mean_selected_size"})

#: The scalar selection metrics a sweep can optimize / rank by.
SWEEP_METRICS = (
    "selection_f1",
    "selection_precision",
    "selection_recall",
    "conditional_commit_rate",
    "abstention_accuracy",
    "mean_selected_size",
)

#: Weight on *sensitivity* (catch true positives) vs *specificity* (reject
#: out-of-scope queries) when :func:`calibrate_min_score` picks the abstention
#: floor: ``objective = w·sensitivity + (1−w)·specificity``. ``0.5`` is balanced
#: (its argmax matches Youden's J / balanced accuracy). Lower it to lean
#: *precision* — abstain more readily — per ir_01 §3's stance that a padded
#: commit is the costlier failure for an agent surface than a dropped gold.
DFLT_CALIB_SENSITIVITY_WEIGHT = 0.5


def _vd_available() -> bool:
    """Whether ``vd`` provides what ``lexical`` / ``hybrid`` ranking needs.

    Probes the exact symbols :mod:`ir.retrieve` imports at query time
    (:class:`vd.BM25Index` for lexical, :func:`vd.reciprocal_rank_fusion` for
    hybrid fusion) — keep this in sync with that module so the eval layer's
    ``vd_degraded`` flag matches the runtime's actual capability.
    """
    try:
        from vd import BM25Index, reciprocal_rank_fusion  # noqa: F401

        return True
    except Exception:
        return False


# =========================================================================== #
# The unit of evaluation
# =========================================================================== #


@dataclass(frozen=True)
class DiscoveryCase:
    """One discovery probe: an intent and the artifact(s) that should answer it.

    Attributes:
        query: the user intent / search query.
        gold: the ``artifact_id``\\ s that correctly answer ``query`` (one for a
            single-answer case, several when capabilities overlap). An **empty**
            tuple marks an *abstention* case — no artifact applies.
        corpus: optional name of the corpus the case targets (provenance / for
            multi-corpus case files).
        source_id: optional ``artifact_id`` the query was generated from
            (back-translation provenance).
        metadata: free-form per-case metadata (difficulty, generator, …).
    """

    query: str
    gold: tuple[str, ...] = ()
    corpus: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gold_is_none(self) -> bool:
        """True when no artifact should match — an abstention case."""
        return not self.gold

    def to_dict(self) -> dict:
        """JSON-serializable form (omitting empty optional fields)."""
        out: dict[str, Any] = {"query": self.query, "gold": list(self.gold)}
        if self.corpus is not None:
            out["corpus"] = self.corpus
        if self.source_id is not None:
            out["source_id"] = self.source_id
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DiscoveryCase":
        """Inverse of :meth:`to_dict`; ``gold`` may be a string or a list."""
        gold = d.get("gold") or ()
        if isinstance(gold, str):
            gold = (gold,)
        return cls(
            query=d["query"],
            gold=tuple(gold),
            corpus=d.get("corpus"),
            source_id=d.get("source_id"),
            metadata=dict(d.get("metadata") or {}),
        )


def save_cases(
    cases: Iterable[DiscoveryCase],
    path: str | Path,
    *,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Write ``cases`` to a JSONL file (one case per line).

    An optional ``meta`` mapping is written as a leading ``{"__meta__": …}``
    line — the natural home for a corpus-version anchor that pins the cases to
    the corpus snapshot they were generated against.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        if meta is not None:
            out.write(json.dumps({"__meta__": dict(meta)}) + "\n")
        for case in cases:
            out.write(json.dumps(case.to_dict()) + "\n")


def load_cases(path: str | Path) -> list[DiscoveryCase]:
    """Read :class:`DiscoveryCase`\\ s from a JSONL file (skips a ``__meta__`` header)."""
    cases: list[DiscoveryCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "__meta__" in obj:
            continue
        cases.append(DiscoveryCase.from_dict(obj))
    return cases


# =========================================================================== #
# Bridge: an ir corpus as an ef retriever
# =========================================================================== #


def _as_corpus(corpus: Any) -> Any:
    """Resolve a corpus *name* to a built :class:`~ir.index.Corpus` (pass-through otherwise)."""
    if isinstance(corpus, str):
        from .index import open_corpus

        return open_corpus(corpus)
    return corpus


def as_doc_retriever(
    corpus: Any,
    *,
    mode: str = DFLT_MODE,
    surfaces: Iterable[str] | None = None,
    **search_kw: Any,
) -> Callable[..., list[str]]:
    """Adapt an ``ir`` corpus to ``ef``'s retriever contract.

    Returns a callable ``retriever(query, *, limit=10) -> [artifact_id, …]`` —
    bare doc-id strings, best-ranked first, one per artifact. That is exactly
    what :func:`ef.evaluation.evaluate_retrieval` consumes (it maps a bare
    string straight to a document id), so the returned callable can be handed to
    ``ef`` with no further adaptation.

    Args:
        corpus: an :class:`~ir.index.Corpus` or a registered corpus *name*.
        mode: ranking mode — ``"dense"`` / ``"lexical"`` / ``"hybrid"``.
        surfaces: restrict to these surface kinds (e.g. ``{"description"}``).
        **search_kw: any other :func:`ir.retrieve.search` keyword (``filter``,
            ``rrf_k``, ``rerank``, ``bm25``, …).
    """
    corpus = _as_corpus(corpus)

    def retriever(query: str, *, limit: int = 10) -> list[str]:
        hits = _search(
            corpus,
            query,
            k=limit,
            mode=mode,
            surfaces=surfaces,
            per_artifact=True,
            **search_kw,
        )
        return [hit.artifact_id for hit in hits]

    return retriever


def to_qrels(
    cases: Sequence[DiscoveryCase], *, grade: float = 1.0
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Build ``ef``'s ``(queries, qrels)`` from the gold-bearing cases.

    Abstention cases (empty ``gold``) are skipped — ``ef``'s retrieval metrics
    require at least one positively-judged document per query. Query ids are
    derived from each case's position so they stay stable across the gap left
    by skipped abstention cases.
    """
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, float]] = {}
    for i, case in enumerate(cases):
        if case.gold_is_none:
            continue
        qid = f"q{i:05d}"
        queries[qid] = case.query
        qrels[qid] = {aid: grade for aid in case.gold}
    return queries, qrels


def retrieval_report(
    corpus: Any,
    cases: Sequence[DiscoveryCase],
    *,
    k_values: Sequence[int] = DFLT_K_VALUES,
    metrics: Sequence[str] = DFLT_METRICS,
    mode: str = DFLT_MODE,
    surfaces: Iterable[str] | None = None,
    limit: int | None = None,
    **search_kw: Any,
) -> Any:
    """Score a corpus's retrieval against the cases — the pure-``ef`` path.

    A thin wrapper that builds the retriever adapter and hands it, with the
    cases' ``(queries, qrels)``, to :func:`ef.evaluation.evaluate_retrieval`.
    Returns its :class:`ef.evaluation.RetrievalEvalReport` (``.primary`` is mean
    NDCG@10). Use this to A/B configurations — dense vs hybrid, with vs without
    a reranker — on the standard BEIR/MTEB metrics.

    For the richer report (taxonomy + abstention) use :func:`evaluate_discovery`.

    Raises:
        ValueError: if no case carries gold (``ef`` requires at least one
            positively-judged query). :func:`evaluate_discovery` instead
            tolerates an all-abstention set — the two paths differ only on that
            degenerate input.
    """
    from ef.evaluation import evaluate_retrieval

    corpus = _as_corpus(corpus)
    queries, qrels = to_qrels(cases)
    retriever = as_doc_retriever(corpus, mode=mode, surfaces=surfaces, **search_kw)
    return evaluate_retrieval(
        retriever,
        qrels,
        queries,
        k_values=tuple(k_values),
        metrics=tuple(metrics),
        limit=limit,
    )


# =========================================================================== #
# Corpus introspection — drift detection
# =========================================================================== #


def corpus_artifact_ids(corpus: Any) -> set[str]:
    """The set of ``artifact_id``\\ s present in a built corpus."""
    corpus = _as_corpus(corpus)
    _ids, _matrix, metas = corpus.store.matrix()
    return {meta["artifact_id"] for meta in metas}


def validate_cases(corpus: Any, cases: Sequence[DiscoveryCase]) -> dict[int, list[str]]:
    """Gold ids a case references that are absent from the corpus.

    Returns ``{case_index: [missing_artifact_id, …]}`` for every case whose gold
    has drifted out of the corpus (an empty dict means the cases are aligned).
    The eval corpora are live and machine-specific, so a committed case file can
    quietly fall out of sync; run this before trusting a report.
    """
    present = corpus_artifact_ids(corpus)
    missing: dict[int, list[str]] = {}
    for i, case in enumerate(cases):
        absent = [aid for aid in case.gold if aid not in present]
        if absent:
            missing[i] = absent
    return missing


# =========================================================================== #
# The rich report: metrics + failure taxonomy + abstention
# =========================================================================== #


def _gold_rank(ranking: Sequence[str], gold: Sequence[str]) -> int:
    """1-based rank of the best-placed gold id in ``ranking`` (0 if none present)."""
    gold_set = set(gold)
    for i, aid in enumerate(ranking):
        if aid in gold_set:
            return i + 1
    return 0


@dataclass(frozen=True)
class DiscoveryReport:
    """The outcome of :func:`evaluate_discovery`.

    Bundles the standard retrieval metrics with the capability-discovery extras:

    Attributes:
        retrieval: the :class:`ef.evaluation.RetrievalEvalReport` over the
            gold-bearing cases.
        failure_classes: counts per failure class (``hit_rank_1`` /
            ``surfaced_low_rank`` / ``retrieval_miss`` for gold cases;
            ``abstention_ok`` / ``false_action`` / ``abstention_unscored`` for
            abstention cases).
        n_cases / n_gold / n_abstention: case counts.
        primary_k: the cutoff ``primary`` reports at (default 10).
        abstention_accuracy: accuracy on the abstention slice when an
            ``abstain_threshold`` was supplied, else ``None``.
        mode: the ranking mode the report was produced with — recorded so a
            number is never mistaken for a different configuration's.
        vd_degraded: True when ``mode`` was ``lexical`` / ``hybrid`` but ``vd``
            was unavailable, so ranking silently fell back to dense (the scores
            are *not* the requested configuration's).
    """

    retrieval: Any
    failure_classes: Mapping[str, int]
    n_cases: int
    n_gold: int
    n_abstention: int
    primary_k: int = 10
    abstention_accuracy: float | None = None
    mode: str = DFLT_MODE
    vd_degraded: bool = False

    @property
    def primary(self) -> float | None:
        """The headline number — mean NDCG@``primary_k`` (``None`` if not computed)."""
        return self.retrieval.primary_at(self.primary_k)

    def __str__(self) -> str:
        primary = self.primary
        head = (
            f"DiscoveryReport — {self.n_cases} cases "
            f"({self.n_gold} gold, {self.n_abstention} abstention)"
        )
        mode_line = f"  mode: {self.mode}" + (
            "  (DEGRADED: vd unavailable -> dense fallback)" if self.vd_degraded else ""
        )
        lines = [
            head,
            mode_line,
            "  primary NDCG@%d: %s"
            % (self.primary_k, f"{primary:.4f}" if primary is not None else "n/a"),
        ]
        for key, value in self.retrieval.metrics.items():
            lines.append(f"    {key}: {value:.4f}")
        if self.failure_classes:
            lines.append("  failure classes:")
            for cls, count in sorted(
                self.failure_classes.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {cls}: {count}")
        if self.abstention_accuracy is not None:
            lines.append(f"  abstention accuracy: {self.abstention_accuracy:.4f}")
        return "\n".join(lines)


def evaluate_discovery(
    corpus: Any,
    cases: Sequence[DiscoveryCase],
    *,
    k_values: Sequence[int] = DFLT_K_VALUES,
    primary_k: int = 10,
    metrics: Sequence[str] = DFLT_METRICS,
    mode: str = DFLT_MODE,
    surfaces: Iterable[str] | None = None,
    abstain_threshold: float | None = None,
    **search_kw: Any,
) -> DiscoveryReport:
    """Score a corpus against ``cases`` — metrics, failure taxonomy, abstention.

    Runs retrieval **once per case** and, from each ranking, computes the
    ``ef.evaluation`` metric primitives (so the numbers match the pure-``ef``
    path) *and* the capability-discovery extras in one pass.

    Args:
        corpus: an :class:`~ir.index.Corpus` or a registered corpus *name*.
        k_values: rank cutoffs to report (``primary_k`` is always included).
        primary_k: the cutoff the headline ``primary`` reports at.
        metrics: which ``ef`` metrics to compute (``ndcg`` / ``recall`` /
            ``precision`` / ``mrr`` / ``map``).
        mode: ranking mode passed to :func:`ir.retrieve.search`.
        surfaces: restrict to these surface kinds.
        abstain_threshold: if given, an abstention case is scored *correct* when
            the top hit's score is below this threshold (a retrieval-side proxy
            for "no artifact applies"); leave ``None`` to only count abstention
            cases without scoring them.
        **search_kw: any other :func:`ir.retrieve.search` keyword.

    Returns:
        a :class:`DiscoveryReport`. If no case carries gold (an empty or
        all-abstention set) the retrieval metrics are omitted — ``report.primary``
        is ``None`` — but the taxonomy and abstention counts are still returned
        (unlike :func:`retrieval_report`, which raises ``ValueError`` on that
        input).
    """
    from ef.evaluation import (
        RetrievalEvalReport,
        average_precision,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )

    # Deliberate mirror of ef.evaluation._RETRIEVAL_METRICS (private, so it cannot
    # be imported). The metric *functions* are ef's — only this name->fn map and
    # the unknown-name check are duplicated; ef exposing a public registry would
    # let this go away.
    metric_fns: dict[str, Callable[..., float]] = {
        "ndcg": ndcg_at_k,
        "recall": recall_at_k,
        "precision": precision_at_k,
        "mrr": reciprocal_rank,
        "map": average_precision,
    }
    unknown = [name for name in metrics if name not in metric_fns]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}. Choose from {sorted(metric_fns)}."
        )

    corpus = _as_corpus(corpus)
    # primary_k is always reported; sort+dedupe so report keys are canonical.
    k_values = tuple(sorted(set(k_values) | {primary_k}))
    limit = max(k_values)
    vd_degraded = mode in ("hybrid", "lexical") and not _vd_available()

    per_query: dict[str, dict[str, float]] = {}
    taxonomy: Counter[str] = Counter()
    n_abstention = 0
    abstain_correct = 0

    for i, case in enumerate(cases):
        # per_artifact=True yields one id per artifact (its best surface) — the
        # same shape ef.evaluate_retrieval produces after its internal _dedup, so
        # the metric primitives below score the same ranking as the pure-ef path.
        hits = _search(
            corpus,
            case.query,
            k=limit,
            mode=mode,
            surfaces=surfaces,
            per_artifact=True,
            **search_kw,
        )
        ranking = [hit.artifact_id for hit in hits]
        top_score = hits[0].score if hits else None

        if case.gold_is_none:
            n_abstention += 1
            if abstain_threshold is None:
                taxonomy["abstention_unscored"] += 1
            elif top_score is None or top_score < abstain_threshold:
                taxonomy["abstention_ok"] += 1
                abstain_correct += 1
            else:
                taxonomy["false_action"] += 1
            continue

        relevant = {aid: 1.0 for aid in case.gold}
        per_query[f"q{i:05d}"] = {
            f"{name}@{k}": metric_fns[name](ranking, relevant, k)
            for name in metrics
            for k in k_values
        }

        rank = _gold_rank(ranking, case.gold)
        if rank == 0 or rank > primary_k:
            taxonomy["retrieval_miss"] += 1
        elif rank == 1:
            taxonomy["hit_rank_1"] += 1
        else:
            taxonomy["surfaced_low_rank"] += 1

    metric_keys = [f"{name}@{k}" for name in metrics for k in k_values]
    n_gold = len(per_query)
    # With no gold-bearing case, leave metrics empty so RetrievalEvalReport.
    # primary_at(...) returns None (the "not computed" contract) rather than NaN.
    aggregated = (
        {
            key: sum(scores[key] for scores in per_query.values()) / n_gold
            for key in metric_keys
        }
        if n_gold
        else {}
    )
    retrieval = RetrievalEvalReport(
        metrics=aggregated,
        per_query=per_query,
        n_queries=n_gold,
        k_values=tuple(k_values),
    )
    abstention_accuracy = (
        abstain_correct / n_abstention
        if abstain_threshold is not None and n_abstention
        else None
    )
    return DiscoveryReport(
        retrieval=retrieval,
        failure_classes=dict(taxonomy),
        n_cases=len(cases),
        n_gold=n_gold,
        n_abstention=n_abstention,
        primary_k=primary_k,
        abstention_accuracy=abstention_accuracy,
        mode=mode,
        vd_degraded=vd_degraded,
    )


# =========================================================================== #
# Distractor-robustness curve
# =========================================================================== #


def distractor_robustness_curve(
    scope: Mapping[str, Any],
    probes: Sequence[tuple[str, str]],
    *,
    strategy: Any = None,
    embedder: Any = DFLT_CURVE_EMBEDDER,
    mode: str = DFLT_CURVE_MODE,
    sizes: Sequence[int] = DFLT_DISTRACTOR_SIZES,
    trials: int = 20,
    seed: int = 0,
    k: int = 1,
) -> dict[int, float]:
    """Retrieval accuracy as the catalog grows — the needle-in-a-haystack curve.

    For each catalog size ``N`` and each ``(query, gold_id)`` probe, build many
    ``N``-artifact sub-corpora (the gold artifact plus ``N−1`` distractors
    sampled from ``scope``) and measure how often the gold artifact lands in the
    top ``k``. The accuracy at each ``N`` is averaged over the probes and over
    ``trials`` random distractor draws. ``N=1`` is the trivial anchor (gold
    alone) and is always 1.0; a steep decline as ``N`` grows is the signature of
    retrieval that does not discriminate.

    Sub-corpora are built in-memory with the ``"light"`` (numpy-only) embedder
    by default, so the curve is fast and offline; pass ``embedder="default"``
    (and ``mode="hybrid"``) to measure the production configuration.

    Sizes larger than the corpus are **capped** at ``len(scope)`` (the largest
    catalog the pool can fill) and de-duplicated, so the returned keys are the
    catalog sizes actually built — the x-axis never claims an ``N`` larger than
    the corpus could supply.

    Args:
        scope: the full ``{artifact_id: raw}`` pool to draw the catalog from.
        probes: ``(query, gold_id)`` pairs; ``gold_id`` must be a key of ``scope``.
        strategy: the :class:`~ir.strategy.IndexingStrategy` to index with
            (default :class:`~ir.strategy.WholeText`).
        embedder: embedder spec for the sub-corpora.
        mode: ranking mode for the probe searches.
        sizes: the catalog sizes to sweep (capped at ``len(scope)``, deduped).
        trials: random distractor draws averaged per size (a size that consumes
            the whole pool has no sampling freedom, so it runs a single trial).
        seed: base RNG seed (reproducible; varied per size).
        k: the top-``k`` the gold must reach to count as a hit.

    Returns:
        ``{N: accuracy}`` for each *achievable* ``N`` (the requested sizes capped
        at ``len(scope)`` and de-duplicated).
    """
    from .index import build
    from .sources import CorpusSource
    from .store import CorpusStore
    from .strategy import WholeText

    scope = dict(scope)
    strategy = strategy or WholeText()
    all_ids = list(scope)
    max_catalog = len(all_ids)
    effective_sizes = sorted({min(n, max_catalog) for n in sizes if n >= 1})
    curve: dict[int, float] = {}

    for size in effective_sizes:
        rng = random.Random(seed + size)
        hits = 0
        total = 0
        for query, gold_id in probes:
            if gold_id not in scope:
                continue
            pool = [aid for aid in all_ids if aid != gold_id]
            n_distractors = min(size - 1, len(pool))
            # The full pool admits no sampling freedom — every trial would build
            # the same catalog — so one representative trial suffices.
            n_trials = 1 if n_distractors >= len(pool) else trials
            for _ in range(n_trials):
                sample = rng.sample(pool, n_distractors) if n_distractors else []
                sub = {aid: scope[aid] for aid in (gold_id, *sample)}
                source = CorpusSource.from_mapping(
                    sub, name="_distractor", strategy=strategy
                )
                corpus = build(source, store=CorpusStore.memory(), embedder=embedder)
                ranking = [
                    hit.artifact_id for hit in _search(corpus, query, k=k, mode=mode)
                ]
                total += 1
                if gold_id in ranking[:k]:
                    hits += 1
        curve[size] = hits / total if total else float("nan")
    return curve


def distractor_curve_from_cases(
    source: Any, cases: Sequence[DiscoveryCase], **kwargs: Any
) -> dict[int, float]:
    """Convenience: a distractor curve over a source's scope from single-gold cases.

    Probes are the single-gold cases (``(query, gold[0])``); the source's
    ``scope`` is the distractor pool and its ``indexing_strategy`` is reused
    unless a ``strategy`` is passed in ``kwargs``.
    """
    probes = [(case.query, case.gold[0]) for case in cases if len(case.gold) == 1]
    kwargs.setdefault("strategy", source.indexing_strategy)
    return distractor_robustness_curve(source.scope, probes, **kwargs)


# =========================================================================== #
# Selection evaluation — measuring the *commit*, isolated from retrieval
# =========================================================================== #


@dataclass(frozen=True)
class SelectionReport:
    """The outcome of :func:`evaluate_selection` — selection quality, isolated.

    Where :func:`evaluate_discovery` scores the *ranking*, this scores the
    *commit*: given the same ``k`` candidates, what subset did the selector keep?
    The headline is :attr:`conditional_commit_rate` — accuracy of the selection
    decision **conditioned on retrieval having surfaced the gold** — which is the
    one number that separates a selection failure from a retrieval failure.

    Every selection-quality metric shares that conditioning: precision, recall
    and F1 are computed **only over the** :attr:`n_gold_retrieved` **cases** whose
    gold reached the ``k`` candidates, so a retrieval miss is never charged to the
    selector (it is :func:`evaluate_discovery`'s to report).

    Attributes:
        selection_precision: mean over committed gold-retrieved cases of
            ``|selected ∩ gold| / |selected|`` (``None`` if nothing was ever
            committed) — how clean the committed sets are.
        selection_recall: mean over gold-retrieved cases of
            ``|selected ∩ gold| / |gold ∩ retrieved|`` — of the gold the selector
            was actually shown, how much it kept (denominator is the *retrievable*
            gold, so retrieval misses do not depress it).
        selection_f1: mean per-case F1 over gold-retrieved cases (an empty commit
            scores 0).
        conditional_commit_rate: among gold cases whose gold was in the ``k``
            candidates, the fraction where the selector committed to ≥1 gold —
            the selection decision isolated from retrieval recall (``None`` if
            retrieval never surfaced any gold).
        n_gold_retrieved: the shared denominator of every selection-quality
            metric above (cases whose gold reached the ``k`` candidates).
        abstention_accuracy: fraction of abstention cases (empty gold) the
            selector correctly committed nothing to (``None`` if none).
        mean_selected_size: mean committed-set size over gold-retrieved cases.
        n_cases / n_gold / n_abstention: case counts.
        strategy / mode / k: the configuration scored (so a number is never
            mistaken for a different setup's).
    """

    selection_precision: float | None
    selection_recall: float | None
    selection_f1: float | None
    conditional_commit_rate: float | None
    n_gold_retrieved: int
    abstention_accuracy: float | None
    mean_selected_size: float | None
    n_cases: int
    n_gold: int
    n_abstention: int
    strategy: str = "conservative"
    mode: str = DFLT_MODE
    k: int = 10

    def to_dict(self) -> dict:
        """JSON-serializable form (for the qh / HTTP surface)."""
        return {
            "selection_precision": self.selection_precision,
            "selection_recall": self.selection_recall,
            "selection_f1": self.selection_f1,
            "conditional_commit_rate": self.conditional_commit_rate,
            "n_gold_retrieved": self.n_gold_retrieved,
            "abstention_accuracy": self.abstention_accuracy,
            "mean_selected_size": self.mean_selected_size,
            "n_cases": self.n_cases,
            "n_gold": self.n_gold,
            "n_abstention": self.n_abstention,
            "strategy": self.strategy,
            "mode": self.mode,
            "k": self.k,
        }

    def __str__(self) -> str:
        def fmt(x: float | None) -> str:
            return f"{x:.4f}" if x is not None else "n/a"

        return "\n".join(
            [
                f"SelectionReport — {self.n_cases} cases "
                f"({self.n_gold} gold, {self.n_abstention} abstention)",
                f"  strategy: {self.strategy}  mode: {self.mode}  k: {self.k}",
                f"  conditional commit rate: {fmt(self.conditional_commit_rate)} "
                f"(over {self.n_gold_retrieved} gold-retrieved cases)",
                f"  selection precision: {fmt(self.selection_precision)}",
                f"  selection recall:    {fmt(self.selection_recall)}",
                f"  selection F1:        {fmt(self.selection_f1)}",
                f"  abstention accuracy: {fmt(self.abstention_accuracy)}",
                f"  mean selected size:  {fmt(self.mean_selected_size)}",
            ]
        )


def _f1(precision: float | None, recall: float) -> float:
    """Per-case F1; an undefined precision (empty commit) scores 0."""
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_selection(
    corpus: Any,
    cases: Sequence[DiscoveryCase],
    *,
    strategy: str = "conservative",
    mode: str = DFLT_MODE,
    k: int = 10,
    surfaces: Iterable[str] | None = None,
    max_k: int = DFLT_MAX_K,
    rel: float = DFLT_REL_THRESHOLD,
    gap_ratio: float = DFLT_GAP_RATIO,
    min_score: float | None = None,
    **search_kw: Any,
) -> SelectionReport:
    """Score a selector against ``cases`` — selection quality, isolated.

    Retrieves ``k`` candidates per case (the window the selector sees), commits
    with :func:`ir.select.select`, and scores the *commit*. The key number,
    :attr:`SelectionReport.conditional_commit_rate`, conditions on retrieval
    having surfaced the gold among those ``k`` — so a low value means the
    *selector* dropped a gold it was shown, not that retrieval missed it.

    ``k`` is the candidate window; hold it equal to the ``k`` used with
    :func:`evaluate_discovery` to compare the two stages on the same footing.

    Args:
        corpus: an :class:`~ir.index.Corpus` or a registered corpus *name*.
        cases: the :class:`DiscoveryCase`\\ s (gold-bearing and/or abstention).
        strategy: selection strategy (see :func:`ir.select.select`).
        mode: ranking mode for retrieval.
        k: candidate depth retrieved before selection.
        surfaces: restrict retrieval to these surface kinds.
        max_k, rel, gap_ratio, min_score: selection parameters.
        **search_kw: any other :func:`ir.retrieve.search` keyword.

    Returns:
        a :class:`SelectionReport`.
    """
    corpus = _as_corpus(corpus)
    scored = _retrieve_for_cases(
        corpus, cases, mode=mode, k=k, surfaces=surfaces, **search_kw
    )
    return _score_selection(
        scored,
        strategy=strategy,
        max_k=max_k,
        rel=rel,
        gap_ratio=gap_ratio,
        min_score=min_score,
        mode=mode,
        k=k,
    )


def _retrieve_for_cases(
    corpus: Any,
    cases: Sequence[DiscoveryCase],
    *,
    mode: str,
    k: int,
    surfaces: Iterable[str] | None,
    **search_kw: Any,
) -> list[tuple[DiscoveryCase, list[SearchHit]]]:
    """Retrieve each case's ``k`` candidate hits — once per case.

    Returns ``[(case, hits), …]``. Selector parameters do not affect retrieval,
    so a sweep over them (:func:`sweep_selector`) reuses one such pass across the
    whole grid; this is the expensive step, paid exactly once.
    """
    return [
        (
            case,
            _search(
                corpus,
                case.query,
                k=k,
                mode=mode,
                surfaces=surfaces,
                per_artifact=True,
                **search_kw,
            ),
        )
        for case in cases
    ]


def _score_selection(
    scored: Sequence[tuple[DiscoveryCase, Sequence[SearchHit]]],
    *,
    strategy: str | Any,
    max_k: int,
    rel: float,
    gap_ratio: float,
    min_score: float | None,
    mode: str,
    k: int,
) -> SelectionReport:
    """Score one selector setting over pre-retrieved ``(case, hits)`` pairs.

    The retrieval-free heart of :func:`evaluate_selection`: commit each case's
    candidates with :func:`ir.select.select` and accumulate the selection-quality
    metrics. Conditioning is identical to :func:`evaluate_selection` (a retrieval
    miss is excluded from every metric, keeping them on the shared
    ``n_gold_retrieved`` denominator). Factored out so :func:`sweep_selector` can
    score a whole parameter grid against a single retrieval pass.
    """
    from .select import select as _select

    precisions: list[float] = []  # only over committed gold cases
    recalls: list[float] = []
    f1s: list[float] = []
    selected_sizes: list[int] = []
    n_gold = 0
    n_gold_retrieved = 0
    n_committed_given_retrieved = 0
    n_abstention = 0
    n_abstained_correct = 0

    for case, hits in scored:
        selection = _select(
            hits,
            strategy=strategy,
            max_k=max_k,
            rel=rel,
            gap_ratio=gap_ratio,
            min_score=min_score,
        )
        selected = set(selection.selected_ids)

        if case.gold_is_none:
            n_abstention += 1
            if not selected:
                n_abstained_correct += 1
            continue

        n_gold += 1
        gold = set(case.gold)
        retrieved = {h.artifact_id for h in hits}
        gold_retrieved = gold & retrieved
        if not gold_retrieved:
            # Retrieval never surfaced the gold → the selector had no chance.
            # That is a *retrieval* miss (measured by evaluate_discovery), so it
            # is excluded from every selection-quality metric here — keeping them
            # all on the one shared denominator, n_gold_retrieved.
            continue
        n_gold_retrieved += 1
        # selected ⊆ retrieved, so selected ∩ gold == selected ∩ gold_retrieved:
        # recall is measured against the gold the selector could actually pick.
        hit_gold = len(selected & gold)
        recall = hit_gold / len(gold_retrieved)
        precision = hit_gold / len(selected) if selected else None
        recalls.append(recall)
        f1s.append(_f1(precision, recall))
        selected_sizes.append(len(selected))
        if precision is not None:
            precisions.append(precision)
        if hit_gold:
            n_committed_given_retrieved += 1

    return SelectionReport(
        selection_precision=(sum(precisions) / len(precisions) if precisions else None),
        selection_recall=(sum(recalls) / len(recalls) if recalls else None),
        selection_f1=(sum(f1s) / len(f1s) if f1s else None),
        conditional_commit_rate=(
            n_committed_given_retrieved / n_gold_retrieved if n_gold_retrieved else None
        ),
        n_gold_retrieved=n_gold_retrieved,
        abstention_accuracy=(
            n_abstained_correct / n_abstention if n_abstention else None
        ),
        mean_selected_size=(
            sum(selected_sizes) / len(selected_sizes) if selected_sizes else None
        ),
        n_cases=len(scored),
        n_gold=n_gold,
        n_abstention=n_abstention,
        strategy=strategy if isinstance(strategy, str) else "custom",
        mode=mode,
        k=k,
    )


# =========================================================================== #
# Selector tuning — sweep the commit knobs against the cases
# =========================================================================== #


def _metric_value(report: SelectionReport, metric: str) -> float | None:
    """A named scalar metric off a :class:`SelectionReport` (``None`` if undefined)."""
    if metric not in SWEEP_METRICS:
        raise ValueError(
            f"unknown selection metric {metric!r}; expected one of {SWEEP_METRICS}"
        )
    return getattr(report, metric)


def _rank_key(value: float | None, metric: str) -> float:
    """Sort key for *maximizing desirability* of a metric (``None`` → worst).

    A metric in :data:`_SWEEP_MINIMIZE` (only ``mean_selected_size``) is negated
    so that "smaller is better" still sorts as "bigger key is better"; a ``None``
    maps to ``-inf`` so an undefined metric never wins.
    """
    if value is None:
        return float("-inf")
    return -value if metric in _SWEEP_MINIMIZE else value


@dataclass(frozen=True)
class SelectionGridPoint:
    """One cell of a selector sweep: a parameter setting and the report it scored.

    Attributes:
        max_k / rel / min_score: the swept selection parameters at this cell.
        report: the :class:`SelectionReport` for those parameters.
    """

    max_k: int
    rel: float
    min_score: float | None
    report: SelectionReport

    @property
    def params(self) -> dict[str, Any]:
        """The swept parameters as a plain dict (``max_k`` / ``rel`` / ``min_score``)."""
        return {"max_k": self.max_k, "rel": self.rel, "min_score": self.min_score}

    def to_dict(self) -> dict:
        """JSON-serializable form: the params plus the full report dict."""
        return {**self.params, "report": self.report.to_dict()}


@dataclass(frozen=True)
class SelectionSweep:
    """The outcome of :func:`sweep_selector` — a grid of selector settings scored.

    Every point shares the same cases, retrieval ``mode`` / candidate depth ``k``
    and ``strategy``; they differ only in the swept selection knobs (``max_k``,
    ``rel``, ``min_score``). :meth:`best` picks a single setting by an objective;
    :meth:`frontier` returns the precision/recall Pareto front for the honest
    trade-off view; :meth:`table` renders the whole grid.

    Attributes:
        points: one :class:`SelectionGridPoint` per grid cell (grid order).
        objective: the default metric :meth:`best` optimizes.
        strategy / mode / k: the shared configuration the grid was scored under.
        n_cases / n_gold / n_abstention / n_gold_retrieved: case counts (echoed
            from the reports; selection metrics share the ``n_gold_retrieved``
            denominator, so it is surfaced here too).
    """

    points: list[SelectionGridPoint]
    objective: str
    strategy: str
    mode: str
    k: int
    n_cases: int
    n_gold: int
    n_abstention: int
    n_gold_retrieved: int

    def best(
        self, metric: str | None = None, *, minimize: bool | None = None
    ) -> SelectionGridPoint:
        """The grid point that optimizes ``metric`` (default :attr:`objective`).

        Ties are broken toward the *cheaper, tighter* commit — smaller
        ``mean_selected_size``, then smaller ``max_k``, then larger ``rel`` —
        encoding the capability-discovery stance that fewer, higher-precision
        commits beat more (ir_01 §3). ``minimize`` is inferred from the metric
        (only ``mean_selected_size`` minimizes) but can be forced.

        Raises:
            ValueError: if the sweep is empty or ``metric`` is unknown.
        """
        if not self.points:
            raise ValueError("cannot pick best of an empty sweep")
        metric = metric or self.objective
        if metric not in SWEEP_METRICS:
            raise ValueError(
                f"unknown selection metric {metric!r}; expected one of {SWEEP_METRICS}"
            )
        do_min = (metric in _SWEEP_MINIMIZE) if minimize is None else minimize

        def desirability(p: SelectionGridPoint) -> float:
            v = _metric_value(p.report, metric)
            if v is None:
                return float("-inf")
            return -v if do_min else v

        def sort_key(p: SelectionGridPoint) -> tuple:
            size = p.report.mean_selected_size
            return (
                -desirability(p),  # primary: best objective first
                size if size is not None else float("inf"),  # then smaller commit
                p.max_k,  # then a tighter cap
                -p.rel,  # then a stricter band
            )

        return min(self.points, key=sort_key)

    def frontier(
        self, *, x: str = "selection_recall", y: str = "selection_precision"
    ) -> list[SelectionGridPoint]:
        """The Pareto-optimal points trading ``x`` off against ``y`` (both maximized).

        A point is on the frontier when no other point is at least as good on both
        axes and strictly better on one. ``mean_selected_size`` on an axis is read
        as "smaller is better" automatically. Returned sorted by ``x`` ascending.
        """
        for m in (x, y):
            if m not in SWEEP_METRICS:
                raise ValueError(
                    f"unknown selection metric {m!r}; expected one of {SWEEP_METRICS}"
                )

        def coords(p: SelectionGridPoint) -> tuple[float, float]:
            return (
                _rank_key(_metric_value(p.report, x), x),
                _rank_key(_metric_value(p.report, y), y),
            )

        front: list[SelectionGridPoint] = []
        for p in self.points:
            px, py = coords(p)
            dominated = any(
                (qx >= px and qy >= py) and (qx > px or qy > py)
                for q in self.points
                for qx, qy in [coords(q)]
                if q is not p
            )
            if not dominated:
                front.append(p)
        return sorted(front, key=lambda p: coords(p)[0])

    def table(self, *, sort_by: str | None = None) -> str:
        """A human-readable grid table, one row per setting (best objective first).

        ``sort_by`` defaults to :attr:`objective`. ``min_score`` is shown only
        when any cell sets one (an all-``None`` column is elided as noise).
        """
        sort_by = sort_by or self.objective
        show_floor = any(p.min_score is not None for p in self.points)
        # _rank_key already maps "more desirable" → larger, so a single
        # descending sort orders the grid most-desirable-first for any metric.
        rows = sorted(
            self.points,
            key=lambda p: _rank_key(_metric_value(p.report, sort_by), sort_by),
            reverse=True,
        )

        def fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "n/a"

        head = ["max_k", "rel"]
        if show_floor:
            head.append("min_score")
        head += ["cond_commit", "precision", "recall", "F1", "sel_size"]
        lines = ["  ".join(f"{h:>11}" for h in head)]
        for p in rows:
            r = p.report
            cells = [str(p.max_k), f"{p.rel:.2f}"]
            if show_floor:
                cells.append("none" if p.min_score is None else f"{p.min_score:.3f}")
            cells += [
                fmt(r.conditional_commit_rate),
                fmt(r.selection_precision),
                fmt(r.selection_recall),
                fmt(r.selection_f1),
                fmt(r.mean_selected_size),
            ]
            lines.append("  ".join(f"{c:>11}" for c in cells))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable form: the grid, the best point, and the config."""
        best = self.best()
        return {
            "objective": self.objective,
            "strategy": self.strategy,
            "mode": self.mode,
            "k": self.k,
            "n_cases": self.n_cases,
            "n_gold": self.n_gold,
            "n_abstention": self.n_abstention,
            "n_gold_retrieved": self.n_gold_retrieved,
            "best": best.to_dict(),
            "points": [p.to_dict() for p in self.points],
        }

    def __str__(self) -> str:
        best = self.best()
        return "\n".join(
            [
                f"SelectionSweep — {len(self.points)} settings  "
                f"(strategy={self.strategy}, mode={self.mode}, k={self.k})",
                f"  {self.n_cases} cases ({self.n_gold} gold, "
                f"{self.n_abstention} abstention; {self.n_gold_retrieved} gold-retrieved)",
                f"  objective: {self.objective}",
                self.table(),
                f"  best ({self.objective}): max_k={best.max_k} rel={best.rel:.2f}"
                + (
                    f" min_score={best.min_score:.3f}"
                    if best.min_score is not None
                    else ""
                )
                + f"  →  {self.objective}="
                + (
                    f"{_metric_value(best.report, self.objective):.4f}"
                    if _metric_value(best.report, self.objective) is not None
                    else "n/a"
                ),
            ]
        )


def sweep_selector(
    corpus: Any,
    cases: Sequence[DiscoveryCase],
    *,
    strategy: str = "conservative",
    mode: str = DFLT_MODE,
    k: int = 10,
    surfaces: Iterable[str] | None = None,
    max_k_grid: Sequence[int] = DFLT_SWEEP_MAX_K,
    rel_grid: Sequence[float] = DFLT_SWEEP_REL,
    min_score_grid: Sequence[float | None] = DFLT_SWEEP_MIN_SCORE,
    gap_ratio: float = DFLT_GAP_RATIO,
    objective: str = DFLT_SWEEP_OBJECTIVE,
    **search_kw: Any,
) -> SelectionSweep:
    """Tune the selector: score a grid of commit knobs against the cases.

    The selection defaults (:data:`~ir.select.DFLT_MAX_K`,
    :data:`~ir.select.DFLT_REL_THRESHOLD`) were tuned on real corpora (see
    ir_06), not guessed. This sweeps ``max_k × rel × min_score`` and scores each
    cell with :func:`evaluate_selection`'s exact metric, so the right defaults
    for a *given* corpus can be read off empirically (precision vs recall vs
    commit size) rather than assumed.

    Retrieval is run **once per case** (selector parameters do not change the
    ranking) and the cached candidates are reused across the whole grid, so a
    6×5 grid costs one retrieval pass, not thirty.

    Args:
        corpus: an :class:`~ir.index.Corpus` or a registered corpus *name*.
        cases: the :class:`DiscoveryCase`\\ s to score against.
        strategy: the selection strategy held fixed across the grid (the swept
            knobs are ``conservative``'s; a non-``conservative`` strategy that
            ignores some knobs simply yields duplicate rows).
        mode: ranking mode for retrieval.
        k: candidate depth retrieved before selection (the selector's window).
        surfaces: restrict retrieval to these surface kinds.
        max_k_grid / rel_grid / min_score_grid: the axes of the sweep.
        gap_ratio: held fixed (used only by the ``score_gap`` strategy).
        objective: the default metric :meth:`SelectionSweep.best` optimizes.
        **search_kw: any other :func:`ir.retrieve.search` keyword.

    Returns:
        a :class:`SelectionSweep` (``.best()`` / ``.frontier()`` / ``.table()`` /
        ``.to_dict()``).
    """
    if objective not in SWEEP_METRICS:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of {SWEEP_METRICS}"
        )
    corpus = _as_corpus(corpus)
    scored = _retrieve_for_cases(
        corpus, cases, mode=mode, k=k, surfaces=surfaces, **search_kw
    )
    points: list[SelectionGridPoint] = []
    for max_k in max_k_grid:
        for rel in rel_grid:
            for min_score in min_score_grid:
                report = _score_selection(
                    scored,
                    strategy=strategy,
                    max_k=max_k,
                    rel=rel,
                    gap_ratio=gap_ratio,
                    min_score=min_score,
                    mode=mode,
                    k=k,
                )
                points.append(
                    SelectionGridPoint(
                        max_k=max_k, rel=rel, min_score=min_score, report=report
                    )
                )
    first = points[0].report if points else None
    return SelectionSweep(
        points=points,
        objective=objective,
        strategy=strategy,
        mode=mode,
        k=k,
        n_cases=first.n_cases if first else len(cases),
        n_gold=first.n_gold if first else 0,
        n_abstention=first.n_abstention if first else 0,
        n_gold_retrieved=first.n_gold_retrieved if first else 0,
    )


# =========================================================================== #
# Abstention-floor calibration — separating in-scope from out-of-scope queries
# =========================================================================== #
#
# The conservative selector is *relative* (ratios to the top score), so it is
# mode-agnostic but cannot tell "all candidates irrelevant" from "all relevant":
# both look like a high-scoring top with the tail falling off. Absolute
# abstention ("nothing applies") needs an *absolute* ``min_score`` floor — and
# that floor is mode-specific (dense cosine ∈ ~[0,1], hybrid RRF ∈ ~[0,0.033],
# lexical BM25 ∈ ~[0,30+]) and corpus/embedder-specific, so it cannot be a
# committed global constant. It must be *calibrated* per (corpus, mode).
#
# Calibration is a 1-D threshold-separation problem, NOT a selector sweep: an
# in-scope (gold-bearing, gold-retrieved) query produces a HIGH top score; an
# out-of-scope (empty-gold) query produces a LOWER top score (the best of a bad
# lot). The floor that best separates the two distributions is the abstention
# floor. (Optimizing ``sweep_selector(objective="abstention_accuracy")`` instead
# is degenerate: floor → ∞ abstains on everything, scoring 100% — at the cost of
# all recall. Separation is the honest objective.)


@dataclass(frozen=True)
class MinScoreCalibration:
    """A calibrated absolute abstention floor for one ``(corpus, mode)``.

    Produced by :func:`calibrate_min_score`. ``min_score`` is the floor to pass to
    :func:`ir.select.select` / :func:`ir.discover` (``discover(min_score="auto")``
    loads it): the conservative selector abstains when even the top hit scores
    below it. ``None`` means no floor could be calibrated (one of the two query
    classes was missing — see :attr:`reason`); the selector then never abstains
    by absolute score, exactly as before.

    The quality numbers treat ``top_score ≥ min_score → commit`` as a binary
    classifier over in-scope (label *commit*) vs out-of-scope (label *abstain*)
    queries:

    Attributes:
        min_score: the calibrated floor (``None`` if not calibratable).
        sensitivity: TPR — fraction of in-scope queries kept (``top ≥ floor``).
        specificity: TNR — fraction of out-of-scope queries abstained
            (``top < floor``).
        youden_j: ``sensitivity + specificity − 1`` (0 = no separation, 1 = perfect).
        balanced_accuracy: ``(sensitivity + specificity) / 2``.
        separable: whether the two score distributions are perfectly separable
            (some floor reaches ``youden_j == 1``).
        sensitivity_weight: the ``w`` used to pick the floor (see
            :data:`DFLT_CALIB_SENSITIVITY_WEIGHT`).
        n_positive: in-scope cases used (gold-bearing **and** gold-retrieved).
        n_abstention: out-of-scope (empty-gold) cases used.
        n_retrieval_miss: gold-bearing cases whose gold never reached the ``k``
            candidates — excluded (a *retrieval* failure, not an abstention
            signal), mirroring :func:`evaluate_selection`'s conditioning.
        positive_scores / abstention_scores: ``{min, median, max}`` summaries of
            each class's top-score distribution (the auditable evidence the floor
            sits between them).
        grid: every evaluated floor, ``{min_score, sensitivity, specificity,
            youden_j, objective}`` — the full sweep behind the choice.
        reason: why ``min_score`` is what it is (``"calibrated"``,
            ``"no_positive_cases"``, ``"no_abstention_cases"``, ``"no_cases"``).
        corpus_name / mode / k / embedder_id: the configuration scored — and the
            ``embedder_id`` stamp lets a consumer detect a stale floor after a
            rebuild with a different embedder.
        vd_degraded: True when ``mode`` was ``lexical`` / ``hybrid`` but ``vd``
            was unavailable, so ranking fell back to dense and the floor is on the
            *dense* scale, not the requested mode's — recalibrate once ``vd`` is
            installed.
    """

    corpus_name: str
    mode: str
    k: int
    min_score: float | None
    sensitivity: float | None
    specificity: float | None
    youden_j: float | None
    balanced_accuracy: float | None
    separable: bool
    sensitivity_weight: float
    n_positive: int
    n_abstention: int
    n_retrieval_miss: int
    positive_scores: Mapping[str, float]
    abstention_scores: Mapping[str, float]
    grid: list[dict]
    reason: str
    embedder_id: str | None = None
    strategy: str = "conservative"
    vd_degraded: bool = False

    def to_dict(self) -> dict:
        """JSON-serializable form (for persistence and the qh / HTTP surface)."""
        return {
            "corpus_name": self.corpus_name,
            "mode": self.mode,
            "k": self.k,
            "min_score": self.min_score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "youden_j": self.youden_j,
            "balanced_accuracy": self.balanced_accuracy,
            "separable": self.separable,
            "sensitivity_weight": self.sensitivity_weight,
            "n_positive": self.n_positive,
            "n_abstention": self.n_abstention,
            "n_retrieval_miss": self.n_retrieval_miss,
            "positive_scores": dict(self.positive_scores),
            "abstention_scores": dict(self.abstention_scores),
            "grid": [dict(g) for g in self.grid],
            "reason": self.reason,
            "embedder_id": self.embedder_id,
            "strategy": self.strategy,
            "vd_degraded": self.vd_degraded,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MinScoreCalibration":
        """Inverse of :meth:`to_dict` (tolerant of a persisted record's defaults)."""
        return cls(
            corpus_name=d.get("corpus_name", ""),
            mode=d.get("mode", DFLT_MODE),
            k=int(d.get("k", 10)),
            min_score=d.get("min_score"),
            sensitivity=d.get("sensitivity"),
            specificity=d.get("specificity"),
            youden_j=d.get("youden_j"),
            balanced_accuracy=d.get("balanced_accuracy"),
            separable=bool(d.get("separable", False)),
            sensitivity_weight=float(
                d.get("sensitivity_weight", DFLT_CALIB_SENSITIVITY_WEIGHT)
            ),
            n_positive=int(d.get("n_positive", 0)),
            n_abstention=int(d.get("n_abstention", 0)),
            n_retrieval_miss=int(d.get("n_retrieval_miss", 0)),
            positive_scores=dict(d.get("positive_scores", {})),
            abstention_scores=dict(d.get("abstention_scores", {})),
            grid=[dict(g) for g in d.get("grid", [])],
            reason=d.get("reason", "calibrated"),
            embedder_id=d.get("embedder_id"),
            strategy=d.get("strategy", "conservative"),
            vd_degraded=bool(d.get("vd_degraded", False)),
        )

    def save(self, corpus: Any) -> "MinScoreCalibration":
        """Persist this calibration on ``corpus`` (keyed by mode); returns self.

        Stored in the corpus's ``calibration`` view (not its build ``config``), so
        it survives reopen and is loaded by ``discover(min_score="auto")``. The
        floor is machine-local (derived from a live corpus + embedder) and is
        never committed to source control.
        """
        corpus = _as_corpus(corpus)
        corpus.store.set_calibration(self.mode, self.to_dict())
        return self

    def __str__(self) -> str:
        def fmt(x: float | None) -> str:
            return f"{x:.4f}" if x is not None else "n/a"

        def span(s: Mapping[str, float]) -> str:
            if not s:
                return "n/a"
            return (
                f"[{s.get('min', float('nan')):.4f} … {s.get('max', float('nan')):.4f}]"
            )

        floor = "none" if self.min_score is None else f"{self.min_score:.4f}"
        return "\n".join(
            [
                f"MinScoreCalibration — corpus={self.corpus_name!r} mode={self.mode} "
                f"k={self.k}",
                f"  min_score floor: {floor}  ({self.reason})",
                f"  in-scope:  {self.n_positive} cases  top-score {span(self.positive_scores)}",
                f"  out-scope: {self.n_abstention} cases  top-score "
                f"{span(self.abstention_scores)}",
                f"  sensitivity: {fmt(self.sensitivity)}  "
                f"specificity: {fmt(self.specificity)}  "
                f"Youden J: {fmt(self.youden_j)}"
                + ("  (perfectly separable)" if self.separable else ""),
                f"  retrieval misses excluded: {self.n_retrieval_miss}  "
                f"(w={self.sensitivity_weight})"
                + (
                    "\n  DEGRADED: vd unavailable → scored on dense fallback, not "
                    f"{self.mode!r}; install vd and recalibrate"
                    if self.vd_degraded
                    else ""
                ),
            ]
        )


def calibrate_min_score(
    corpus: Any,
    cases: Sequence[DiscoveryCase],
    *,
    mode: str = DFLT_MODE,
    k: int = 10,
    surfaces: Iterable[str] | None = None,
    sensitivity_weight: float = DFLT_CALIB_SENSITIVITY_WEIGHT,
    floor_grid: Sequence[float] | None = None,
    persist: bool = False,
    **search_kw: Any,
) -> MinScoreCalibration:
    """Calibrate the absolute abstention ``min_score`` floor for one ``mode``.

    Retrieves ``k`` candidates per case **once** (reusing :func:`evaluate_selection`'s
    retrieval pass), then treats floor-picking as a 1-D separation between the
    *in-scope* top scores (gold-bearing cases whose gold reached the candidates)
    and the *out-of-scope* top scores (empty-gold abstention cases). The floor
    that best separates them — by ``w·sensitivity + (1−w)·specificity`` — is the
    calibrated ``min_score``.

    ``cases`` must contain **both** gold-bearing and abstention cases; with only
    one class the floor is undefined (``min_score=None``, ``reason`` records which
    class was missing). Generate abstention cases with :mod:`ir.eval_gen` (its
    ``abstention`` slice) — this module does not synthesize them.

    Args:
        corpus: an :class:`~ir.index.Corpus` or a registered corpus *name*.
        cases: mixed :class:`DiscoveryCase`\\ s (gold-bearing + abstention).
        mode: ranking mode to calibrate (a floor is mode-specific).
        k: candidate depth retrieved before selection (match your ``discover`` k).
        surfaces: restrict retrieval to these surface kinds.
        sensitivity_weight: ``w`` in the pick objective (see
            :data:`DFLT_CALIB_SENSITIVITY_WEIGHT`); ``0.5`` is balanced.
        floor_grid: candidate floors to evaluate; default derives them from the
            observed top scores (midpoints between adjacent values + the
            commit-all / abstain-all extremes), which is exact for a 1-D split.
        persist: when True, :meth:`MinScoreCalibration.save` the result on the
            corpus so ``discover(min_score="auto")`` will load it.
        **search_kw: any other :func:`ir.retrieve.search` keyword.

    Returns:
        a :class:`MinScoreCalibration`.
    """
    corpus = _as_corpus(corpus)
    if floor_grid is not None:
        floor_grid = list(floor_grid)
        bad = [f for f in floor_grid if not math.isfinite(f)]
        if not floor_grid or bad:
            raise ValueError(
                "floor_grid must be a non-empty sequence of finite floats "
                f"(got {floor_grid!r}); pass floor_grid=None to derive candidates "
                "from the observed scores."
            )
    # lexical / hybrid ranking silently falls back to dense when vd is missing
    # (ir.retrieve), so the scores — and thus the floor — would be on the dense
    # scale, not the requested mode's. Flag it so the calibration is auditable
    # and a consumer can recalibrate once vd is installed.
    vd_degraded = mode in ("hybrid", "lexical") and not _vd_available()
    scored = _retrieve_for_cases(
        corpus, cases, mode=mode, k=k, surfaces=surfaces, **search_kw
    )

    # Split cases into the two score distributions the floor must separate.
    # An empty retrieval already commits nothing, so it abstains at any finite
    # floor → represent its top score as −inf (below every candidate floor).
    pos: list[float] = []  # in-scope: gold-bearing AND gold reached the candidates
    neg: list[float] = []  # out-of-scope: abstention (empty-gold) cases
    n_retrieval_miss = 0
    for case, hits in scored:
        top = float(hits[0].score) if hits else float("-inf")
        if case.gold_is_none:
            neg.append(top)
            continue
        retrieved = {h.artifact_id for h in hits}
        if set(case.gold) & retrieved:
            pos.append(top)
        else:
            # Gold never surfaced → a retrieval miss, not an abstention signal.
            # Excluded so the floor is not blamed for retrieval's recall gap
            # (the same conditioning evaluate_selection uses).
            n_retrieval_miss += 1

    reason = "calibrated"
    if not pos and not neg:
        reason = "no_cases"
    elif not pos:
        reason = "no_positive_cases"
    elif not neg:
        reason = "no_abstention_cases"

    pos_summary = _score_summary(pos)
    neg_summary = _score_summary([s for s in neg if s != float("-inf")])

    if reason != "calibrated":
        # Cannot separate without both classes → no floor (selector never
        # abstains by absolute score, exactly as with min_score=None).
        return MinScoreCalibration(
            corpus_name=corpus.name,
            mode=mode,
            k=k,
            min_score=None,
            sensitivity=(1.0 if pos else None),
            specificity=(1.0 if neg else None),
            youden_j=None,
            balanced_accuracy=None,
            separable=False,
            sensitivity_weight=sensitivity_weight,
            n_positive=len(pos),
            n_abstention=len(neg),
            n_retrieval_miss=n_retrieval_miss,
            positive_scores=pos_summary,
            abstention_scores=neg_summary,
            grid=[],
            reason=reason,
            embedder_id=getattr(corpus, "embedder_id", None),
            vd_degraded=vd_degraded,
        )

    floors = (
        list(floor_grid) if floor_grid is not None else _candidate_floors(pos + neg)
    )
    grid = [_score_floor(f, pos, neg) for f in floors]
    best = _pick_floor(grid, sensitivity_weight)
    separable = any(g["youden_j"] >= 1.0 - 1e-9 for g in grid)

    calib = MinScoreCalibration(
        corpus_name=corpus.name,
        mode=mode,
        k=k,
        min_score=best["min_score"],
        sensitivity=best["sensitivity"],
        specificity=best["specificity"],
        youden_j=best["youden_j"],
        balanced_accuracy=(best["sensitivity"] + best["specificity"]) / 2,
        separable=separable,
        sensitivity_weight=sensitivity_weight,
        n_positive=len(pos),
        n_abstention=len(neg),
        n_retrieval_miss=n_retrieval_miss,
        positive_scores=pos_summary,
        abstention_scores=neg_summary,
        grid=grid,
        reason=reason,
        embedder_id=getattr(corpus, "embedder_id", None),
        vd_degraded=vd_degraded,
    )
    if persist:
        calib.save(corpus)
    return calib


def load_calibration(corpus: Any, mode: str) -> MinScoreCalibration | None:
    """Load a persisted :class:`MinScoreCalibration` for ``(corpus, mode)``.

    Returns ``None`` when no calibration has been stored for that mode. Does not
    validate the embedder stamp — that staleness check lives at the
    ``discover(min_score="auto")`` boundary, which knows the live corpus's
    embedder id.
    """
    corpus = _as_corpus(corpus)
    rec = corpus.store.get_calibration(mode)
    return MinScoreCalibration.from_dict(rec) if rec else None


def _score_summary(scores: Sequence[float]) -> dict[str, float]:
    """``{min, median, max}`` of a score list (non-finite dropped; empty → ``{}``).

    Drops ``±inf`` / ``NaN`` (a NaN never equals itself) so the summary stays
    JSON-serializable and never poisons the persisted record.
    """
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return {}
    return {
        "min": float(min(finite)),
        "median": float(statistics.median(finite)),
        "max": float(max(finite)),
    }


def _candidate_floors(scores: Sequence[float]) -> list[float]:
    """Candidate floors for a 1-D split: midpoints of adjacent observed scores.

    The optimal threshold separating two sets of reals always lies between two
    adjacent observed values, so the midpoints (plus a commit-all floor below the
    minimum and an abstain-all floor above the maximum) are an *exact* search
    grid — no resolution to tune. ``−inf`` scores (empty retrievals) are dropped:
    they abstain at any finite floor and so never constrain the choice.
    """
    finite = sorted({s for s in scores if s != float("-inf") and s == s})
    if not finite:
        return [0.0]
    if len(finite) == 1:
        v = finite[0]
        margin = abs(v) * 0.5 + 1.0
        return [v - margin, v + margin]
    span = finite[-1] - finite[0]
    margin = (span / len(finite)) if span > 0 else 1.0
    mids = [(finite[i] + finite[i + 1]) / 2 for i in range(len(finite) - 1)]
    return [finite[0] - margin, *mids, finite[-1] + margin]


def _score_floor(floor: float, pos: Sequence[float], neg: Sequence[float]) -> dict:
    """Confusion of ``top_score ≥ floor → commit`` over the two classes.

    Sensitivity (TPR) is the in-scope commit rate; specificity (TNR) is the
    out-of-scope abstain rate. A ``−inf`` abstention score (empty retrieval)
    is below every finite floor, so it always abstains correctly.
    """
    sens = sum(s >= floor for s in pos) / len(pos) if pos else 0.0
    spec = sum(s < floor for s in neg) / len(neg) if neg else 0.0
    return {
        "min_score": float(floor),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "youden_j": float(sens + spec - 1.0),
    }


def _pick_floor(grid: Sequence[dict], sensitivity_weight: float) -> dict:
    """Pick the grid floor maximizing ``w·sensitivity + (1−w)·specificity``.

    Ties (common — every floor inside one gap scores identically) break toward
    the *higher* floor: the precision-leaning choice (ir_01 §3 — a padded commit
    costs more than a dropped gold), and the more robust separator when the gap
    above the out-of-scope scores is wide.
    """
    w = sensitivity_weight

    def objective(g: dict) -> float:
        return w * g["sensitivity"] + (1.0 - w) * g["specificity"]

    return max(grid, key=lambda g: (objective(g), g["min_score"]))
