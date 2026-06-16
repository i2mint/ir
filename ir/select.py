"""Selection & progressive disclosure — committing to a subset, then revealing it.

Retrieval (:func:`ir.retrieve.search`) ranks *candidates*; **selection** commits
to the small, high-precision *subset* an agent should actually act on, and
**disclosure** loads the heavy payload (a SKILL.md body, a package pointer, a
file's full text) for only those committed items — and only when asked.

The split is deliberate and grounded in the capability-discovery research
(``misc/docs/ir_01``):

- *Fewer, higher-precision candidates beat more.* The distractor problem — the
  central selection risk — worsens, not improves, as you retrieve more. So the
  default selector is **conservative and adaptive**, not a fixed top-*k*: it
  keeps the best candidate and admits the next only while the score stays close
  to the top (relative threshold), capped at a small ``max_k``. A score-gap
  "elbow" cut is offered as its own strategy (:func:`score_gap`) rather than
  folded into the default, where it would only duplicate the relative threshold.
- *Progressive disclosure is append-only.* Naively injecting full payloads into
  context destroys prompt-cache hits. Disclosure here is a *pure* function that
  reads pointers already stored on each hit and returns **new** payload objects;
  it never mutates the ranked hits and never re-embeds. The append-only *benefit*
  is the caller's to realize, not ``ir``'s to enforce: ``ir`` only hands back
  additive :class:`Disclosure` objects. To keep the prompt cache warm in an
  agent loop, append each disclosed payload into the **message history** (the
  cheap, append-only end of the prefix) rather than mutating the tool
  definitions or an earlier system block — exactly the discipline ir_01 §5
  prescribes.

Three composable entry points, smallest surface first:

- :func:`select` — ranked hits → a :class:`Selection` (a committed subset, or
  abstention). Pure; offline; no model.
- :func:`disclose` — a :class:`Selection` → per-item :class:`Disclosure`
  payloads at a chosen level (``"metadata"`` / ``"body"`` / ``"bundled"``).
  Orthogonally, ``expand=`` (a :data:`~ir.expand.NeighborhoodPolicy`, with
  ``corpus=``) stitches each hit's stored neighborhood into
  :attr:`Disclosure.passage` — the mid-granularity payload between the matched
  surface and the pointer's full body (see :mod:`ir.expand`).
- :func:`discover` — the single agent-callable tool: retrieve → select →
  (optionally) disclose, returning a JSON-serializable :class:`DiscoveryResult`.
  This is the qh-exposable surface (pass a corpus *name*; get back ``.to_dict()``).
  Pass a **list** of names for single-shot *federated* discovery: each source is
  searched and gated on its own calibrated floor (pre-fusion), then the
  survivors rank-fuse via :func:`ir.fuse_hits` — raw scores never cross a
  source boundary (ir_07/ir_08). The caller names the sources; ir never
  chooses the set.

Selection scores are compared **relatively** (ratios to the top score), so the
same selector works across ``dense`` cosine, ``hybrid`` RRF, and ``lexical``
BM25 — whose absolute scales differ by orders of magnitude — without
per-mode calibration. Absolute abstention (``"nothing applies"``) requires
either an explicit ``min_score`` floor or an LLM selector; pure relative
structure cannot tell "all irrelevant" from "all relevant". Because a useful
floor is mode- and corpus-specific, it is *calibrated* rather than guessed:
:func:`ir.eval.calibrate_min_score` separates in-scope from out-of-scope query
scores and persists the floor, and :func:`discover` loads it on
``min_score="auto"`` — the opt-in that turns relative selection into one that
can also abstain.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import POINTER_KEYS, SearchHit, best_per_artifact
from .expand import NeighborhoodPolicy, SeedNotFound
from .expand import expand as _expand
from .retrieve import NoLedgerEntry
from .retrieve import search as _search

#: A selector: ranked hits → the committed subset (a pure ranking decision).
Selector = Callable[[Sequence[SearchHit]], list[SearchHit]]

#: A body loader: a hit's metadata → its disclosed payload text (or ``None``).
BodyLoader = Callable[[Mapping[str, Any]], "str | None"]

#: A resource store: ``pointer -> payload`` (ir_09 §5). Any ``Mapping`` works —
#: a ``dol`` file/blob/URL store, an in-memory dict, etc. — so lazy disclosure
#: (``store[pointer]``) is decoupled from local disk.
ResourceStore = Mapping[str, Any]

#: Max items the conservative selector will ever commit to. Small on purpose:
#: ir_01 §3 ("fewer, higher-precision candidates beat more"), echoing MCP-Atlas's
#: 3–7 target tools per task. Tuned to 3 in ir_06 (the F1-optimal commit cap over
#: real skills/packages/reports corpora); leaves headroom for genuine multi-gold
#: queries while the strict ``rel`` band keeps padding out.
DFLT_MAX_K = 3

#: Keep a follow-on hit only if its score is at least this fraction of the top.
#: 0.9 is a strict "near-tie only" band: for MiniLM cosine and RRF hybrid the
#: informative second-best sits within ~10–15% of the top, while the distractor
#: tail starts below ~0.8×top, so a looser band just rakes in distractors. Tuned
#: empirically in ir_06 (``sweep_selector`` over three real corpora, both modes —
#: ``rel=0.6`` was dominated everywhere). Loosen toward 0 to admit more recall at
#: the cost of precision; ``ir sweep-select`` re-tunes against your own corpus.
DFLT_REL_THRESHOLD = 0.9

#: Cut before a hit whose score drops below this fraction of the previous one
#: (the score-gap "elbow"): a sharp relative drop marks the end of the signal.
#: Used by the ``score_gap`` strategy only — the conservative default uses the
#: relative-to-top threshold above, which would otherwise subsume a weaker elbow.
DFLT_GAP_RATIO = 0.5

#: Candidate depth retrieved before selection in :func:`discover`.
DFLT_FETCH_K = 10

#: Disclosure levels, cheapest → richest (each a superset of the previous).
DISCLOSURE_LEVELS = ("metadata", "body", "bundled")

#: Truncate a disclosed body to this many characters (a guard, not a feature).
DFLT_MAX_BODY_CHARS = 20000

# POINTER_KEYS is imported from ir.base (the data-model SSOT) and re-exported
# here for backward compatibility with ``ir.select.POINTER_KEYS`` callers.


# =========================================================================== #
# Selection result
# =========================================================================== #


@dataclass(frozen=True)
class Selection:
    """A selector's commitment: the chosen subset of a ranked candidate list.

    Attributes:
        selected: the committed hits, best-first (empty iff ``abstained``).
        candidates: the full ranked input, kept for provenance / audit.
        abstained: True iff the selector committed to nothing by policy.
        reason: which rule ended the commit (e.g. ``"rel_threshold"``,
            ``"score_gap"``, ``"max_k"``, ``"abstain:below_floor"``).
        signals: concrete, defined numbers behind the decision (``top_score``,
            ``n_candidates``, ``n_selected``, ``min_ratio``) — the auditable
            replacement for an opaque "confidence" float.
    """

    selected: list[SearchHit]
    candidates: list[SearchHit]
    abstained: bool
    reason: str
    signals: Mapping[str, Any] = field(default_factory=dict)

    @property
    def selected_ids(self) -> list[str]:
        """The committed artifact ids, best-first."""
        return [h.artifact_id for h in self.selected]

    @property
    def sufficient(self) -> bool:
        """A model-free sufficiency *hint* for an agent's Evaluator (ir_09 §3).

        ``True`` when this selection committed to at least one item (i.e. did not
        abstain). It is a **signal, not a directive**: the re-query / ``refinement``
        decision and the loop belong to the agent layer (the back-edge, ir_09 §4)
        — ``ir`` derives this from its own outcome and never acts on it.
        """
        return not self.abstained and len(self.selected) > 0

    def to_dict(self) -> dict:
        """JSON-serializable form (scores cast to ``float``)."""
        return {
            "selected": [_hit_to_dict(h) for h in self.selected],
            "selected_ids": self.selected_ids,
            "abstained": self.abstained,
            "sufficient": self.sufficient,
            "reason": self.reason,
            "signals": dict(self.signals),
            "n_candidates": len(self.candidates),
        }


def _hit_to_dict(hit: SearchHit) -> dict:
    """A SearchHit as a plain JSON-serializable dict."""
    return hit.to_dict()


# =========================================================================== #
# Selectors — plain, composable factories
# =========================================================================== #


def top_k(k: int = DFLT_MAX_K) -> Selector:
    """The naive baseline: always commit to the first ``k`` hits."""

    def selector(hits: Sequence[SearchHit]) -> list[SearchHit]:
        return list(hits[:k])

    return selector


def abs_threshold(min_score: float) -> Selector:
    """Keep every hit scoring at or above an absolute ``min_score``.

    Mode-specific (cosine / RRF / BM25 scales differ); use when you have
    calibrated a floor for one ranking mode. Like :func:`rel_threshold` /
    :func:`score_gap`, this does **not** cap how many hits it keeps — bound the
    commit size with the caller's ``max_k`` (:func:`select` already applies it).
    """

    def selector(hits: Sequence[SearchHit]) -> list[SearchHit]:
        return [h for h in hits if h.score >= min_score]

    return selector


def rel_threshold(ratio: float = DFLT_REL_THRESHOLD) -> Selector:
    """Keep hits scoring at least ``ratio`` × the top score (mode-agnostic).

    Degenerate-input guard: a non-positive top score makes the ratio test
    meaningless (e.g. a hashing embedder yielding negative cosines for an
    unrelated query), so only the single best hit is kept.
    """

    def selector(hits: Sequence[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        top = hits[0].score
        if top <= 0:
            return [hits[0]]
        return [h for h in hits if h.score >= ratio * top]

    return selector


def score_gap(gap_ratio: float = DFLT_GAP_RATIO) -> Selector:
    """Elbow cut: keep hits until one drops below ``gap_ratio`` × the previous.

    A sharp relative drop marks where the ranked signal ends. Flat or
    non-positive distributions have no such elbow, so the whole list passes
    (the floor for over-selection is the caller's ``max_k`` / threshold).
    """

    def selector(hits: Sequence[SearchHit]) -> list[SearchHit]:
        chosen: list[SearchHit] = []
        prev = None
        for h in hits:
            if prev is not None and (prev <= 0 or h.score < gap_ratio * prev):
                break
            chosen.append(h)
            prev = h.score
        return chosen

    return selector


def _conservative_select(
    hits: Sequence[SearchHit],
    *,
    max_k: int,
    ratio: float,
    min_score: float | None,
) -> tuple[list[SearchHit], str, dict]:
    """The default distractor-robust commit: ``(selected, reason, signals)``.

    Keep the best hit, then admit each next hit only while it stays *close* to
    the top score (``score_i >= ratio * top``), up to ``max_k``. The test is
    relative-to-top, so it is mode-agnostic (cosine / RRF / BM25 scales differ).
    Abstain only when there are no candidates, or an explicit ``min_score`` floor
    is set and even the top falls below it — relative structure alone cannot
    justify "nothing applies". (For an elbow cut on a consecutive cliff, use the
    :func:`score_gap` strategy instead.)
    """
    n = len(hits)
    signals: dict[str, Any] = {
        "n_candidates": n,
        "top_score": None,
        "min_ratio": None,
        "n_selected": 0,
    }
    if n == 0:
        return [], "abstain:no_candidates", signals

    top = float(hits[0].score)
    signals["top_score"] = top
    if min_score is not None and top < min_score:
        signals["min_score"] = float(min_score)
        return [], "abstain:below_floor", signals

    chosen = [hits[0]]
    limit = min(n, max_k)
    min_ratio = 1.0
    cut: str | None = None
    if top > 0:
        for i in range(1, limit):
            r = float(hits[i].score) / top
            if r < ratio:
                cut = "rel_threshold"
                break
            chosen.append(hits[i])
            # Track the edge only over *accepted* hits, so the signal reports how
            # close the commit came to its threshold — not the rejected cliff.
            min_ratio = min(min_ratio, r)
    # top <= 0 → the ratio test is meaningless; keep only the single best hit.

    if cut:
        reason = cut
    elif n == 1:
        reason = "single"
    elif top <= 0:
        reason = "nonpositive_top"
    elif limit < n:
        reason = "max_k"
    else:
        reason = "exhausted"
    signals["min_ratio"] = min_ratio
    signals["n_selected"] = len(chosen)
    return chosen, reason, signals


#: Named selector strategies that take no query (built from :func:`select` kwargs).
_NAMED: dict[str, Callable[..., Selector]] = {
    "top_k": top_k,
    "abs_threshold": abs_threshold,
    "rel_threshold": rel_threshold,
    "score_gap": score_gap,
}


def select(
    hits: Sequence[SearchHit],
    *,
    strategy: str | Selector = "conservative",
    max_k: int = DFLT_MAX_K,
    rel: float = DFLT_REL_THRESHOLD,
    gap_ratio: float = DFLT_GAP_RATIO,
    min_score: float | None = None,
) -> Selection:
    """Commit to a subset of ranked ``hits`` — the selection stage.

    Args:
        hits: ranked :class:`~ir.base.SearchHit`\\ s (best first), as returned by
            :func:`ir.retrieve.search`.
        strategy: ``"conservative"`` (the default distractor-robust commit),
            one of ``"top_k"`` / ``"abs_threshold"`` / ``"rel_threshold"`` /
            ``"score_gap"``, or any :data:`Selector` callable (``hits ->
            subset``) — e.g. one built by :func:`make_llm_selector`.
        max_k: never commit to more than this many (caps distractor exposure).
        rel: relative-to-top keep threshold for ``"conservative"`` / the ratio
            for ``"rel_threshold"``.
        gap_ratio: score-gap elbow ratio — used by the ``"score_gap"`` strategy
            only (``"conservative"`` deliberately uses ``rel`` alone, not an
            elbow; see this module's docstring).
        min_score: optional absolute floor; with ``"conservative"`` the selector
            abstains when even the top hit falls below it (also usable as the
            ``"abs_threshold"`` floor).

    Returns:
        a :class:`Selection`. ``abstained`` is True iff ``selected`` is empty.
    """
    hits = list(hits)
    if strategy == "conservative":
        chosen, reason, signals = _conservative_select(
            hits, max_k=max_k, ratio=rel, min_score=min_score
        )
        return Selection(
            selected=chosen,
            candidates=hits,
            abstained=not chosen,
            reason=reason,
            signals=signals,
        )

    selector = _resolve_selector(
        strategy, max_k=max_k, rel=rel, gap_ratio=gap_ratio, min_score=min_score
    )
    chosen = list(selector(hits))[:max_k]
    top = float(hits[0].score) if hits else None
    name = strategy if isinstance(strategy, str) else "custom"
    return Selection(
        selected=chosen,
        candidates=hits,
        abstained=not chosen,
        reason=name if chosen else f"abstain:{name}",
        signals={
            "n_candidates": len(hits),
            "n_selected": len(chosen),
            "top_score": top,
        },
    )


def _resolve_selector(
    strategy: str | Selector,
    *,
    max_k: int,
    rel: float,
    gap_ratio: float,
    min_score: float | None,
) -> Selector:
    """Resolve a strategy name (or pass a callable through) to a :data:`Selector`."""
    if callable(strategy):
        return strategy
    if strategy not in _NAMED:
        raise ValueError(
            f"unknown selection strategy {strategy!r}; expected 'conservative', "
            f"a callable, or one of {sorted(_NAMED)}"
        )
    factory = _NAMED[strategy]
    if strategy == "top_k":
        return factory(max_k)
    if strategy == "abs_threshold":
        if min_score is None:
            raise ValueError("strategy='abs_threshold' requires min_score=...")
        return factory(min_score)
    if strategy == "rel_threshold":
        return factory(rel)
    return factory(gap_ratio)  # score_gap


# =========================================================================== #
# LLM-as-selector — optional, lazy, injectable (mirrors ir.eval_gen)
# =========================================================================== #

SELECTION_PROMPT = """\
A user made this request:

{query}

Below are candidate capabilities (id and description). Choose ONLY the ones that
genuinely help with the request — prefer few, high-precision choices over many.
If none apply, choose none.

{candidates}

Reply with the chosen ids, one per line, and nothing else. No numbering, no prose.
"""


def make_llm_selector(
    query: str,
    *,
    chooser: Callable[..., Sequence[str]] | None = None,
    prompt: str = SELECTION_PROMPT,
    max_candidates: int = DFLT_FETCH_K,
    fallback: Selector | None = None,
    **prompt_function_kwargs: Any,
) -> Selector:
    """An LLM-as-selector :data:`Selector`, grounded in the research's caveats.

    The model reads the candidates' descriptions and commits to a subset.
    ``chooser`` is an injectable ``(query, candidates) -> [id, …]`` callable
    (a test double, or your own router); when omitted it is built lazily on
    :mod:`aix` (``aix.prompt_func``), so importing this module stays offline.

    Robustness: any error or empty/garbled reply falls back to ``fallback``
    (default: the ``"conservative"`` heuristic), because LLM selection is known
    to be swayed by description phrasing (ir_01 §3) — it must never be the only
    line of defense.
    """

    def choose(hits: Sequence[SearchHit]) -> Sequence[str]:
        cands = list(hits)[:max_candidates]
        rendered = "\n".join(f"- {h.artifact_id}: {h.text}" for h in cands)
        fn = (
            chooser
            if chooser is not None
            else _default_llm_chooser(prompt, **prompt_function_kwargs)
        )
        return fn(query=query, candidates=rendered)

    def selector(hits: Sequence[SearchHit]) -> list[SearchHit]:
        by_id = {h.artifact_id: h for h in hits}
        try:
            picked_ids = list(choose(hits))
        except Exception:
            picked_ids = []
        chosen = [by_id[i] for i in picked_ids if i in by_id]
        if chosen:
            return chosen
        fb = fallback if fallback is not None else _conservative_selector()
        return fb(hits)

    return selector


def _conservative_selector() -> Selector:
    """The default heuristic selector as a bare :data:`Selector` (for fallback)."""

    def selector(hits: Sequence[SearchHit]) -> list[SearchHit]:
        chosen, _reason, _signals = _conservative_select(
            hits, max_k=DFLT_MAX_K, ratio=DFLT_REL_THRESHOLD, min_score=None
        )
        return chosen

    return selector


def _default_llm_chooser(prompt: str, **prompt_function_kwargs: Any):
    """Build the default LLM chooser on :mod:`aix` (lazy import)."""
    import aix

    def _parse_ids(text: str) -> list[str]:
        return [line.strip(" -\t") for line in str(text).splitlines() if line.strip()]

    fn = aix.prompt_func(
        prompt, egress=_parse_ids, name="select_capabilities", **prompt_function_kwargs
    )

    def choose(*, query: str, candidates: str) -> list[str]:
        return list(fn(query=query, candidates=candidates))

    return choose


# =========================================================================== #
# Progressive disclosure
# =========================================================================== #


@dataclass(frozen=True)
class Disclosure:
    """The progressively-disclosed payload for one selected artifact.

    Attributes:
        artifact_id: the artifact this payload belongs to.
        level: how much was loaded — ``"metadata"`` (no I/O), ``"body"`` (the
            pointer's full text), or ``"bundled"`` (body + extras).
        name: a display name (the ``name`` filter field, else the id).
        score: the selecting hit's score.
        summary: the matched surface text — always present, always cheap.
        body: the full payload (SKILL.md / file text); ``None`` below
            ``"body"`` level or when the pointer could not be read.
        pointer: the source pointer (``skill_path`` / ``path``) — the "package
            pointer" an agent follows to act; ``None`` if the hit has none.
        metadata: the hit's filter metadata, plus a ``disclosure`` note when a
            pointer was present but unreadable (stale/moved/deleted), and an
            ``expansion`` note when expansion was requested but not possible
            for this hit.
        source: the corpus/source name the selecting hit came from (``None``
            when unattributed) — the attribution a federated caller needs to
            tell two same-id artifacts from different corpora apart.
        passage: the expanded neighborhood text (``disclose(..., expand=...)``)
            — the mid-granularity payload between ``summary`` (the matched
            surface) and ``body`` (the pointer's full text); ``None`` when
            expansion was not requested or not possible. Assembled from the
            corpus's *stored records* (see :mod:`ir.expand`), unlike ``body``,
            which dereferences the pointer to an external resource.
    """

    artifact_id: str
    level: str
    name: str
    score: float
    summary: str
    body: str | None = None
    pointer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    passage: str | None = None

    def to_dict(self) -> dict:
        """JSON-serializable form (score cast to ``float``)."""
        return {
            "artifact_id": self.artifact_id,
            "level": self.level,
            "name": self.name,
            "score": float(self.score),
            "summary": self.summary,
            "body": self.body,
            "pointer": self.pointer,
            "metadata": dict(self.metadata),
            "source": self.source,
            "passage": self.passage,
        }


def _pointer_of(metadata: Mapping[str, Any]) -> str | None:
    """The disclosure pointer in a hit's metadata, if any (see :data:`POINTER_KEYS`)."""
    for key in POINTER_KEYS:
        val = metadata.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _store_body_loader(store: ResourceStore) -> BodyLoader:
    """A :data:`BodyLoader` that dereferences ``store[pointer]`` (ir_09 §5).

    Stale-tolerant exactly like the default disk loader: a missing pointer or a
    missing key yields ``None`` (disclosure never raises). This is what lets lazy
    disclosure work over any ``Mapping`` resource store — a ``dol`` file/blob/URL
    store — not just local disk.
    """

    def load(metadata: Mapping[str, Any]) -> str | None:
        pointer = _pointer_of(metadata)
        if pointer is None:
            return None
        try:
            return store.get(pointer)
        except Exception:
            return None

    return load


def _default_body_loader(
    metadata: Mapping[str, Any], *, max_chars: int = DFLT_MAX_BODY_CHARS
) -> str | None:
    """Read the body a hit's pointer names, tolerating missing/moved targets.

    A ``skill_path`` / file ``path`` is read directly; a directory ``path`` (a
    package) discloses its README if present (the body), else stays ``None`` and
    leaves the directory to serve as the actionable pointer. Any I/O error
    returns ``None`` (the caller records the stale pointer); disclosure never
    raises.
    """
    import os

    pointer = _pointer_of(metadata)
    if not pointer:
        return None
    try:
        if os.path.isdir(pointer):
            for cand in ("README.md", "README.rst", "README.txt"):
                rp = os.path.join(pointer, cand)
                if os.path.isfile(rp):
                    with open(rp, encoding="utf-8", errors="ignore") as f:
                        return f.read(max_chars)
            return None
        if os.path.isfile(pointer):
            with open(pointer, encoding="utf-8", errors="ignore") as f:
                return f.read(max_chars)
    except OSError:
        return None
    return None


def _resolve_expansion_corpus(corpus: Any) -> Any:
    """Resolve ``disclose``'s ``corpus=`` spec once, upfront.

    A corpus *name* resolves straight to its local store — deliberately not
    via :func:`ir.open_corpus`, which would eagerly load the corpus's
    embedder; expansion only reads stored records and never embeds. Mapping
    values resolve the same way. Resolving once per :func:`disclose` call
    means an N-hit disclosure never re-opens a store (or re-loads a model)
    per hit.
    """
    if isinstance(corpus, str):
        from .store import CorpusStore

        return CorpusStore.local(corpus)
    if isinstance(corpus, Mapping):
        return {name: _resolve_expansion_corpus(c) for name, c in corpus.items()}
    return corpus


def _expand_passage(
    hit: SearchHit,
    *,
    corpus: Any,
    policy: NeighborhoodPolicy,
    corpus_name: str | None,
) -> "tuple[str | None, str | None]":
    """``(passage, None)`` for *hit*, or ``(None, why_not)``.

    Per-hit tolerance mirrors the stale-pointer handling — these are *data*
    conditions, and disclosure never raises on them:

    - a federated source absent from the ``corpus`` mapping, or a single
      named corpus that is not the hit's source (expanding against the wrong
      corpus would silently stitch a same-id stranger's text — artifact
      identity is only unique within a source) → ``"no_corpus_for_source"``;
    - no ledger entry for the artifact → ``"no_ledger_entry"``;
    - the seed no longer among its siblings (stale hit) → ``"seed_not_found"``.

    Everything else still raises: a torn store (a ledger entry listing
    missing records — corruption whose error names the remedy) and policy
    bugs are programming/integrity errors, not data conditions.
    """
    if isinstance(corpus, Mapping):
        c = corpus.get(hit.source)
        if c is None:
            return None, "no_corpus_for_source"
    else:
        c = corpus
        if (
            corpus_name is not None
            and hit.source is not None
            and hit.source != corpus_name
        ):
            return None, "no_corpus_for_source"
    try:
        return _expand(hit, c, policy=policy).text, None
    except NoLedgerEntry:
        return None, "no_ledger_entry"
    except SeedNotFound:
        return None, "seed_not_found"


def disclose(
    selection: Selection,
    *,
    level: str = "body",
    loader: BodyLoader | None = None,
    store: ResourceStore | None = None,
    expand: NeighborhoodPolicy | None = None,
    corpus: Any = None,
) -> list[Disclosure]:
    """Reveal the payload of each selected hit at ``level`` — append-only, pure.

    Args:
        selection: a committed :class:`Selection`.
        level: ``"metadata"`` (no I/O — summary + pointer only), ``"body"`` (load
            the pointer's full text), or ``"bundled"`` (body + extras; today the
            same as ``"body"``, reserved for bundled scripts/references).
        loader: override the body resolver — ``metadata -> str | None``. The
            default reads the ``skill_path`` / ``path`` pointer from disk and
            tolerates a missing target (returns ``None``, never raises).
        store: a :data:`ResourceStore` (``pointer -> payload`` ``Mapping``) to
            dereference instead of disk — ir_09 §5 pointer-passing over a ``dol``
            store / URL map / blob storage. Mutually exclusive with ``loader``.
        expand: a :data:`~ir.expand.NeighborhoodPolicy` to also stitch each
            hit's neighborhood from the corpus's stored records into
            :attr:`Disclosure.passage` (see :mod:`ir.expand`). Orthogonal to
            ``level``, which governs *pointer* payloads: e.g.
            ``level="metadata", expand=sentence_window_policy()`` reads no
            pointer at all but still returns mid-granularity passages.
            Requires ``corpus=``.
        corpus: where ``expand`` finds each hit's stored siblings — a
            :class:`~ir.index.Corpus` / :class:`~ir.store.CorpusStore` / name,
            or, for cross-source selections, a ``{source_name: corpus}``
            ``Mapping`` resolved per hit via ``hit.source``. Only meaningful
            with ``expand=``.

    Returns:
        one :class:`Disclosure` per selected hit, best-first. This is a pure
        read: the :class:`Selection` and its hits are never mutated, so a caller
        can disclose append-only without disturbing a cached ranked prefix.
    """
    if level not in DISCLOSURE_LEVELS:
        raise ValueError(
            f"unknown disclosure level {level!r}; expected one of {DISCLOSURE_LEVELS}"
        )
    if loader is not None and store is not None:
        raise ValueError("pass either loader= or store=, not both")
    if expand is not None and corpus is None:
        raise ValueError(
            "disclose(expand=...) needs corpus= (a Corpus / CorpusStore / name, "
            "or a {source_name: corpus} mapping) to fetch each hit's siblings"
        )
    if corpus is not None and expand is None:
        raise ValueError("corpus= is only meaningful together with expand=")
    if store is not None:
        load = _store_body_loader(store)
    elif loader is not None:
        load = loader
    else:
        load = _default_body_loader
    corpus_name = None
    if expand is not None:
        if isinstance(corpus, str):
            corpus_name = corpus
        elif not isinstance(corpus, Mapping):
            corpus_name = getattr(corpus, "name", None)
        corpus = _resolve_expansion_corpus(corpus)
    out: list[Disclosure] = []
    for hit in selection.selected:
        meta = dict(hit.metadata)
        pointer = _pointer_of(meta)
        body = None
        if level in ("body", "bundled"):
            body = load(meta)
            if body is None and pointer is not None:
                meta["disclosure"] = "pointer_unreadable"
        passage = None
        if expand is not None:
            passage, note = _expand_passage(
                hit, corpus=corpus, policy=expand, corpus_name=corpus_name
            )
            if note is not None:
                meta["expansion"] = note
        out.append(
            Disclosure(
                artifact_id=hit.artifact_id,
                level=level,
                name=str(meta.get("name") or hit.artifact_id),
                score=float(hit.score),
                summary=hit.text,
                body=body,
                pointer=pointer,
                metadata=meta,
                source=hit.source,
                passage=passage,
            )
        )
    return out


# =========================================================================== #
# discover — the single agent-callable search-and-select tool
# =========================================================================== #


@dataclass(frozen=True)
class DiscoveryResult:
    """The result of :func:`discover` — retrieve → select → (optional) disclose.

    The qh-exposable payload: :meth:`to_dict` is fully JSON-serializable (lists
    of dicts, floats, strings, bools — no numpy, no objects), so a FastAPI
    facade can return it directly.
    """

    query: str
    mode: str
    strategy: str
    disclose_level: str
    results: list[Disclosure]
    abstained: bool
    reason: str
    n_retrieved: int
    signals: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        """The committed artifact ids, best-first."""
        return [d.artifact_id for d in self.results]

    def to_dict(self) -> dict:
        """JSON-serializable result for the qh / HTTP surface."""
        return {
            "query": self.query,
            "mode": self.mode,
            "strategy": self.strategy,
            "disclose_level": self.disclose_level,
            "results": [d.to_dict() for d in self.results],
            "abstained": self.abstained,
            "reason": self.reason,
            "n_retrieved": self.n_retrieved,
            "n_selected": len(self.results),
            "signals": dict(self.signals),
        }


def _resolve_auto_min_score(corpus: Any, mode: str) -> float | None:
    """Load the calibrated abstention floor for ``(corpus, mode)``, or ``None``.

    Reads the per-mode record persisted by :func:`ir.eval.calibrate_min_score`
    from the corpus's ``calibration`` store. Returns ``None`` (and warns) when no
    calibration is stored for the mode, or when the stored ``embedder_id`` no
    longer matches the live corpus (a stale floor after a rebuild) — so
    ``min_score="auto"`` degrades safely to "no absolute abstention" rather than
    abstaining on a mis-scaled floor. Kept here (not in :mod:`ir.eval`) so the
    common ``discover`` path stays free of the eval module's heavier imports.
    """
    # stacklevel=3: warn → _resolve_auto_min_score → discover → the user's call.
    store = getattr(corpus, "store", None)
    get = getattr(store, "get_calibration", None)
    rec = get(mode) if callable(get) else None
    name = getattr(corpus, "name", "?")
    if not rec:
        warnings.warn(
            f"discover(min_score='auto'): no calibration stored for corpus "
            f"{name!r} mode {mode!r}; not abstaining by absolute score. Calibrate "
            f"with ir.eval.calibrate_min_score(corpus, cases, mode={mode!r}, "
            f"persist=True).",
            stacklevel=3,
        )
        return None
    stored_emb = rec.get("embedder_id")
    live_emb = getattr(corpus, "embedder_id", None)
    # Any presence/value mismatch is treated as stale: an unstamped or
    # differently-stamped floor cannot be confirmed to match the live scale.
    if (stored_emb or live_emb) and stored_emb != live_emb:
        warnings.warn(
            f"discover(min_score='auto'): calibration for corpus {name!r} mode "
            f"{mode!r} was made with embedder {stored_emb!r} but the corpus now "
            f"uses {live_emb!r}; ignoring the stale (possibly mis-scaled) floor. "
            f"Re-run ir.eval.calibrate_min_score(..., persist=True).",
            stacklevel=3,
        )
        return None
    floor = rec.get("min_score")
    if floor is None:
        return None
    floor = float(floor)
    if not math.isfinite(floor):
        # A corrupted/hand-edited ±inf floor would abstain (or commit) on
        # everything; refuse it rather than silently breaking abstention.
        warnings.warn(
            f"discover(min_score='auto'): calibration for corpus {name!r} mode "
            f"{mode!r} has a non-finite floor ({floor}); ignoring it. Re-run "
            f"ir.eval.calibrate_min_score(..., persist=True).",
            stacklevel=3,
        )
        return None
    return floor


def discover(
    corpus: Any,
    query: str,
    *,
    k: int = DFLT_FETCH_K,
    mode: str = "hybrid",
    strategy: str | Selector = "conservative",
    disclose_level: str = "metadata",
    filter: Mapping[str, Any] | None = None,
    surfaces: Iterable[str] | None = None,
    max_k: int = DFLT_MAX_K,
    rel: float = DFLT_REL_THRESHOLD,
    gap_ratio: float = DFLT_GAP_RATIO,
    min_score: float | str | Mapping[str, float | str | None] | None = None,
    merge: str | Callable = "rrf",
    merge_weights: Mapping[str, float] | None = None,
    merge_rrf_k: int | None = None,
    loader: BodyLoader | None = None,
    store: ResourceStore | None = None,
    expand: NeighborhoodPolicy | None = None,
    **search_kw: Any,
) -> DiscoveryResult:
    """Find and commit to the capabilities for ``query`` — the one search tool.

    Retrieves ``k`` candidates, commits to a distractor-robust subset, and
    (optionally) discloses each committed item's payload. This is the single
    agent-callable surface the capability-discovery research argues for: one
    tool that returns *few, high-precision* answers rather than a long candidate
    list the model must then filter under context rot.

    Args:
        corpus: a built :class:`~ir.index.Corpus`, or a registered corpus
            **name** (resolved with :func:`ir.open_corpus`). Pass a *name* for
            the qh / HTTP surface — it is the JSON-friendly form. Pass a
            **list/tuple of names (or Corpus objects)** for single-shot
            *federated* discovery across several corpora: each is searched,
            per-source abstention floors gate before any merging, and the
            survivors are rank-fused (see ``merge``). The caller names the
            sources explicitly; ir never chooses the set (source planning is
            the agent layer's job, ir_09 §3).
        query: the user intent.
        k: candidate depth retrieved before selection. Federated: ``k``
            candidates are retrieved *per source*, and the fused ranking is
            also truncated to ``k`` before selection.
        mode: ranking mode — ``"hybrid"`` (default; ``ir``'s strongest overall),
            ``"dense"``, or ``"lexical"``.
        strategy: selection strategy (see :func:`select`).
        disclose_level: ``"metadata"`` (default; cheap, no body I/O), ``"body"``,
            or ``"bundled"``.
        filter, surfaces: retrieval constraints (forwarded to
            :func:`ir.retrieve.search`).
        max_k, rel, gap_ratio, min_score: selection parameters (see
            :func:`select`). ``min_score="auto"`` loads the floor calibrated for
            this ``(corpus, mode)`` by :func:`ir.eval.calibrate_min_score` and
            persisted on the corpus — the opt-in that turns on absolute
            abstention; it falls back to no floor (with a warning) when no
            calibration is stored or it is stale (a different embedder).
            **Federated**: floors are per-(corpus, mode, embedder), so a single
            number cannot apply across corpora — pass ``"auto"`` (each source's
            own calibrated floor), a ``{name: floor_or_"auto"}`` mapping, or
            ``None``; a bare float raises. Floors gate each source on its own
            raw scores *before* fusion; the fused ranking is never floored
            (rank-fused scores are ordinal — ir_07/ir_08).
        merge: federated only — how the per-source rankings combine: ``"rrf"``
            (default; rank-based, scale-free — see
            :func:`ir.retrieve.fuse_hits`), ``"score"`` (raw-score merge,
            valid only when all corpora share an embedder — verified, raises
            on mismatch), or a callable ``{name: hits} -> hits``.
        merge_weights: federated only — per-source trust weights for
            ``merge="rrf"`` (default 1.0 each).
        merge_rrf_k: federated only — the cross-source RRF rank constant
            (default: :data:`~ir.retrieve.DFLT_RRF_K`; distinct from the
            within-corpus hybrid ``rrf_k`` in ``search_kw``).
        loader: optional body resolver for disclosure (see :func:`disclose`).
        expand: a :data:`~ir.expand.NeighborhoodPolicy` — also stitch each
            committed hit's neighborhood from its corpus's stored records into
            :attr:`Disclosure.passage` (retrieval-time context expansion, see
            :mod:`ir.expand`). Works at any ``disclose_level``; the federated
            form resolves each hit's corpus via its ``source``.
        **search_kw: any other :func:`ir.retrieve.search` keyword (``rrf_k``,
            ``rerank``, ``bm25``, …).

    Returns:
        a :class:`DiscoveryResult` (``.to_dict()`` for JSON / qh). Federated
        results add ``signals["per_source"]`` (per-corpus ``n_retrieved`` /
        ``top_score`` / ``floor`` / ``abstained``) and each disclosure carries
        its ``source``.
    """
    if isinstance(corpus, (list, tuple)):
        return _discover_federated(
            list(corpus),
            query,
            k=k,
            mode=mode,
            strategy=strategy,
            disclose_level=disclose_level,
            filter=filter,
            surfaces=surfaces,
            max_k=max_k,
            rel=rel,
            gap_ratio=gap_ratio,
            min_score=min_score,
            merge=merge,
            merge_weights=merge_weights,
            merge_rrf_k=merge_rrf_k,
            loader=loader,
            store=store,
            expand=expand,
            **search_kw,
        )
    if merge != "rrf" or merge_weights is not None or merge_rrf_k is not None:
        raise ValueError(
            "merge / merge_weights / merge_rrf_k apply to the federated form — "
            "pass a list of corpora, e.g. discover(['skills', 'packages'], ...)."
        )
    if isinstance(min_score, Mapping):
        raise ValueError(
            "a {name: floor} min_score mapping applies to the federated form — "
            "pass a list of corpora, or a float/'auto' for a single corpus."
        )
    if isinstance(corpus, str):
        from .index import open_corpus

        corpus = open_corpus(corpus)

    if isinstance(min_score, str):
        if min_score != "auto":
            raise ValueError(
                f"invalid min_score {min_score!r}; expected a float, None, or the "
                f"sentinel 'auto' (load the calibrated floor for this corpus/mode)."
            )
        min_score = _resolve_auto_min_score(corpus, mode)

    hits = _search(
        corpus,
        query,
        k=k,
        mode=mode,
        filter=filter,
        surfaces=surfaces,
        per_artifact=True,
        **search_kw,
    )
    selection = select(
        hits,
        strategy=strategy,
        max_k=max_k,
        rel=rel,
        gap_ratio=gap_ratio,
        min_score=min_score,
    )
    results = disclose(
        selection,
        level=disclose_level,
        loader=loader,
        store=store,
        expand=expand,
        corpus=corpus if expand is not None else None,
    )
    return DiscoveryResult(
        query=query,
        mode=mode,
        strategy=strategy if isinstance(strategy, str) else "custom",
        disclose_level=disclose_level,
        results=results,
        abstained=selection.abstained,
        reason=selection.reason,
        n_retrieved=len(hits),
        signals=selection.signals,
    )


def _resolve_federated_floors(
    resolved: "list[tuple[str, Any]]",
    min_score: float | str | Mapping[str, float | str | None] | None,
    mode: str,
) -> dict[str, float | None]:
    """Per-source abstention floors for federated :func:`discover`.

    Floors are per-(corpus, mode, embedder) — ir_07's central finding — so a
    single number across corpora is the exact mis-scaled comparison the
    calibration machinery exists to prevent: it is refused loudly, never
    applied. Each accepted form resolves to ``{source_name: floor_or_None}``.
    """
    if min_score is None:
        return {}
    if isinstance(min_score, (bool, int, float)):
        raise ValueError(
            "a single numeric min_score cannot apply across corpora — abstention "
            "floors are per-(corpus, mode, embedder) (ir_07). Pass 'auto' (each "
            "source's own calibrated floor), a {name: floor_or_'auto'} mapping, "
            "or None."
        )
    if isinstance(min_score, str):
        if min_score != "auto":
            raise ValueError(
                f"invalid min_score {min_score!r}; expected None, 'auto', or a "
                f"{{name: floor_or_'auto'}} mapping for federated discover."
            )
        return {name: _resolve_auto_min_score(c, mode) for name, c in resolved}
    if isinstance(min_score, Mapping):
        by_name = dict(resolved)
        unknown = sorted(set(min_score) - set(by_name))
        if unknown:
            raise ValueError(
                f"min_score names corpora not in this discover call: {unknown}; "
                f"the call federates {sorted(by_name)}."
            )
        floors: dict[str, float | None] = {}
        for name, spec in min_score.items():
            if spec is None:
                floors[name] = None
            elif spec == "auto":
                floors[name] = _resolve_auto_min_score(by_name[name], mode)
            elif isinstance(spec, bool) or not isinstance(spec, (int, float)):
                raise ValueError(
                    f"invalid min_score for source {name!r}: {spec!r}; expected "
                    f"None, 'auto', or a number (each source's own calibrated "
                    f"scale)."
                )
            else:
                floors[name] = float(spec)
        return floors
    raise ValueError(
        f"invalid min_score {min_score!r}; expected None, 'auto', or a "
        f"{{name: floor_or_'auto'}} mapping for federated discover."
    )


def _merge_federated(
    surviving: "dict[str, list[SearchHit]]",
    *,
    merge: str | Callable,
    merge_weights: Mapping[str, float] | None,
    merge_rrf_k: int | None,
    k: int,
    corpora_by_name: "dict[str, Any]",
) -> list[SearchHit]:
    """Combine the per-source rankings — only ranks cross the source boundary."""
    from .retrieve import DFLT_RRF_K, fuse_hits

    if callable(merge):
        return list(merge(surviving))
    if merge == "rrf":
        return fuse_hits(
            surviving,
            rrf_k=DFLT_RRF_K if merge_rrf_k is None else merge_rrf_k,
            weights=merge_weights,
            k=k,
        )
    if merge == "score":
        # A raw-score merge is the caller asserting one score scale across
        # sources; verify the assertion instead of trusting it (the same
        # detect-and-refuse posture as the stale-calibration check).
        emb_ids = {
            name: getattr(corpora_by_name[name], "embedder_id", None)
            for name in surviving
        }
        if len(set(emb_ids.values())) > 1:
            raise ValueError(
                f"merge='score' compares raw scores across corpora, which "
                f"requires one shared embedder; got {emb_ids}. Use merge='rrf' "
                f"(rank-based, scale-free) for heterogeneous embedders."
            )
        merged = best_per_artifact([h for hits in surviving.values() for h in hits])
        return merged[:k]
    raise ValueError(
        f"unknown merge {merge!r}; expected 'rrf', 'score', or a callable."
    )


def _discover_federated(
    corpora_list: "list[Any]",
    query: str,
    *,
    k: int,
    mode: str,
    strategy: str | Selector,
    disclose_level: str,
    filter: Mapping[str, Any] | None,
    surfaces: Iterable[str] | None,
    max_k: int,
    rel: float,
    gap_ratio: float,
    min_score: float | str | Mapping[str, float | str | None] | None,
    merge: str | Callable,
    merge_weights: Mapping[str, float] | None,
    merge_rrf_k: int | None,
    loader: BodyLoader | None,
    store: ResourceStore | None,
    expand: NeighborhoodPolicy | None = None,
    **search_kw: Any,
) -> DiscoveryResult:
    """Single-shot federated discover: fan-out → gate → fuse → select → disclose.

    The federation of one corpus equals the single-corpus call (the fuse is a
    passthrough); the caller's list order is the deterministic priority order
    for rank ties. All names resolve to open corpora *before* any retrieval, so
    registry drift mid-call cannot yield a partial fan-out.
    """
    from .index import open_corpus

    if not corpora_list:
        raise ValueError(
            "federated discover needs at least one corpus name (ir never "
            "chooses the source set — pass it explicitly)."
        )
    if strategy == "abs_threshold":
        raise ValueError(
            "strategy='abs_threshold' does not apply to federated discover: "
            "fused scores are rank-derived (ordinal), so an absolute floor "
            "post-fusion is meaningless. Absolute floors are per-source and "
            "pre-fusion — pass min_score='auto' or a {name: floor} mapping, "
            "and use a relative post-fusion strategy ('conservative', "
            "'top_k', 'rel_threshold', 'score_gap')."
        )
    resolved: list[tuple[str, Any]] = []
    for entry in corpora_list:
        # Validate the name BEFORE resolving: open_corpus('') would create
        # and load a shared 'unnamed' store on the way to the error.
        name = entry if isinstance(entry, str) else getattr(entry, "name", None)
        if name is None or not str(name).strip():
            raise ValueError(
                f"cannot name corpus {entry!r} for federation; pass registered "
                f"names, or Corpus objects with a non-empty .name."
            )
        c = open_corpus(entry) if isinstance(entry, str) else entry
        resolved.append((str(name), c))
    names = [name for name, _ in resolved]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate corpus names in federated discover: {names}")
    if merge_weights:
        unknown = sorted(set(merge_weights) - set(names))
        if unknown:
            raise ValueError(
                f"merge_weights names corpora not in this discover call: "
                f"{unknown}; the call federates {sorted(names)}."
            )

    floors = _resolve_federated_floors(resolved, min_score, mode)

    per_source: dict[str, dict] = {}
    surviving: dict[str, list[SearchHit]] = {}
    n_retrieved = 0
    any_gated = False
    for name, c in resolved:
        hits = _search(
            c,
            query,
            k=k,
            mode=mode,
            filter=filter,
            surfaces=surfaces,
            per_artifact=True,
            **search_kw,
        )
        n_retrieved += len(hits)
        floor = floors.get(name)
        top = float(hits[0].score) if hits else None
        # The gate consumes raw, per-source magnitudes — the only place an
        # absolute floor is meaningful; the fused ranking below is ordinal.
        gated = floor is not None and (top is None or top < floor)
        any_gated = any_gated or gated
        per_source[name] = {
            "n_retrieved": len(hits),
            "top_score": top,
            "floor": floor,
            "abstained": bool(gated or not hits),
        }
        if hits and not gated:
            surviving[name] = hits

    fused = _merge_federated(
        surviving,
        merge=merge,
        merge_weights=merge_weights,
        merge_rrf_k=merge_rrf_k,
        k=k,
        corpora_by_name=dict(resolved),
    )
    # Floors were already applied per source, on comparable scales; the fused
    # scores are rank-derived, so an absolute floor here would be meaningless.
    selection = select(
        fused,
        strategy=strategy,
        max_k=max_k,
        rel=rel,
        gap_ratio=gap_ratio,
        min_score=None,
    )
    reason = selection.reason
    if not fused and any_gated:
        # Claim "all below floor" only when every source actually had a floor;
        # a mix of floor-gated and merely-empty sources is reported honestly.
        if all(s["floor"] is not None for s in per_source.values()):
            reason = "abstain:all_sources_below_floor"
        else:
            reason = "abstain:no_surviving_sources"
    results = disclose(
        selection,
        level=disclose_level,
        loader=loader,
        store=store,
        expand=expand,
        # Resolve each hit's corpus by its source attribution — the fused
        # selection mixes sources, and sibling lookups must hit the right one.
        corpus=dict(resolved) if expand is not None else None,
    )
    merge_name = merge if isinstance(merge, str) else "custom"
    signals: dict[str, Any] = {
        **dict(selection.signals),
        "merge": merge_name,
        "sources": names,
        "per_source": per_source,
    }
    if merge_name == "rrf":
        from .retrieve import DFLT_RRF_K

        signals["merge_rrf_k"] = DFLT_RRF_K if merge_rrf_k is None else merge_rrf_k
    return DiscoveryResult(
        query=query,
        mode=mode,
        strategy=strategy if isinstance(strategy, str) else "custom",
        disclose_level=disclose_level,
        results=results,
        abstained=selection.abstained,
        reason=reason,
        n_retrieved=n_retrieved,
        signals=signals,
    )
