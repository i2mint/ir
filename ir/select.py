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
- :func:`discover` — the single agent-callable tool: retrieve → select →
  (optionally) disclose, returning a JSON-serializable :class:`DiscoveryResult`.
  This is the qh-exposable surface (pass a corpus *name*; get back ``.to_dict()``).

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

from .base import POINTER_KEYS, SearchHit
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

    def to_dict(self) -> dict:
        """JSON-serializable form (scores cast to ``float``)."""
        return {
            "selected": [_hit_to_dict(h) for h in self.selected],
            "selected_ids": self.selected_ids,
            "abstained": self.abstained,
            "reason": self.reason,
            "signals": dict(self.signals),
            "n_candidates": len(self.candidates),
        }


def _hit_to_dict(hit: SearchHit) -> dict:
    """A SearchHit as a plain JSON-serializable dict."""
    return {
        "artifact_id": hit.artifact_id,
        "surface_kind": hit.surface_kind,
        "score": float(hit.score),
        "text": hit.text,
        "metadata": dict(hit.metadata),
    }


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
    :mod:`oa` (``oa.prompt_function``), so importing this module stays offline.

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
    """Build the default LLM chooser on :mod:`oa` (lazy import)."""
    import oa

    def _parse_ids(text: str) -> list[str]:
        return [line.strip(" -\t") for line in str(text).splitlines() if line.strip()]

    fn = oa.prompt_function(
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
            pointer was present but unreadable (stale/moved/deleted).
    """

    artifact_id: str
    level: str
    name: str
    score: float
    summary: str
    body: str | None = None
    pointer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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


def disclose(
    selection: Selection,
    *,
    level: str = "body",
    loader: BodyLoader | None = None,
    store: ResourceStore | None = None,
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
    if store is not None:
        load = _store_body_loader(store)
    elif loader is not None:
        load = loader
    else:
        load = _default_body_loader
    out: list[Disclosure] = []
    for hit in selection.selected:
        meta = dict(hit.metadata)
        pointer = _pointer_of(meta)
        body = None
        if level in ("body", "bundled"):
            body = load(meta)
            if body is None and pointer is not None:
                meta["disclosure"] = "pointer_unreadable"
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
    min_score: float | str | None = None,
    loader: BodyLoader | None = None,
    store: ResourceStore | None = None,
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
            the qh / HTTP surface — it is the JSON-friendly form.
        query: the user intent.
        k: candidate depth retrieved before selection.
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
        loader: optional body resolver for disclosure (see :func:`disclose`).
        **search_kw: any other :func:`ir.retrieve.search` keyword (``rrf_k``,
            ``rerank``, ``bm25``, …).

    Returns:
        a :class:`DiscoveryResult` (``.to_dict()`` for JSON / qh).
    """
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
    results = disclose(selection, level=disclose_level, loader=loader, store=store)
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
