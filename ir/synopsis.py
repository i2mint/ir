"""Synopsis surfaces — LLM-derived summaries as an indexed surface (report 12).

The document-summary-index / collapsed-tree *fuel* (ADR #43): run a summarizer
over each artifact at **build time** to produce a short *synopsis*, index it as a
``"synopsis"`` surface, and let the collapsed-tree policy (:func:`ir.traverse`)
route a synopsis match down to that artifact's chunks. Build-time cost, ≈free at
query time — and incremental, so only new / changed artifacts are re-synthesized.

:func:`with_synopsis` wraps *any* :class:`~ir.strategy.IndexingStrategy` and adds
one synopsis surface per artifact::

    strat  = ir.with_synopsis(ir.Chunked(), synthesize=my_summarizer)
    corpus = ir.build(ir.CorpusSource.from_mapping(docs, name="d", strategy=strat))
    hits   = ir.traverse(q, corpus, policy=ir.collapsed_tree_policy())  # routes via synopsis

The synopsis is **prepended** (plan position 0) so it is the *first* summary
surface — hence the collapsed-tree *router* (on ``with_synopsis(Package())`` the
synopsis, not the terse ``description``, routes). An empty synopsis is dropped, so
a synth that returns ``""`` simply leaves the artifact with its other surfaces.

``synthesize: Callable[[Artifact], str]`` is **injectable** (a test double, or
your own summarizer); omitted, it is built lazily on :mod:`oa` via
:func:`make_llm_synthesizer` (the ``make_llm_*`` idiom — ``import ir`` stays
offline, ``oa`` is imported only on the first synthesis).

**Staleness.** The wrapper exposes its identity as scalar attributes
(``synthesizer_id``, ``synopsis_kind``) and holds the inner strategy, so
:func:`ir.index._strategy_id` (which recurses into nested strategies) folds both
the inner strategy's parameters *and* the synthesizer identity into the corpus's
``strategy_id``. A prompt / model change — or an inner-strategy change — therefore
re-synthesizes exactly the affected artifacts, the same way an ``embedder_id``
change does; no silent staleness.

**Routing needs no edges.** collapsed-tree descends synopsis→chunks *within* an
artifact via :func:`ir.retrieve.records_for_artifact` (shared ``artifact_id``), so
synopsis routing works with no entries in the :mod:`ir.graph` ``links`` view —
that view models *cross-artifact* edges (REF / PARENT between artifacts), a
different grain than surface→surface within one artifact.

**Caveat — ``edge_extractor``.** Eager edge ingest (``build(edge_extractor=...)``)
calls ``decompose`` for *every* artifact each build, and synthesis lives in
``decompose``; so combining :func:`with_synopsis` with an ``edge_extractor``
re-runs synthesis on every rebuild. The common path — synopsis routing with no
``edge_extractor`` — stays fully incremental.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from .base import Artifact, IndexPlan, Surface
from .strategy import IndexingStrategy, text_of

#: A synthesizer: an :class:`~ir.base.Artifact` → its synopsis text (``""`` to
#: skip — no synopsis surface is added for that artifact).
Synthesizer = Callable[[Artifact], str]

#: The surface kind a synopsis is indexed under — a member of
#: :data:`ir.traverse.DFLT_SUMMARY_KINDS`, so collapsed-tree routes from it.
SYNOPSIS_KIND = "synopsis"

#: Default prompt for :func:`make_llm_synthesizer` (a routing-oriented summary).
SYNOPSIS_PROMPT = (
    "Write a concise synopsis (2-4 sentences) of the document below: what it is "
    "about and what questions it answers, so that a search over synopses can route "
    "to it. Output only the synopsis, no preamble.\n\nDocument:\n{text}"
)


def _prompt_hash(prompt: str) -> str:
    """Short stable hash of a prompt, for the default synthesizer identity."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _default_llm_summarizer(
    prompt: str, model: str | None, **prompt_function_kwargs: Any
):
    """Build the default text→synopsis summarizer on :mod:`oa` (lazy import)."""
    import oa

    kwargs = dict(prompt_function_kwargs)
    if model is not None:
        kwargs.setdefault("model", model)
    fn = oa.prompt_function(prompt, name="synthesize_synopsis", **kwargs)

    def summarize(text: str) -> str:
        return str(fn(text=text) or "").strip()

    return summarize


def make_llm_synthesizer(
    *,
    summarize: Callable[[str], str] | None = None,
    prompt: str = SYNOPSIS_PROMPT,
    model: str | None = None,
    synthesizer_id: str | None = None,
    **prompt_function_kwargs: Any,
) -> Synthesizer:
    """An LLM-backed :data:`Synthesizer` (:class:`~ir.base.Artifact` → synopsis).

    ``summarize`` is an injectable ``text -> str`` callable (a test double, or
    your own summarizer); when omitted it is built lazily on :mod:`oa`
    (``oa.prompt_function``) on the **first** synthesis and reused — so importing
    this module, and even constructing the synthesizer, stays offline. The
    artifact's text is extracted with :func:`ir.strategy.text_of`; an empty text,
    or any synthesis error, yields ``""`` (the surface is then skipped, never a
    fabricated summary).

    The returned callable carries a ``synthesizer_id`` attribute (default
    ``"oa:{model}:{sha(prompt)[:12]}"``) that :func:`with_synopsis` reads into the
    corpus's ``strategy_id`` for staleness — a prompt or model change re-synthesizes.
    """
    cache: dict[str, Callable[[str], str]] = {}

    def _summarizer() -> Callable[[str], str]:
        if summarize is not None:
            return summarize
        if "fn" not in cache:
            cache["fn"] = _default_llm_summarizer(
                prompt, model, **prompt_function_kwargs
            )
        return cache["fn"]

    def synthesize(artifact: Artifact) -> str:
        text = text_of(artifact.raw).strip()
        if not text:
            return ""
        try:
            out = _summarizer()(text)
        except Exception:
            out = ""
        return out.strip() if isinstance(out, str) else ""

    synthesize.synthesizer_id = (
        synthesizer_id or f"oa:{model or 'default'}:{_prompt_hash(prompt)}"
    )
    return synthesize


class _SynopsisStrategy:
    """An :class:`~ir.strategy.IndexingStrategy` that prepends a synopsis surface.

    Delegates decomposition to the wrapped ``strategy``, then prepends one
    ``synopsis`` surface (skipped if the synthesizer returns empty). Exposes
    ``synthesizer_id`` / ``synopsis_kind`` as scalar attributes and holds the
    inner strategy, so :func:`ir.index._strategy_id` captures the full identity.
    """

    def __init__(
        self,
        strategy: IndexingStrategy,
        *,
        synthesize: Synthesizer | None = None,
        synthesizer_id: str | None = None,
        synopsis_kind: str = SYNOPSIS_KIND,
    ):
        self.strategy = strategy
        self.synthesize: Synthesizer = (
            synthesize if synthesize is not None else make_llm_synthesizer()
        )
        # Identity for staleness: an explicit id wins; else a stamp the
        # synthesizer carries; else its qualname (stable for a named double).
        self.synthesizer_id = (
            synthesizer_id
            or getattr(self.synthesize, "synthesizer_id", None)
            or getattr(self.synthesize, "__qualname__", None)
            or "custom"
        )
        self.synopsis_kind = synopsis_kind

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        plan = self.strategy.decompose(artifact_id, raw, metadata)
        artifact = Artifact(id=artifact_id, raw=raw, metadata=dict(metadata or {}))
        text = self.synthesize(artifact)
        text = text.strip() if isinstance(text, str) else ""
        if text:
            synopsis = Surface(
                artifact_id,
                self.synopsis_kind,
                text,
                granularity="document",
                metadata={"synthesizer_id": self.synthesizer_id},
            )
            plan.surfaces = [synopsis, *plan.surfaces]
        return plan


def with_synopsis(
    strategy: IndexingStrategy,
    *,
    synthesize: Synthesizer | None = None,
    synthesizer_id: str | None = None,
    synopsis_kind: str = SYNOPSIS_KIND,
) -> IndexingStrategy:
    """Wrap *strategy* to add one LLM-derived ``synopsis`` surface per artifact.

    Args:
        strategy: the inner :class:`~ir.strategy.IndexingStrategy` (``Chunked``,
            ``Package``, ...). Its surfaces are kept; the synopsis is prepended.
        synthesize: an injectable ``Artifact -> str`` (test double / custom
            summarizer). Omitted → :func:`make_llm_synthesizer` (lazy ``oa``).
        synthesizer_id: explicit identity stamp for staleness (recommended when
            injecting an unnamed callable / lambda). Omitted → the synthesizer's
            own ``synthesizer_id`` / ``__qualname__``.
        synopsis_kind: the surface kind (default ``"synopsis"``, a summary kind).

    Returns:
        an :class:`~ir.strategy.IndexingStrategy` usable anywhere a strategy is —
        ``ir.CorpusSource.from_mapping(docs, name=..., strategy=with_synopsis(...))``.

    >>> strat = with_synopsis(Chunked(), synthesize=lambda a: "a summary")  # doctest: +SKIP
    """
    return _SynopsisStrategy(
        strategy,
        synthesize=synthesize,
        synthesizer_id=synthesizer_id,
        synopsis_kind=synopsis_kind,
    )
