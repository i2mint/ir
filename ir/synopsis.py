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
your own summarizer); omitted, it is built lazily on :mod:`aix` (the
multi-provider LLM facade) via :func:`make_llm_synthesizer` (the ``make_llm_*``
idiom — ``import ir`` stays offline, ``aix`` is imported only on the first
synthesis).

**Staleness.** The wrapper exposes its identity as scalar attributes
(``synthesizer_id``, ``synopsis_kind``) and holds the inner strategy, so
:func:`ir.index._strategy_id` (which recurses into nested strategies) folds both
the inner strategy's parameters *and* the synthesizer identity into the corpus's
``strategy_id``. A prompt / model change — or an inner-strategy change — therefore
re-synthesizes exactly the affected artifacts, the same way an ``embedder_id``
change does; no silent staleness. (An injected *unnamed* synthesizer — a lambda
or a local closure — has no stable identity to fold in, so ``with_synopsis``
**warns** and disables its staleness tracking unless you pass ``synthesizer_id=``.)

**Routing needs no edges.** collapsed-tree descends synopsis→chunks *within* an
artifact via :func:`ir.retrieve.records_for_artifact` (shared ``artifact_id``), so
synopsis routing works with no entries in the :mod:`ir.graph` ``links`` view —
that view models *cross-artifact* edges (REF / PARENT between artifacts), a
different grain than surface→surface within one artifact.

**Conditional synopsis.** ``when=`` (default ``"always"``) restricts synthesis to the
artifacts a predicate selects, so LLM cost is paid only where a synopsis helps —
e.g. ``with_synopsis(ir.Package(), when=thin_description_predicate)`` synthesizes
only for packages whose ``[project].description`` is empty or short (their
README is the only topical signal), and skips the rest *without calling the
synthesizer*. The predicate's identity folds into ``strategy_id`` (``when_id``)
like the synthesizer's, so changing it re-processes the corpus (``strategy_id`` is
corpus-wide) and re-synthesizes every artifact the new gate selects.

**Caveat — ``edge_extractor``.** Eager edge ingest (``build(edge_extractor=...)``)
calls ``decompose`` for *every* artifact each build, and synthesis lives in
``decompose``; so combining :func:`with_synopsis` with an ``edge_extractor``
re-runs synthesis on every rebuild. The common path — synopsis routing with no
``edge_extractor`` — stays fully incremental.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Union

from .base import Artifact, IndexPlan, Surface
from .strategy import IndexingStrategy, text_of

#: A synthesizer: an :class:`~ir.base.Artifact` → its synopsis text (``""`` to
#: skip — no synopsis surface is added for that artifact).
Synthesizer = Callable[[Artifact], str]

#: A synthesis gate: an :class:`~ir.base.Artifact` → whether to synthesize for it.
Predicate = Callable[[Artifact], bool]

#: ``when`` value meaning "synthesize for every artifact" (the default).
ALWAYS = "always"

#: Descriptions shorter than this many characters count as thin by default.
DFLT_THIN_DESCRIPTION_CHARS = 40

#: The surface kind a synopsis is indexed under — a member of
#: :data:`ir.traverse.DFLT_SUMMARY_KINDS`, so collapsed-tree routes from it.
SYNOPSIS_KIND = "synopsis"

#: Default prompt for :func:`make_llm_synthesizer` (a routing-oriented summary).
SYNOPSIS_PROMPT = (
    "Write a concise synopsis (2-4 sentences) of the document below: what it is "
    "about and what questions it answers, so that a search over synopses can route "
    "to it. Output only the synopsis, no preamble.\n\nDocument:\n{text}"
)

#: A topical / classification-oriented alternative to :data:`SYNOPSIS_PROMPT`,
#: suited to thin-description packages (opt-in: ``make_llm_synthesizer(prompt=...)``).
TOPICAL_SYNOPSIS_PROMPT = (
    "Write a concise synopsis (2-3 sentences) of the software package described "
    "below: which domain or problem class it belongs to (e.g. audio, graphs, "
    "http clients, data validation) and what it is used for, so that a search "
    "for that domain finds it. Output only the synopsis, no preamble.\n\n"
    "Package:\n{text}"
)


def _prompt_hash(prompt: str) -> str:
    """Short stable hash of a prompt, for the default synthesizer identity."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _default_llm_summarizer(
    prompt: str, model: str | None, **prompt_function_kwargs: Any
):
    """Build the default text→synopsis summarizer on :mod:`aix` (lazy import)."""
    import aix

    kwargs = dict(prompt_function_kwargs)
    if model is not None:
        kwargs.setdefault("model", model)
    fn = aix.prompt_func(prompt, name="synthesize_synopsis", **kwargs)

    def summarize(text: str) -> str:
        return str(fn(text=text) or "").strip()

    return summarize


def make_llm_synthesizer(
    *,
    summarize: Callable[[str], str] | None = None,
    prompt: str = SYNOPSIS_PROMPT,
    model: str | None = None,
    synthesizer_id: str | None = None,
    text_key: str | None = None,
    text_fields: Sequence[str] | None = None,
    **prompt_function_kwargs: Any,
) -> Synthesizer:
    """An LLM-backed :data:`Synthesizer` (:class:`~ir.base.Artifact` → synopsis).

    ``summarize`` is an injectable ``text -> str`` callable (a test double, or
    your own summarizer); when omitted it is built lazily on :mod:`aix`
    (``aix.prompt_func``) on the **first** synthesis and reused — so importing
    this module, and even constructing the synthesizer, stays offline. The
    artifact's text is extracted with :func:`ir.strategy.text_of` using
    ``text_key`` — which :func:`with_synopsis` threads from the inner strategy, so
    the synopsis summarizes the *same* field the strategy indexes.

    ``text_fields`` (mutually exclusive with ``text_key``) names the fields of a
    mapping artifact whose text is sent to the LLM, and **only those**: without it,
    a mapping with no ``text`` / ``text_key`` has *every* string field joined and
    sent (see :func:`ir.strategy.text_of`) — including, for a
    :class:`~ir.strategy.Package` record, its local ``path``. ``with_synopsis``
    threads the inner strategy's ``synopsis_fields`` (``Package`` declares
    ``("name", "description", "readme")``), so the Package default never leaks the
    path. The fields fold into the default ``synthesizer_id``, so changing them
    re-synthesizes. An empty text,
    or any synthesis error, yields ``""`` (the surface is then skipped, never a
    fabricated summary).

    The returned callable carries a ``synthesizer_id`` attribute (default
    ``"aix:{model}:{sha(prompt)[:12]}"``) that :func:`with_synopsis` reads into the
    corpus's ``strategy_id`` for staleness — a prompt or model change re-synthesizes.
    """
    if text_fields is not None and text_key is not None:
        raise ValueError("make_llm_synthesizer: pass text_key or text_fields, not both")
    if isinstance(text_fields, str):
        raise TypeError(
            f"make_llm_synthesizer: text_fields must be a sequence of names, not {text_fields!r}"
        )
    fields = tuple(text_fields) if text_fields is not None else None
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
        raw = artifact.raw
        if fields is not None and isinstance(raw, Mapping):
            parts = (str(raw.get(f) or "").strip() for f in fields)
            text = "\n\n".join(p for p in parts if p)
        else:
            text = text_of(raw, text_key).strip()
        if not text:
            return ""
        try:
            out = _summarizer()(text)
        except Exception:
            out = ""
        return out.strip() if isinstance(out, str) else ""

    default_id = f"aix:{model or 'default'}:{_prompt_hash(prompt)}"
    if fields is not None:
        default_id += ":" + ",".join(fields)
    synthesize.synthesizer_id = synthesizer_id or default_id
    return synthesize


def make_thin_description_predicate(
    *, min_chars: int = DFLT_THIN_DESCRIPTION_CHARS, field: str = "description"
) -> Predicate:
    """A :data:`Predicate` true when the artifact's ``field`` is empty or thin.

    Reads ``artifact.raw[field]`` (a :class:`~ir.strategy.Package` record's
    ``description``); a raw that is not a mapping is a thin source when its text
    is shorter than ``min_chars``. Whitespace is ignored. The returned callable
    carries a ``when_id`` stamp so the threshold folds into ``strategy_id``.

    >>> thin = make_thin_description_predicate(min_chars=10)
    >>> thin(Artifact(id="a", raw={"name": "a", "description": ""}))
    True
    >>> thin(Artifact(id="b", raw={"name": "b", "description": "Tools to parse dates"}))
    False
    """

    def thin(artifact: Artifact) -> bool:
        raw = artifact.raw
        value = raw.get(field) if isinstance(raw, Mapping) else raw
        return len(str(value or "").strip()) < min_chars

    thin.when_id = f"thin:{field}:{min_chars}"
    return thin


#: Ready-made gate: synthesize only for packages with an empty / short description.
thin_description_predicate = make_thin_description_predicate()


def _default_synthesizer(strategy: IndexingStrategy) -> Synthesizer:
    """The LLM synthesizer for ``strategy``: its ``text_key``, else its ``synopsis_fields``."""
    while isinstance(
        strategy, _SynopsisStrategy
    ):  # nested wrappers: read the real strategy
        strategy = strategy.strategy
    text_key = getattr(strategy, "text_key", None)
    fields = (
        None if text_key is not None else getattr(strategy, "synopsis_fields", None)
    )
    return make_llm_synthesizer(text_key=text_key, text_fields=fields)


class _SynopsisStrategy:
    """An :class:`~ir.strategy.IndexingStrategy` that prepends a synopsis surface.

    Delegates decomposition to the wrapped ``strategy``, then prepends one
    ``synopsis`` surface (skipped if the synthesizer returns empty). Exposes
    ``synthesizer_id`` / ``synopsis_kind`` as scalar attributes and holds the
    inner strategy, so :func:`ir.index._strategy_id` captures the full identity.
    """

    when: Predicate | None = None  # class-level defaults: see __init__

    def __init__(
        self,
        strategy: IndexingStrategy,
        *,
        synthesize: Synthesizer | None = None,
        synthesizer_id: str | None = None,
        synopsis_kind: str = SYNOPSIS_KIND,
        when: Union[str, Predicate] = ALWAYS,
        when_id: str | None = None,
    ):
        self.strategy = strategy
        # Gate attributes are set only when a gate is given: ``_strategy_id`` reads
        # instance attributes, so an ungated wrapper keeps its pre-``when`` id and
        # existing synopsis corpora are not re-synthesized by this feature.
        resolved = self._resolve_when_id(when, when_id)
        if not isinstance(when, str):  # gated (decided on `when`, not its id)
            self.when = when
            self.when_id = resolved
        # The default synthesizer summarizes the same field the inner strategy
        # indexes (thread its text_key, if any), so synopsis routing matches.
        self.synthesize: Synthesizer = (
            synthesize if synthesize is not None else _default_synthesizer(strategy)
        )
        self.synthesizer_id = self._resolve_synthesizer_id(synthesizer_id)
        self.synopsis_kind = synopsis_kind

    @staticmethod
    def _resolve_when_id(when: Union[str, Predicate], explicit: str | None) -> str:
        """Identity of the gate for staleness (``"always"`` for no gate)."""
        if isinstance(when, str):
            if explicit is not None:
                raise ValueError("with_synopsis: when_id requires a `when` predicate")
            if when != ALWAYS:
                raise ValueError(
                    f"with_synopsis: when must be {ALWAYS!r} or a callable "
                    f"Artifact -> bool, got {when!r}"
                )
            return ALWAYS
        if not callable(when):
            raise TypeError(f"with_synopsis: when must be callable, got {when!r}")
        stamped = explicit or getattr(when, "when_id", None)
        if stamped:
            return stamped
        qualname = getattr(when, "__qualname__", "") or ""
        if qualname and "<lambda>" not in qualname and "<locals>" not in qualname:
            return qualname
        warnings.warn(
            "with_synopsis: could not derive a stable identity for the `when` "
            "predicate (an unnamed lambda or a local closure), so changing it will "
            "not re-synthesize — pass when_id=... to track it.",
            stacklevel=4,
        )
        return "custom"

    def _resolve_synthesizer_id(self, explicit: str | None) -> str:
        """Identity for staleness: explicit id, else a stamp, else a stable name.

        An explicit ``synthesizer_id`` wins; else a ``synthesizer_id`` the
        synthesizer carries (e.g. :func:`make_llm_synthesizer`'s). Failing both,
        a *named* callable's ``__qualname__`` is a stable identity — but an
        unnamed lambda (``"<lambda>"``) or a local closure (``"<locals>"``) is
        **not**: distinct such callables share one qualname, so swapping them
        would silently *not* re-synthesize. In that case we warn and fall back to
        a sentinel, surfacing the lost guarantee at construction time.
        """
        stamped = explicit or getattr(self.synthesize, "synthesizer_id", None)
        if stamped:
            return stamped
        qualname = getattr(self.synthesize, "__qualname__", "") or ""
        if qualname and "<lambda>" not in qualname and "<locals>" not in qualname:
            return qualname
        warnings.warn(
            "with_synopsis: could not derive a stable identity for the injected "
            "synthesizer (an unnamed lambda or a local closure), so synopsis "
            "staleness tracking is disabled — pass synthesizer_id=... so that "
            "swapping the synthesizer re-synthesizes the corpus on the next build.",
            stacklevel=3,
        )
        return "custom"

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        plan = self.strategy.decompose(artifact_id, raw, metadata)
        artifact = Artifact(id=artifact_id, raw=raw, metadata=dict(metadata or {}))
        if self.when is not None and not self.when(artifact):
            return plan  # gated out: no synthesis call, no synopsis surface
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
    when: Union[str, Predicate] = ALWAYS,
    when_id: str | None = None,
) -> IndexingStrategy:
    """Wrap *strategy* to add one LLM-derived ``synopsis`` surface per artifact.

    Args:
        strategy: the inner :class:`~ir.strategy.IndexingStrategy` (``Chunked``,
            ``Package``, ...). Its surfaces are kept; the synopsis is prepended.
        synthesize: an injectable ``Artifact -> str`` (test double / custom
            summarizer). Omitted → :func:`make_llm_synthesizer` (lazy ``aix``).
        synthesizer_id: explicit identity stamp for staleness (recommended when
            injecting an unnamed callable / lambda). Omitted → the synthesizer's
            own ``synthesizer_id`` / ``__qualname__``.
        synopsis_kind: the surface kind (default ``"synopsis"``, a summary kind).
        when: ``"always"`` (default: every artifact) or a predicate
            ``Artifact -> bool``; the synthesizer is called, and a synopsis
            surface added, only where it returns true (e.g.
            :data:`thin_description_predicate`). A predicate that raises
            propagates.
        when_id: explicit identity stamp for an unnamed predicate (staleness);
            omitted → the predicate's ``when_id`` / ``__qualname__``.

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
        when=when,
        when_id=when_id,
    )
