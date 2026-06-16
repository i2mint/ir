"""Query formulation — the Formulator seam (ir_09 §3).

Turn a user query into one or more concrete low-level queries *before* retrieval:
rewrite, expand, paraphrase, HyDE. This is the single most-acknowledged gap in
ir's standalone retrieval — the raw query is otherwise embedded as-is — and a
good formulation lifts recall on short, identifier-heavy capability text with
**no agent and no loop**.

The seam is **opt-in and identity by default**: ``ir.search(corpus, q)`` with no
``formulate=`` embeds ``q`` verbatim (exactly today's behavior). A formulator
returns a ``str`` (one query) or a sequence of strings (multi-query fan-out);
when it returns several, :func:`ir.retrieve.search` runs each and fuses the
candidate lists (best surface per artifact across all queries).

LOAD-BEARING BOUNDARY: a :data:`Formulator` returns **queries, never SubTasks**.
Decomposing a goal into sub-tasks + source selection is the *Planner's* job — that
lives in the agent layer (``raglab``), not here.

:func:`make_llm_formulator` mirrors :func:`ir.select.make_llm_selector`: an
injectable ``rewriter`` callable, built lazily on :mod:`aix` when omitted (so
importing ir stays offline), falling back to identity on any failure — a
formulator must never make retrieval *worse* than the raw query.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

#: A formulator: a query string -> one query (``str``) or several
#: (``Sequence[str]``). Identity by default; an LLM rewriter / HyDE / multi-query
#: producer when injected.
Formulator = Callable[[str], "str | Sequence[str]"]

#: Default prompt for :func:`make_llm_formulator` — diverse paraphrases for recall.
FORMULATION_PROMPT = (
    "Rewrite the search query into {n} short, diverse alternative search queries "
    "that would retrieve the same target documents: fix typos, expand jargon, and "
    "add synonyms, but keep each a terse search phrase. One query per line, no "
    "numbering.\n\nQuery: {query}"
)


def identity_formulator(query: str) -> str:
    """The default formulator: return the query unchanged (embed it verbatim)."""
    return query


def _default_llm_rewriter(prompt: str, n: int, **prompt_function_kwargs: Any):
    """Build the default LLM rewriter on :mod:`aix` (lazy import)."""
    import aix

    def _parse_lines(text: str) -> list[str]:
        return [line.strip(" -\t") for line in str(text).splitlines() if line.strip()]

    fn = aix.prompt_func(
        prompt, egress=_parse_lines, name="formulate_queries", **prompt_function_kwargs
    )

    def rewrite(query: str) -> list[str]:
        return list(fn(query=query, n=n))

    return rewrite


def make_llm_formulator(
    *,
    rewriter: Callable[[str], "str | Sequence[str]"] | None = None,
    prompt: str = FORMULATION_PROMPT,
    n: int = 3,
    fallback: Formulator | None = None,
    **prompt_function_kwargs: Any,
) -> Formulator:
    """An LLM-backed :data:`Formulator` (rewrite / expand / multi-query).

    ``rewriter`` is an injectable ``query -> str | [str, ...]`` callable (a test
    double, or your own router); when omitted it is built lazily on :mod:`aix`
    (``aix.prompt_func``), so importing this module stays offline. ``n`` is the
    multi-query fan-out width. Any error or empty reply falls back to ``fallback``
    (default: :func:`identity_formulator`).
    """

    def formulate(query: str) -> str | Sequence[str]:
        fn = (
            rewriter
            if rewriter is not None
            else _default_llm_rewriter(prompt, n, **prompt_function_kwargs)
        )
        try:
            out = fn(query)
        except Exception:
            out = None
        queries = [out] if isinstance(out, str) else list(out or [])
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if queries:
            return queries
        fb = fallback if fallback is not None else identity_formulator
        return fb(query)

    return formulate
