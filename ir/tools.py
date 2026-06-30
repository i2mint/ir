"""Agent-callable tool surface over ``ir`` — plain functions returning JSON-ready
dicts, deliberately MCP/HTTP-agnostic.

``ir`` knows nothing about MCP, HTTP, or any agent host. This module is the SSOT
for "expose ir's retrieval as a single agent-callable tool": a wrapper
(``py2mcp``, ``qh``, a hand-written agent tool) references e.g. ``ir.tools:search``
and gets a clean JSON ``dict`` back. The corpus is a **parameter**, so one
function serves every corpus with no per-corpus code — and :func:`make_search`
returns a *corpus-bound* tool when a connector should expose exactly one corpus.

This pairs with ``ir.discover`` (which ``ir`` already calls "the single
agent-callable tool"); :func:`search` is just its JSON-returning, tool-shaped
front door.
"""

from __future__ import annotations

from typing import Any, Callable

from .select import discover

__all__ = ["search", "make_search"]


def search(
    query: str,
    *,
    corpus: Any,
    k: int = 8,
    mode: str = "hybrid",
    filter: dict | None = None,
) -> dict:
    """Search a named ``ir`` corpus and return a JSON-serializable result dict.

    Thin agent-callable wrapper over :func:`ir.discover` — returns its
    ``.to_dict()`` (committed results, scores, disclosures), fit to hand straight
    back from an MCP tool or HTTP endpoint. The *corpus* is a parameter, so this
    single function serves any corpus.

    Args:
        query: the natural-language query.
        corpus: a registered corpus **name** (str), a list of names (federated
            search), or a built :class:`ir.Corpus`.
        k: maximum number of results.
        mode: ``"dense"`` | ``"lexical"`` | ``"hybrid"``.
        filter: optional ``vd`` Mongo-style metadata filter (hard pre-filter).
    """
    return discover(corpus, query, k=k, mode=mode, filter=filter).to_dict()


def make_search(
    corpus: Any,
    *,
    name: str | None = None,
    description: str | None = None,
    k: int = 8,
    mode: str = "hybrid",
) -> Callable[..., dict]:
    """Return a **corpus-bound** ``search(query, k=...) -> dict`` tool.

    The returned function exposes only ``query`` (and ``k``) — the corpus is fixed
    — so a connector built over it surfaces exactly one corpus and nothing else.
    Its ``__name__`` / ``__doc__`` are set so an MCP/agent host shows a clean tool
    name and description. Use this when wiring a single-corpus connector; use
    :func:`search` when the caller should choose the corpus.
    """
    label = name or (corpus if isinstance(corpus, str) else "corpus")
    safe = "".join(c if c.isalnum() else "_" for c in str(label)).strip("_") or "corpus"

    def bound_search(query: str, k: int = k) -> dict:
        return discover(corpus, query, k=k, mode=mode).to_dict()

    bound_search.__name__ = f"search_{safe}"
    bound_search.__doc__ = description or (
        f"Search the {label!r} knowledge base; returns ranked, JSON results."
    )
    return bound_search
