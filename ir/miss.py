"""Lexical miss signals: did a keyword search actually find what was asked for?

A deterministic search returns *something* for almost any query, so "got hits"
says little about whether the right thing was found. These are model-free,
auditable facts an agent's Evaluator (raglab) can use to decide to reformulate
(ir_13 §1.3, ir_09 §3):

- ``zero_hits``: retrieval returned nothing;
- ``missing_required``: terms the answer must mention (an entity the user named)
  that appear in none of the committed hits;
- ``missing_in_candidates``: the subset of those absent from every *retrieved*
  candidate too, which separates "retrieval missed it, reformulate" from "the
  selection was too tight, widen it".

Matching is whole-word, case-insensitive, per ``\\w`` boundary. Known limits: no
word boundaries in CJK text (``東京`` is not found inside a sentence), no plural or
spelling folding (``invoice`` vs ``invoices``, ``ci-cd`` vs ``CI/CD``), no Unicode
normalisation. Fold both sides first if that matters.

They are **signals, never directives**: ir reports them (``Selection.signals``,
``Selection.sufficient``) and never re-queries; the back-edge belongs to the
agent layer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .base import SearchHit

_WORD = re.compile(r"\w+", re.UNICODE)


def _mentions(haystack: str, term: str) -> bool:
    """Whole-word, case-insensitive containment (a multi-word term must appear as a phrase)."""
    words = [re.escape(w) for w in term.casefold().split()]
    return re.search(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", haystack) is not None


def _text_of(hits: Iterable[SearchHit]) -> str:
    return "\n".join(h.text for h in hits).casefold()


def lexical_miss(
    hits: Sequence[SearchHit],
    *,
    required: Iterable[str] = (),
    candidates: Sequence[SearchHit] | None = None,
) -> dict[str, Any]:
    """Miss signals for the committed ``hits``; ``{"miss": bool, ...}``.

    ``required`` are terms the answer must mention. ``candidates`` (all retrieved
    hits, a superset of ``hits``) enables ``zero_hits`` over retrieval and
    ``missing_in_candidates``; without it the committed hits stand in for both.

    >>> hit = SearchHit("a", "doc", 1.0, "the acme invoice")
    >>> lexical_miss([hit], required=["acme", "globex"])["missing_required"]
    ['globex']
    >>> lexical_miss([])["miss"]
    True
    """
    if isinstance(required, str):
        required = [required]
    required = [t for t in required if t.strip()]
    pool = hits if candidates is None else candidates
    text, pool_text = _text_of(hits), _text_of(pool)
    missing = [t for t in required if not _mentions(text, t)]
    zero = not pool
    return {
        "miss": zero or bool(missing),
        "zero_hits": zero,
        "missing_required": missing,
        "missing_in_candidates": [t for t in missing if not _mentions(pool_text, t)],
    }
