"""Lexical miss signals: did a keyword search actually find what was asked for?

A deterministic search returns *something* for almost any query, so "got hits"
says little about whether the right thing was found. These are model-free,
auditable facts an agent's Evaluator (raglab) can use to decide to reformulate
(ir_13 §1.3, ir_09 §3):

- ``zero_hits``: nothing came back;
- ``missing_required``: terms the answer must mention (an entity the user named)
  that appear in none of the top hits;
- ``only_via_variants``: the hits mention none of the original query's terms and
  only terms of an expansion variant (a synonym-ledger variant may have matched
  by accident).

They are **signals, never directives**: ir reports them (``Selection.signals``,
``Selection.sufficient``) and never re-queries; the back-edge belongs to the
agent layer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .base import SearchHit

#: Query words shorter than this are ignored by ``only_via_variants`` (stop-ish noise).
MIN_TERM_LEN = 3

_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(text: str, *, min_len: int = MIN_TERM_LEN) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if len(w) >= min_len}


def _mentions(haystack: str, term: str) -> bool:
    """Whole-word, case-insensitive containment (a multi-word term must appear as a phrase)."""
    pattern = (
        r"(?<!\w)" + r"\s+".join(map(re.escape, term.casefold().split())) + r"(?!\w)"
    )
    return re.search(pattern, haystack) is not None


def lexical_miss(
    hits: Sequence[SearchHit],
    *,
    query: str = "",
    required: Iterable[str] = (),
    variants: Iterable[str] = (),
) -> dict[str, Any]:
    """Miss signals for ``hits`` (the top-k of one search); ``{"miss": bool, ...}``.

    ``required`` are terms the answer must mention; ``variants`` are the expansion
    queries that were searched alongside ``query`` (``only_via_variants`` is only
    evaluated when both ``query`` and ``variants`` are given).

    >>> hit = SearchHit("a", "doc", 1.0, "the acme invoice")
    >>> lexical_miss([hit], required=["acme", "globex"])["missing_required"]
    ['globex']
    >>> lexical_miss([])["miss"]
    True
    >>> lexical_miss([hit], query="billing", variants=["invoice"])["only_via_variants"]
    True
    """
    text = "\n".join(h.text for h in hits).casefold()
    missing = [t for t in required if t.strip() and not _mentions(text, t)]
    only_via_variants = False
    if hits and query and variants:
        original = _terms(query)
        variant = set().union(*(_terms(v) for v in variants)) - original
        only_via_variants = (
            bool(original)
            and not any(_mentions(text, t) for t in original)
            and any(_mentions(text, t) for t in variant)
        )
    zero = not hits
    return {
        "miss": zero or bool(missing) or only_via_variants,
        "zero_hits": zero,
        "missing_required": missing,
        "only_via_variants": only_via_variants,
    }
