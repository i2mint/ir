"""The synonym ledger: corpus vocabulary an agent learns, for keyword and regex search.

Lexical search (BM25, grep) misses when the searcher's word is not the document's
word. An LLM closes most of that gap on its own by writing variants of the query,
but it cannot know a corpus's private vocabulary: a project's nickname, a renamed
package, a person's handle, in-house jargon. The ledger keeps exactly those, learned
from misses, so the next search starts where the last one ended (research report
``misc/docs/ir_13``).

One JSON object per line in ``synonyms.jsonl`` under ir's config dir (override the
path with ``$IR_SYNONYMS``)::

    {"terms": ["acmecorp", "acme corp", "ac"], "note": "client; repo dir ac/", "added": "2026-11-11"}
    {"terms": ["vd", "vector database"], "oneway": true, "note": "vd expands, not back"}

``terms`` are equivalent (each expands to the others); with ``oneway`` only the first
expands to the rest. ``corpus`` (optional) limits an entry to one corpus. The ledger
is the user's private data: it lives in their config dir, never in a repo.

Three uses, one source: :func:`term_regex` builds an alternation for grep,
:func:`expand_query` produces query variants, and :func:`make_synonym_formulator`
plugs those into :func:`ir.retrieve.search`'s ``formulate=`` seam (deterministic, no
LLM; the original query always comes first).

>>> entries = [SynonymEntry(("acmecorp", "acme corp", "ac"))]
>>> aliases("Acme Corp", entries)
['Acme Corp', 'acmecorp', 'ac']
>>> term_regex(aliases("acmecorp", entries))
'acme[-_ ]?corp|acmecorp|\\\\bac\\\\b'
>>> expand_query("acme corp invoice", entries)
['acme corp invoice', 'acmecorp invoice', 'ac invoice']
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

#: The ledger's file name inside :func:`ir.config.config_dir`.
DFLT_LEDGER_NAME = "synonyms.jsonl"

#: Terms at most this long that are plain word characters get ``\b`` guards in a
#: regex, so ``ac`` does not match inside ``react``.
SHORT_TERM_CHARS = 4

#: Default cap on the number of query variants (the original included).
DFLT_MAX_VARIANTS = 6


@dataclass(frozen=True)
class SynonymEntry:
    """One ledger line: equivalent ``terms`` (or ``oneway`` from the first)."""

    terms: tuple[str, ...]
    oneway: bool = False
    note: str = ""
    added: str = ""
    corpus: str | None = None

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v not in (None, "", False)}
        d["terms"] = list(self.terms)
        return json.dumps(d, ensure_ascii=False)


def ledger_path() -> Path:
    """Where the ledger lives: ``$IR_SYNONYMS``, else ``<config_dir>/synonyms.jsonl``."""
    env = os.environ.get("IR_SYNONYMS")
    if env:
        return Path(env).expanduser()
    from .config import config_dir

    return config_dir() / DFLT_LEDGER_NAME


def load_synonyms(
    path: str | Path | None = None, *, corpus: str | None = None
) -> list[SynonymEntry]:
    """The ledger's entries (those for all corpora, plus ``corpus``'s own).

    A missing ledger is an empty one; a malformed line is skipped, so a hand edit
    gone wrong never breaks search.
    """
    p = Path(path) if path else ledger_path()
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            d = json.loads(line)
            terms = tuple(str(t) for t in d["terms"] if str(t).strip())
        except (ValueError, KeyError, TypeError):
            continue
        if len(terms) < 2:
            continue
        e = SynonymEntry(
            terms=terms,
            oneway=bool(d.get("oneway", False)),
            note=str(d.get("note", "")),
            added=str(d.get("added", "")),
            corpus=d.get("corpus"),
        )
        if e.corpus is None or corpus is None or e.corpus == corpus:
            out.append(e)
    return out


def add_synonyms(
    terms: Sequence[str],
    *,
    note: str = "",
    oneway: bool = False,
    corpus: str | None = None,
    path: str | Path | None = None,
) -> SynonymEntry:
    """Append one entry to the ledger (creating it) and return it.

    Add an entry only after a verified find that the corpus-specific word made
    possible; general-language synonyms are the searcher's job, not the ledger's.
    """
    terms = tuple(t.strip() for t in terms if t and t.strip())
    if len(terms) < 2:
        raise ValueError(f"a ledger entry needs at least two terms, got {terms!r}")
    e = SynonymEntry(
        terms=terms,
        oneway=oneway,
        note=note,
        added=date.today().isoformat(),
        corpus=corpus,
    )
    p = Path(path) if path else ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(e.to_json() + "\n")
    return e


def _norm(term: str) -> str:
    return re.sub(r"[-_\s]+", " ", term.strip().lower())


def aliases(term: str, entries: Iterable[SynonymEntry]) -> list[str]:
    """``term`` first, then every term the ledger makes equivalent to it (deduped)."""
    key = _norm(term)
    out = [term]
    seen = {key}
    for e in entries:
        normed = [_norm(t) for t in e.terms]
        if key not in (normed[:1] if e.oneway else normed):
            continue
        for t, n in zip(e.terms, normed, strict=True):
            if n not in seen:
                seen.add(n)
                out.append(t)
    return out


def term_regex(terms: Iterable[str]) -> str:
    """One alternation matching any of ``terms``, tolerant of ``-``/``_``/space.

    Longest first (so a regex engine prefers the full phrase); short plain words get
    word boundaries.

    >>> term_regex(["db", "data base"])
    'data[-_ ]?base|\\\\bdb\\\\b'
    """
    parts, seen = [], set()
    for t in sorted({t for t in terms if t.strip()}, key=len, reverse=True):
        words = [re.escape(w) for w in re.split(r"[-_\s]+", t.strip()) if w]
        rx = "[-_ ]?".join(words)
        if len(t) <= SHORT_TERM_CHARS and re.fullmatch(r"\w+", t):
            rx = rf"\b{rx}\b"
        if rx.lower() not in seen:
            seen.add(rx.lower())
            parts.append(rx)
    return "|".join(parts)


def expand_query(
    query: str,
    entries: Iterable[SynonymEntry],
    *,
    max_variants: int = DFLT_MAX_VARIANTS,
) -> list[str]:
    """``query`` first, then variants with one known term swapped for an alias.

    Terms are matched as whole words, case-insensitively, separator-tolerant.
    """
    entries = list(entries)
    out = [query]
    for e in entries:
        sources = e.terms[:1] if e.oneway else e.terms
        for src in sorted(sources, key=len, reverse=True):
            rx = re.compile(rf"(?<!\w){term_regex([src]).strip('|')}(?!\w)", re.I)
            if not rx.search(query):
                continue
            for alt in aliases(src, [e])[1:]:
                variant = rx.sub(alt, query, count=1)
                if variant not in out:
                    out.append(variant)
                if len(out) >= max_variants:
                    return out
            break
    return out


def make_synonym_formulator(
    entries: Iterable[SynonymEntry] | None = None,
    *,
    path: str | Path | None = None,
    corpus: str | None = None,
    max_variants: int = DFLT_MAX_VARIANTS,
):
    """A deterministic :data:`ir.formulate.Formulator` that expands from the ledger.

    With ``entries=None`` the ledger is re-read on every call, so an entry an agent
    just added applies to its next search. Returns the original query alone when
    nothing in it is in the ledger (identity, the seam's default behavior).
    """
    fixed = list(entries) if entries is not None else None

    def formulate(query: str) -> list[str]:
        es = fixed if fixed is not None else load_synonyms(path, corpus=corpus)
        return expand_query(query, es, max_variants=max_variants)

    return formulate
