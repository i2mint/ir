"""Replay the synonym ledger through ``ir.eval``: keep the entries that earn their place.

A ledger entry is added after one verified find. Whether it keeps helping, or ever
helped beyond that find, is testable: an entry's ``provenance`` names the query that
missed and the artifact ids the vocabulary found (see :mod:`ir.synonyms`), which is a
:class:`ir.eval.DiscoveryCase`. :func:`replay_ledger` scores each such case twice,
with and without the entry's expansion, through :func:`ir.eval.evaluate_discovery`
(so the numbers are the harness's), and says whether the entry ``helps``, has
``no_effect`` or ``hurts``. With ``apply=True`` it records the win in ``uses`` and
demotes entries that do not help (they are kept in the file, with ``status:
"demoted"``, and are no longer loaded).

Entries without provenance are ``unverifiable`` and left alone. Replay returns
verdicts and, optionally, rewrites the ledger; it never searches on its own
initiative and holds no loop (ir/raglab boundary).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .eval import DiscoveryCase, evaluate_discovery
from .synonyms import (
    ACTIVE,
    DEMOTED,
    SynonymEntry,
    load_synonyms,
    make_synonym_formulator,
    write_synonyms,
)

#: Verdicts of one entry's replay.
HELPS, NO_EFFECT, HURTS, UNVERIFIABLE = "helps", "no_effect", "hurts", "unverifiable"

#: A replay score gain at or below this counts as no change (float noise guard).
DFLT_EPSILON = 1e-9


@dataclass(frozen=True)
class EntryVerdict:
    """One entry's replay: scores of its miss case without and with the expansion."""

    entry: SynonymEntry
    verdict: str
    without: float | None = None
    with_entry: float | None = None


def case_of(entry: SynonymEntry, *, corpus: str | None = None) -> DiscoveryCase | None:
    """The :class:`~ir.eval.DiscoveryCase` an entry's provenance encodes, if complete."""
    prov = entry.provenance or {}
    query, found = prov.get("miss_query"), prov.get("found_ids")
    if isinstance(found, str):
        found = [found]
    if not (isinstance(query, str) and query.strip() and found):
        return None
    return DiscoveryCase(
        query=query,
        gold=tuple(str(i) for i in found),
        corpus=corpus or entry.corpus,
    )


def _score(
    corpus: Any, case: DiscoveryCase, *, mode: str, k: int, **search_kw
) -> float:
    report = evaluate_discovery(
        corpus, [case], mode=mode, k_values=[k], primary_k=k, **search_kw
    )
    return float(report.primary or 0.0)


def replay_entry(
    corpus: Any,
    entry: SynonymEntry,
    *,
    mode: str = "lexical",
    k: int = 10,
    epsilon: float = DFLT_EPSILON,
    **search_kw: Any,
) -> EntryVerdict:
    """Replay one entry: NDCG@k of its miss case without, then with, its expansion."""
    case = case_of(entry)
    if case is None:
        return EntryVerdict(entry, UNVERIFIABLE)
    base = _score(corpus, case, mode=mode, k=k, **search_kw)
    formulate = make_synonym_formulator([entry])
    with_entry = _score(corpus, case, mode=mode, k=k, formulate=formulate, **search_kw)
    delta = with_entry - base
    verdict = HELPS if delta > epsilon else HURTS if delta < -epsilon else NO_EFFECT
    return EntryVerdict(entry, verdict, base, with_entry)


def replay_ledger(
    corpus: Any,
    *,
    entries: Sequence[SynonymEntry] | None = None,
    path: str | Path | None = None,
    apply: bool = False,
    mode: str = "lexical",
    k: int = 10,
    **search_kw: Any,
) -> list[EntryVerdict]:
    """Replay every active entry of the ledger against ``corpus``; see the module doc.

    ``entries`` defaults to the ledger's active entries for this corpus. With
    ``apply=True`` the *whole* ledger at ``path`` is rewritten: a ``helps`` entry
    gets ``uses + 1``; ``no_effect`` and ``hurts`` entries become ``status:
    "demoted"``; unverifiable entries (and entries of other corpora) are untouched.
    """
    name = getattr(corpus, "name", corpus if isinstance(corpus, str) else None)
    active = list(entries) if entries is not None else load_synonyms(path, corpus=name)
    verdicts = [replay_entry(corpus, e, mode=mode, k=k, **search_kw) for e in active]
    if apply and entries is None:
        # entries hold a dict (provenance), so they are matched by their ledger line
        verdict_of = {v.entry.to_json(): v.verdict for v in verdicts}
        updated = []
        for e in load_synonyms(path, include_demoted=True):
            verdict = verdict_of.get(e.to_json())
            if verdict in (None, UNVERIFIABLE):
                updated.append(e)
            elif verdict == HELPS:
                updated.append(replace(e, uses=e.uses + 1, status=ACTIVE))
            else:
                updated.append(replace(e, status=DEMOTED))
        write_synonyms(updated, path)
    return verdicts
