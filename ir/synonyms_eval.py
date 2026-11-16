"""Replay the synonym ledger through ``ir.eval``: keep the entries that earn their place.

A ledger entry is added after one verified find. Whether it keeps helping is testable:
an entry's ``provenance`` names the query that missed and the artifact ids the
vocabulary found (see :mod:`ir.synonyms`), which is a :class:`ir.eval.DiscoveryCase`.
:func:`replay_ledger` scores each such case twice, with and without the entry's
expansion, through :func:`ir.eval.evaluate_discovery` (so the numbers are the
harness's), and says whether the entry ``helps``, has ``no_effect`` or ``hurts``.

What it will not do is judge on thin evidence: an entry is ``unverifiable`` when it has
no provenance, when its gold ids are not in this corpus (a global entry replayed on
the wrong corpus, or a corpus rebuilt since), or when ``vd`` is unavailable so lexical
scores are not the real ones. With ``apply=True`` the verdicts are written back
(:func:`apply_verdicts`): a win bumps ``verified`` and re-activates a demoted entry; a
``hurts`` demotes; a ``no_effect`` demotes only an entry that has never won a replay
(the miss may simply be fixed by now). Demoted entries stay in the file and are no
longer loaded by default. Replay returns verdicts and optionally patches the ledger; it
never searches on its own initiative and holds no loop (ir/raglab boundary).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .eval import DiscoveryCase, evaluate_discovery, validate_cases
from .synonyms import (
    ACTIVE,
    DEMOTED,
    SynonymEntry,
    load_synonyms,
    make_synonym_formulator,
    update_entries,
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


def case_of(entry: SynonymEntry) -> DiscoveryCase | None:
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
        corpus=prov.get("corpus") or entry.corpus,
    )


def _report(corpus: Any, case: DiscoveryCase, *, mode: str, k: int, **search_kw):
    return evaluate_discovery(
        corpus, [case], mode=mode, k_values=[k], primary_k=k, **search_kw
    )


def replay_entry(
    corpus: Any,
    entry: SynonymEntry,
    *,
    mode: str = "lexical",
    k: int = 10,
    epsilon: float = DFLT_EPSILON,
    **search_kw: Any,
) -> EntryVerdict:
    """Replay one entry: NDCG@k of its miss case without, then with, its expansion.

    ``mode`` defaults to ``lexical`` (the ledger serves keyword search; ``ir.eval``
    itself defaults to ``hybrid``). ``formulate`` cannot be passed: the entry's own
    expansion is the thing under test.
    """
    if "formulate" in search_kw:
        raise ValueError(
            "replay_entry tests the entry's own formulator; drop formulate="
        )
    case = case_of(entry)
    if case is None:
        return EntryVerdict(entry, UNVERIFIABLE)
    if case.corpus and case.corpus != getattr(corpus, "name", corpus):
        return EntryVerdict(entry, UNVERIFIABLE)  # the miss happened elsewhere
    if validate_cases(corpus, [case]):  # gold ids absent from this corpus
        return EntryVerdict(entry, UNVERIFIABLE)
    base = _report(corpus, case, mode=mode, k=k, **search_kw)
    withe = _report(
        corpus,
        case,
        mode=mode,
        k=k,
        formulate=make_synonym_formulator([entry]),
        **search_kw,
    )
    if base.vd_degraded or withe.vd_degraded:  # scores are not the requested mode's
        return EntryVerdict(entry, UNVERIFIABLE)
    before, after = float(base.primary or 0.0), float(withe.primary or 0.0)
    delta = after - before
    verdict = HELPS if delta > epsilon else HURTS if delta < -epsilon else NO_EFFECT
    return EntryVerdict(entry, verdict, before, after)


def apply_verdicts(
    verdicts: Sequence[EntryVerdict],
) -> list[tuple[SynonymEntry, SynonymEntry]]:
    """The ``(old, new)`` entry changes a set of verdicts implies (pure; no I/O).

    >>> e = SynonymEntry(("a1", "b1"))
    >>> [(n.verified, n.status) for _, n in apply_verdicts([EntryVerdict(e, HELPS)])]
    [(1, 'active')]
    >>> [(n.verified, n.status) for _, n in apply_verdicts([EntryVerdict(e, HURTS)])]
    [(0, 'demoted')]
    """
    changes = []
    for v in verdicts:
        e = v.entry
        if v.verdict == HELPS:
            new = replace(e, verified=e.verified + 1, status=ACTIVE)
        elif v.verdict == HURTS or (v.verdict == NO_EFFECT and e.verified == 0):
            new = replace(e, status=DEMOTED)
        else:
            continue
        if (new.verified, new.status) != (e.verified, e.status):  # skip no-ops
            changes.append((e, new))
    return changes


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
    """Replay the ledger's entries for ``corpus`` (demoted ones too, so a win re-activates).

    ``entries`` overrides what to replay (a dry run only: ``apply`` with it raises,
    since the ledger lines are then not the ones being judged). With ``apply=True``
    the verdicts are patched into the ledger at ``path`` line by line
    (:func:`ir.synonyms.update_entries`); unverifiable entries, entries of other
    corpora and every line this code does not understand are left untouched.
    """
    if entries is not None and apply:
        raise ValueError("apply=True replays the ledger itself; do not pass entries=")
    name = getattr(corpus, "name", corpus if isinstance(corpus, str) else None)
    pool = (
        list(entries)
        if entries is not None
        else load_synonyms(path, corpus=name, include_demoted=True)
    )
    verdicts = [replay_entry(corpus, e, mode=mode, k=k, **search_kw) for e in pool]
    if apply:
        update_entries(apply_verdicts(verdicts), path)
    return verdicts
