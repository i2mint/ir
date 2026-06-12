"""Retrieval-time context expansion — the ``expand(hit)`` operator (report 13).

A retriever returns the matched chunk in isolation; the unit downstream wants
is usually neither the bare chunk nor the whole document. Sentence-window,
parent-document, auto-merging, and topic-bounded expansion are **the same
operator with different neighborhood policies** over NEXT/PREV/PARENT segment
relationships, so ir ships ONE operator parameterized by an injectable
:data:`NeighborhoodPolicy` — the same callable-seam style as ``Selector`` and
``Formulator``.

Pipeline position: **retrieve → expand → rerank** — expansion composes after
retrieval (and selection) and before reranking/generation, enriching what the
reader sees without touching hit identity ``(source, artifact_id)`` or scores.
The disclosure seam exposes it as ``disclose(..., expand=policy, corpus=...)``
(the :class:`~ir.select.Disclosure` gains a ``passage``), and
``discover(..., expand=...)`` opts in end-to-end.

Sibling segments are resolved through the ledger only
(:func:`ir.retrieve.records_for_artifact` — never by re-deriving record ids;
see :meth:`ir.base.Record.make_id`), which is what makes multi-kind strategies
(:class:`~ir.strategy.Package`) expand correctly.

Stitching is **overlap-aware and conservative**: chunkers carry an overlap tail
between consecutive chunks, so naive concatenation duplicates text. Adjacent
same-kind segments are deduped by their longest suffix→prefix match; anything
non-adjacent or cross-kind joins with a blank line and is never deduped, so an
accidental match can never *truncate* text — when in doubt, the stitcher
prefers a little duplication over any loss.

Shipped policies: :func:`sentence_window_policy` (±k same-kind neighbors — the
zero-config default) and :func:`parent_policy` (the whole artifact,
small-to-big). Auto-merging and topic-bounded policies are deliberately not
shipped until evaluation shows queries spanning sibling leaves (report 13's
stage-3 gate); both fit the same :data:`NeighborhoodPolicy` seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .base import Record, SearchHit
from .retrieve import records_for_artifact

#: Default sentence-window half-width (±chunks around the seed). Segments are
#: ~chunk_size paragraph-packed chunks (not sentences), so ±1 already roughly
#: triples the seed snippet; raise ``k`` for sentence-granularity corpora.
DFLT_WINDOW = 1

#: Minimum suffix→prefix match length treated as real chunker overlap when
#: stitching. Below this, a match is more likely coincidence than carry-over;
#: the stitcher then joins with a blank line instead (duplication over
#: truncation — see module docstring).
DFLT_MIN_STITCH_OVERLAP = 8

#: A neighborhood policy: ``(seed_hit, ordered_siblings) -> the records to
#: stitch``. Pure membership selection over the artifact's ledger-ordered
#: sibling records — the operator owns fetching, ordering, and stitching, so a
#: policy cannot corrupt assembly (non-sibling records are rejected, output is
#: re-ordered to plan order, an empty selection degrades to the seed's text).
NeighborhoodPolicy = Callable[[SearchHit, Sequence[Record]], Sequence[Record]]


class SeedNotFound(ValueError):
    """The seed hit's ``(surface_kind, surface_index)`` is not among its
    artifact's stored records — typically a corpus re-chunked between search
    and expansion (a stale hit), or a hit expanded against the wrong corpus.

    A ``ValueError`` subclass so existing ``except ValueError`` handling keeps
    working; the disclosure seam catches it per hit (a data condition, not a
    programming error) and notes ``expansion="seed_not_found"``.
    """


@dataclass(frozen=True)
class Passage:
    """An expanded hit: the seed's identity + the stitched neighborhood text.

    ``artifact_id`` / ``surface_kind`` / ``score`` / ``source`` /
    ``surface_index`` are the **seed hit's** — expansion never disturbs hit
    identity ``(source, artifact_id)`` or scores. ``text`` is the assembled
    neighborhood (overlap-deduped, plan order) and ``record_ids`` the ordered
    stored segments it was stitched from (empty when expansion degraded to the
    seed's own text).
    """

    artifact_id: str
    surface_kind: str
    score: float
    text: str
    record_ids: tuple[str, ...] = ()
    source: str | None = None
    surface_index: int | None = None

    def to_dict(self) -> dict:
        """JSON-serializable form (``score`` cast to ``float``, ids as a list)."""
        return {
            "artifact_id": self.artifact_id,
            "surface_kind": self.surface_kind,
            "score": float(self.score),
            "text": self.text,
            "record_ids": list(self.record_ids),
            "source": self.source,
            "surface_index": self.surface_index,
        }


def sentence_window_policy(k: int = DFLT_WINDOW) -> NeighborhoodPolicy:
    """±*k* same-kind neighbors around the seed (NEXT/PREV expansion).

    The window runs over the artifact's surfaces of the **seed's kind only**,
    in plan order — a ``readme_chunk`` window never swallows the
    ``description`` surface. ``k=0`` selects just the seed's own record.
    """
    if k < 0:
        raise ValueError(f"window half-width k must be >= 0; got {k}")

    def policy(hit: SearchHit, siblings: Sequence[Record]) -> Sequence[Record]:
        if hit.surface_index is None:
            raise ValueError(
                "sentence_window_policy needs hit.surface_index to locate the "
                "seed among its siblings, but this hit has none (a hand-built "
                "hit?). Hits from ir.search carry it; otherwise use "
                "parent_policy(), which needs no seed position."
            )
        run = [r for r in siblings if r.surface_kind == hit.surface_kind]
        pos = next(
            (i for i, r in enumerate(run) if r.surface_index == hit.surface_index),
            None,
        )
        if pos is None:
            raise SeedNotFound(
                f"no stored record of kind {hit.surface_kind!r} with "
                f"surface_index {hit.surface_index} for artifact "
                f"{hit.artifact_id!r} — is this hit from a different corpus "
                f"(or a stale build)?"
            )
        return run[max(0, pos - k) : pos + k + 1]

    return policy


def parent_policy() -> NeighborhoodPolicy:
    """The whole artifact (small-to-big): every stored surface, plan order.

    The mid-granularity analogue of ``disclose(level="body")`` — but assembled
    from the *indexed* segments rather than dereferencing the pointer, so it
    works for corpora whose artifacts have no on-disk body.
    """

    def policy(hit: SearchHit, siblings: Sequence[Record]) -> Sequence[Record]:
        return siblings

    return policy


def _shared_overlap(prev: str, nxt: str, *, min_overlap: int) -> int:
    """Length of the longest suffix of *prev* that prefixes *nxt*.

    Returns 0 when the longest match is shorter than *min_overlap* (treated as
    coincidence, not chunker carry-over).
    """
    for length in range(min(len(prev), len(nxt)), min_overlap - 1, -1):
        if prev.endswith(nxt[:length]):
            return length
    return 0


def _stitch(
    records: Sequence[Record], *, min_overlap: int = DFLT_MIN_STITCH_OVERLAP
) -> str:
    """Assemble plan-ordered records into one text, deduping chunker overlap.

    Only **sequence-adjacent, same-kind** pairs (consecutive
    ``surface_index``) are candidates for overlap dedupe — those are the pairs
    a chunker actually carried an overlap tail between. Everything else joins
    with a blank line, untouched.
    """
    text = records[0].text
    # strict=False: pairwise iteration over (r[i], r[i+1]) is intentionally
    # one shorter than records.
    for prev, cur in zip(records, records[1:], strict=False):
        if (
            cur.surface_kind == prev.surface_kind
            and cur.surface_index == prev.surface_index + 1
        ):
            # Match against prev's OWN text, not the accumulated string: real
            # chunker carry-over is bounded by prev, while a longer match
            # against the accumulation necessarily spans the synthetic "\n\n"
            # joiner or earlier records — guaranteed coincidence that would
            # truncate genuinely repeated document text.
            shared = _shared_overlap(prev.text, cur.text, min_overlap=min_overlap)
            if shared:
                text += cur.text[shared:]
                continue
        text += "\n\n" + cur.text
    return text


def expand(
    hit: SearchHit, corpus: Any, *, policy: NeighborhoodPolicy | None = None
) -> Passage:
    """Expand *hit* into a :class:`Passage` of its neighborhood in *corpus*.

    Fetches the hit's sibling records through the ledger
    (:func:`ir.retrieve.records_for_artifact`), asks *policy* which to keep,
    and stitches them in plan order with overlap-aware dedupe. The default
    policy is a ±:data:`DFLT_WINDOW` sentence window; pass
    :func:`parent_policy` for the whole artifact, or any
    :data:`NeighborhoodPolicy`.

    Args:
        hit: the seed :class:`~ir.base.SearchHit` (its identity and score pass
            through to the :class:`Passage` untouched).
        corpus: a :class:`~ir.index.Corpus`, :class:`~ir.store.CorpusStore`,
            or corpus name — whatever :func:`~ir.retrieve.records_for_artifact`
            accepts. Must be the corpus the hit came from.
        policy: which siblings make up the neighborhood (default: sentence
            window). A policy that selects nothing degrades the passage to the
            hit's own text (``record_ids=()``) rather than returning nothing.

    Raises:
        KeyError: the corpus has no ledger entry for the hit's artifact
            (:class:`ir.retrieve.NoLedgerEntry`), or the ledger is stale —
            an entry listing records missing from the store.
        SeedNotFound: the default window policy cannot find the seed among
            its artifact's stored records (stale hit / wrong corpus).
        ValueError: the policy returned records that are not siblings of the
            hit's artifact (operator-enforced safety), or the seed hit lacks
            ``surface_index`` (hand-built hit) under the default window
            policy — use :func:`parent_policy`, which needs no seed position.
    """
    policy = sentence_window_policy() if policy is None else policy
    siblings = records_for_artifact(corpus, hit.artifact_id)
    chosen = list(policy(hit, siblings))
    sibling_ids = {r.id for r in siblings}
    foreign = [r.id for r in chosen if r.id not in sibling_ids]
    if foreign:
        raise ValueError(
            f"neighborhood policy selected records that are not siblings of "
            f"artifact {hit.artifact_id!r} (e.g. {foreign[:3]}); a policy may "
            f"only choose from the records it was given"
        )
    seen: set[str] = set()
    ordered: list[Record] = []
    for r in sorted(chosen, key=lambda r: r.surface_index):
        if r.id not in seen:
            seen.add(r.id)
            ordered.append(r)
    if not ordered:
        return Passage(
            artifact_id=hit.artifact_id,
            surface_kind=hit.surface_kind,
            score=float(hit.score),
            text=hit.text,
            record_ids=(),
            source=hit.source,
            surface_index=hit.surface_index,
        )
    return Passage(
        artifact_id=hit.artifact_id,
        surface_kind=hit.surface_kind,
        score=float(hit.score),
        text=_stitch(ordered),
        record_ids=tuple(r.id for r in ordered),
        source=hit.source,
        surface_index=hit.surface_index,
    )
