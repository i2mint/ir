"""Tests for synopsis surfaces (#48) — LLM-derived summaries as indexed surfaces.

Pins the #48 acceptance:

- ``with_synopsis`` adds one ``synopsis`` surface per artifact, *prepended* (so it
  is the collapsed-tree router); an empty synopsis is dropped.
- search restricted to synopsis surfaces + ``traverse`` routes to the right chunks
  end-to-end (the synopsis is the routing signal; a trap whose synopsis does not
  match is excluded though its chunk matches the query).
- incremental rebuild re-synthesizes only changed artifacts, and a
  synthesizer-identity (or inner-strategy) change re-synthesizes — staleness via
  the ledger ``strategy_id`` and the recursive ``_strategy_id``.
- offline import preserved: ``oa`` is lazy, an injected synthesizer never needs it.

Hermetic: light embedder + memory store + injected synthesizers.
"""

import ir
from ir.base import Artifact
from ir.index import _strategy_id
from ir.retrieve import records_for_artifact
from ir.store import CorpusStore
from ir.synopsis import make_llm_synthesizer, with_synopsis


def _synopsis_from_field(artifact):
    """An injected synthesizer double: read the artifact's ``synopsis`` field."""
    raw = artifact.raw
    return raw.get("synopsis", "") if isinstance(raw, dict) else ""


# --------------------------------------------------------------------------- #
# Surface: one prepended synopsis surface per artifact
# --------------------------------------------------------------------------- #


def test_with_synopsis_indexes_one_prepended_synopsis_surface():
    docs = {"a": {"text": "body alpha here", "synopsis": "a short routing summary"}}
    strat = with_synopsis(
        ir.Chunked(), synthesize=_synopsis_from_field, synthesizer_id="v1"
    )
    src = ir.CorpusSource.from_mapping(docs, name="s", strategy=strat)
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")

    recs = records_for_artifact(corpus, "a")
    assert recs[0].surface_kind == "synopsis"  # prepended -> plan position 0
    assert recs[0].surface_index == 0
    assert recs[0].text == "a short routing summary"
    assert recs[0].metadata["synthesizer_id"] == "v1"  # provenance stamp
    assert any(r.surface_kind == "chunk" for r in recs)  # inner surfaces kept

    syn = ir.search(corpus, "short routing summary", surfaces=("synopsis",))
    assert syn and syn[0].artifact_id == "a" and syn[0].surface_kind == "synopsis"


def test_empty_synopsis_is_dropped_artifact_keeps_other_surfaces():
    docs = {"a": {"text": "body alpha here", "synopsis": ""}}
    strat = with_synopsis(
        ir.Chunked(), synthesize=_synopsis_from_field, synthesizer_id="v1"
    )
    corpus = ir.build(
        ir.CorpusSource.from_mapping(docs, name="s", strategy=strat),
        store=CorpusStore.memory(),
        embedder="light",
    )
    recs = records_for_artifact(corpus, "a")
    assert all(r.surface_kind != "synopsis" for r in recs)
    assert any(r.surface_kind == "chunk" for r in recs)


# --------------------------------------------------------------------------- #
# Routing: search-on-synopsis + traverse routes to the artifact's chunks
# --------------------------------------------------------------------------- #

# Gold A: synopsis matches the routing tokens, the answer is in its chunk.
# Trap B: synopsis does NOT match, but its chunk matches the routing tokens.
# Fillers C, D: non-matching synopses, present so B falls below seed_k.
ROUTING_DOCS = {
    "A": {
        "text": "ANSTOK answer payload here. filler one two three.",
        "synopsis": "rtok1 rtok2 rtok3 rtok4 alpha beta",
    },
    "B": {
        "text": "rtok1 rtok2 rtok3 rtok4 trap distractor strong body.",
        "synopsis": "omega phi chi psi unrelated",
    },
    "C": {"text": "neutral cccc content.", "synopsis": "gamma delta epsilon zeta"},
    "D": {"text": "neutral dddd content.", "synopsis": "eta theta iota kappa"},
}
ROUTING_QUERY = "rtok1 rtok2 rtok3 rtok4 ANSTOK"


def _routing_corpus():
    strat = with_synopsis(
        ir.Chunked(chunk_size=80, overlap=10),
        synthesize=_synopsis_from_field,
        synthesizer_id="v1",
    )
    return ir.build(
        ir.CorpusSource.from_mapping(ROUTING_DOCS, name="rt", strategy=strat),
        store=CorpusStore.memory(),
        embedder="light",
    )


def test_synopsis_routes_traverse_to_chunks_end_to_end():
    corpus = _routing_corpus()
    trav = ir.traverse(
        ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(seed_k=2), k=10
    )
    assert trav
    # routed via the synopsis (summary) down to chunk leaves — synopses themselves
    # are routers, never emitted.
    assert all(h.surface_kind == "chunk" for h in trav)
    # gold A (synopsis matched) surfaces its answer chunk...
    assert trav[0].artifact_id == "A"
    assert "ANSTOK" in trav[0].text
    # ...and trap B (synopsis didn't match -> not seeded) is excluded, though its
    # chunk matches the query terms.
    assert all(h.artifact_id != "B" for h in trav)
    # walk provenance: a routed leaf at depth 1, seeded by A's synopsis.
    assert trav[0].metadata["walk_depth"] == 1
    assert trav[0].metadata["seed"] == "A"


def test_flat_search_buries_the_answer_that_synopsis_routing_surfaces():
    # The discriminator: among chunks, flat ranks B's trap chunk first; synopsis
    # routing excludes B entirely — proof the synopsis route does real work.
    corpus = _routing_corpus()
    flat = ir.search(corpus, ROUTING_QUERY, k=10, per_artifact=False)
    flat_chunks = [h for h in flat if h.surface_kind == "chunk"]
    assert flat_chunks and flat_chunks[0].artifact_id == "B"


# --------------------------------------------------------------------------- #
# Staleness: incremental re-synthesis keyed on synthesizer + strategy identity
# --------------------------------------------------------------------------- #


def test_incremental_resynthesizes_only_changed_and_on_identity_change():
    calls = []

    def synth(artifact):
        calls.append(artifact.id)
        return f"summary of {artifact.id}"

    def source(docs, sid):
        strat = with_synopsis(ir.Chunked(), synthesize=synth, synthesizer_id=sid)
        return ir.CorpusSource.from_mapping(docs, name="inc", strategy=strat)

    store = CorpusStore.memory()
    docs = {"A": {"text": "a body"}, "B": {"text": "b body"}}

    ir.build(source(docs, "v1"), store=store, embedder="light")
    assert sorted(calls) == ["A", "B"]  # first build synthesizes both
    calls.clear()

    ir.build(source(docs, "v1"), store=store, embedder="light")
    assert calls == []  # unchanged rebuild -> no synthesis

    docs2 = {"A": {"text": "a body CHANGED"}, "B": {"text": "b body"}}
    ir.build(source(docs2, "v1"), store=store, embedder="light")
    assert calls == ["A"]  # only the changed artifact re-synthesized
    calls.clear()

    ir.build(source(docs2, "v2"), store=store, embedder="light")
    assert sorted(calls) == ["A", "B"]  # synthesizer identity changed -> all


def test_strategy_id_recurses_into_inner_strategy_and_synthesizer():
    base = with_synopsis(
        ir.Chunked(chunk_size=500), synthesize=_synopsis_from_field, synthesizer_id="v1"
    )
    inner_changed = with_synopsis(
        ir.Chunked(chunk_size=900), synthesize=_synopsis_from_field, synthesizer_id="v1"
    )
    synth_changed = with_synopsis(
        ir.Chunked(chunk_size=500), synthesize=_synopsis_from_field, synthesizer_id="v2"
    )
    # an inner-strategy param change AND a synthesizer-id change each shift the id
    assert _strategy_id(base) != _strategy_id(inner_changed)
    assert _strategy_id(base) != _strategy_id(synth_changed)
    # the inner strategy's own identity is embedded (recursion), not the callable
    assert "Chunked:" in _strategy_id(base)


# --------------------------------------------------------------------------- #
# Offline: oa is lazy; an injected synthesizer never needs it
# --------------------------------------------------------------------------- #


def test_make_llm_synthesizer_uses_injected_summarize_without_oa():
    synth = make_llm_synthesizer(summarize=lambda text: "S:" + text[:5])
    assert synth(Artifact("x", "hello world")) == "S:hello"


def test_make_llm_synthesizer_empty_text_returns_empty():
    synth = make_llm_synthesizer(summarize=lambda text: "never reached")
    assert synth(Artifact("x", "   ")) == ""


def test_make_llm_synthesizer_swallows_summarizer_errors():
    def boom(text):
        raise RuntimeError("no LLM available")

    assert make_llm_synthesizer(summarize=boom)(Artifact("x", "body text")) == ""


def test_default_synthesizer_constructs_offline_and_stamps_identity():
    # No synthesize injected -> the lazy-oa default; constructing must not need oa,
    # and an empty-text artifact short-circuits before the oa path is reached.
    strat = with_synopsis(ir.Chunked())
    assert strat.synthesizer_id.startswith("oa:")
    assert strat.synthesize(Artifact("x", "")) == ""
    # a prompt change shifts the default identity (re-synthesis trigger)
    a = make_llm_synthesizer()
    b = make_llm_synthesizer(prompt="a very different prompt {text}")
    assert a.synthesizer_id.startswith("oa:") and a.synthesizer_id != b.synthesizer_id
