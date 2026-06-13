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

import pytest

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


# --------------------------------------------------------------------------- #
# Review-driven hardening (#48 adversarial review)
# --------------------------------------------------------------------------- #


def test_default_synthesizer_threads_inner_strategy_text_key(monkeypatch):
    # The default synthesizer must summarize the SAME field the inner strategy
    # indexes: with_synopsis(Chunked(text_key="body")) summarizes "body", not a
    # competing "text" field. Capture the summarized text via a patched default.
    seen = []
    import ir.synopsis as _syn

    monkeypatch.setattr(
        _syn,
        "_default_llm_summarizer",
        lambda prompt, model, **kw: (lambda t: (seen.append(t) or "S")),
    )
    strat = with_synopsis(ir.Chunked(text_key="body"))  # default synth, no id arg
    docs = {"a": {"body": "REAL body content here", "text": "WRONG text field"}}
    ir.build(
        ir.CorpusSource.from_mapping(docs, name="tk", strategy=strat),
        store=CorpusStore.memory(),
        embedder="light",
    )
    assert seen == ["REAL body content here"]


def test_unnamed_synthesizer_warns_and_disables_staleness_tracking():
    # A lambda / local closure has no stable identity -> swapping it would be
    # silently stale; with_synopsis warns and uses a sentinel id instead.
    with pytest.warns(UserWarning, match="stable identity"):
        s = with_synopsis(ir.Chunked(), synthesize=lambda a: "x")
    assert s.synthesizer_id == "custom"
    # a named top-level function has a stable qualname -> tracks, no warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s2 = with_synopsis(ir.Chunked(), synthesize=_synopsis_from_field)
    assert s2.synthesizer_id == "_synopsis_from_field"


def test_with_synopsis_package_prepends_synopsis_before_description_and_routes():
    # On Package (which already has a "description" summary surface), the synopsis
    # is prepended ahead of it -> the synopsis is the router. An artifact whose
    # synopsis matches (description does not) routes to its readme chunk.
    docs = {
        "A": {
            "name": "alpha",
            "description": "terse unrelated blurb",
            "readme": "ANSTOK the answer lives in the body here. filler one two.",
            "synopsis": "rtok1 rtok2 rtok3 rtok4 routing match",
        },
        "B": {
            "name": "beta",
            "description": "gamma delta",
            "readme": "neutral bbbb body.",
            "synopsis": "omega phi unrelated",
        },
        "C": {
            "name": "c",
            "description": "x",
            "readme": "neutral cccc body.",
            "synopsis": "eta theta iota",
        },
    }
    strat = with_synopsis(
        ir.Package(chunk_size=80, overlap=10),
        synthesize=_synopsis_from_field,
        synthesizer_id="v1",
    )
    corpus = ir.build(
        ir.CorpusSource.from_mapping(docs, name="pkg", strategy=strat),
        store=CorpusStore.memory(),
        embedder="light",
    )
    kinds = [r.surface_kind for r in records_for_artifact(corpus, "A")]
    assert kinds[0] == "synopsis"  # prepended ahead of...
    assert "description" in kinds  # ...the Package description surface
    assert "readme_chunk" in kinds
    # routes via A's matching synopsis to A's answer chunk (description doesn't match)
    trav = ir.traverse(
        "rtok1 rtok2 rtok3 rtok4 ANSTOK",
        corpus,
        policy=ir.collapsed_tree_policy(seed_k=1),
        k=5,
    )
    assert trav and trav[0].artifact_id == "A"
    assert trav[0].surface_kind == "readme_chunk"
    assert "ANSTOK" in trav[0].text


def test_default_synthesizer_construction_is_lazy_and_offline(monkeypatch):
    # Mutation-resistant offline guarantee: poison oa so any use raises, then
    # confirm constructing the default synthesizer (and the wrapper) and the
    # injected path never reach it. An eager-import regression would fail here.
    import sys
    import types

    poison = types.ModuleType("oa")

    def _boom(*a, **k):
        raise AssertionError("oa.prompt_function reached on an offline path")

    poison.prompt_function = _boom
    monkeypatch.setitem(sys.modules, "oa", poison)

    synth = make_llm_synthesizer()  # lazy: builds nothing yet
    strat = with_synopsis(ir.Chunked())  # default synth: still no oa
    assert synth(Artifact("x", "")) == ""  # empty text short-circuits the oa path
    assert strat.synthesize(Artifact("y", "")) == ""
    # an injected summarizer is wholly oa-free even on non-empty text
    assert make_llm_synthesizer(summarize=lambda t: "S")(Artifact("z", "body")) == "S"


def test_default_synthesizer_id_is_content_stable():
    # The default id is content-derived (prompt/model), so two constructions with
    # the same prompt agree — staleness keys on content, not object identity.
    a = make_llm_synthesizer()
    b = make_llm_synthesizer()
    assert a.synthesizer_id == b.synthesizer_id
    c = make_llm_synthesizer(prompt="a different prompt {text}")
    assert c.synthesizer_id != a.synthesizer_id


def test_non_str_synthesize_yields_no_synopsis_surface():
    # The decompose-level guard (independent of make_llm_synthesizer): a synth
    # returning a non-str adds no synopsis surface, never crashes the build.
    strat = with_synopsis(
        ir.Chunked(), synthesize=lambda a: 123, synthesizer_id="nonstr"
    )
    corpus = ir.build(
        ir.CorpusSource.from_mapping(
            {"a": {"text": "body alpha"}}, name="ns", strategy=strat
        ),
        store=CorpusStore.memory(),
        embedder="light",
    )
    recs = records_for_artifact(corpus, "a")
    assert all(r.surface_kind != "synopsis" for r in recs)
    assert any(r.surface_kind == "chunk" for r in recs)


def test_file_backed_incremental_round_trip(tmp_path, monkeypatch):
    # Staleness through a real file store + reopen: the recursive strategy_id
    # (nested inner id + synthesizer id) survives JSON round-trip in the ledger.
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    calls = []

    def synth(artifact):
        calls.append(artifact.id)
        return f"synopsis of {artifact.id}"

    docs = {"A": {"text": "a body"}, "B": {"text": "b body"}}

    def source(sid):
        strat = with_synopsis(ir.Chunked(), synthesize=synth, synthesizer_id=sid)
        return ir.CorpusSource.from_mapping(docs, name="syn_fb", strategy=strat)

    ir.build(source("v1"), store=CorpusStore.local("syn_fb"), embedder="light")
    assert sorted(calls) == ["A", "B"]
    # the synopsis surface persisted to disk and reopens
    recs = records_for_artifact(CorpusStore.local("syn_fb"), "A")
    assert any(r.surface_kind == "synopsis" for r in recs)
    calls.clear()

    ir.build(source("v1"), store=CorpusStore.local("syn_fb"), embedder="light")
    assert calls == []  # unchanged rebuild through a fresh handle -> no synthesis
    ir.build(source("v2"), store=CorpusStore.local("syn_fb"), embedder="light")
    assert sorted(calls) == ["A", "B"]  # synthesizer identity changed -> all
