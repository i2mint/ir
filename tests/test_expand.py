"""Tests for retrieval-time context expansion (#45) — hermetic.

Pins the operator contracts: overlap-aware stitching (dedupe only between
sequence-adjacent same-kind segments; duplication over truncation), the two
shipped policies (sentence window / parent), operator-enforced safety
(foreign-record rejection, empty-selection degradation), seed identity/score
passthrough, the ledger-only multi-kind path (Package), and the disclosure /
discover wiring (additive ``passage``). Light embedder + memory stores.
"""

import json

import numpy as np
import pytest

import ir
from ir.base import Record, SearchHit
from ir.expand import _stitch
from ir.store import CorpusStore

# Distinctive sentinel tokens: zero-padded so "para001" never substring-matches
# "para011", and appearing exactly ONCE per paragraph so occurrence counts in a
# stitched passage detect duplication precisely.
SENTINELS = [f"para{i:03d}" for i in range(12)]
LONG = "\n\n".join(f"{s} server deployment systemd caddy" for s in SENTINELS)

PKG = {
    "dol": {
        "name": "dol",
        "description": "dict-like facades over storage backends",
        "readme": LONG,
        "owner": "i2mint",
    },
}


def _rec(idx, text, *, kind="chunk", aid="a"):
    return Record(
        id=f"r{kind}{idx}",
        artifact_id=aid,
        surface_kind=kind,
        surface_index=idx,
        text=text,
        vector=np.zeros(4, dtype=np.float32),
    )


def _chunked_corpus(name="xc"):
    src = ir.CorpusSource.from_mapping(
        {"big": {"text": LONG}},
        name=name,
        strategy=ir.Chunked(chunk_size=120, overlap=60),
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _package_corpus(name="xp"):
    src = ir.CorpusSource.from_mapping(
        PKG, name=name, strategy=ir.Package(chunk_size=120, overlap=60)
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


# --------------------------------------------------------------------------- #
# Stitching unit contracts
# --------------------------------------------------------------------------- #


def test_stitch_dedupes_adjacent_overlap():
    a = _rec(0, "alpha beta gamma SHARED-OVERLAP-TAIL")
    b = _rec(1, "SHARED-OVERLAP-TAIL delta epsilon")
    out = _stitch([a, b])
    assert out == "alpha beta gamma SHARED-OVERLAP-TAIL delta epsilon"
    assert out.count("SHARED-OVERLAP-TAIL") == 1


def test_stitch_keeps_text_when_not_sequence_adjacent():
    a = _rec(0, "alpha beta gamma SHARED-OVERLAP-TAIL")
    c = _rec(2, "SHARED-OVERLAP-TAIL delta epsilon")  # gap: index 0 -> 2
    out = _stitch([a, c])
    assert out.count("SHARED-OVERLAP-TAIL") == 2  # no dedupe across a gap
    assert "\n\n" in out


def test_stitch_never_dedupes_across_kinds():
    a = _rec(0, "alpha SHARED-OVERLAP-TAIL", kind="description")
    b = _rec(1, "SHARED-OVERLAP-TAIL beta", kind="chunk")
    out = _stitch([a, b])
    assert out.count("SHARED-OVERLAP-TAIL") == 2


def test_stitch_ignores_coincidental_short_match():
    # Adjacent chunks sharing only a tiny suffix/prefix ("me") must NOT be
    # deduped — below the minimum, a match is coincidence, and dropping it
    # would truncate real text.
    a = _rec(0, "the home")
    b = _rec(1, "me too entirely-different text")
    out = _stitch([a, b])
    assert out == "the home\n\nme too entirely-different text"


def test_stitch_matches_prev_record_not_accumulated_text():
    # Regression: dedupe must compare against the PREVIOUS record's own text.
    # Matching against the accumulated string admits matches that span the
    # synthetic "\n\n" joiner into earlier records — guaranteed coincidence
    # that truncates genuinely repeated document text (here, the second
    # HELLOWORLD and C's leading "A\n\n" would vanish).
    a = _rec(0, "AAAA")
    b = _rec(1, "HELLOWORLD")
    c = _rec(2, "A\n\nHELLOWORLDXYZ")
    out = _stitch([a, b, c])
    assert out.count("HELLOWORLD") == 2
    assert out == "AAAA\n\nHELLOWORLD\n\nA\n\nHELLOWORLDXYZ"


# --------------------------------------------------------------------------- #
# expand — operator contracts on a Chunked corpus
# --------------------------------------------------------------------------- #


def _top_chunk_hit(corpus, query):
    hits = ir.search(corpus, query, k=1)
    assert hits and hits[0].surface_index is not None
    return hits[0]


def test_expand_zero_config_window():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    passage = ir.expand(hit, corpus)
    # The default genuinely expands: a mid-document seed gains both neighbors.
    assert len(passage.record_ids) >= 2
    assert len(passage.text) > len(hit.text)
    # Seed identity and score pass through untouched.
    assert passage.artifact_id == hit.artifact_id
    assert passage.source == hit.source
    assert passage.score == float(hit.score)
    assert passage.surface_index == hit.surface_index


def test_expand_window_never_duplicates_overlap_text():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    passage = ir.expand(hit, corpus, policy=ir.sentence_window_policy(2))
    for s in SENTINELS:
        assert passage.text.count(s) <= 1, f"sentinel {s} duplicated"
    # The window is wider than the seed chunk alone.
    covered = [s for s in SENTINELS if s in passage.text]
    seed_covered = [s for s in SENTINELS if s in hit.text]
    assert len(covered) > len(seed_covered)


def test_expand_window_k0_is_seed_only():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    passage = ir.expand(hit, corpus, policy=ir.sentence_window_policy(0))
    assert passage.text == hit.text
    assert len(passage.record_ids) == 1


def test_expand_parent_policy_returns_whole_artifact():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    passage = ir.expand(hit, corpus, policy=ir.parent_policy())
    for s in SENTINELS:
        assert passage.text.count(s) == 1, f"sentinel {s} missing or duplicated"
    siblings = ir.records_for_artifact(corpus, hit.artifact_id)
    assert list(passage.record_ids) == [r.id for r in siblings]


def test_expand_empty_policy_degrades_to_seed_text():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    passage = ir.expand(hit, corpus, policy=lambda h, sibs: [])
    assert passage.text == hit.text
    assert passage.record_ids == ()


def test_expand_rejects_foreign_records():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    foreign = _rec(99, "not a sibling", aid="elsewhere")
    with pytest.raises(ValueError, match="not siblings"):
        ir.expand(hit, corpus, policy=lambda h, sibs: [foreign])


def test_window_policy_requires_surface_index():
    corpus = _chunked_corpus()
    bare = SearchHit("big", "chunk", 1.0, "text")  # no surface_index
    with pytest.raises(ValueError, match="surface_index"):
        ir.expand(bare, corpus)


def test_expand_unknown_artifact_raises_keyerror():
    corpus = _chunked_corpus()
    ghost = SearchHit("ghost", "chunk", 1.0, "text", surface_index=0)
    with pytest.raises(KeyError):
        ir.expand(ghost, corpus)


def test_expand_window_larger_than_run_returns_whole_run():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    passage = ir.expand(hit, corpus, policy=ir.sentence_window_policy(100))
    siblings = ir.records_for_artifact(corpus, hit.artifact_id)
    assert list(passage.record_ids) == [r.id for r in siblings]


def test_window_policy_rejects_negative_k():
    with pytest.raises(ValueError, match=">= 0"):
        ir.sentence_window_policy(-1)


def test_expand_single_record_artifact_degenerates_to_itself():
    src = ir.CorpusSource.from_mapping(
        {"solo": {"text": "one short document"}}, name="solo", strategy=ir.WholeText()
    )
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    hit = ir.search(corpus, "short document", k=1)[0]
    passage = ir.expand(hit, corpus)
    assert passage.text == hit.text
    assert len(passage.record_ids) == 1


def test_expand_stale_surface_index_raises_seed_not_found():
    from ir.expand import SeedNotFound

    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    stale = SearchHit(
        hit.artifact_id, hit.surface_kind, 1.0, hit.text, {}, hit.source, 999
    )
    with pytest.raises(SeedNotFound):
        ir.expand(stale, corpus)


def test_passage_to_dict_is_json_clean():
    corpus = _chunked_corpus()
    hit = _top_chunk_hit(corpus, "para005 server deployment")
    d = ir.expand(hit, corpus).to_dict()
    json.dumps(d)
    assert isinstance(d["record_ids"], list)
    assert isinstance(d["score"], float)
    assert d["artifact_id"] == hit.artifact_id


# --------------------------------------------------------------------------- #
# Multi-kind (Package): ledger path, window stays within the seed's kind
# --------------------------------------------------------------------------- #


def test_expand_package_window_stays_within_kind():
    corpus = _package_corpus()
    # Seed at the FIRST readme chunk — plan position 1, directly adjacent to
    # the description surface at plan position 0. A kind-ignoring window
    # would swallow the description; the same-kind run must not.
    first_chunk = ir.records_for_artifact(corpus, "dol", surface_kind="readme_chunk")[0]
    assert first_chunk.surface_index == 1  # description occupies position 0
    hit = SearchHit(
        "dol",
        "readme_chunk",
        1.0,
        first_chunk.text,
        first_chunk.metadata,
        "xp",
        first_chunk.surface_index,
    )
    passage = ir.expand(hit, corpus, policy=ir.sentence_window_policy(1))
    assert "dict-like facades" not in passage.text
    assert len(passage.record_ids) == 2  # chunk 0 + chunk 1, nothing before
    for s in SENTINELS:
        assert passage.text.count(s) <= 1


def test_expand_package_parent_spans_kinds():
    corpus = _package_corpus()
    hits = ir.search(
        corpus,
        "para005 server deployment",
        k=5,
        per_artifact=False,
        surfaces=["readme_chunk"],
    )
    passage = ir.expand(hits[0], corpus, policy=ir.parent_policy())
    assert "dict-like facades" in passage.text  # description included
    for s in SENTINELS:
        assert passage.text.count(s) == 1


# --------------------------------------------------------------------------- #
# Disclosure / discover wiring (additive passage)
# --------------------------------------------------------------------------- #


def test_disclose_expand_fills_passage_and_keeps_summary():
    corpus = _chunked_corpus()
    hits = ir.search(corpus, "para005 server deployment", k=5)
    selection = ir.select(hits)
    results = ir.disclose(
        selection,
        level="metadata",
        expand=ir.sentence_window_policy(1),
        corpus=corpus,
    )
    assert results
    for d, hit in zip(results, selection.selected, strict=True):
        assert d.summary == hit.text  # summary contract untouched
        assert isinstance(d.passage, str) and len(d.passage) >= len(d.summary)
        assert d.score == float(hit.score)  # identity/score undisturbed
        dd = d.to_dict()
        json.dumps(dd)
        assert dd["passage"] == d.passage


def test_disclose_without_expand_has_none_passage():
    corpus = _chunked_corpus()
    selection = ir.select(ir.search(corpus, "para005 server deployment", k=3))
    results = ir.disclose(selection, level="metadata")
    assert all(d.passage is None for d in results)
    assert all("passage" in d.to_dict() for d in results)


def test_disclose_expand_requires_corpus_and_vice_versa():
    corpus = _chunked_corpus()
    selection = ir.select(ir.search(corpus, "para005", k=3))
    with pytest.raises(ValueError, match="corpus="):
        ir.disclose(selection, expand=ir.sentence_window_policy())
    with pytest.raises(ValueError, match="expand="):
        ir.disclose(selection, corpus=corpus)


def test_disclose_expand_tolerates_unknown_artifact():
    corpus = _chunked_corpus()
    ghost = SearchHit("ghost", "chunk", 1.0, "text", surface_index=0)
    selection = ir.select([ghost])
    (d,) = ir.disclose(
        selection,
        level="metadata",
        expand=ir.sentence_window_policy(),
        corpus=corpus,
    )
    assert d.passage is None
    assert d.metadata["expansion"] == "no_ledger_entry"


def test_disclose_expand_tolerates_missing_source_corpus():
    corpus = _chunked_corpus()
    hit = ir.search(corpus, "para005 server deployment", k=1)[0]
    selection = ir.select([hit])
    (d,) = ir.disclose(
        selection,
        level="metadata",
        expand=ir.sentence_window_policy(),
        corpus={"some-other-source": corpus},
    )
    assert d.passage is None
    assert d.metadata["expansion"] == "no_corpus_for_source"


def test_disclose_expand_guards_against_wrong_single_corpus():
    # A cross-source hit must not silently expand against a same-id stranger
    # in a DIFFERENT single corpus (artifact identity is per-source).
    corpus = _chunked_corpus(name="xc")
    other = _chunked_corpus(name="other")  # also has artifact "big"
    hit = ir.search(corpus, "para005 server deployment", k=1)[0]
    assert hit.source == "xc"
    selection = ir.select([hit])
    (d,) = ir.disclose(
        selection, level="metadata", expand=ir.sentence_window_policy(), corpus=other
    )
    assert d.passage is None
    assert d.metadata["expansion"] == "no_corpus_for_source"


def test_disclose_expand_tolerates_stale_hit():
    corpus = _chunked_corpus()
    hit = ir.search(corpus, "para005 server deployment", k=1)[0]
    stale = SearchHit(
        hit.artifact_id, hit.surface_kind, 1.0, hit.text, {}, hit.source, 999
    )
    selection = ir.select([stale])
    (d,) = ir.disclose(
        selection, level="metadata", expand=ir.sentence_window_policy(), corpus=corpus
    )
    assert d.passage is None
    assert d.metadata["expansion"] == "seed_not_found"


def test_disclose_expand_propagates_torn_store_keyerror():
    # A ledger entry listing a record missing from the store is data
    # corruption (rebuild the corpus) — NOT a tolerated per-hit condition.
    from ir.base import ledger_key

    corpus = _chunked_corpus()
    hit = ir.search(corpus, "para005 server deployment", k=1)[0]
    entry = corpus.store.get_ledger_entry(ledger_key(hit.artifact_id))
    corpus.store.delete_record(entry["record_ids"][0])  # tear the store
    selection = ir.select([hit])
    with pytest.raises(KeyError, match="stale ledger"):
        ir.disclose(
            selection,
            level="metadata",
            expand=ir.parent_policy(),
            corpus=corpus,
        )


def test_discover_expand_single_corpus():
    corpus = _chunked_corpus()
    result = ir.discover(
        corpus, "para005 server deployment", expand=ir.sentence_window_policy(1)
    )
    assert result.results
    for d in result.results:
        assert isinstance(d.passage, str)
    json.dumps(result.to_dict())


def test_discover_expand_federated_resolves_per_source():
    c1 = _chunked_corpus(name="fed-one")
    c2 = _package_corpus(name="fed-two")
    result = ir.discover(
        [c1, c2],
        "para005 server deployment",
        expand=ir.sentence_window_policy(1),
        surfaces=["chunk", "readme_chunk"],
    )
    assert result.results
    for d in result.results:
        assert d.source in ("fed-one", "fed-two")
        assert isinstance(d.passage, str)
    # Both sources contribute (pins that per-source resolution actually ran
    # against BOTH corpora, not just one).
    assert {d.source for d in result.results} == {"fed-one", "fed-two"}
