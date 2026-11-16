"""Ledger provenance and replay through ir.eval (#136)."""

import pytest

import ir
from ir.store import CorpusStore
from ir.synonyms import (
    DEMOTED,
    SynonymEntry,
    add_synonyms,
    load_synonyms,
    write_synonyms,
)
from ir.synonyms_eval import (
    HELPS,
    NO_EFFECT,
    UNVERIFIABLE,
    case_of,
    replay_entry,
    replay_ledger,
)

DOCS = {
    "d1": "acmecorp quarterly summary",
    "d2": "ac quarterly report ac ac",
    "d3": "bake a cake with flour",
}
PROV = {"miss_query": "ac", "found_ids": ["d1"]}


@pytest.fixture
def corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="led", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def test_provenance_round_trips_through_the_ledger_file(tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    (e,) = load_synonyms(p)
    assert e.provenance == PROV and e.uses == 0 and e.status == "active"
    # defaults stay out of the file, so existing ledgers keep their shape
    add_synonyms(["x1", "y1"], path=p)
    assert "provenance" not in p.read_text().splitlines()[1]
    assert "status" not in p.read_text().splitlines()[1]


def test_demoted_entries_are_kept_but_not_loaded(tmp_path):
    p = tmp_path / "s.jsonl"
    write_synonyms([SynonymEntry(("aa", "bb"), status=DEMOTED)], p)
    assert load_synonyms(p) == []
    assert len(load_synonyms(p, include_demoted=True)) == 1


def test_case_of_needs_a_complete_provenance():
    assert case_of(SynonymEntry(("a1", "b1"))) is None
    assert case_of(SynonymEntry(("a1", "b1"), provenance={"miss_query": "q"})) is None
    c = case_of(SynonymEntry(("a1", "b1"), provenance=PROV))
    assert c.query == "ac" and c.gold == ("d1",)


def test_an_entry_that_found_the_gold_helps(corpus):
    v = replay_entry(corpus, SynonymEntry(("acmecorp", "ac"), provenance=PROV))
    assert v.verdict == HELPS and v.with_entry > v.without


def test_an_entry_that_does_not_apply_has_no_effect(corpus):
    v = replay_entry(corpus, SynonymEntry(("zzz", "yyy"), provenance=PROV))
    assert v.verdict == NO_EFFECT and v.with_entry == v.without


def test_entry_without_provenance_is_unverifiable(corpus):
    assert replay_entry(corpus, SynonymEntry(("a1", "b1"))).verdict == UNVERIFIABLE


def test_replay_apply_counts_wins_and_demotes_the_rest(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    add_synonyms(["zzz", "yyy"], provenance=PROV, path=p)
    add_synonyms(["no1", "no2"], path=p)  # no provenance
    verdicts = replay_ledger(corpus, path=p, apply=True)
    assert [v.verdict for v in verdicts] == [HELPS, NO_EFFECT, UNVERIFIABLE]
    by = {e.terms[0]: e for e in load_synonyms(p, include_demoted=True)}
    assert by["acmecorp"].uses == 1 and by["acmecorp"].status == "active"
    assert by["zzz"].status == DEMOTED
    assert by["no1"].status == "active" and by["no1"].uses == 0
    # the demoted one no longer expands searches; the survivor does
    assert [e.terms[0] for e in load_synonyms(p)] == ["acmecorp", "no1"]


def test_replay_without_apply_does_not_touch_the_file(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["zzz", "yyy"], provenance=PROV, path=p)
    before = p.read_text()
    replay_ledger(corpus, path=p)
    assert p.read_text() == before
