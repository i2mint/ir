"""Lexical miss signals (#136): model-free facts for an Evaluator, never a loop."""

import pytest

import ir
from ir.base import SearchHit
from ir.miss import lexical_miss
from ir.select import select
from ir.store import CorpusStore


def _hit(text, aid="a", score=1.0):
    return SearchHit(aid, "doc", score, text)


def test_zero_hits_is_a_miss():
    m = lexical_miss([])
    assert m["zero_hits"] and m["miss"]


def test_required_terms_are_whole_word_case_insensitive_phrases():
    hits = [_hit("Acme Corp sent the invoice")]
    assert lexical_miss(hits, required=["acme corp"])["missing_required"] == []
    assert lexical_miss(hits, required=["ACME"])["missing_required"] == []
    # a substring of another word is not a mention
    assert lexical_miss(hits, required=["voice"])["missing_required"] == ["voice"]
    assert lexical_miss(hits, required=["corp acme"])["missing_required"] == [
        "corp acme"
    ]
    assert lexical_miss(hits, required=["acme", ""])["miss"] is False


def test_required_terms_are_checked_across_all_the_hits():
    hits = [_hit("acme paid", "a"), _hit("globex refund", "b")]
    assert lexical_miss(hits, required=["acme", "globex"])["miss"] is False
    assert lexical_miss(hits, required=["acme", "initech"])["missing_required"] == [
        "initech"
    ]


def test_candidates_separate_a_retrieval_miss_from_a_tight_selection():
    committed = [_hit("acme invoice", "a")]
    pool = [*committed, _hit("globex refund", "b")]
    m = lexical_miss(committed, required=["globex", "initech"], candidates=pool)
    assert m["missing_required"] == ["globex", "initech"]
    assert m["missing_in_candidates"] == ["initech"]  # globex was retrieved, not chosen
    assert not m["zero_hits"]


def test_abstained_selection_with_candidates_is_not_zero_hits():
    m = lexical_miss([], required=["acme"], candidates=[_hit("acme")])
    assert m["zero_hits"] is False and m["missing_required"] == ["acme"]
    assert m["missing_in_candidates"] == []


def test_a_bare_string_is_one_term_not_letters():
    assert lexical_miss([_hit("acme")], required="acme")["miss"] is False
    sel = select([_hit("acme")], required_terms="acme")
    assert sel.sufficient and sel.signals["lexical_miss"]["missing_required"] == []


def test_select_without_required_terms_is_unchanged():
    hits = [_hit("acme", "a", 1.0), _hit("zzz", "b", 0.2)]
    sel = select(hits)
    assert "lexical_miss" not in sel.signals and sel.sufficient


def test_select_required_terms_flips_sufficient_but_not_the_selection():
    hits = [_hit("acme invoice", "a", 1.0)]
    base = select(hits)
    sel = select(hits, required_terms=["globex"])
    assert sel.selected_ids == base.selected_ids  # a signal, not a directive
    assert sel.signals["lexical_miss"]["missing_required"] == ["globex"]
    assert base.sufficient and not sel.sufficient
    assert select(hits, required_terms=["acme"]).sufficient


@pytest.fixture
def corpus():
    docs = {"d1": "acme invoice paid", "d2": "bake a cake with flour"}
    src = ir.CorpusSource.from_mapping(docs, name="miss", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def test_discover_reports_the_signal(corpus):
    ok = ir.discover(corpus, "acme invoice", mode="lexical", required_terms=["acme"])
    assert (
        ok.sufficient
        and ok.to_dict()["sufficient"] is True
        and ok.to_dict()["signals"]["lexical_miss"]["miss"] is False
    )
    bad = ir.discover(corpus, "acme invoice", mode="lexical", required_terms=["globex"])
    assert not bad.sufficient and bad.ids == ok.ids
    assert bad.signals["lexical_miss"]["missing_required"] == ["globex"]
    plain = ir.discover(corpus, "acme invoice", mode="lexical")
    assert plain.sufficient and "lexical_miss" not in plain.signals


def test_federated_discover_forwards_required_terms(corpus):
    res = ir.discover([corpus], "acme invoice", mode="lexical", required_terms=["x"])
    assert res.ids and not res.sufficient
    assert res.signals["lexical_miss"]["missing_required"] == ["x"]
    assert res.to_dict()["sufficient"] is False
