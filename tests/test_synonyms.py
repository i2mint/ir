"""Tests for the synonym ledger (ir.synonyms) and its CLI / formulate= wiring.

Hermetic: a ledger in ``tmp_path`` (via ``$IR_SYNONYMS``), the light embedder.
"""

import re

import pytest

import ir
from ir import cli
from ir.store import CorpusStore
from ir.synonyms import (
    SynonymEntry,
    add_synonyms,
    aliases,
    expand_query,
    ledger_path,
    load_synonyms,
    make_synonym_formulator,
    term_regex,
)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "synonyms.jsonl"
    monkeypatch.setenv("IR_SYNONYMS", str(p))
    return p


def test_ledger_path_honours_env(ledger):
    assert ledger_path() == ledger


def test_missing_ledger_is_empty(ledger):
    assert load_synonyms() == []


def test_add_then_load_round_trips(ledger):
    e = add_synonyms(["acmecorp", " acme corp ", "", "ac"], note="client")
    assert e.terms == ("acmecorp", "acme corp", "ac") and e.added
    assert load_synonyms() == [e]


def test_add_needs_two_terms(ledger):
    with pytest.raises(ValueError):
        add_synonyms(["solo"])


def test_malformed_lines_are_skipped(ledger):
    ledger.write_text('not json\n{"terms": ["one"]}\n{"terms": ["a1", "b1"]}\n')
    assert [e.terms for e in load_synonyms()] == [("a1", "b1")]


def test_corpus_scoping(ledger):
    add_synonyms(["x1", "y1"], corpus="sessions")
    add_synonyms(["x2", "y2"])
    assert len(load_synonyms(corpus="sessions")) == 2
    assert [e.terms for e in load_synonyms(corpus="reports")] == [("x2", "y2")]


def test_aliases_equivalent_and_oneway():
    es = [
        SynonymEntry(("acmecorp", "acme corp", "ac")),
        SynonymEntry(("vd", "vector database"), oneway=True),
    ]
    assert aliases("ACME-corp", es) == ["ACME-corp", "acmecorp", "ac"]
    assert aliases("vd", es) == ["vd", "vector database"]
    assert aliases("vector database", es) == ["vector database"]  # one way only
    assert aliases("unknown", es) == ["unknown"]


def test_term_regex_matches_variants_and_guards_short_terms():
    rx = re.compile(term_regex(["acme corp", "ac"]), re.I)
    for s in ["Acme-Corp", "acme_corp", "acmecorp", "the AC deal"]:
        assert rx.search(s), s
    assert not rx.search("react")


def test_expand_query_original_first_and_capped():
    es = [SynonymEntry(("acmecorp", "acme corp", "ac"))]
    assert expand_query("acme corp invoice", es) == [
        "acme corp invoice",
        "acmecorp invoice",
        "ac invoice",
    ]
    assert expand_query("nothing known", es) == ["nothing known"]
    assert len(expand_query("acmecorp", es, max_variants=2)) == 2


def test_formulator_finds_a_doc_only_the_alias_names(ledger):
    docs = {
        "memo": "the zorblax venture memo and its invoice",
        "other": "a cake recipe with flour",
    }
    src = ir.CorpusSource.from_mapping(docs, name="f", strategy=ir.WholeText())
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    q = "quuxco venture"
    plain = ir.search(corpus, "quuxco", mode="lexical", k=2)
    assert all(h.artifact_id != "memo" for h in plain)
    add_synonyms(["quuxco", "zorblax"], note="renamed")
    hits = ir.search(
        corpus, q, mode="lexical", k=1, formulate=make_synonym_formulator()
    )
    assert hits[0].artifact_id == "memo"


def test_cli_synonyms_show_and_add(ledger):
    out = cli.synonyms_add("acmecorp, acme corp, ac", note="client")
    assert "added to" in out
    shown = cli.synonyms("acme corp")
    assert "acmecorp" in shown and r"\bac\b" in shown
    assert "acmecorp = acme corp = ac" in cli.synonyms()


def test_expand_flag_off_is_todays_behavior():
    assert cli._expand_kw(False, "x") == {}
    assert "formulate" in cli._expand_kw(True, "x")
