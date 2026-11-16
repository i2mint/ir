"""Tests for the query-formulation seam (ir.formulate / ir_09 §3).

Identity by default (search unchanged); a single rewrite redirects retrieval;
multi-query fan-out unions+fuses; the LLM formulator is injectable and falls back
to identity on any failure. Hermetic: the light embedder, no ``aix``.
"""

import pytest

import ir
from ir.formulate import (
    identity_formulator,
    make_keyword_formulator,
    make_llm_formulator,
)
from ir.store import CorpusStore

DOCS = {
    "deploy": "deploy the app to the server with systemd units",
    "bake": "bake a cake in the oven with flour and sugar",
    "vector": "vector similarity search with metadata filters",
}


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="f", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


# ----- the formulate= seam on search --------------------------------------- #


def test_no_formulate_equals_identity_equals_today():
    corpus = _corpus()
    base = [h.artifact_id for h in ir.search(corpus, "deploy the app")]
    ident = [
        h.artifact_id
        for h in ir.search(corpus, "deploy the app", formulate=identity_formulator)
    ]
    assert ident == base


def test_single_query_rewrite_redirects_retrieval():
    corpus = _corpus()
    # a rewriter that replaces the query entirely -> retrieval follows the rewrite
    hits = ir.search(
        corpus, "irrelevant words", formulate=lambda q: "bake a cake in the oven", k=1
    )
    assert hits[0].artifact_id == "bake"


def test_multi_query_fan_out_unions_results():
    corpus = _corpus()

    def formulate(q):
        return ["deploy the app to the server", "vector similarity search"]

    ids = {h.artifact_id for h in ir.search(corpus, "x", formulate=formulate, k=5)}
    assert {"deploy", "vector"} <= ids


def test_empty_formulation_falls_back_to_original_query():
    corpus = _corpus()
    base = [h.artifact_id for h in ir.search(corpus, "deploy the app")]
    hits = [
        h.artifact_id
        for h in ir.search(corpus, "deploy the app", formulate=lambda q: [])
    ]
    assert hits == base


def test_discover_threads_formulate_via_search_kw():
    corpus = _corpus()
    res = ir.discover(
        corpus,
        "x",
        k=3,
        strategy="top_k",
        formulate=lambda q: "bake a cake in the oven",
    )
    assert any(d.name == "bake" for d in res.results)


# ----- make_llm_formulator (injectable, identity fallback) ------------------ #


def test_make_llm_formulator_uses_injected_rewriter():
    f = make_llm_formulator(rewriter=lambda q: [q, q + " server"])
    assert f("deploy") == ["deploy", "deploy server"]


def test_make_llm_formulator_falls_back_to_identity_on_error():
    def boom(q):
        raise RuntimeError("no LLM available")

    assert make_llm_formulator(rewriter=boom)("deploy") == "deploy"


def test_make_llm_formulator_falls_back_on_empty_reply():
    assert make_llm_formulator(rewriter=lambda q: [])("deploy") == "deploy"


def test_make_llm_formulator_with_injected_rewriter_needs_no_aix():
    # The injected-rewriter path must not touch the lazy aix builder; a single-str
    # rewrite is normalized to a one-element list (a valid Formulator output).
    f = make_llm_formulator(rewriter=lambda q: q)
    assert f("anything") == ["anything"]


# ----- variant_weight / keyword formulator (#136) -------------------------- #


def _two_query(q):
    return [q, "bake a cake in the oven"]


def test_variant_weight_default_is_todays_merge():
    corpus = _corpus()
    kw = dict(formulate=_two_query, mode="lexical", per_artifact=False, k=6)
    base = ir.search(corpus, "deploy the app", **kw)
    same = ir.search(corpus, "deploy the app", variant_weight=1.0, **kw)
    assert [(h.artifact_id, h.score) for h in base] == [
        (h.artifact_id, h.score) for h in same
    ]


def test_variant_weight_demotes_formulations_not_the_original():
    corpus = _corpus()
    kw = dict(formulate=_two_query, mode="lexical", k=6)
    full = {h.artifact_id: h.score for h in ir.search(corpus, "deploy the app", **kw)}
    low = {
        h.artifact_id: h.score
        for h in ir.search(corpus, "deploy the app", variant_weight=0.5, **kw)
    }
    assert low["deploy"] == full["deploy"]  # the original query's hit is untouched
    assert low["bake"] == pytest.approx(full["bake"] * 0.5)  # the variant's is halved


def test_variant_weight_zero_keeps_only_the_original_ranking_on_top():
    corpus = _corpus()
    hits = ir.search(
        corpus,
        "deploy the app",
        formulate=_two_query,
        mode="lexical",
        variant_weight=0.0,
        k=1,
    )
    assert hits[0].artifact_id == "deploy"


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_variant_weight_validated(bad):
    with pytest.raises(ValueError, match="variant_weight"):
        ir.search(_corpus(), "x", formulate=_two_query, variant_weight=bad)


def test_keyword_formulator_keeps_original_first_and_dedupes():
    f = make_keyword_formulator(rewriter=lambda q: ["a b", q, "c"])
    assert f("orig") == ["orig", "a b", "c"]
    assert make_keyword_formulator(rewriter=lambda q: ["a"], include_original=False)(
        "orig"
    ) == ["a"]


def test_keyword_formulator_falls_back_to_the_query_on_failure():
    def boom(q):
        raise RuntimeError("no llm")

    assert make_keyword_formulator(rewriter=boom)("orig") == ["orig"]


def test_keyword_formulator_uses_the_keyword_prompt_by_default():
    from ir.formulate import KEYWORD_FORMULATION_PROMPT

    assert "BM25" in KEYWORD_FORMULATION_PROMPT
    assert (
        "{query}" in KEYWORD_FORMULATION_PROMPT and "{n}" in KEYWORD_FORMULATION_PROMPT
    )
