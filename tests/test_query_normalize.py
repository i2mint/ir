"""A corpus folds its queries the way its strategy folded the indexed text (#146).

``Fields(normalize=...)`` normalizes the text it indexes. Without the same
function on the query side, folded text (``creme brulee``) is never matched by
the query a person types (``Crème Brûlée``). :func:`ir.build` records the
strategy's normalizer on the corpus, and :func:`ir.retrieve.search` applies it
to every query, in every mode.
"""

import pytest

import ir
from ir.index import open_corpus
from ir.store import CorpusStore
from ir.strategy import Fields, fold_text, strategy_normalizer

DOCS = {
    "creme": {"name": "Crème Brûlée", "about": "a custard dessert"},
    "pho": {"name": "Phở Bò", "about": "a noodle soup"},
}


def _source(normalize, **kwargs):
    return ir.CorpusSource.from_mapping(
        DOCS,
        name="dishes",
        strategy=Fields(["about"], head=["name"], normalize=normalize, **kwargs),
    )


def _top(corpus, query, mode="lexical"):
    hits = ir.search(corpus, query, mode=mode, k=1)
    return hits[0].artifact_id if hits else None


def test_an_in_memory_corpus_folds_its_queries():
    corpus = ir.build(_source(fold_text), store=CorpusStore.memory(), embedder="light")
    assert corpus.query_normalize is fold_text
    assert _top(corpus, "CRÈME brûlée") == "creme"
    assert _top(corpus, "Phở") == "pho"


def test_a_corpus_without_a_normalizer_leaves_queries_alone():
    corpus = ir.build(_source(None), store=CorpusStore.memory(), embedder="light")
    assert corpus.query_normalize is None
    assert _top(corpus, "creme") is None  # the index kept its accents
    assert _top(corpus, "crème") == "creme"


def test_hybrid_and_dense_queries_are_folded_too():
    corpus = ir.build(_source(fold_text), store=CorpusStore.memory(), embedder="light")
    seen = []
    corpus.query_normalize = lambda q: seen.append(q) or fold_text(q)
    for mode in ("dense", "hybrid"):
        ir.search(corpus, "Crème", mode=mode, k=1)
    assert seen == ["Crème", "Crème"]


def test_a_reference_survives_reopening(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path))
    ir.build(_source("ir.strategy:fold_text"), embedder="light")
    reopened = open_corpus("dishes")
    assert reopened.query_normalize is fold_text
    assert _top(reopened, "CRÈME") == "creme"


def test_a_callable_cannot_be_reopened_so_reopening_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path))
    ir.build(_source(lambda t: fold_text(t), normalize_id="fold-v1"), embedder="light")
    with pytest.warns(UserWarning, match="fold-v1.*not folded"):
        reopened = open_corpus("dishes")
    assert reopened.query_normalize is None
    # ...and the caller can hand it back
    given = open_corpus("dishes", query_normalize=fold_text)
    assert _top(given, "CRÈME") == "creme"


def test_strategy_normalizer_looks_through_a_synopsis_wrapper():
    inner = Fields(["about"], normalize="ir.strategy:fold_text")
    wrapped = ir.with_synopsis(inner, synthesize=lambda raw: "summary")
    assert strategy_normalizer(wrapped) == "ir.strategy:fold_text"
    assert strategy_normalizer(ir.WholeText()) is None


def test_a_stale_reference_warns_instead_of_breaking_open(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path))
    ir.build(_source("ir.strategy:fold_text"), embedder="light")
    store = CorpusStore.local("dishes")
    store.set_config({**store.get_config(), "query_normalize": "gone_xyz:fold"})
    with pytest.warns(UserWarning, match="cannot be imported"):
        reopened = open_corpus("dishes")
    assert reopened.query_normalize is None
