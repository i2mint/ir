"""Documents outside the caller's access change nothing a caller can see (i2mint/ir#169).

Filtering happens before the BM25 statistics are built, so the ids, the scores and the
number of hits of a filtered search are identical whether or not the hidden documents
are in the corpus at all. Words that appear only in hidden documents leak nothing
through the IDF either.
"""

import pytest

import ir
from ir.store import CorpusStore

ALLOWED = {"a": "alpha apple pie", "b": "alpha banana split", "c": "cherry tart"}
HIDDEN = {
    "x": "alpha alpha alpha secret",
    "y": "alpha secret secret",
    "z": "apple apple secret",
}
ACCESS = {"acl": "ok"}


def _build(docs, meta):
    src = ir.CorpusSource.from_mapping(
        docs, name="acl", strategy=ir.WholeText(), metadata_of=lambda i, raw: meta[i]
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


@pytest.fixture(scope="module")
def corpora():
    meta = {k: {"acl": "ok"} for k in ALLOWED}
    without = _build(ALLOWED, meta)
    with_hidden = _build(
        {**ALLOWED, **HIDDEN}, {**meta, **{k: {"acl": "no"} for k in HIDDEN}}
    )
    return without, with_hidden


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
@pytest.mark.parametrize("query", ["alpha", "apple secret", "secret"])
def test_hidden_documents_change_neither_ids_scores_nor_count(corpora, mode, query):
    without, with_hidden = corpora
    expected = without.search(query, mode=mode, filter=ACCESS)
    got = with_hidden.search(query, mode=mode, filter=ACCESS)
    assert [(h.artifact_id, h.score) for h in got] == [
        (h.artifact_id, h.score) for h in expected
    ]
    assert len(got) == len(expected)
    assert not {h.artifact_id for h in got} & set(HIDDEN)


def test_a_query_matching_only_hidden_words_returns_nothing(corpora):
    _, with_hidden = corpora
    assert with_hidden.search("secret", mode="lexical", filter=ACCESS) == []
