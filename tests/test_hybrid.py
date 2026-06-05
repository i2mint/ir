"""Hybrid retrieval tests — dense + BM25 fused via RRF, plus the rerank hook.

Hermetic: the light (numpy-only) hashing embedder, no model download, no
network. The core claim under test is that lexical (BM25) signal recovers an
*exact rare identifier* that dense-only retrieval misses, and that hybrid
fusion lifts that doc's rank relative to dense.
"""

import pytest

import ir
from ir.store import CorpusStore

# `svc` is the only doc carrying the rare identifier ``zxqv``; the d* docs flood
# the corpus with the common query terms (deploy/systemd/unit), so their IDF is
# low and the frequency-weighted hashing embedder is distracted by d3's repeats.
DOCS = {
    "svc": "zxqv service config",
    "d1": "deploy systemd unit alpha",
    "d2": "deploy systemd unit beta",
    "d3": "deploy systemd unit gamma deploy systemd unit",
    "d4": "deploy systemd unit delta",
}
QUERY = "deploy systemd unit zxqv"


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="hyb", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _order(corpus, **kw):
    return [h.artifact_id for h in ir.search(corpus, QUERY, k=10, **kw)]


def test_lexical_finds_rare_identifier():
    # BM25 ranks the rare-identifier doc first; dense buries it.
    corpus = _corpus()
    assert _order(corpus, mode="lexical")[0] == "svc"
    assert _order(corpus, mode="dense")[0] != "svc"


def test_hybrid_lifts_what_dense_misses():
    corpus = _corpus()
    dense_order = _order(corpus, mode="dense")
    hybrid_order = _order(corpus, mode="hybrid")
    # Hybrid recovers the lexically-strong doc: its rank improves vs dense.
    assert hybrid_order.index("svc") < dense_order.index("svc")


def test_dense_is_the_default():
    # Omitting mode must equal mode="dense" (backward compatibility).
    corpus = _corpus()
    assert _order(corpus) == _order(corpus, mode="dense")


def test_unknown_mode_raises():
    corpus = _corpus()
    with pytest.raises(ValueError):
        ir.search(corpus, QUERY, mode="bogus")


def test_rerank_hook_reorders():
    # A reranker that prefers a specific token must reorder the results.
    docs = {"a": "red apple fruit", "b": "blue sky cloud", "c": "green grass field"}
    src = ir.CorpusSource.from_mapping(docs, name="rr", strategy=ir.WholeText())
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")

    def prefer_grass(query, segments):
        return [10.0 if "grass" in s["text"] else 0.0 for s in segments]

    reranked = [
        h.artifact_id for h in ir.search(corpus, "apple", k=3, rerank=prefer_grass)
    ]
    assert reranked[0] == "c"


def test_bm25_params_forwarded():
    # Passing Okapi params must not break lexical search (smoke test).
    corpus = _corpus()
    hits = ir.search(corpus, QUERY, mode="lexical", bm25={"k1": 1.2, "b": 0.5})
    assert hits and hits[0].artifact_id == "svc"


def _prefer_svc(query, segments):
    # Reranker that promotes the doc whose text mentions "service".
    return [10.0 if "service" in s["text"] else 0.0 for s in segments]


def test_rerank_applies_across_modes():
    # The rerank hook must run after ranking in every mode, not just dense.
    corpus = _corpus()
    for mode in ("dense", "lexical", "hybrid"):
        top = ir.search(corpus, QUERY, k=5, mode=mode, rerank=_prefer_svc)[0]
        assert top.artifact_id == "svc", f"rerank did not take effect for {mode=}"


def test_rerank_with_per_artifact_false():
    # rerank must also work when surfaces are not collapsed per artifact.
    corpus = _corpus()
    hits = ir.search(
        corpus, QUERY, k=5, mode="hybrid", per_artifact=False, rerank=_prefer_svc
    )
    assert hits and hits[0].artifact_id == "svc"
