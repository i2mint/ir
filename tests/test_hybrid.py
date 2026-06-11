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


# --------------------------------------------------------------------------- #
# Magnitude-preserving "blend" fusion (ir_08) — opt-in alternative to RRF
# --------------------------------------------------------------------------- #


def test_blend_fuse_unit_math():
    from ir.retrieve import _blend_fuse

    dense = [("a", 0.8), ("b", 0.2)]
    lexical = [("a", 0.0), ("b", 8.0)]  # b's bm25 saturates: 8/(8+8) = 0.5
    out = dict(_blend_fuse(dense, lexical, alpha=0.5, bm25_sat_k=8.0, fetch=10))
    assert out["a"] == pytest.approx(0.5 * 0.8 + 0.5 * 0.0)  # 0.40
    assert out["b"] == pytest.approx(0.5 * 0.2 + 0.5 * 0.5)  # 0.35


def test_blend_fuse_falls_back_to_single_side():
    from ir.retrieve import _blend_fuse

    dense = [("a", 0.9), ("b", 0.1)]
    assert _blend_fuse(dense, [], 0.5, 8.0, 10) == dense  # no lexical -> dense
    assert _blend_fuse([], dense, 0.5, 8.0, 10) == dense  # no dense -> lexical


def test_blend_preserves_magnitude_vs_rrf():
    # The core property: RRF collapses the top hit to the ~1/(k+1) rank scale,
    # while blend keeps the dense/lexical magnitude — what abstention needs.
    corpus = _corpus()
    rrf_top = ir.search(corpus, QUERY, mode="hybrid", fusion="rrf")[0].score
    blend_top = ir.search(corpus, QUERY, mode="hybrid", fusion="blend")[0].score
    assert rrf_top < 0.1
    assert blend_top > 0.3


def test_blend_does_not_push_rare_identifier_below_dense():
    # Blend is magnitude-preserving, NOT rank-boosting: unlike RRF it does not
    # strongly lift a dense-weak / lexical-strong rare identifier (that is RRF's
    # strength — the recall/abstention tradeoff documented in ir_08). It must at
    # least not push it *below* its pure-dense rank.
    corpus = _corpus()
    blend_order = [
        h.artifact_id
        for h in ir.search(corpus, QUERY, k=10, mode="hybrid", fusion="blend")
    ]
    dense_order = _order(corpus, mode="dense")
    assert blend_order.index("svc") <= dense_order.index("svc")


def test_blend_falls_back_to_dense_without_vd(monkeypatch):
    import sys

    corpus = _corpus()
    monkeypatch.setitem(sys.modules, "vd", None)
    with pytest.warns(UserWarning):
        blend = [
            h.artifact_id
            for h in ir.search(corpus, QUERY, mode="hybrid", fusion="blend")
        ]
    assert blend == _order(corpus, mode="dense")


def test_unknown_fusion_raises():
    corpus = _corpus()
    with pytest.raises(ValueError, match="unknown fusion"):
        ir.search(corpus, QUERY, mode="hybrid", fusion="bogus")
