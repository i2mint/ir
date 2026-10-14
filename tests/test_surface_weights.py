"""Per-surface-kind weighting at fusion (#68): ``search(surface_weights=...)``.

A weight multiplies a surface-hit's first-stage score (the fused score in hybrid
mode) before the fetch cut, the reranker and the per-artifact collapse, so a
high-precision surface such as ``deps`` can outrank prose. ``None`` (the
default) is today's equal weighting. Hermetic: light embedder, in-memory store.
"""

import pytest

import ir
from ir import retrieve
from ir.store import CorpusStore

PACKAGES = {
    "allude": {
        "name": "allude",
        "description": "Allusions and references in text documents",
        "deps": ["meshed", "dol"],
    },
    "grapher": {
        "name": "grapher",
        "description": "Draw a meshed graph of nodes and edges for meshed networks",
        "deps": ["numpy"],
    },
    "other": {
        "name": "other",
        "description": "Bake bread and cakes in the oven",
        "deps": ["flour"],
    },
}
QUERY = "meshed graph networks"


def _corpus():
    src = ir.CorpusSource.from_mapping(
        PACKAGES, name="pk", strategy=ir.Package(embed_deps=True)
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _surface_scores(corpus, **kw):
    hits = ir.search(corpus, QUERY, k=50, per_artifact=False, **kw)
    return {(h.artifact_id, h.surface_kind): h.score for h in hits}


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
@pytest.mark.parametrize("fusion", ["rrf", "blend"])
def test_default_and_empty_weights_are_todays_behavior(mode, fusion):
    corpus = _corpus()
    base = ir.search(corpus, QUERY, k=10, mode=mode, fusion=fusion)
    for weights in (None, {}, {"no_such_kind": 5.0}):
        assert (
            ir.search(
                corpus, QUERY, k=10, mode=mode, fusion=fusion, surface_weights=weights
            )
            == base
        )


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
@pytest.mark.parametrize("fusion", ["rrf", "blend"])
def test_weight_multiplies_only_that_kinds_scores(mode, fusion):
    corpus = _corpus()
    plain = _surface_scores(corpus, mode=mode, fusion=fusion)
    weighted = _surface_scores(
        corpus, mode=mode, fusion=fusion, surface_weights={"deps": 3.0}
    )
    assert set(plain) == set(weighted)
    assert any(kind == "deps" for _, kind in plain)
    for key, score in plain.items():
        factor = 3.0 if key[1] == "deps" else 1.0
        assert weighted[key] == pytest.approx(score * factor)


def test_deps_weight_lifts_the_dependent_over_prose_in_hybrid():
    corpus = _corpus()
    plain = [h.artifact_id for h in ir.search(corpus, QUERY, k=3, mode="hybrid")]
    boosted = [
        h.artifact_id
        for h in ir.search(
            corpus, QUERY, k=3, mode="hybrid", surface_weights={"deps": 3.0}
        )
    ]
    assert plain[0] == "grapher"  # its prose matches every query term
    assert boosted[0] == "allude"  # depends on meshed: the deps surface now wins


def test_weighting_happens_before_the_fetch_cut(monkeypatch):
    """A weighted surface fused just below the cut still gets its chance."""
    corpus = _corpus()
    ids, metas = corpus.store.metas()
    rid = {(m["artifact_id"], m["surface_kind"]): i for i, m in zip(ids, metas)}
    x = rid[("grapher", "description")]
    y = rid[("other", "description")]
    z = rid[("allude", "description")]
    w = rid[("allude", "deps")]
    # Dense top-2 = x, y; lexical top-2 = z, w. RRF over the union ranks w last.
    monkeypatch.setattr(retrieve, "_dense_ranked", lambda *a, **k: [(x, 0.9), (y, 0.8)])
    monkeypatch.setattr(
        retrieve, "_lexical_ranked", lambda *a, **k: [(z, 5.0), (w, 4.0)]
    )
    kw = dict(k=2, mode="hybrid", per_artifact=False, fetch_k=2)
    plain = ir.search(corpus, QUERY, **kw)
    assert w not in {rid[(h.artifact_id, h.surface_kind)] for h in plain}
    boosted = ir.search(corpus, QUERY, surface_weights={"deps": 10.0}, **kw)
    assert (boosted[0].artifact_id, boosted[0].surface_kind) == ("allude", "deps")


def test_weights_reach_formulated_subsearches():
    corpus = _corpus()

    def two_queries(q):
        return [q, q + " graph"]

    plain = ir.search(corpus, QUERY, k=3, mode="hybrid", formulate=two_queries)
    boosted = ir.search(
        corpus,
        QUERY,
        k=3,
        mode="hybrid",
        formulate=two_queries,
        surface_weights={"deps": 3.0},
    )
    assert plain != boosted


@pytest.mark.parametrize("bad", [{"deps": -1.0}, {"deps": float("nan")}, {"deps": "2"}])
def test_invalid_weights_raise(bad):
    corpus = _corpus()
    with pytest.raises(ValueError, match="surface_weights"):
        ir.search(corpus, QUERY, mode="hybrid", surface_weights=bad)


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_weights_act_at_small_k_without_collapse(mode):
    """Weights over-fetch: a boosted surface beyond k still wins (review finding)."""
    corpus = _corpus()
    top = ir.search(
        corpus,
        QUERY,
        k=1,
        mode=mode,
        per_artifact=False,
        surface_weights={"deps": 100.0},
    )[0]
    assert (top.artifact_id, top.surface_kind) == ("allude", "deps")


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_weights_act_at_small_k_through_formulate(mode):
    corpus = _corpus()
    top = ir.search(
        corpus,
        QUERY,
        k=1,
        mode=mode,
        formulate=lambda q: [q, q + " edges"],
        surface_weights={"deps": 100.0},
    )[0]
    assert top.artifact_id == "allude"


def test_numpy_weights_are_accepted():
    import numpy as np

    corpus = _corpus()
    as_float = ir.search(corpus, QUERY, mode="hybrid", surface_weights={"deps": 2.0})
    as_numpy = ir.search(
        corpus, QUERY, mode="hybrid", surface_weights={"deps": np.float32(2.0)}
    )
    assert [h.artifact_id for h in as_float] == [h.artifact_id for h in as_numpy]


def test_discover_auto_floor_is_skipped_for_weighted_scores(recwarn):
    corpus = _corpus()
    corpus.store.set_calibration(
        "hybrid", {"min_score": 99.0, "embedder_id": corpus.embedder_id}
    )
    plain = ir.discover(corpus, QUERY, mode="hybrid", min_score="auto")
    assert plain.abstained  # the (absurd) calibrated floor applies to plain scores
    weighted = ir.discover(
        corpus,
        QUERY,
        mode="hybrid",
        min_score="auto",
        surface_weights={"deps": 2.0},
    )
    assert not weighted.abstained
    assert any("rescale the scores" in str(w.message) for w in recwarn)


def test_a_rescaled_calibration_cannot_overwrite_the_plain_floor():
    from ir import eval as ev

    corpus = _corpus()
    cases = [
        ev.DiscoveryCase(query=QUERY, gold=("allude",)),
        ev.DiscoveryCase(query="bake bread", gold=()),
    ]
    with pytest.raises(ValueError, match="cannot persist"):
        ev.calibrate_min_score(
            corpus, cases, mode="hybrid", persist=True, surface_weights={"deps": 2.0}
        )
    assert corpus.store.calibration_modes() == []
    # Without persist it still reports a floor for the caller to use.
    ev.calibrate_min_score(corpus, cases, mode="hybrid", surface_weights={"deps": 2.0})
