"""Dependency-as-ranking-signal (#64): reverse-dependency walk + dep-evidence boost.

Two caller-seeded mechanisms, both structural and model-free:

- recall time: :meth:`ir.CorpusGraph.reverse_neighbors` / ``fan_in`` and the
  :func:`ir.reverse_dependency_policy` walk, which surfaces the corpus artifacts
  that depend on a caller-supplied set of libraries;
- ranking time: ``ir.search(..., dep_boost=...)`` with the shipped
  :func:`ir.dependency_evidence_boost`, which adds score to candidates whose
  ``deps`` overlap a caller-supplied library set.

``ir`` never decides which libraries are relevant to a goal: that is the
planner's call (raglab). Hermetic: light embedder, in-memory store.
"""

import math

import pytest

import ir
from ir.base import SearchHit
from ir.graph import REF, CorpusGraph
from ir.store import CorpusStore
from ir.traverse import traverse

PACKAGES = {
    "meshed": {
        "name": "meshed",
        "description": "Function composition into a DAG framework",
        "deps": ["i2"],
    },
    "allude": {
        "name": "allude",
        "description": "Allusions and references in text documents",
        "deps": ["meshed>=0.1", "dol"],
    },
    "unbox": {
        "name": "unbox",
        "description": "Unpack containers into their parts",
        "deps": ["Meshed", "networkx"],
    },
    "au": {
        "name": "au",
        "description": "Async task runner for DAG graph workflows",
        "deps": ["asyncio-extras"],
    },
    "cooking": {
        "name": "cooking",
        "description": "Bake bread and cakes in the oven",
        "deps": ["flour"],
    },
}
QUERY = "DAG graph workflows"


def _corpus():
    src = ir.CorpusSource.from_mapping(PACKAGES, name="pk", strategy=ir.Package())
    return ir.build(
        src,
        store=CorpusStore.memory(),
        embedder="light",
        edge_extractor=ir.default_edge_extractor,
    )


# --------------------------------------------------------------------------- #
# Reverse index on CorpusGraph
# --------------------------------------------------------------------------- #


def test_reverse_neighbors_and_fan_in():
    graph = CorpusGraph(_corpus())
    assert sorted(graph.reverse_neighbors("meshed")) == ["allude", "unbox"]
    assert graph.reverse_neighbors("networkx") == ["unbox"]  # a third-party target
    assert graph.reverse_neighbors("cooking") == []
    assert graph.fan_in("meshed") == 2
    assert graph.fan_in("cooking") == 0
    # Stored REF targets are lower-cased bare names; a seed's case does not matter.
    assert sorted(graph.reverse_neighbors("Meshed")) == ["allude", "unbox"]
    # Another edge type has its own (here empty) reverse index.
    assert graph.reverse_neighbors("meshed", edge_type="PARENT") == []


def test_reverse_index_sees_the_store_at_first_use():
    corpus = _corpus()
    graph = CorpusGraph(corpus)
    assert graph.fan_in("dol") == 1
    corpus.store.set_links("cooking", {REF: ["dol"]})
    assert graph.fan_in("dol") == 1  # snapshot: derived once per graph instance
    assert CorpusGraph(corpus).fan_in("dol") == 2  # a fresh graph sees the edit


# --------------------------------------------------------------------------- #
# reverse_dependency_policy (recall time)
# --------------------------------------------------------------------------- #


def _walk(seeds, **kw):
    corpus = _corpus()
    walk_kw = {k: kw.pop(k) for k in ("max_depth", "node_budget") if k in kw}
    policy = ir.reverse_dependency_policy(seeds=seeds, **kw)
    return traverse(QUERY, corpus, policy=policy, k=10, **walk_kw)


def test_walk_surfaces_the_dependents_of_the_seeds():
    hits = _walk(["meshed"])
    by_id = {h.artifact_id: h for h in hits}
    # The seed is in the corpus, so it is a result too; its dependents follow.
    assert set(by_id) == {"meshed", "allude", "unbox"}
    assert by_id["meshed"].metadata["walk_depth"] == 0
    assert by_id["allude"].metadata["walk_depth"] == 1
    assert by_id["allude"].metadata["seed"] == "meshed"
    assert by_id["meshed"].metadata["fan_in"] == 2
    # The prose-similar distractor does not depend on a seed, so it never shows.
    assert "au" not in by_id
    assert all(h.source == "pk" for h in hits)


def test_a_seed_outside_the_corpus_routes_but_is_not_a_result():
    assert [h.artifact_id for h in _walk(["networkx"])] == ["unbox"]
    assert _walk([]) == []
    assert _walk(["no-such-lib"]) == []


def test_max_depth_zero_keeps_only_the_seeds():
    assert [h.artifact_id for h in _walk(["meshed"], max_depth=0)] == ["meshed"]


def test_scores_are_cosine_plus_weighted_fan_in_and_proximity():
    plain = {h.artifact_id: h.score for h in _walk(["meshed"])}
    weighted = {
        h.artifact_id: h.score
        for h in _walk(["meshed"], fan_in_weight=1.0, proximity_weight=0.5)
    }
    # meshed: fan-in 2 is the max (normalized to 1.0); depth 0 -> proximity 1.
    assert weighted["meshed"] == pytest.approx(plain["meshed"] + 1.0 + 0.5)
    # allude: fan-in 0; depth 1 -> proximity 1/2.
    assert weighted["allude"] == pytest.approx(plain["allude"] + 0.25)


def test_walk_hits_fuse_with_flat_hits():
    corpus = _corpus()
    flat = ir.search(corpus, QUERY, k=5, mode="hybrid")
    walked = traverse(
        QUERY, corpus, policy=ir.reverse_dependency_policy(seeds=["meshed"]), k=5
    )
    fused = ir.fuse_hits({"flat": flat, "walk": walked})
    assert {"allude", "unbox"} <= {h.artifact_id for h in fused}


def test_policy_rejects_negative_weights():
    with pytest.raises(ValueError, match="fan_in_weight"):
        ir.reverse_dependency_policy(seeds=["x"], fan_in_weight=-1.0)


# --------------------------------------------------------------------------- #
# dependency_evidence_boost + search(dep_boost=) (ranking time)
# --------------------------------------------------------------------------- #


def _hit(deps):
    return SearchHit("a", "description", 0.0, "t", metadata={"deps": deps})


def test_boost_function_reads_bare_dep_names():
    boost = ir.dependency_evidence_boost({"meshed", "scikit_learn"}, weight=2.0)
    assert boost(_hit(["meshed>=0.1", "dol"])) == 2.0
    assert boost(_hit(["Scikit-Learn[extra]"])) == 2.0  # PEP 503-style match
    assert boost(_hit(["numpy"])) == 0.0
    assert boost(_hit([])) == 0.0
    assert boost(SearchHit("a", "d", 0.0, "t")) == 0.0  # no deps field at all
    halves = ir.dependency_evidence_boost({"meshed", "dol"}, weight=1.0, saturate=4)
    assert halves(_hit(["meshed", "dol", "numpy"])) == 0.5
    with pytest.raises(ValueError, match="weight"):
        ir.dependency_evidence_boost({"meshed"}, weight=math.inf)
    with pytest.raises(ValueError, match="saturate"):
        ir.dependency_evidence_boost({"meshed"}, weight=1.0, saturate=0)


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_dep_boost_adds_to_every_surface_score(mode):
    corpus = _corpus()
    boost = ir.dependency_evidence_boost({"meshed"}, weight=1.0)
    kw = dict(k=50, mode=mode, per_artifact=False)
    plain = {
        (h.artifact_id, h.surface_kind): h.score for h in ir.search(corpus, QUERY, **kw)
    }
    boosted = {
        (h.artifact_id, h.surface_kind): h.score
        for h in ir.search(corpus, QUERY, dep_boost=boost, **kw)
    }
    assert set(plain) == set(boosted)
    for (aid, kind), score in plain.items():
        extra = 1.0 if aid in {"allude", "unbox"} else 0.0
        assert boosted[(aid, kind)] == pytest.approx(score + extra)


def test_dep_boost_demotes_the_prose_distractor():
    corpus = _corpus()
    plain = [h.artifact_id for h in ir.search(corpus, QUERY, k=5, mode="hybrid")]
    boost = ir.dependency_evidence_boost({"meshed"}, weight=1.0)
    boosted = [
        h.artifact_id
        for h in ir.search(corpus, QUERY, k=5, mode="hybrid", dep_boost=boost)
    ]
    assert plain[0] == "au"
    assert boosted[:2] == ["allude", "unbox"] or boosted[:2] == ["unbox", "allude"]


def test_dep_boost_default_is_todays_behavior():
    corpus = _corpus()
    assert ir.search(corpus, QUERY, k=5, mode="hybrid", dep_boost=None) == ir.search(
        corpus, QUERY, k=5, mode="hybrid"
    )


def _prefer_last(query, passages):
    return [float(i) / 10 for i, _ in enumerate(passages)]


def test_dep_boost_applies_after_the_reranker():
    corpus = _corpus()
    boost = ir.dependency_evidence_boost({"meshed"}, weight=1.0)
    kw = dict(k=50, mode="hybrid", per_artifact=False, rerank=_prefer_last)
    reranked = {
        (h.artifact_id, h.surface_kind): h.score for h in ir.search(corpus, QUERY, **kw)
    }
    both = {
        (h.artifact_id, h.surface_kind): h.score
        for h in ir.search(corpus, QUERY, dep_boost=boost, **kw)
    }
    for (aid, kind), score in reranked.items():
        extra = 1.0 if aid in {"allude", "unbox"} else 0.0
        assert both[(aid, kind)] == pytest.approx(score + extra)


def test_dep_boost_applies_once_to_formulated_queries():
    corpus = _corpus()
    boost = ir.dependency_evidence_boost({"meshed"}, weight=1.0)

    def two(q):
        return [q, q + " runner"]

    plain = {
        h.artifact_id: h.score for h in ir.search(corpus, QUERY, k=5, formulate=two)
    }
    boosted = {
        h.artifact_id: h.score
        for h in ir.search(corpus, QUERY, k=5, formulate=two, dep_boost=boost)
    }
    for aid in {"allude", "unbox"} & set(plain):
        assert boosted[aid] == pytest.approx(plain[aid] + 1.0)


def test_dep_boost_must_return_a_finite_number():
    corpus = _corpus()
    with pytest.raises(ValueError, match="dep_boost"):
        ir.search(corpus, QUERY, dep_boost=lambda hit: math.nan)
