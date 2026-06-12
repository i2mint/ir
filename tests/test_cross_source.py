"""Cross-source fusion tests (ir#34's federated half + the fan-in primitive).

Covers `SearchHit.source` provenance (stamped by `search`, carried through
`to_dict`), the `(source, artifact_id)` identity in `best_per_artifact` (the
skills-corpus "dol" is not the packages-corpus "dol"), `ir.fuse_hits` (weighted
N-ary RRF — only ranks cross the source boundary), and federated
`ir.discover([names], query)` (fan-out → per-source gate → fuse → select →
disclose). Hermetic: light embedder, in-memory stores.
"""

import dataclasses
import json

import pytest

import ir
from ir import SearchHit, fuse_hits, tag_source
from ir.base import best_per_artifact
from ir.store import CorpusStore


def _hit(artifact_id, score, *, source=None, text="t", metadata=None):
    return SearchHit(
        artifact_id=artifact_id,
        surface_kind="description",
        score=score,
        text=text,
        metadata=metadata or {},
        source=source,
    )


def _corpus(docs, name):
    src = ir.CorpusSource.from_mapping(docs, name=name, strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


# ----- SearchHit.source provenance ----------------------------------------- #


def test_searchhit_source_defaults_none_and_serializes():
    hit = SearchHit("a", "description", 1.0, "snippet", {})
    assert hit.source is None  # trailing default: existing constructors unchanged
    d = hit.to_dict()
    assert d["source"] is None
    assert json.dumps(d)  # stays JSON-clean


def test_search_stamps_corpus_name_as_source():
    corpus = _corpus({"alpha": "alpha apple avocado"}, name="prov")
    hits = ir.search(corpus, "alpha apple")
    assert hits and all(h.source == "prov" for h in hits)


def test_tag_source_stamps_only_untagged():
    hits = [_hit("a", 1.0), _hit("b", 0.5, source="original")]
    tagged = tag_source(hits, "new")
    assert tagged[0].source == "new"
    assert tagged[1].source == "original"  # existing tags win


# ----- best_per_artifact: (source, artifact_id) identity ------------------- #


def test_best_per_artifact_keeps_colliding_ids_across_sources():
    hits = [_hit("dol", 0.9, source="skills"), _hit("dol", 0.5, source="packages")]
    survivors = best_per_artifact(hits)
    assert len(survivors) == 2  # same id, different corpus = different artifact


def test_best_per_artifact_still_collapses_within_one_source():
    hits = [_hit("dol", 0.5, source="skills"), _hit("dol", 0.9, source="skills")]
    survivors = best_per_artifact(hits)
    assert len(survivors) == 1
    assert survivors[0].score == 0.9  # keeps the max-score surface


# ----- fuse_hits ------------------------------------------------------------ #


def test_fuse_hits_empty_input():
    assert fuse_hits({}) == []
    assert fuse_hits({"a": [], "b": []}) == []


def test_fuse_hits_single_source_passes_raw_scores_through():
    hits = [_hit("x", 0.9), _hit("y", 0.4)]
    fused = fuse_hits({"only": hits})
    assert [h.artifact_id for h in fused] == ["x", "y"]
    assert [h.score for h in fused] == [0.9, 0.4]  # no RRF rescaling for one list
    assert all(h.source == "only" for h in fused)


def test_fuse_hits_interleaves_by_rank_not_score():
    # Source "cosine" scores ~[0,1]; source "bm25" scores ~[0,50]: raw-score
    # ordering would let bm25 bury cosine entirely — rank fusion must not.
    cosine = [_hit("c1", 0.92), _hit("c2", 0.85)]
    bm25 = [_hit("b1", 31.0), _hit("b2", 24.0)]
    fused = fuse_hits({"cosine": cosine, "bm25": bm25})
    assert {h.artifact_id for h in fused[:2]} == {"c1", "b1"}  # both rank-1s lead
    assert {h.artifact_id for h in fused[2:]} == {"c2", "b2"}


def test_fuse_hits_rank_ties_break_by_caller_source_order():
    fused = fuse_hits({"first": [_hit("f", 0.1)], "second": [_hit("s", 99.0)]})
    # Equal rank, equal weight: the caller's order is the priority order —
    # never the (incomparable) raw scores.
    assert [h.artifact_id for h in fused] == ["f", "s"]


def test_fuse_hits_colliding_ids_stay_distinct_results():
    fused = fuse_hits({"skills": [_hit("dol", 0.9)], "packages": [_hit("dol", 28.0)]})
    assert len(fused) == 2
    assert {h.source for h in fused} == {"skills", "packages"}


def test_fuse_hits_weights_bias_the_merge():
    a = [_hit("a1", 0.9), _hit("a2", 0.8)]
    b = [_hit("b1", 0.9), _hit("b2", 0.8)]
    fused = fuse_hits({"a": a, "b": b}, weights={"b": 2.0})
    assert fused[0].artifact_id == "b1"  # 2/(60+1) beats 1/(60+1)


def test_fuse_hits_within_source_duplicates_collapse_before_ranking():
    # The same artifact twice in one source (multi-round / multi-query pool)
    # must not double its RRF mass: it collapses to its best rank first.
    dup = [_hit("x", 0.9), _hit("x", 0.7), _hit("y", 0.8)]
    other = [_hit("z", 0.9)]
    fused = fuse_hits({"a": dup, "b": other})
    by_id = {h.artifact_id: h for h in fused}
    assert len([h for h in fused if h.artifact_id == "x"]) == 1
    assert by_id["x"].score == pytest.approx(1 / 61)  # rank-1 mass only, not 1/61+1/62
    assert by_id["x"].metadata["source_rank"] == 1


def test_fuse_hits_preserves_prefusion_magnitudes_in_metadata():
    fused = fuse_hits({"a": [_hit("x", 0.9)], "b": [_hit("y", 30.0)]})
    for h in fused:
        assert isinstance(h.metadata["source_score"], float)
        assert isinstance(h.metadata["source_rank"], int)
    assert {h.metadata["source_score"] for h in fused} == {0.9, 30.0}


def test_fuse_hits_output_is_best_first():
    fused = fuse_hits({"a": [_hit("a1", 0.9), _hit("a2", 0.5)], "b": [_hit("b1", 9.0)]})
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)  # ir.select's precondition


def test_fuse_hits_identity_merges_cross_source_duplicates():
    a = [_hit("doc_a", 0.9, metadata={"path": "/shared/readme"})]
    b = [_hit("doc_b", 22.0, metadata={"path": "/shared/readme"}), _hit("solo", 21.0)]
    fused = fuse_hits({"a": a, "b": b}, identity="pointer")
    merged = [h for h in fused if h.metadata.get("fused_sources")]
    assert len(merged) == 1
    assert merged[0].metadata["fused_sources"] == ["a", "b"]
    # Two rank-1 memberships sum: the merged hit outranks the rank-2 singleton
    # and either single rank-1 contribution.
    assert fused[0] is merged[0]
    assert merged[0].score == pytest.approx(1 / 61 + 1 / 61)


def test_fuse_hits_truncates_to_k():
    a = [_hit(f"a{i}", 1.0 - i / 10) for i in range(5)]
    b = [_hit(f"b{i}", 50.0 - i) for i in range(5)]
    assert len(fuse_hits({"a": a, "b": b}, k=3)) == 3


def test_fuse_hits_matches_vd_rrf_scores():
    vd = pytest.importorskip("vd")
    a = [_hit("x", 0.9), _hit("y", 0.8)]
    b = [_hit("y", 30.0), _hit("z", 20.0)]
    # Merge cross-source by artifact_id to mirror vd's flat-id fusion.
    fused = fuse_hits({"a": a, "b": b}, identity=lambda h: h.artifact_id)
    expected = vd.reciprocal_rank_fusion(
        [
            [{"id": "x", "score": 0.9}, {"id": "y", "score": 0.8}],
            [{"id": "y", "score": 30.0}, {"id": "z", "score": 20.0}],
        ],
        k=60,
    )
    assert {h.artifact_id: pytest.approx(h.score) for h in fused} == {
        e["id"]: e["rrf_score"] for e in expected
    }


def test_fuse_hits_rejects_unknown_identity():
    with pytest.raises(ValueError):
        fuse_hits({"a": [_hit("x", 1.0)]}, identity="bogus")


# ----- federated discover --------------------------------------------------- #


DOCS_ONE = {
    "alpha": "alpha apple avocado almond",
    "shared": "zebra zephyr zucchini",
}
DOCS_TWO = {
    "beta": "beta banana blueberry",
    "shared": "zebra zephyr zucchini",
}


@pytest.fixture()
def corpora():
    return _corpus(DOCS_ONE, "one"), _corpus(DOCS_TWO, "two")


def test_federation_of_one_equals_single_corpus(corpora):
    one, _ = corpora
    single = ir.discover(one, "alpha apple")
    federated = ir.discover([one], "alpha apple")
    assert federated.ids == single.ids
    assert [d.score for d in federated.results] == [d.score for d in single.results]


def test_federated_discover_merges_and_attributes_sources(corpora):
    one, two = corpora
    result = ir.discover([one, two], "zebra zephyr zucchini", max_k=10, rel=0.0)
    assert result.signals["merge"] == "rrf"
    assert result.signals["sources"] == ["one", "two"]
    assert set(result.signals["per_source"]) == {"one", "two"}
    sources = {(d.artifact_id, d.source) for d in result.results}
    # The colliding "shared" id survives from BOTH corpora, distinguishable.
    assert {("shared", "one"), ("shared", "two")} <= sources


def test_federated_discover_result_is_json_clean(corpora):
    one, two = corpora
    result = ir.discover([one, two], "alpha apple")
    assert json.loads(json.dumps(result.to_dict()))  # round-trips


def test_federated_discover_rejects_bare_float_floor(corpora):
    one, two = corpora
    with pytest.raises(ValueError, match="per-\\(corpus, mode, embedder\\)"):
        ir.discover([one, two], "alpha", min_score=0.5)


def test_federated_discover_floor_mapping_gates_per_source(corpora):
    one, two = corpora
    result = ir.discover([one, two], "alpha apple", min_score={"one": 1e9, "two": 1e9})
    assert result.abstained
    assert result.reason == "abstain:all_sources_below_floor"
    assert all(s["abstained"] for s in result.signals["per_source"].values())


def test_federated_discover_floor_mapping_rejects_unknown_names(corpora):
    one, two = corpora
    with pytest.raises(ValueError, match="not in this discover call"):
        ir.discover([one, two], "alpha", min_score={"nonesuch": 0.5})


def test_federated_discover_rejects_empty_and_duplicate_sources(corpora):
    one, _ = corpora
    with pytest.raises(ValueError, match="at least one corpus"):
        ir.discover([], "alpha")
    with pytest.raises(ValueError, match="duplicate"):
        ir.discover([one, one], "alpha")


def test_federated_merge_score_requires_shared_embedder(corpora):
    one, two = corpora
    assert ir.discover([one, two], "alpha apple", merge="score").ids is not None
    impostor = dataclasses.replace(two, embedder_id="other-embedder")
    with pytest.raises(ValueError, match="shared embedder"):
        ir.discover([one, impostor], "alpha apple", merge="score")


def test_federated_merge_accepts_a_callable(corpora):
    one, two = corpora
    calls = {}

    def merge(surviving):
        calls["sources"] = sorted(surviving)
        return [h for hits in surviving.values() for h in hits]

    result = ir.discover([one, two], "alpha apple", merge=merge)
    assert calls["sources"] == ["one", "two"]
    assert result.signals["merge"] == "custom"


def test_federated_merge_rejects_unknown_method(corpora):
    one, two = corpora
    with pytest.raises(ValueError, match="unknown merge"):
        ir.discover([one, two], "alpha", merge="bogus")


def test_single_corpus_discover_rejects_federated_only_params(corpora):
    one, _ = corpora
    with pytest.raises(ValueError, match="federated form"):
        ir.discover(one, "alpha", merge_weights={"one": 2.0})
    with pytest.raises(ValueError, match="federated form"):
        ir.discover(one, "alpha", min_score={"one": 0.5})
