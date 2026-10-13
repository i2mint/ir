"""The precision stage (ir#65): ``rerank=`` end to end on a Package corpus.

Hermetic: the light embedder, memory stores, and a *stub* ``sentence_transformers`` for the
cross-encoder path -- no torch, no model download. The stub sits at the real boundary
(``sentence_transformers.CrossEncoder``), so ef's actual ``cross_encoder_reranker`` runs.
"""

import sys
import types
import warnings

import pytest

import ir
import ir.eval as ev
from ir import rerank as rr
from ir.store import CorpusStore

README = (
    "The thing is a general toolkit. " * 6
    + "It schedules directed acyclic graph pipelines for audio processing. "
    + "More filler about nothing in particular. " * 6
)

PKG = {
    # thin description: only the README says what it is about
    "thin": {"name": "thin", "description": "a toolkit", "readme": README},
    # a distractor whose short description shares jargon
    "jargon": {
        "name": "jargon",
        "description": "async task framework with a single DAG bullet",
        "readme": "Runs queued jobs. " * 20,
    },
    "other": {
        "name": "other",
        "description": "pdf tools",
        "readme": "Merge pdf. " * 20,
    },
}
QUERY = "audio directed acyclic graph pipelines"


def _corpus(name="rp"):
    src = ir.CorpusSource.from_mapping(
        PKG, name=name, strategy=ir.Package(chunk_size=150, overlap=30)
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


class FakeCrossEncoder:
    """A sentence_transformers.CrossEncoder stand-in: scores topical word overlap."""

    instances: list = []

    def __init__(self, model_name, device=None, **kwargs):
        self.model_name = model_name
        self.pairs_seen: list = []
        FakeCrossEncoder.instances.append(self)

    def predict(self, pairs, batch_size=32, **kwargs):
        self.pairs_seen.extend(pairs)
        out = []
        for query, passage in pairs:
            hit = len(set(query.lower().split()) & set(passage.lower().split()))
            out.append(2.0 * hit - 3.0)  # raw logits: negative when off-topic
        return out


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(rr, "_model_cache", {})
    FakeCrossEncoder.instances = []
    return module


@pytest.fixture
def no_st(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import raises
    monkeypatch.setattr(rr, "_model_cache", {})
    monkeypatch.setattr(rr, "_warned", set())


def _ids(hits):
    return [h.artifact_id for h in hits]


def _seen_passages(reranker):
    seen = []

    def spy(query, segments):
        seen.extend(s["text"] for s in segments)
        return reranker(query, segments)

    return spy, seen


# ------------------------------------------------------------------ passage text


def test_default_rerank_text_is_the_matched_surface_as_before():
    corpus = _corpus()
    spy, seen = _seen_passages(lambda q, segs: [0.0] * len(segs))
    hits = ir.search(corpus, QUERY, k=3, mode="hybrid", rerank=spy)
    assert hits and seen
    assert all(len(t) < 400 for t in seen)  # single surfaces, not whole artifacts


def test_artifact_text_feeds_the_whole_readme_and_scores_each_artifact_once():
    corpus = _corpus()
    spy, seen = _seen_passages(lambda q, segs: [0.0] * len(segs))
    ir.search(corpus, QUERY, k=3, mode="hybrid", rerank=spy, rerank_text="artifact")
    assert any("directed acyclic graph" in t and "a toolkit" in t for t in seen)
    assert len(seen) == len(set(seen)) == len(PKG)  # one passage per artifact


def test_artifact_text_scores_each_passage_once_even_without_collapse():
    corpus = _corpus()
    spy, seen = _seen_passages(lambda q, segs: [0.0] * len(segs))
    hits = ir.search(
        corpus, QUERY, k=10, rerank=spy, rerank_text="artifact", per_artifact=False
    )
    assert len(seen) == len(set(seen)) == len(PKG)
    assert len(hits) > len(PKG)  # every surface is still returned


def test_artifact_text_is_capped_and_overlap_is_not_repeated():
    corpus = _corpus()
    spy, seen = _seen_passages(lambda q, segs: [0.0] * len(segs))
    ir.search(
        corpus, QUERY, k=3, rerank=spy, rerank_text="artifact", rerank_max_chars=120
    )
    assert seen and all(len(t) <= 120 for t in seen)


def test_artifact_text_rescues_a_thin_description():
    corpus = _corpus()
    reader = lambda q, segs: [  # noqa: E731
        len(set(q.split()) & set(s["text"].split())) / 10 for s in segs
    ]
    by_surface = _ids(ir.search(corpus, QUERY, k=3, mode="hybrid", rerank=reader))
    by_artifact = _ids(
        ir.search(
            corpus, QUERY, k=3, mode="hybrid", rerank=reader, rerank_text="artifact"
        )
    )
    assert by_artifact[0] == "thin"
    assert set(by_surface) == set(by_artifact)


def test_a_callable_rerank_text_is_the_seam():
    corpus = _corpus()
    spy, seen = _seen_passages(lambda q, segs: [0.0] * len(segs))
    ir.search(
        corpus,
        QUERY,
        k=3,
        rerank=spy,
        rerank_text=lambda corpus, meta: f"NAME={meta['artifact_id']}",
    )
    assert seen and all(t.startswith("NAME=") for t in seen)


def test_unknown_rerank_text_is_rejected():
    with pytest.raises(ValueError, match="unknown rerank_text"):
        ir.search(
            _corpus(), QUERY, rerank=lambda q, s: [0.0] * len(s), rerank_text="bogus"
        )


def test_rerank_over_fetches_fetch_k_then_trims_to_k():
    corpus = _corpus()
    sizes = []

    def spy(query, segments):
        sizes.append(len(segments))
        return [0.0] * len(segments)

    for per_artifact in (True, False):
        sizes.clear()
        hits = ir.search(
            corpus, QUERY, k=1, mode="hybrid", rerank=spy, per_artifact=per_artifact
        )
        assert len(hits) == 1 and sizes[0] > 1
    ir.search(corpus, QUERY, k=1, rerank=spy, fetch_k=2)
    assert sizes[-1] == 2


# ------------------------------------------------------------ the cross-encoder spec


def test_cross_encoder_spec_runs_efs_reranker_over_the_stub_model(fake_st):
    corpus = _corpus()
    hits = ir.search(
        corpus,
        QUERY,
        k=3,
        mode="hybrid",
        rerank="cross-encoder",
        rerank_text="artifact",
    )
    (model,) = FakeCrossEncoder.instances
    assert model.model_name == rr.DFLT_RERANK_MODEL
    assert _ids(hits)[0] == "thin"
    # raw logits are squashed into (0, 1): relative selectors compare ratios
    assert all(0.0 < h.score < 1.0 for h in hits)
    assert all(q == QUERY for q, _ in model.pairs_seen)


def test_cross_encoder_model_is_named_and_loaded_once(fake_st):
    corpus = _corpus()
    for _ in range(3):
        ir.search(corpus, QUERY, rerank="cross-encoder:my/model")
    (model,) = FakeCrossEncoder.instances
    assert model.model_name == "my/model"


def test_missing_sentence_transformers_warns_once_and_keeps_first_stage(no_st):
    corpus = _corpus()
    plain = ir.search(corpus, QUERY, k=3, mode="hybrid")
    with pytest.warns(UserWarning, match=r"ir\[rerank\]"):
        degraded = ir.search(corpus, QUERY, k=3, mode="hybrid", rerank="cross-encoder")
    assert [(h.artifact_id, h.score) for h in degraded] == [
        (h.artifact_id, h.score) for h in plain
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second search must not warn again
        ir.search(corpus, QUERY, k=3, mode="hybrid", rerank="cross-encoder")


def test_unknown_rerank_spec_is_rejected():
    with pytest.raises(ValueError, match="unknown rerank spec"):
        ir.search(_corpus(), QUERY, rerank="bm25")
    with pytest.raises(TypeError):
        ir.search(_corpus(), QUERY, rerank=3)


def test_formulated_queries_are_reranked_once_against_the_original_query(fake_st):
    corpus = _corpus()
    queries, sizes = [], []

    def spy(query, segments):
        queries.append(query)
        sizes.append(len(segments))
        return [1.0] * len(segments)

    hits = ir.search(
        corpus, "x", k=3, rerank=spy, formulate=lambda q: [QUERY, "pdf merge tools"]
    )
    assert hits and queries == ["x"]  # one call, against what the caller asked
    assert sizes[0] > 3  # over-fetched candidates, not just k per sub-query


def test_a_single_rewritten_query_is_also_judged_against_the_original(fake_st):
    queries = []

    def spy(query, segments):
        queries.append(query)
        return [0.0] * len(segments)

    ir.search(_corpus(), "orig", rerank=spy, formulate=lambda q: "rewritten " + QUERY)
    assert queries == ["orig"]


def test_the_cross_encoder_model_is_shared_across_fanout(fake_st):
    ir.search(
        _corpus(), "x", rerank="cross-encoder", formulate=lambda q: [QUERY, "pdf"]
    )
    assert len(FakeCrossEncoder.instances) == 1


# ------------------------------------------------------------------- calibration


def test_calibration_key_separates_reranked_from_first_stage():
    plain = ir.calibration_key("hybrid")
    tagged = ir.calibration_key("hybrid", rr.rerank_tag(len, "artifact"))
    assert plain == "hybrid" and tagged == "hybrid+rerank-builtins.len+artifact"
    assert "/" not in ir.calibration_key("h", rr.rerank_tag(_named("a/b"), "surface"))


def test_tags_do_not_collide_across_configurations():
    import functools

    tag = rr.rerank_tag
    assert tag(_named("m"), "artifact") != tag(_named("m"), "surface")
    assert tag(_named("m"), "artifact", max_chars=500) != tag(_named("m"), "artifact")

    def artifact(corpus, meta):  # a callable that merely shares a built-in's name
        return ""

    assert tag(_named("m"), artifact) != tag(_named("m"), "artifact")
    assert rr.is_anonymous(tag(lambda q, s: [], "surface"))
    assert rr.is_anonymous(tag(functools.partial(len), "surface"))
    assert rr.is_anonymous(tag(_named("m"), lambda corpus, meta: ""))
    assert not rr.is_anonymous(tag(_named("m"), "surface"))
    stable = lambda q, s: []  # noqa: E731
    stable.rerank_id = "v1"
    assert not rr.is_anonymous(tag(stable, "surface"))


def _named(name):
    def fn(query, segments):
        return [0.0] * len(segments)

    fn.model_name = name
    return fn


def test_persisting_a_floor_for_an_anonymous_reranker_is_refused():
    with pytest.raises(ValueError, match="stable identity"):
        ev.calibrate_min_score(
            _corpus(),
            _cases(),
            mode="hybrid",
            k=3,
            persist=True,
            rerank=lambda q, segs: [0.0] * len(segs),
        )


def test_discover_auto_with_an_anonymous_reranker_does_not_borrow_a_floor():
    corpus = _corpus()
    ev.calibrate_min_score(corpus, _cases(), mode="hybrid", k=3, persist=True)
    with pytest.warns(UserWarning, match="stable identity"):
        ir.discover(
            corpus,
            QUERY,
            mode="hybrid",
            min_score="auto",
            rerank=lambda q, segs: [1.0] * len(segs),
        )


def test_missing_extra_warning_points_at_the_caller(no_st):
    with pytest.warns(UserWarning) as record:
        ir.search(_corpus(), QUERY, rerank="cross-encoder")
    assert record[0].filename == __file__


def test_a_stale_ledger_degrades_the_passage_not_the_search():
    corpus = _corpus()
    spy, seen = _seen_passages(lambda q, segs: [0.0] * len(segs))
    corpus.store.get_ledger_entry = lambda key: {"record_ids": ["ghost"]}
    assert ir.search(corpus, QUERY, k=3, rerank=spy, rerank_text="artifact")
    assert seen


def _cases():
    return [
        ev.DiscoveryCase(QUERY, gold=("thin",)),
        ev.DiscoveryCase("pdf merge tools", gold=("other",)),
        ev.DiscoveryCase("quantum chromodynamics lattice", gold=()),
        ev.DiscoveryCase("medieval falconry manuals", gold=()),
    ]


def test_a_reranked_floor_is_stored_apart_and_not_applied_to_first_stage(fake_st):
    corpus = _corpus()
    plain = ev.calibrate_min_score(corpus, _cases(), mode="hybrid", k=3, persist=True)
    ce = ev.calibrate_min_score(
        corpus,
        _cases(),
        mode="hybrid",
        k=3,
        persist=True,
        rerank="cross-encoder",
        rerank_text="artifact",
    )
    assert plain.rerank is None and ce.rerank == (
        "rerank-cross-encoder_2f_ms-marco-MiniLM-L-6-v2_2b_sigmoid+artifact"
    )
    keys = set(corpus.store.calibration)
    assert keys == {"hybrid", f"hybrid+{ce.rerank}"}
    assert ev.load_calibration(corpus, "hybrid").min_score == plain.min_score
    loaded = ev.load_calibration(
        corpus, "hybrid", rerank="cross-encoder", rerank_text="artifact"
    )
    assert loaded.min_score == ce.min_score and loaded.rerank == ce.rerank
    # the record round-trips through the persisted dict
    assert ev.MinScoreCalibration.from_dict(ce.to_dict()).rerank == ce.rerank


def test_discover_auto_floor_uses_the_reranked_calibration(fake_st):
    corpus = _corpus()
    ev.calibrate_min_score(corpus, _cases(), mode="hybrid", k=3, persist=True)
    # No reranked floor stored: auto must warn and not borrow the first-stage floor.
    with pytest.warns(UserWarning, match=r"hybrid\+rerank-"):
        ir.discover(
            corpus,
            QUERY,
            mode="hybrid",
            min_score="auto",
            rerank="cross-encoder",
            rerank_text="artifact",
        )
    ev.calibrate_min_score(
        corpus,
        _cases(),
        mode="hybrid",
        k=3,
        persist=True,
        rerank="cross-encoder",
        rerank_text="artifact",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = ir.discover(
            corpus,
            QUERY,
            mode="hybrid",
            min_score="auto",
            rerank="cross-encoder",
            rerank_text="artifact",
        )
        absent = ir.discover(
            corpus,
            "quantum chromodynamics lattice",
            mode="hybrid",
            min_score="auto",
            rerank="cross-encoder",
            rerank_text="artifact",
        )
    assert [d.artifact_id for d in found.results][:1] == ["thin"]
    assert absent.abstained


def test_a_degraded_cross_encoder_calibrates_under_the_plain_key(no_st):
    corpus = _corpus()
    with pytest.warns(UserWarning, match=r"ir\[rerank\]"):
        calib = ev.calibrate_min_score(
            corpus, _cases(), mode="hybrid", k=3, persist=True, rerank="cross-encoder"
        )
    assert calib.rerank is None
    assert set(corpus.store.calibration) == {"hybrid"}


def test_federated_auto_floors_follow_the_rerank_tag(fake_st):
    a, b = _corpus("fa"), _corpus("fb")
    for c in (a, b):
        ev.calibrate_min_score(
            c, _cases(), mode="hybrid", k=3, persist=True, rerank="cross-encoder"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ir.discover(
            [a, b], QUERY, mode="hybrid", min_score="auto", rerank="cross-encoder"
        )
    assert result.signals["per_source"]["fa"]["floor"] is not None


# ------------------------------------------------ second review round (ir#103)


def test_closures_bound_methods_and_punctuation_do_not_collide():
    def make(scale):
        def reranker(query, segments):
            return [scale] * len(segments)

        return reranker

    class Scorer:
        def score(self, query, segments):
            return [0.0] * len(segments)

    tag = rr.rerank_tag
    assert rr.is_anonymous(tag(make(1), "surface"))  # a factory's closure
    assert rr.is_anonymous(tag(Scorer().score, "surface"))  # a bound method
    assert (
        len({tag(_named(n), "surface") for n in ("org/m", "org-m", "org m", "org_m")})
        == 4
    )
    assert rr.is_anonymous(tag(_named("m"), lambda c, m: ""))
    assert not rr.is_anonymous(tag(_named("my-anonymous"), "surface"))
    named = make(1)
    named.rerank_id = "scale-1"
    assert not rr.is_anonymous(tag(named, "surface"))


def test_a_closure_floor_is_not_borrowed_by_its_sibling():
    corpus = _corpus()

    def make(scale):
        return lambda q, segs: [scale] * len(segs)

    with pytest.raises(ValueError, match="stable identity"):
        ev.calibrate_min_score(
            corpus, _cases(), mode="hybrid", k=3, persist=True, rerank=make(1000.0)
        )
    with pytest.warns(UserWarning, match="stable identity"):
        ir.discover(corpus, QUERY, mode="hybrid", min_score="auto", rerank=make(1.0))


def test_a_nan_logit_does_not_become_the_top_hit(fake_st, monkeypatch):
    corpus = _corpus()
    calls = {"n": 0}
    real = FakeCrossEncoder.predict

    def predict(self, pairs, batch_size=32, **kwargs):
        scores = real(self, pairs, batch_size, **kwargs)
        calls["n"] += 1
        return scores[:-1] + [float("nan")]

    monkeypatch.setattr(FakeCrossEncoder, "predict", predict)
    hits = ir.search(corpus, QUERY, k=3, rerank="cross-encoder", rerank_text="artifact")
    assert calls["n"] and hits[-1].score == 0.0 or all(h.score < 1.0 for h in hits)
    assert all(0.0 <= h.score < 1.0 for h in hits)


def test_long_tags_are_shortened_to_a_usable_key_and_stay_distinct():
    a, b = _named("m" * 300 + "a"), _named("m" * 300 + "b")
    ta, tb = rr.rerank_tag(a, "surface"), rr.rerank_tag(b, "surface")
    assert len(ta) <= rr.MAX_TAG_CHARS and ta != tb


def test_loading_a_floor_for_a_rerank_that_cannot_run_returns_none(no_st):
    corpus = _corpus()
    ev.calibrate_min_score(corpus, _cases(), mode="hybrid", k=3, persist=True)
    assert ev.load_calibration(corpus, "hybrid") is not None
    assert ev.load_calibration(corpus, "hybrid", rerank="cross-encoder") is None


def test_loading_a_cross_encoder_floor_does_not_load_the_model(fake_st):
    corpus = _corpus()
    ev.calibrate_min_score(
        corpus, _cases(), mode="hybrid", k=3, persist=True, rerank="cross-encoder"
    )
    FakeCrossEncoder.instances.clear()
    rr._model_cache.clear()
    assert ev.load_calibration(corpus, "hybrid", rerank="cross-encoder") is not None
    assert FakeCrossEncoder.instances == []


def test_rerank_max_chars_must_be_positive():
    with pytest.raises(ValueError, match="rerank_max_chars"):
        ir.search(
            _corpus(),
            QUERY,
            rerank=lambda q, s: [0.0] * len(s),
            rerank_text="artifact",
            rerank_max_chars=0,
        )


def test_anonymity_survives_the_shortening_of_a_long_tag():
    long_name = _named("x" * 100)
    tag = rr.rerank_tag(long_name, lambda c, m: "")
    assert len(tag) <= rr.MAX_TAG_CHARS and rr.is_anonymous(tag)
    with pytest.raises(ValueError, match="stable identity"):
        ev.calibrate_min_score(
            _corpus(),
            _cases(),
            mode="hybrid",
            k=3,
            persist=True,
            rerank=_named("y" * 100),
            rerank_text=lambda c, m: m["text"],
        )


def test_same_named_functions_in_different_modules_do_not_collide():
    def score(query, segments):
        return []

    other = type(score)(score.__code__, {}, "score")
    other.__module__ = "another.module"
    other.__qualname__ = "score"
    score.__qualname__ = "score"
    a, b = rr.rerank_tag(score, "surface"), rr.rerank_tag(other, "surface")
    assert a != b and not rr.is_anonymous(a)


def test_an_anonymous_calibration_cannot_be_saved_or_loaded_by_hand():
    corpus = _corpus()
    calib = ev.calibrate_min_score(
        corpus, _cases(), mode="hybrid", k=3, rerank=lambda q, s: [0.0] * len(s)
    )
    with pytest.raises(ValueError, match="stable identity"):
        calib.save(corpus)
    assert (
        ev.load_calibration(corpus, "hybrid", rerank=lambda q, s: [0.0] * len(s))
        is None
    )


def test_a_record_stored_under_the_wrong_key_is_not_used(fake_st):
    corpus = _corpus()
    calib = ev.calibrate_min_score(
        corpus, _cases(), mode="hybrid", k=3, persist=True, rerank="cross-encoder"
    )
    key = ir.calibration_key("hybrid", calib.rerank)
    record = corpus.store.get_calibration(key)
    corpus.store.set_calibration(key, {**record, "rerank": "something-else"})
    with pytest.warns(UserWarning, match="another configuration"):
        ir.discover(
            corpus, QUERY, mode="hybrid", min_score="auto", rerank="cross-encoder"
        )
