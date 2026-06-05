"""Eval-harness tests — hermetic: the light (numpy-only) embedder, no network.

The demo corpus uses **disjoint vocabularies** so dense retrieval is
unambiguous: each query shares content tokens with exactly one document, which
therefore ranks first. That lets the metric assertions be exact (recall@1 == 1)
without depending on a downloaded model.
"""

from pathlib import Path

import pytest

import ir
from ir import eval as ev
from ir.store import CorpusStore

# Disjoint-vocab demo corpus; keys match tests/fixtures/demo_cases.jsonl gold ids.
DOCS = {
    "python": "python programming language scripting interpreter",
    "javascript": "javascript browser web frontend dom rendering",
    "database": "sql relational tables rows columns indexing",
    "docker": "docker container image kubernetes orchestration",
}
FIXTURE = Path(__file__).parent / "fixtures" / "demo_cases.jsonl"


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="demo", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _gold_cases():
    return [c for c in ev.load_cases(FIXTURE) if not c.gold_is_none]


# --------------------------------------------------------------------------- #
# DiscoveryCase + (de)serialization
# --------------------------------------------------------------------------- #


def test_case_roundtrip(tmp_path):
    import json

    cases = [
        ev.DiscoveryCase("q1", gold=("a",), source_id="a", metadata={"d": 1}),
        ev.DiscoveryCase("q2", gold=("b", "c")),
        ev.DiscoveryCase("q3", gold=()),  # abstention
    ]
    path = tmp_path / "cases.jsonl"
    ev.save_cases(cases, path, meta={"corpus": "x"})
    loaded = ev.load_cases(path)
    assert loaded == cases  # frozen dataclasses compare by value
    assert loaded[2].gold_is_none
    # the meta header is actually written as line 1 and is recoverable
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line) == {"__meta__": {"corpus": "x"}}

    # omitting meta writes no header line
    path2 = tmp_path / "no_meta.jsonl"
    ev.save_cases(cases, path2)
    assert "__meta__" not in path2.read_text(encoding="utf-8")
    assert ev.load_cases(path2) == cases


def test_from_dict_accepts_scalar_gold():
    case = ev.DiscoveryCase.from_dict({"query": "q", "gold": "a"})
    assert case.gold == ("a",)


def test_load_fixture_skips_meta_header():
    cases = ev.load_cases(FIXTURE)
    assert len(cases) == 5
    assert sum(c.gold_is_none for c in cases) == 1


# --------------------------------------------------------------------------- #
# Retriever adapter + qrels
# --------------------------------------------------------------------------- #


def test_as_doc_retriever_returns_artifact_ids():
    retriever = ev.as_doc_retriever(_corpus(), mode="dense")
    ranking = retriever("python programming", limit=4)
    assert ranking[0] == "python"
    assert set(ranking) <= set(DOCS)


def test_to_qrels_skips_abstention():
    cases = ev.load_cases(FIXTURE)
    queries, qrels = ev.to_qrels(cases)
    assert len(queries) == 4 and len(qrels) == 4  # the abstention case is dropped
    assert set(queries) == set(qrels)
    # ids are position-stable: the abstention case at index 4 leaves a q00004 gap.
    assert set(qrels) == {"q00000", "q00001", "q00002", "q00003"}
    assert "q00004" not in qrels and "q00004" not in queries


# --------------------------------------------------------------------------- #
# Metrics (delegating to ef) — exact on the disjoint corpus
# --------------------------------------------------------------------------- #


def test_retrieval_report_perfect_on_disjoint_corpus():
    report = ev.retrieval_report(_corpus(), _gold_cases(), mode="dense")
    assert report.n_queries == 4
    assert report.metrics["recall@1"] == pytest.approx(1.0)
    assert report.metrics["ndcg@1"] == pytest.approx(1.0)
    assert report.primary_at(1) == pytest.approx(1.0)


def test_retrieval_report_detects_a_miss():
    # Deliberately mislabel: the query is about docker, gold says database.
    cases = [ev.DiscoveryCase("container image kubernetes", gold=("database",))]
    report = ev.retrieval_report(_corpus(), cases, mode="dense", k_values=(1,))
    assert report.metrics["recall@1"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# evaluate_discovery — report + taxonomy + abstention
# --------------------------------------------------------------------------- #


def test_evaluate_discovery_taxonomy_all_rank1():
    report = ev.evaluate_discovery(_corpus(), _gold_cases(), mode="dense", primary_k=1)
    assert report.n_gold == 4 and report.n_abstention == 0
    assert report.failure_classes.get("hit_rank_1") == 4
    assert report.primary == pytest.approx(1.0)
    assert "DiscoveryReport" in str(report)


def test_evaluate_discovery_matches_pure_ef_path():
    cases = _gold_cases()
    rich = ev.evaluate_discovery(cases=cases, corpus=_corpus(), mode="dense")
    pure = ev.retrieval_report(_corpus(), cases, mode="dense")
    for key in pure.metrics:
        assert rich.retrieval.metrics[key] == pytest.approx(pure.metrics[key])


def test_abstention_separation_and_accuracy():
    corpus = _corpus()
    cases = ev.load_cases(FIXTURE)
    # The abstention query shares no tokens with any doc -> near-zero top score.
    def top_score(query):
        hits = ir.search(corpus, query, k=1, mode="dense")
        return hits[0].score if hits else 0.0

    gold_tops = [top_score(c.query) for c in cases if not c.gold_is_none]
    abstain_top = next(top_score(c.query) for c in cases if c.gold_is_none)
    assert abstain_top < min(gold_tops)  # the separating signal exists

    threshold = (abstain_top + min(gold_tops)) / 2
    report = ev.evaluate_discovery(
        corpus, cases, mode="dense", abstain_threshold=threshold
    )
    assert report.n_abstention == 1
    assert report.abstention_accuracy == pytest.approx(1.0)
    assert report.failure_classes.get("abstention_ok") == 1


def test_abstention_unscored_without_threshold():
    report = ev.evaluate_discovery(_corpus(), ev.load_cases(FIXTURE), mode="dense")
    assert report.failure_classes.get("abstention_unscored") == 1
    assert report.abstention_accuracy is None


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        ev.evaluate_discovery(_corpus(), _gold_cases(), metrics=("bogus",))


# --------------------------------------------------------------------------- #
# Drift detection
# --------------------------------------------------------------------------- #


def test_validate_cases_flags_drift():
    cases = [
        ev.DiscoveryCase("q", gold=("python",)),      # present
        ev.DiscoveryCase("q", gold=("gone", "docker")),  # 'gone' absent
    ]
    missing = ev.validate_cases(_corpus(), cases)
    assert missing == {1: ["gone"]}


# --------------------------------------------------------------------------- #
# Distractor-robustness curve
# --------------------------------------------------------------------------- #


def test_distractor_curve_shape_and_anchor():
    probes = [("python programming language", "python")]
    curve = ev.distractor_robustness_curve(
        DOCS, probes, sizes=(1, 2, 4), trials=3, embedder="light", mode="dense"
    )
    assert set(curve) == {1, 2, 4}
    assert curve[1] == pytest.approx(1.0)  # gold alone is always found
    assert all(0.0 <= v <= 1.0 for v in curve.values())
    # Disjoint vocab => gold wins regardless of distractors.
    assert curve[4] == pytest.approx(1.0)


def test_distractor_curve_from_cases():
    src = ir.CorpusSource.from_mapping(DOCS, name="demo", strategy=ir.WholeText())
    curve = ev.distractor_curve_from_cases(
        src, _gold_cases(), sizes=(1, 3), trials=2, mode="dense"
    )
    assert set(curve) == {1, 3}
    assert curve[1] == pytest.approx(1.0)


def test_distractor_curve_declines_with_confusable_distractors():
    # Each distractor shares MORE of the query's tokens than gold does, so any
    # distractor present outranks gold -> accuracy collapses as the catalog grows.
    scope = {"gold": "alpha gamma"}
    for i in range(4):
        scope[f"d{i}"] = "alpha beta delta"
    curve = ev.distractor_robustness_curve(
        scope,
        [("alpha beta", "gold")],
        sizes=(1, 2, 4),
        trials=4,
        embedder="light",
        mode="dense",
    )
    assert curve[1] == pytest.approx(1.0)  # gold alone
    assert curve[max(curve)] < curve[1]  # a genuine decline (the headline signal)


def test_distractor_curve_caps_sizes_to_corpus():
    # Sizes beyond the corpus must be capped+deduped, not relabelled.
    curve = ev.distractor_robustness_curve(
        DOCS, [("python programming", "python")], sizes=(1, 4, 8, 128), trials=2
    )
    assert set(curve) == {1, 4}  # corpus has 4 docs; 8 and 128 collapse to 4


# --------------------------------------------------------------------------- #
# Degenerate inputs: no gold-bearing cases
# --------------------------------------------------------------------------- #


def test_no_gold_yields_none_primary_not_nan():
    cases = [ev.DiscoveryCase("q1", gold=()), ev.DiscoveryCase("q2", gold=())]
    report = ev.evaluate_discovery(_corpus(), cases, mode="dense")
    assert report.n_gold == 0
    assert report.primary is None  # the None-means-not-computed contract holds
    assert report.retrieval.metrics == {}
    assert "n/a" in str(report) and "nan" not in str(report).lower()


def test_retrieval_report_raises_on_all_abstention():
    # The pure-ef path raises (ef requires a positively-judged query); the rich
    # path tolerates it (see test above). The asymmetry is documented.
    with pytest.raises(ValueError):
        ev.retrieval_report(_corpus(), [ev.DiscoveryCase("q", gold=())], mode="dense")


def test_abstention_false_action_when_threshold_too_low():
    # A threshold below the abstention query's top score => the system "acts" when
    # it should abstain: false_action, accuracy 0.
    report = ev.evaluate_discovery(
        _corpus(), ev.load_cases(FIXTURE), mode="dense", abstain_threshold=-1.0
    )
    assert report.failure_classes.get("false_action") == 1
    assert report.abstention_accuracy == pytest.approx(0.0)


def test_per_query_keys_skip_abstention_index():
    report = ev.evaluate_discovery(_corpus(), ev.load_cases(FIXTURE), mode="dense")
    assert set(report.retrieval.per_query) == {"q00000", "q00001", "q00002", "q00003"}


# --------------------------------------------------------------------------- #
# surfaces= restriction threads through the eval path
# --------------------------------------------------------------------------- #


def test_surfaces_restriction_threads_through():
    docs = {
        "alpha": {"name": "alpha", "description": "alpha tool for foo tasks"},
        "beta": {"name": "beta", "description": "beta tool for bar tasks"},
    }
    src = ir.CorpusSource.from_mapping(docs, name="skl", strategy=ir.Skill())
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    # Skill surfaces are kind "capability".
    keep = ev.as_doc_retriever(corpus, mode="dense", surfaces={"capability"})
    assert keep("alpha tool for foo tasks", limit=2)[0] == "alpha"
    drop = ev.as_doc_retriever(corpus, mode="dense", surfaces={"nonexistent"})
    assert drop("alpha tool for foo tasks", limit=2) == []


# --------------------------------------------------------------------------- #
# CLI eval — drift warning, empty corpus, no cases (hermetic, light embedder)
# --------------------------------------------------------------------------- #


def test_cli_eval_paths(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    from ir import cli

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text("deploy the app to the server with systemd units")
    (docs / "baking.md").write_text("bake a cake in the oven with flour and sugar")
    cli.register("notes", "files", root=str(docs), pattern=r".*\.md$")
    cli.build("notes", embedder="light")  # built with light -> open_corpus stays offline

    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"query": "deploy app systemd", "gold": ["deploy.md"]})
        + "\n"
        + json.dumps({"query": "irrelevant", "gold": ["gone.md"]})  # drift
        + "\n"
    )
    out = cli.eval("notes", str(cases), mode="dense", k=3)
    assert "DiscoveryReport" in out
    assert "WARNING" in out  # drift on the absent gold id is surfaced

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert "no cases found" in cli.eval("notes", str(empty), mode="dense")

    blank_dir = tmp_path / "blankdir"
    blank_dir.mkdir()
    cli.register("blank", "files", root=str(blank_dir), pattern=r".*\.md$")
    cli.build("blank", embedder="light")  # 0 records, config pins light embedder
    assert "build it first" in cli.eval("blank", str(cases), mode="dense")


# --------------------------------------------------------------------------- #
# ir.eval is reachable as a submodule attribute
# --------------------------------------------------------------------------- #


def test_eval_reachable_via_ir_namespace():
    assert hasattr(ir, "eval")
    assert ir.eval.DiscoveryCase is ev.DiscoveryCase
