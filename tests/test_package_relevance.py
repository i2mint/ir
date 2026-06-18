"""Tests for the graded package-relevance harness (``ir.eval``, issue #66).

Hermetic: the public-name-only fixture
(``tests/fixtures/package_relevance_fixture.jsonl``) plus the light, numpy-only
embedder — no private package names, no network, no model download. Covers the
schema round-trip, graded qrels, the named-set metric functions, the
deterministic ``derive_named_sets`` derivation, the ``compare_indexings``
A/B gate, and the ``regressions`` gate logic.
"""

from pathlib import Path

import pytest

import ir
from ir import eval as ev
from ir.store import CorpusStore

FIXTURE = Path(__file__).parent / "fixtures" / "package_relevance_fixture.jsonl"

# A tiny disjoint-vocab corpus whose keys match the fixture artifact ids, so the
# light (hashing) embedder ranks each theme's probe sensibly and deterministically.
DOCS = {
    "sentence-transformers": "sentence embedding semantic vector text similarity model",
    "chromadb": "vector embedding similarity search store database",
    "transformers": "transformer language model text embedding nlp tokenizer",
    "torch": "tensor deep learning autograd gpu training backend",
    "sklearn": "feature vectorizer machine learning classifier regression",
    "fasttext": "fasttext word embedding subword vectors",
    "librosa": "audio sound waveform spectrogram signal music",
    "networkx": "graph network node edge directed algorithms",
    "igraph": "graph vertex edge community network analysis",
    "graphviz": "graph visualization dot digraph node edge layout",
    "kroki": "diagram digraph mermaid rendering node edge",
    "airflow": "workflow pipeline scheduler tasks orchestration cron",
}
THEMES = ("embeddings", "graphs")


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="pkgfix", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _cases():
    return ev.load_package_cases(FIXTURE)


def _probes():
    return ev.read_package_meta(FIXTURE)["probes"]


# --------------------------------------------------------------------------- #
# Schema + (de)serialization
# --------------------------------------------------------------------------- #


def test_package_case_roundtrip(tmp_path):
    import json

    cases = [
        ev.PackageRelevanceCase(
            "a",
            labels={"embeddings": "core", "graphs": "none"},
            evidence={"embeddings": "why"},
            observed={"embeddings": 0.5},
            thin_description=True,
            metadata={"d": 1},
        ),
        ev.PackageRelevanceCase("b", labels={"graphs": "strong"}),
    ]
    path = tmp_path / "cases.jsonl"
    ev.save_package_cases(cases, path, meta={"probes": {"embeddings": "x"}})
    loaded = ev.load_package_cases(path)
    assert loaded == cases  # frozen dataclasses compare by value
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header == {"__meta__": {"probes": {"embeddings": "x"}}}
    assert ev.read_package_meta(path) == {"probes": {"embeddings": "x"}}


def test_from_dict_rejects_unknown_level():
    with pytest.raises(ValueError, match="unknown relevance level"):
        ev.PackageRelevanceCase.from_dict(
            {"artifact_id": "a", "labels": {"embeddings": "kinda"}}
        )


def test_fixture_loads_and_is_public_only():
    cases = _cases()
    assert len(cases) == 12
    # Sanity: the fixture must not leak private package names — only public ones.
    assert {c.artifact_id for c in cases} == set(DOCS)


def test_level_and_gain():
    case = ev.PackageRelevanceCase("x", labels={"embeddings": "core"})
    assert case.level("embeddings") == "core"
    assert case.level("graphs") == "none"  # unlabeled -> none
    assert case.gain("embeddings") == 3.0
    assert case.gain("graphs") == 0.0


# --------------------------------------------------------------------------- #
# Graded qrels
# --------------------------------------------------------------------------- #


def test_to_graded_qrels_uses_level_gains():
    cases = _cases()
    queries, qrels = ev.to_graded_qrels(cases, "graphs", probe="graph network")
    qid = "theme:graphs"
    assert queries == {qid: "graph network"}
    grades = qrels[qid]
    # core=3 (networkx, igraph), strong=2 (graphviz, kroki); none/airflow omitted.
    assert grades["networkx"] == 3.0
    assert grades["igraph"] == 3.0
    assert grades["graphviz"] == 2.0
    assert grades["kroki"] == 2.0
    assert "airflow" not in grades  # none -> not judged positive
    # grade histogram reproduces the known group sizes
    from collections import Counter

    assert Counter(grades.values()) == {3.0: 2, 2.0: 2}


def test_level_histogram_covers_all_levels():
    hist = ev.level_histogram(_cases(), "embeddings")
    assert set(hist) == set(ev.RELEVANCE_LEVELS)
    assert hist["core"] == 3  # sentence-transformers, chromadb, fasttext
    assert hist["strong"] == 1
    assert hist["uses-tools"] == 2


# --------------------------------------------------------------------------- #
# Named-set metrics (pure functions) + derivation
# --------------------------------------------------------------------------- #


def test_fp_rate_and_recall_pure():
    ranking = ["a", "b", "c", "d", "e"]
    # 1 of 2 distractors in top-3
    assert ev.fp_rate_on_distractors(ranking, ["b", "z"], k=3) == 0.5
    # 2 of 2 hard positives in top-5, 1 of 2 in top-2
    assert ev.recall_on_hard_positives(ranking, ["a", "d"], k=5) == 1.0
    assert ev.recall_on_hard_positives(ranking, ["a", "d"], k=2) == 0.5
    # empty named set -> 0.0, never divide by zero
    assert ev.fp_rate_on_distractors(ranking, [], k=3) == 0.0
    assert ev.recall_on_hard_positives(ranking, [], k=3) == 0.0


def test_derive_named_sets_is_deterministic():
    cases = _cases()
    emb = ev.derive_named_sets(cases, "embeddings", observed_floor=0.4)
    # librosa (none, observed 0.46 >= 0.4) is a distractor; uses-tools are positive.
    assert emb.distractors == ("librosa",)
    # fasttext (core + thin_description) is the hard positive.
    assert emb.hard_positives == ("fasttext",)
    gr = ev.derive_named_sets(cases, "graphs", observed_floor=0.4)
    assert gr.distractors == ("airflow",)
    assert gr.hard_positives == ("kroki",)
    assert ev.NamedSets.from_dict(gr.to_dict()) == gr  # round-trips


# --------------------------------------------------------------------------- #
# evaluate_named_sets + compare_indexings (real tiny corpus, light embedder)
# --------------------------------------------------------------------------- #


def test_evaluate_named_sets_is_auditable():
    corpus = _corpus()
    ns = ev.derive_named_sets(_cases(), "graphs", observed_floor=0.4)
    report = ev.evaluate_named_sets(
        corpus, ns, "graphs", probe=_probes()["graphs"], mode="dense", k=5
    )
    assert 0.0 <= report.fp_rate <= 1.0
    assert 0.0 <= report.hard_positive_recall <= 1.0
    # every number is auditable back to specific packages
    assert set(report.distractors_seen) <= set(ns.distractors)
    assert set(report.hard_positives_missed) <= set(ns.hard_positives)


def test_compare_indexings_self_has_no_regression():
    import json

    corpus = _corpus()
    cases = _cases()
    named = {t: ev.derive_named_sets(cases, t, observed_floor=0.4) for t in THEMES}
    report = ev.compare_indexings(
        {"baseline": corpus, "candidate": corpus},
        cases,
        themes=THEMES,
        probes=_probes(),
        named_sets=named,
        mode="dense",
        k=5,
    )
    # identical corpora => no named-id moved => zero regressions
    assert report.regressions() == []
    # graded nDCG present for both labels and themes; a real number in [0, 1]
    for label in ("baseline", "candidate"):
        for theme in THEMES:
            ndcg = report.metrics[label][theme]["ndcg"]
            assert 0.0 <= ndcg <= 1.0
            assert "fp_rate" in report.metrics[label][theme]
            assert "hard_positive_recall" in report.metrics[label][theme]
    # to_dict is JSON-clean (the qh / HTTP surface)
    assert json.loads(json.dumps(report.to_dict()))["baseline"] == "baseline"


def test_regressions_flags_dropped_positive_and_risen_distractor():
    # Hand-built report: a hard positive dropped (rank 3 -> 30) and a distractor
    # rose (rank 40 -> 2). Both must be flagged; an unchanged id must not.
    report = ev.ComparisonReport(
        k=20,
        themes=("graphs",),
        labels=("baseline", "candidate"),
        baseline="baseline",
        metrics={
            "baseline": {"graphs": {"ndcg": 0.5}},
            "candidate": {"graphs": {"ndcg": 0.4}},
        },
        deltas={
            "graphs": {
                "kroki": {
                    "role": "hard_positive",
                    "by_label": {
                        "baseline": {"rank": 3, "score": 0.4},
                        "candidate": {"rank": 30, "score": 0.1},
                    },
                },
                "airflow": {
                    "role": "distractor",
                    "by_label": {
                        "baseline": {"rank": 40, "score": 0.1},
                        "candidate": {"rank": 2, "score": 0.5},
                    },
                },
                "networkx": {
                    "role": "hard_positive",
                    "by_label": {
                        "baseline": {"rank": 1, "score": 0.9},
                        "candidate": {"rank": 1, "score": 0.9},
                    },
                },
            }
        },
    )
    regs = report.regressions()
    flagged = {(r["artifact_id"], r["role"]) for r in regs}
    assert ("kroki", "hard_positive") in flagged
    assert ("airflow", "distractor") in flagged
    assert ("networkx", "hard_positive") not in flagged
    # a tolerance threshold suppresses small moves
    assert report.regressions(threshold=100) == []
    # absent/0 ranks are reported as None (1-based ranks: 0 would read as "top")
    report2 = ev.ComparisonReport(
        k=20, themes=("graphs",), labels=("baseline", "candidate"),
        baseline="baseline",
        metrics={"baseline": {"graphs": {"ndcg": 0.5}},
                 "candidate": {"graphs": {"ndcg": 0.4}}},
        deltas={"graphs": {"kroki": {"role": "hard_positive", "by_label": {
            "baseline": {"rank": 3, "score": 0.4},
            "candidate": {"rank": 0, "score": None}}}}},  # vanished
    )
    assert report2.regressions()[0]["candidate_rank"] is None


# --------------------------------------------------------------------------- #
# Hardening (review follow-ups): defensive defaults & input guards
# --------------------------------------------------------------------------- #


def test_gain_with_partial_gains_defaults_to_zero():
    case = ev.PackageRelevanceCase("x", labels={"embeddings": "none"})
    # a partial gains mapping omitting "none" must not KeyError
    assert case.gain("embeddings", gains={"core": 3.0}) == 0.0


def test_derive_named_sets_honors_cached_is_distractor():
    cases = [
        # cached True overrides the observed-floor rule (here observed is absent)
        ev.PackageRelevanceCase("x", labels={"embeddings": "none"},
                                is_distractor={"embeddings": True}),
        # cached False suppresses what the floor rule would otherwise flag
        ev.PackageRelevanceCase("y", labels={"embeddings": "none"},
                                observed={"embeddings": 0.9},
                                is_distractor={"embeddings": False}),
    ]
    ns = ev.derive_named_sets(cases, "embeddings", observed_floor=0.4)
    assert ns.distractors == ("x",)


def test_named_set_rates_dedup_ids():
    ranking = ["a", "b", "c"]
    # "b" listed twice must count once (rate 1/1, not 2/2 miscount or 1/2)
    assert ev.fp_rate_on_distractors(ranking, ["b", "b"], k=3) == 1.0
    assert ev.recall_on_hard_positives(ranking, ["z", "z"], k=3) == 0.0


def test_compare_indexings_guards_bad_args():
    corpus = _corpus()
    cases = _cases()
    with pytest.raises(ValueError, match="rank_depth"):
        ev.compare_indexings({"b": corpus}, cases, themes=("graphs",),
                             probes=_probes(), k=20, rank_depth=5)
    with pytest.raises(ValueError, match="no probe text"):
        ev.compare_indexings({"b": corpus}, cases, themes=("graphs", "missing"),
                             probes={"graphs": "g"}, k=5, rank_depth=50)


def test_evaluate_named_sets_theme_defaults_to_named_sets_theme():
    corpus = _corpus()
    ns = ev.derive_named_sets(_cases(), "graphs", observed_floor=0.4)
    report = ev.evaluate_named_sets(corpus, ns, probe=_probes()["graphs"],
                                    mode="dense", k=5)
    assert report.theme == "graphs"
