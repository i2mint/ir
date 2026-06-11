"""End-to-end eval on a **frozen, committed, real-content** fixture.

The other eval tests (``test_eval.py``) run on an inline *synthetic disjoint-
vocab* corpus engineered so each query matches exactly one document — good for
pinning the metric math, but it never exercises ranking on realistic confusable
text. This module closes that gap with a small corpus of **real** ecosystem
package descriptions (``tests/fixtures/mini_corpus.json``) and golden cases
(``tests/fixtures/mini_cases.jsonl``). The descriptions share vocabulary ("tools",
"data", "functions"), so retrieval has genuine distractors — yet the run is fully
deterministic and offline (the light, numpy-only hashing embedder, in-memory
store), which lets us *freeze a baseline*: a regression in indexing, retrieval,
selection, or calibration moves these numbers and trips the test.

Assertions are deliberately robust — rank membership, separation ratios, and
ranges — not brittle exact floats, so a harmless embedder tweak does not break
the build while a real regression still does.
"""

import json
from pathlib import Path

import pytest

import ir
from ir import eval as ev
from ir.select import discover
from ir.store import CorpusStore

FIXTURES = Path(__file__).parent / "fixtures"
MINI_CORPUS = FIXTURES / "mini_corpus.json"
MINI_CASES = FIXTURES / "mini_cases.jsonl"


@pytest.fixture(scope="module")
def mini_corpus():
    """Build the frozen real-content corpus deterministically (light embedder)."""
    mapping = json.loads(MINI_CORPUS.read_text())
    src = ir.CorpusSource.from_mapping(mapping, name="mini", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


@pytest.fixture(scope="module")
def mini_cases():
    return ev.load_cases(MINI_CASES)


def test_fixture_is_well_formed(mini_corpus, mini_cases):
    # 7 real artifacts, 6 gold-bearing cases + 1 abstention, and no drift between
    # the cases' gold ids and the corpus (the committed pair stays in sync).
    assert len(mini_corpus) == 7
    assert sum(not c.gold_is_none for c in mini_cases) == 6
    assert sum(c.gold_is_none for c in mini_cases) == 1
    assert ev.validate_cases(mini_corpus, mini_cases) == {}


def test_every_gold_ranks_first_with_separation(mini_corpus, mini_cases):
    # On real confusable text the gold still ranks #1 — but real distractors do
    # surface in the top-3 (this is NOT the disjoint-vocab toy), and the gold's
    # score stands clear of the runner-up.
    saw_a_distractor = False
    for case in (c for c in mini_cases if not c.gold_is_none):
        hits = mini_corpus.search(case.query, k=3, mode="dense", per_artifact=True)
        assert hits[0].artifact_id == case.gold[0]
        assert float(hits[0].score) > 1.3 * float(hits[1].score)  # clear separation
        if any(h.artifact_id not in case.gold for h in hits[1:]):
            saw_a_distractor = True
    assert saw_a_distractor, "fixture should have real distractors in the top-k"


def test_retrieval_baseline_is_frozen(mini_corpus, mini_cases):
    report = ev.evaluate_discovery(
        mini_corpus, mini_cases, mode="dense", primary_k=3, k_values=(1, 3, 5)
    )
    m = report.retrieval.metrics
    # Perfect ranking baseline (gold at rank 1 for all six gold cases)...
    assert report.primary == pytest.approx(1.0)
    for key in ("ndcg@1", "recall@1", "mrr@1", "precision@1", "recall@5"):
        assert m[key] == pytest.approx(1.0)
    # ...with the mechanical precision@k given exactly one gold per query.
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert m["precision@5"] == pytest.approx(0.2)


def test_selection_baseline_is_frozen(mini_corpus, mini_cases):
    report = ev.evaluate_selection(
        mini_corpus, mini_cases, strategy="conservative", mode="dense", k=5
    )
    # Given retrieval surfaced the gold, the conservative selector keeps it every
    # time (conditional commit rate) at perfect precision/recall.
    assert report.conditional_commit_rate == pytest.approx(1.0)
    assert report.selection_f1 == pytest.approx(1.0)
    assert report.mean_selected_size == pytest.approx(1.0)


def test_calibration_separates_and_drives_abstention(mini_corpus, mini_cases):
    # The out-of-scope query scores below every in-scope query, so calibration
    # finds a cleanly separating floor — the heart of the abstention story.
    calib = ev.calibrate_min_score(mini_corpus, mini_cases, mode="dense", k=5)
    assert calib.separable
    assert calib.youden_j == pytest.approx(1.0)
    # The floor sits between the out-of-scope top score and the weakest in-scope
    # top score (a robust band, not a brittle exact value).
    assert 0.194 < calib.min_score < 0.313

    # And the floor actually abstains on out-of-scope while committing in-scope.
    abstention = next(c for c in mini_cases if c.gold_is_none)
    out = discover(mini_corpus, abstention.query, mode="dense", min_score=calib.min_score)
    assert out.abstained

    in_scope = next(c for c in mini_cases if not c.gold_is_none)
    committed = discover(
        mini_corpus, in_scope.query, mode="dense", min_score=calib.min_score
    )
    assert not committed.abstained
    assert committed.results[0].name == in_scope.gold[0]
