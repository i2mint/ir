"""The real-benchmark #64 script (misc/evals/dep_signal_real.py) runs and scores right.

The benchmark itself is private, so this drives the script's arms and scoring on the
synthetic corpus of ``dep_signal_synthetic.py``, labelled the benchmark's way, and
checks the arithmetic of :func:`score` on a hand-made ranking.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from ir import eval as ev

_EVALS = Path(__file__).parents[1] / "misc" / "evals"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _EVALS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real():
    return _load("dep_signal_real")


def _case(aid, **levels):
    labels = {t: levels.get(t, "none") for t in ("graphs",)}
    return ev.PackageRelevanceCase(artifact_id=aid, labels=labels)


def test_score_counts_what_it_names(real):
    cases = [
        _case("a", graphs="core"),
        _case("b", graphs="strong"),
        _case("c", graphs="uses-tools"),
        _case("d"),
    ]
    named = ev.NamedSets(theme="graphs", distractors=("d",), hard_positives=("c",))
    metrics, ranks = real.score(["d", "c", "a"], cases, "graphs", named, seeds={"c"})
    assert metrics["recall@20"] == 0.5  # a of {a, b}; c is uses-tools, not relevant
    assert metrics["hard positives in top-20"] == 1
    assert metrics["of them not seeds"] == 0  # c is a seed
    assert metrics["distractor fp_rate@20"] == 1.0
    assert metrics["distractors in top-10"] == 1
    assert metrics["uses-tools in top-20"] == 1
    # linear gains core=3, strong=2, uses-tools=1: c at rank 2, a at rank 3
    dcg = 1 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 + 2 / math.log2(3) + 1 / math.log2(4)
    assert metrics["nDCG@20"] == pytest.approx(dcg / idcg)
    assert ranks == {"c": 2, "d": 1}


def test_arms_on_a_synthetic_corpus(real):
    syn = _load("dep_signal_synthetic")
    packages, gold = syn.make_corpus(seed=0)
    corpus = syn.build(packages)
    spec, g = syn.THEMES["graphs"], gold["graphs"]
    levels = {a: "core" if r == 2 else "strong" for a, r in g["relevance"].items()}
    cases = [_case(a, graphs=levels.get(a, "none")) for a in packages]
    named = ev.NamedSets(
        theme="graphs",
        distractors=tuple(g["distractors"]),
        hard_positives=tuple(g["hard"]),
    )
    ranked = real.arms(corpus, spec["query"], set(spec["libs"]), rerank=None)
    assert not any("cross-encoder" in arm for arm in ranked)
    scored = {
        arm: real.score(r, cases, "graphs", named)[0] for arm, r in ranked.items()
    }
    base = scored["baseline (hybrid)"]
    for arm in ("+ reverse-dep walk (RRF)", "+ dep_boost w=0.02"):
        assert (
            scored[arm]["hard positives in top-20"] > base["hard positives in top-20"]
        )
        assert scored[arm]["nDCG@20"] > base["nDCG@20"]
    assert "### graphs" in real.as_markdown(
        {
            "fan_in_histogram": {},
            "themes": {
                "graphs": {
                    "n_hard_positives": 8,
                    "n_distractors": 8,
                    "metrics": scored,
                }
            },
        }
    )


def test_build_reports_missing_and_duplicated_checkouts(real, tmp_path):
    dirs = []
    for rel in ("t/a", "tt/a", "t/b"):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "package.json").write_text(json.dumps({"description": f"pkg {rel}"}))
        dirs.append(d)
    corpus, missing, duplicated = real.build(
        ["a", "b", "c"], embedder="light", repo_dirs=lambda: dirs
    )
    assert missing == {"c"} and duplicated == {"a"}
    assert ev.corpus_artifact_ids(corpus) == {"a", "b"}
