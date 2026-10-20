"""The synthetic #64 experiment (misc/evals/dep_signal_synthetic.py) keeps its findings.

Only the qualitative claims are pinned, on one seeded corpus: both dependency
mechanisms lift graded nDCG@20 and the hard positives over the flat hybrid
baseline, and neither lets more prose look-alikes into the top 10.
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "misc" / "evals" / "dep_signal_synthetic.py"


@pytest.fixture(scope="module")
def results():
    spec = importlib.util.spec_from_file_location("dep_signal_synthetic", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(seed=0)[0]


@pytest.mark.parametrize("theme", ["graphs", "embeddings"])
@pytest.mark.parametrize("arm", ["+ reverse-dep walk (RRF)", "+ dep_boost w=0.02"])
def test_dependency_signal_beats_the_flat_baseline(results, theme, arm):
    base, treated = results[theme]["baseline (hybrid)"], results[theme][arm]
    assert base["hard in top-20"] == 0  # prose alone never finds them, by construction
    assert treated["nDCG@20"] > base["nDCG@20"]
    assert treated["hard in top-20"] >= 3
    assert treated["distractors in top-10"] <= base["distractors in top-10"]
