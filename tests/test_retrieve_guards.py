"""Guard-path tests for ``ir.retrieve`` — the user-facing failure & degradation modes.

Two production-relevant guards previously had no direct coverage:

- the **dimension-mismatch** ``ValueError`` raised when a corpus is queried with a
  different embedder than it was built with (the stale-embedder trap), and
- the graceful **vd-unavailable degradation**: lexical mode returns nothing with
  a warning, hybrid falls back to dense with a warning — rather than hard-failing.

All hermetic: the light (numpy-only) embedder over a tiny in-memory corpus, no
network. The vd-absence is simulated by poisoning ``sys.modules['vd']`` so the
lazy ``from vd import ...`` inside retrieval raises ``ImportError``.
"""

import sys

import numpy as np
import pytest

import ir
from ir.store import CorpusStore

DOCS = {
    "a": "alpha apple avocado almond",
    "b": "beta banana blueberry blackberry",
    "c": "gamma grape guava greengage",
}


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="guards", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def test_dim_mismatch_raises_actionable_error():
    corpus = _corpus()

    def tiny_embedder(texts, input_type=None):
        # 3-dim vectors, mismatching the hashing index's width
        return np.ones((len(list(texts)), 3), dtype=np.float32)

    corpus.embedder = tiny_embedder
    with pytest.raises(ValueError, match="different embedder"):
        corpus.search("alpha apple", mode="dense")


def test_lexical_without_vd_returns_empty_and_warns(monkeypatch):
    corpus = _corpus()  # build first (build needs no vd)
    monkeypatch.setitem(sys.modules, "vd", None)  # `import vd` now raises
    with pytest.warns(UserWarning, match="vd.BM25Index unavailable"):
        hits = corpus.search("alpha apple", mode="lexical")
    assert hits == []


def test_hybrid_without_vd_falls_back_to_dense(monkeypatch):
    corpus = _corpus()
    monkeypatch.setitem(sys.modules, "vd", None)
    with pytest.warns(UserWarning):
        hits = corpus.search("alpha apple", mode="hybrid")
    # Hybrid degrades to the dense ranking: non-empty, with the lexical match on top.
    assert hits
    assert hits[0].artifact_id == "a"
