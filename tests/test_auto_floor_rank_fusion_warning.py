"""min_score='auto' on hybrid rank fusion warns; blend and other modes do not (i2mint/ir#168)."""

import warnings

import pytest

import ir
from ir.store import CorpusStore

DOCS = {"a": "alpha apple", "b": "alpha banana", "c": "cherry tart"}


@pytest.fixture(scope="module")
def corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="auto", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _warnings_about_rank_fusion(corpus, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ir.discover(corpus, "alpha", min_score="auto", **kw)
    return [w for w in caught if "fusion='rrf'" in str(w.message)]


def test_hybrid_default_rrf_warns_and_points_at_the_caller(corpus):
    (w,) = _warnings_about_rank_fusion(corpus)
    assert "fusion='blend'" in str(w.message)
    assert w.filename.endswith("test_auto_floor_rank_fusion_warning.py")


@pytest.mark.parametrize("kw", [{"fusion": "blend"}, {"mode": "lexical"}, {"mode": "dense"}])
def test_blend_and_non_hybrid_modes_do_not_warn(corpus, kw):
    assert _warnings_about_rank_fusion(corpus, **kw) == []


def test_without_auto_there_is_no_warning(corpus):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ir.discover(corpus, "alpha")
    assert not [w for w in caught if "fusion='rrf'" in str(w.message)]
