"""ir's lexical search ranks the same through vd's bm25s engine as through Python (#171).

``ir.retrieve.BM25_ENGINE = "auto"`` picks vd's ``"bm25s"`` engine whenever the
``bm25s`` package imports. That is only safe if it changes speed and nothing
else: the same top-50 for every query, scores equal to float32 rounding (the
per-corpus abstention floors are calibrated on score values).
"""

from pathlib import Path

import pytest

import ir
from ir import retrieve
from ir.store import CorpusStore

QUERIES = (
    "abstention floor calibration",
    "reciprocal rank fusion of dense and lexical rankings",
    "incremental reindex of a changed corpus",
    "surface weights per artifact",
    "the single run maintenance lock",
    "chunk text by tokens",
    "metadata filter on the candidate set",
    "rerank the fetched candidates",
    "register a corpus by name",
    "query normalization",
)
TOP = 50


@pytest.fixture(scope="module")
def corpus():
    """ir's own source, chunked: a fixture corpus of real prose and code."""
    root = Path(ir.__file__).parent
    docs = {
        str(path.relative_to(root)): {"text": path.read_text(encoding="utf-8")}
        for path in sorted(root.rglob("*.py"))
    }
    source = ir.CorpusSource.from_mapping(
        docs, name="ir-src", strategy=ir.Chunked(chunk_size=600, overlap=60)
    )
    return ir.build(source, store=CorpusStore.memory(), embedder="light")


def _ids(hits):
    return [(h.artifact_id, h.surface_kind, h.surface_index) for h in hits]


def _lexical(corpus, query, engine, monkeypatch, **kwargs):
    monkeypatch.setattr(retrieve, "BM25_ENGINE", engine)
    return retrieve.search(corpus, query, mode="lexical", k=TOP, fetch_k=TOP, **kwargs)


def test_auto_resolves_to_python_without_bm25s(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_bm25s(name, *args, **kwargs):
        if name == "bm25s":
            raise ImportError("no bm25s here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_bm25s)
    assert retrieve._resolve_bm25_engine("auto") == "python"
    assert retrieve._resolve_bm25_engine("python") == "python"


@pytest.mark.parametrize("query", QUERIES)
def test_bm25s_engine_gives_the_python_engines_top_50(corpus, query, monkeypatch):
    pytest.importorskip("bm25s")
    assert retrieve._resolve_bm25_engine("auto") == "bm25s"
    slow = _lexical(corpus, query, "python", monkeypatch)
    fast = _lexical(corpus, query, "bm25s", monkeypatch)
    assert slow, "the fixture query should match something"
    assert _ids(fast) == _ids(slow)
    for a, b in zip(fast, slow, strict=True):
        assert a.score == pytest.approx(b.score, rel=1e-5)


def test_per_query_okapi_parameters_agree_too(corpus, monkeypatch):
    pytest.importorskip("bm25s")
    params = {"bm25": {"k1": 1.2, "b": 0.5}}
    slow = _lexical(corpus, QUERIES[1], "python", monkeypatch, **params)
    fast = _lexical(corpus, QUERIES[1], "bm25s", monkeypatch, **params)
    assert _ids(fast) == _ids(slow)


def test_switching_engines_does_not_reuse_the_other_engines_index(
    corpus, monkeypatch
):
    pytest.importorskip("bm25s")
    _lexical(corpus, QUERIES[0], "python", monkeypatch)
    _lexical(corpus, QUERIES[0], "bm25s", monkeypatch)
    engines = {index.engine for index in corpus._bm25_index_cache.values()}
    assert engines == {"python", "bm25s"}
