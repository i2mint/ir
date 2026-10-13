"""A reopened corpus is pinned to the embedder it was built with (#90).

The stored ``embedder_spec`` is often ``"default"``, which means the 384-dim
sentence-transformers model on one machine and the 512-dim hashing fallback on
another. Reopening must follow the stored ``embedder_id``, not re-resolve the
spec; and a mismatch that still happens must say how to fix it, not crash.
"""

import pytest

import ir
import ir.embed as emb
from ir import cli
from ir.embed import EmbedderMismatch, EmbedderUnavailable, make_embedder_for_id
from ir.index import open_corpus

DOCS = {"a": "alpha apple avocado", "b": "beta banana blueberry"}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    for var in ("DATA", "CONFIG", "CACHE"):
        monkeypatch.setenv(f"IR_{var}_DIR", str(tmp_path / var.lower()))


def _build(name="pin", embedder="light"):
    src = ir.CorpusSource.from_mapping(DOCS, name=name, strategy=ir.WholeText())
    return ir.build(src, embedder=embedder)


def test_reopen_ignores_a_spec_that_now_resolves_differently(dirs, monkeypatch):
    _build()
    # Rewrite the stored spec to "default", as a registry-built corpus has it.
    corpus = open_corpus("pin")
    corpus.store.set_config({**corpus.store.get_config(), "embedder_spec": "default"})
    # Querying machine would resolve "default" to a 384-dim model.
    monkeypatch.setattr(emb, "_sentence_transformers", lambda m: 1 / 0)
    hits = open_corpus("pin").search("alpha apple", mode="dense")
    assert hits and hits[0].artifact_id == "a"


def test_make_embedder_for_id_round_trips_and_declines():
    e, i = make_embedder_for_id("hashing__dim64")
    assert i == "hashing__dim64" and e(["x"]).shape == (1, 64)
    assert make_embedder_for_id("custom") is None
    assert make_embedder_for_id("") is None


def test_pinned_model_unavailable_is_an_actionable_error(monkeypatch):
    monkeypatch.setattr(emb, "_sentence_transformers", lambda m: 1 / 0)
    with pytest.raises(EmbedderUnavailable, match="ir build"):
        make_embedder_for_id("st__some-model")


def test_mismatch_message_names_corpus_and_fix(dirs):
    corpus = _build()
    corpus.embedder = lambda texts, **kw: __import__("numpy").ones((1, 3))
    with pytest.raises(EmbedderMismatch, match=r"ir build pin.*--mode lexical"):
        corpus.search("alpha", mode="dense")


def test_cli_search_reports_mismatch_instead_of_crashing(dirs, monkeypatch, capsys):
    _build()
    import numpy as np

    real_open = cli.open_corpus

    def open_mismatched(name, **kw):
        c = real_open(name, **kw)
        c.embedder = lambda texts, **k: np.ones((1, 3))
        return c

    monkeypatch.setattr(cli, "open_corpus", open_mismatched)
    for command in (cli.search, cli.discover):
        with pytest.raises(SystemExit) as exit_info:
            command("pin", "alpha", mode="dense")
        assert exit_info.value.code == 2
        assert "ir build pin" in capsys.readouterr().err
