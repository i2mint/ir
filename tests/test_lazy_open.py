"""``open_corpus`` defers the embedding-model load until a query embeds (#56).

The model load is the dominant per-process cost; ``ls`` / ``info`` and
lexical-only search never embed, so they must not pay it. A corpus knows its
``embedder_id`` from stored config immediately, but resolves the model lazily.
"""

import numpy as np

import ir
import ir.index as idx
from ir.index import _LazyEmbedder


def test_lazy_embedder_resolves_only_on_first_call(monkeypatch):
    calls = []

    def fake_make_embedder(spec, **kw):
        calls.append(spec)

        def emb(texts, **kwargs):
            return np.ones((len(list(texts)), 4), dtype=np.float32)

        return emb, "fake-id"

    monkeypatch.setattr(idx, "make_embedder", fake_make_embedder)

    lazy = _LazyEmbedder("default")
    assert calls == []  # constructing does not resolve the model

    out = lazy(["x"])  # first call resolves
    assert calls == ["default"]
    assert out.shape == (1, 4)

    lazy(["y"])  # cached: no second resolve
    assert calls == ["default"]
    assert lazy.embedder_id == "fake-id"


def test_open_corpus_does_not_load_model_for_len_or_lexical(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))

    src = ir.CorpusSource.from_mapping(
        {"a": "alpha apple avocado", "b": "beta banana blueberry"},
        name="lz",
        strategy=ir.WholeText(),
    )
    ir.build(src, embedder="light")  # builds into the tmp data dir

    # Count model resolutions from here on (build's own load is already done).
    real = idx.make_embedder
    calls = []

    def counting(spec, **kw):
        calls.append(spec)
        return real(spec, **kw)

    monkeypatch.setattr(idx, "make_embedder", counting)

    corpus = ir.open_corpus("lz")
    assert corpus.embedder_id  # known from config without resolving the model
    assert len(corpus) == 2
    assert calls == []  # opening + len(): no model load

    corpus.search("alpha", mode="lexical")
    assert calls == []  # lexical ranks on text alone: still no model load

    corpus.search("alpha", mode="dense")
    assert len(calls) == 1  # dense embeds the query: resolves exactly once
