"""Tests for embedder resolution (``ir.embed.make_embedder``).

The resolved ``embedder_id`` is a real contract, not a cosmetic label: it is
written to the ledger to detect model drift (``ir.index``) and stamped into
calibration records for staleness checks (``ir.select``). These tests pin the
resolution branches and — most importantly — the graceful fallback to the light
hashing embedder when ``sentence-transformers`` is unavailable, all without any
network or model download.
"""

import numpy as np
import pytest

from ir import embed


@pytest.mark.parametrize("spec", ["light", "hashing", "hash"])
def test_light_aliases_resolve_to_hashing(spec):
    emb, emb_id = embed.make_embedder(spec)
    assert emb_id == "hashing__dim512"
    assert callable(emb)


def test_callable_passthrough_keeps_custom_id():
    def my_embedder(texts, input_type=None):
        return np.zeros((len(list(texts)), 4), dtype=np.float32)

    my_embedder._ir_id = "myid"
    emb, emb_id = embed.make_embedder(my_embedder)
    assert emb_id == "myid"
    assert callable(emb)


def test_callable_without_id_defaults_to_custom():
    emb, emb_id = embed.make_embedder(lambda texts, input_type=None: texts)
    assert emb_id == "custom"


def test_default_falls_back_to_hashing_when_st_unavailable(monkeypatch):
    # Force the sentence-transformers path to fail: resolution must degrade to
    # the hashing embedder WITH a warning, not raise.
    def boom(model_name):
        raise ImportError("no sentence-transformers here")

    monkeypatch.setattr(embed, "_sentence_transformers", boom)
    with pytest.warns(UserWarning, match="falling back to the light hashing"):
        emb, emb_id = embed.make_embedder("default")
    assert emb_id == "hashing__dim512"
    assert callable(emb)


def test_embedder_id_is_stable_across_calls():
    # The id keys the ledger / calibration, so it must be deterministic.
    _, id1 = embed.make_embedder("light")
    _, id2 = embed.make_embedder("light")
    assert id1 == id2
