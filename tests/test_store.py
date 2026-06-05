"""Unit tests for the CorpusStore repository layer (in-memory)."""

import numpy as np

from ir.base import Record
from ir.store import CorpusStore


def _rec(rid="r1", aid="a1", kind="document", idx=0, vec=(1.0, 0.0, 0.0)):
    return Record(
        id=rid,
        artifact_id=aid,
        surface_kind=kind,
        surface_index=idx,
        text="some text",
        vector=np.asarray(vec, dtype=np.float32),
        metadata={"owner": "ours"},
    )


def test_record_make_id_is_deterministic():
    assert Record.make_id("a", "k", 0) == Record.make_id("a", "k", 0)
    assert Record.make_id("a", "k", 0) != Record.make_id("a", "k", 1)


def test_put_get_delete_roundtrip():
    store = CorpusStore.memory()
    rec = _rec()
    store.put_record(rec)
    assert len(store) == 1
    got = store.get_record(rec.id)
    assert got.artifact_id == "a1"
    assert got.metadata["owner"] == "ours"
    np.testing.assert_allclose(got.vector, rec.vector)
    store.delete_record(rec.id)
    assert len(store) == 0


def test_matrix_is_l2_normalized_and_aligned():
    store = CorpusStore.memory()
    store.put_record(_rec(rid="r1", vec=(3.0, 0.0, 0.0)))
    store.put_record(_rec(rid="r2", vec=(0.0, 4.0, 0.0)))
    ids, mat, metas = store.matrix()
    assert mat.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), [1.0, 1.0], atol=1e-6)
    assert len(ids) == len(metas) == 2


def test_matrix_cache_invalidated_on_write():
    store = CorpusStore.memory()
    store.put_record(_rec(rid="r1"))
    assert store.matrix()[1].shape == (1, 3)
    store.put_record(_rec(rid="r2"))
    assert store.matrix()[1].shape == (2, 3)


def test_ledger_and_config():
    store = CorpusStore.memory()
    store.set_ledger_entry(
        "k", {"artifact_id": "a", "version": "v1", "record_ids": ["r1"]}
    )
    assert store.get_ledger_entry("k")["version"] == "v1"
    store.set_config({"embedder_id": "hashing__dim512"})
    assert store.get_config()["embedder_id"] == "hashing__dim512"
    store.delete_ledger_entry("k")
    assert store.get_ledger_entry("k") is None
