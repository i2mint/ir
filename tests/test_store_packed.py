"""Packed-matrix disk cache + vector-free ``metas()`` — the #56 perf paths.

These exercise the file-backed (``packed_dir``) behavior that the in-memory
store in ``test_store.py`` cannot: persisting one normalized matrix so a fresh
process reopens it with a few reads instead of a per-record vector-file storm,
and serving ``metas()`` without touching vectors (for lexical-only ranking).
"""

import numpy as np

from ir.base import Record
from ir.store import CorpusStore, _json_store, _ndarray_store


def _rec(rid, *, vec=(1.0, 0.0, 0.0), text="t"):
    return Record(
        id=rid,
        artifact_id=f"art_{rid}",
        surface_kind="document",
        surface_index=0,
        text=text,
        vector=np.asarray(vec, dtype=np.float32),
        metadata={"owner": "ours"},
    )


def _file_store(tmp_path):
    """A file-backed store with the on-disk packed cache enabled."""
    root = tmp_path / "corpus"
    store = CorpusStore(
        meta=_json_store(root / "meta"),
        vectors=_ndarray_store(root / "vectors"),
        ledger=_json_store(root / "ledger"),
        config=_json_store(root / "config"),
        calibration=_json_store(root / "calibration"),
        links=_json_store(root / "links"),
        packed_dir=root / "matrix",
    )
    return store, root


def test_packed_cache_written_then_reloaded_by_fresh_store(tmp_path):
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    store.put_record(_rec("r2", vec=(0.0, 4.0, 0.0)))
    ids, mat, metas = store.matrix()  # builds from records + persists packed
    assert (root / "matrix" / "sig.json").exists()

    # A brand-new store over the same dir must reload from the packed cache and
    # return identical ids / normalized matrix / metas.
    store2, _ = _file_store(tmp_path)
    ids2, mat2, metas2 = store2.matrix()
    assert ids2 == ids
    np.testing.assert_allclose(np.asarray(mat2), np.asarray(mat), atol=1e-6)
    assert metas2 == metas


def test_packed_cache_cleared_on_write_then_rebuilt(tmp_path):
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1"))
    store.matrix()
    sig = root / "matrix" / "sig.json"
    assert sig.exists()

    store.put_record(_rec("r2"))  # any write invalidates the packed cache
    assert not sig.exists()

    ids, mat, _ = store.matrix()  # rebuilds with both rows + re-persists
    assert mat.shape[0] == 2
    assert sig.exists()


def test_metas_matches_matrix_metas(tmp_path):
    store, _ = _file_store(tmp_path)
    store.put_record(_rec("r1", text="alpha"))
    store.put_record(_rec("r2", text="beta"))
    ids_m, metas_m = store.metas()  # vector-free path
    ids, _mat, metas = store.matrix()
    assert ids_m == ids
    assert metas_m == metas


def test_corrupt_packed_cache_falls_back_to_rebuild(tmp_path):
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1"))
    store.matrix()
    # A torn / corrupt sig must read as invalid → rebuild, never raise.
    (root / "matrix" / "sig.json").write_text("not json", encoding="utf-8")
    store2, _ = _file_store(tmp_path)
    ids, mat, metas = store2.matrix()
    assert mat.shape[0] == 1


def test_memory_store_keeps_purely_in_process(tmp_path):
    store = CorpusStore.memory()
    assert store._packed_dir is None  # no disk cache for the in-memory store
    store.put_record(_rec("r1"))
    assert store.matrix()[1].shape == (1, 3)
    ids, metas = store.metas()
    assert ids == ["r1"] and len(metas) == 1
