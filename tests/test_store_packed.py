"""Packed-matrix disk cache + vector-free ``metas()`` — the #56 perf paths.

These exercise the file-backed (``packed_dir``) behavior that the in-memory
store in ``test_store.py`` cannot: persisting one normalized matrix so a fresh
process reopens it with a few reads instead of a per-record vector-file storm,
and serving ``metas()`` without touching vectors (for lexical-only ranking).
"""

import json
import os

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


def _packed_file(root, kind):
    """Path of the ``kind`` (matrix/ids/metas) file the current sig points at."""
    packed = root / "matrix"
    sig = json.loads((packed / "sig.json").read_text(encoding="utf-8"))
    gen = sig.get("generation")
    legacy = {"matrix": "matrix.npy", "ids": "ids.json", "metas": "metas.json"}
    if gen is None:
        return packed / legacy[kind]
    ext = "npy" if kind == "matrix" else "json"
    return packed / f"{kind}-{gen}.{ext}"


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


def test_torn_packed_cache_is_rejected_not_served(tmp_path):
    """A matrix from one writer + ids/metas from another must never be served.

    The four packed files are written independently, so two writers rebuilding a
    same-size corpus can leave ``matrix.npy`` from A beside ``ids.json`` /
    ``metas.json`` from B. Every length check still passes, but row *i* no
    longer belongs to ``ids[i]`` — searches would answer confidently wrong.
    """
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    store.put_record(_rec("r2", vec=(0.0, 4.0, 0.0)))
    store.matrix()  # builds from records + persists packed

    # Writer B rewrites ids/metas (same length, different order) after writer A
    # wrote the matrix — the common same-size incremental case.
    ids_path, metas_path = _packed_file(root, "ids"), _packed_file(root, "metas")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    metas = json.loads(metas_path.read_text(encoding="utf-8"))
    ids_path.write_text(json.dumps(ids[::-1]), encoding="utf-8")
    metas_path.write_text(json.dumps(metas[::-1]), encoding="utf-8")

    store2, _ = _file_store(tmp_path)
    ids2, mat2, _metas2 = store2.matrix()
    expected = {"r1": (1.0, 0.0, 0.0), "r2": (0.0, 1.0, 0.0)}
    assert sorted(ids2) == ["r1", "r2"]
    for i, rid in enumerate(ids2):
        np.testing.assert_allclose(np.asarray(mat2[i]), expected[rid], atol=1e-6)


def test_packed_cache_with_mismatched_matrix_shape_is_rejected(tmp_path):
    """A matrix whose columns disagree with the recorded shape is a torn set."""
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    store.put_record(_rec("r2", vec=(0.0, 4.0, 0.0)))
    store.matrix()

    # Writer B's matrix (a different embedding dim) over writer A's sig/ids.
    np.save(_packed_file(root, "matrix"), np.zeros((2, 5), dtype=np.float32))

    store2, _ = _file_store(tmp_path)
    _ids2, mat2, _metas2 = store2.matrix()
    assert mat2.shape == (2, 3)  # rebuilt from records, not the torn cache


def test_packed_cache_without_content_sig_still_loads(tmp_path):
    """A cache written by an older ``ir`` (no signature) stays usable as-is."""
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    ids, _mat, metas = store.matrix()

    sig_path = root / "matrix" / "sig.json"
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    sig.pop("content_sig", None)
    sig.pop("shape", None)
    sig_path.write_text(json.dumps(sig), encoding="utf-8")

    store2, _ = _file_store(tmp_path)

    def _no_rebuild():
        raise AssertionError("legacy packed cache should be used, not rebuilt")

    store2._build_matrix = _no_rebuild
    ids2, _mat2, metas2 = store2.matrix()
    assert ids2 == ids
    assert metas2 == metas


def test_interleaved_packed_writers_never_serve_mismatched_rows(tmp_path, monkeypatch):
    """Writer A saves its matrix, writer B saves a whole set, then A finishes.

    With four independently written files, the disk ends up holding B's matrix
    beside A's ids/metas/sig. A signature over ids/metas alone vouches for that
    mixture (both writers' matrices have the same shape), so rows get served
    under the wrong ids. Whatever the interleaving, a reload must either miss
    (and rebuild) or return one writer's consistent set.
    """
    import ir.store as ir_store

    store_a, root = _file_store(tmp_path)
    store_a.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    store_a.put_record(_rec("r2", vec=(0.0, 4.0, 0.0)))
    result_a = store_a._build_matrix()
    ids_a, mat_a, metas_a = result_a
    # Writer B: the same corpus listed in the other order (a legitimate build).
    order = [ids_a.index(rid) for rid in reversed(ids_a)]
    result_b = (
        [ids_a[i] for i in order],
        np.asarray(mat_a)[order],
        [metas_a[i] for i in order],
    )
    store_b, _ = _file_store(tmp_path)

    real_save = np.save
    state = {"interleaved": False}

    def save_then_let_b_run(*args, **kwargs):
        real_save(*args, **kwargs)
        if not state["interleaved"]:
            state["interleaved"] = True
            store_b._save_packed(result_b)  # B runs start to finish here

    monkeypatch.setattr(ir_store.np, "save", save_then_let_b_run)
    store_a._save_packed(result_a)
    monkeypatch.setattr(ir_store.np, "save", real_save)
    assert state["interleaved"]

    store2, _ = _file_store(tmp_path)
    ids2, mat2, _metas2 = store2.matrix()
    expected = {"r1": (1.0, 0.0, 0.0), "r2": (0.0, 1.0, 0.0)}
    assert sorted(ids2) == ["r1", "r2"]
    for i, rid in enumerate(ids2):
        np.testing.assert_allclose(np.asarray(mat2[i]), expected[rid], atol=1e-6)


def test_legacy_flat_packed_layout_still_loads(tmp_path):
    """A cache in the pre-generation flat layout (``matrix.npy`` ...) is used."""
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    ids, mat, metas = store._build_matrix()
    packed = root / "matrix"
    packed.mkdir(parents=True, exist_ok=True)
    np.save(packed / "matrix.npy", np.asarray(mat, dtype=np.float32))
    (packed / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (packed / "metas.json").write_text(json.dumps(metas), encoding="utf-8")
    (packed / "sig.json").write_text(
        json.dumps({"format": 1, "count": len(ids)}), encoding="utf-8"
    )

    store2, _ = _file_store(tmp_path)

    def _no_rebuild():
        raise AssertionError("legacy packed cache should be used, not rebuilt")

    store2._build_matrix = _no_rebuild
    ids2, _mat2, metas2 = store2.matrix()
    assert ids2 == ids and metas2 == metas


def test_republishing_sweeps_older_generations(tmp_path):
    """Each save leaves exactly one generation (plus sig.json) on disk."""
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1"))
    result = store._build_matrix()
    for _ in range(3):
        store._save_packed(result)
    names = sorted(p.name for p in (root / "matrix").iterdir())
    assert len(names) == 4 and "sig.json" in names

    store.put_record(_rec("r2"))  # a write clears the cache entirely
    assert list((root / "matrix").iterdir()) == []


def test_writer_sees_its_own_write_despite_another_process_publishing(tmp_path):
    """A packed set published by another process must not hide our own writes.

    Writer P1 clears the cache on its first write (and then skips the clear on
    later writes, which is what keeps bulk builds cheap). If reader P2 publishes
    a matrix in between, P1's next write leaves that set on disk, and P1 then
    loaded it -- missing the record it had just written.
    """
    p1, _root = _file_store(tmp_path)
    p1.put_record(_rec("r1", vec=(1.0, 0.0, 0.0)))
    p2, _ = _file_store(tmp_path)
    assert p2.matrix()[0] == ["r1"]  # P2 builds and publishes {r1}
    p1.put_record(_rec("r2", vec=(0.0, 1.0, 0.0)))
    assert sorted(p1.matrix()[0]) == ["r1", "r2"]
    assert sorted(p1.metas()[0]) == ["r1", "r2"]
    # ...and P1's rebuild republished, so a fresh reader now sees both too.
    p3, _ = _file_store(tmp_path)
    assert sorted(p3.matrix()[0]) == ["r1", "r2"]


def test_empty_or_truncated_packed_matrix_is_a_miss_not_an_error(tmp_path):
    """A sig naming a zero-length/truncated ``.npy`` (crash before the data hit
    disk) must rebuild, not raise ``EOFError`` from every ``matrix()`` call."""
    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1", vec=(3.0, 0.0, 0.0)))
    store.put_record(_rec("r2", vec=(0.0, 4.0, 0.0)))
    store.matrix()
    matrix_file = _packed_file(root, "matrix")
    full = matrix_file.read_bytes()
    for corrupt in (b"", full[:20], full[:-4]):
        matrix_file.write_bytes(corrupt)
        store2, _ = _file_store(tmp_path)
        store2._save_packed = lambda result: None  # keep the corrupt set on disk
        ids2, mat2, _metas2 = store2.matrix()
        assert sorted(ids2) == ["r1", "r2"]
        assert np.asarray(mat2).shape == (2, 3)


def test_packed_files_are_fsynced_before_sig_is_published(tmp_path, monkeypatch):
    """Each data file is fsynced before ``sig.json`` names it."""
    import ir.store as ir_store

    store, root = _file_store(tmp_path)
    store.put_record(_rec("r1"))
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(
        ir_store.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))
    )
    monkeypatch.setattr(
        ir_store.os,
        "replace",
        lambda a, b: (events.append("replace"), real_replace(a, b)),
    )
    store.matrix()
    assert "replace" in events
    assert events.index("replace") >= 4  # matrix, ids, metas, sig tmp first
    assert _packed_file(root, "matrix").exists()
