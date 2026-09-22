"""Reading a file-backed corpus while another process writes it (i2mint/ir#85).

``matrix()``/``metas()`` rebuild from the per-record ``meta``/``vectors`` files
whenever the packed cache is missing, which is exactly while a writer is adding
records (its first write clears the cache). These tests pin down that such a
read never raises: a record is written atomically and vector-first, a record
that vanishes mid-read is treated as deleted, and a record that still can't be
read is left out of that one answer -- which is then neither cached nor
published as the packed set.
"""

import json
import logging
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import ir.store as ir_store
from ir.base import Record
from ir.store import CorpusStore, _json_store, _ndarray_store


def _rec(rid, *, vec=(1.0, 0.0, 0.0)):
    return Record(
        id=rid,
        artifact_id=f"art_{rid}",
        surface_kind="document",
        surface_index=0,
        text=f"text of {rid}",
        vector=np.asarray(vec, dtype=np.float32),
        metadata={},
    )


def _file_store(root):
    return CorpusStore(
        meta=_json_store(root / "meta"),
        vectors=_ndarray_store(root / "vectors"),
        ledger=_json_store(root / "ledger"),
        config=_json_store(root / "config"),
        packed_dir=root / "matrix",
    )


def test_put_record_writes_vector_before_meta():
    """A reader lists ids from ``meta``, so a listed id must already have a vector."""
    order = []

    class Recording(dict):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __setitem__(self, k, v):
            order.append(self.name)
            super().__setitem__(k, v)

    store = CorpusStore(
        meta=Recording("meta"), vectors=Recording("vectors"), ledger={}, config={}
    )
    store.put_record(_rec("r1"))
    assert order == ["vectors", "meta"]


def test_record_with_meta_but_no_vector_is_left_out_not_published(tmp_path, caplog):
    root = tmp_path / "corpus"
    store = _file_store(root)
    store.put_record(_rec("r1"))
    store.put_record(_rec("r2", vec=(0.0, 1.0, 0.0)))
    (root / "vectors" / "r2").unlink()  # meta without vector: unreadable record

    reader = _file_store(root)
    with caplog.at_level(logging.WARNING, logger="ir.store"):
        ids, mat, metas = reader.matrix()
    assert ids == ["r1"] and mat.shape == (1, 3) and len(metas) == 1
    assert "r2" in caplog.text
    assert not (root / "matrix" / "sig.json").exists()  # partial set not published
    assert _file_store(root).matrix()[0] == ["r1"]  # a fresh process reads again

    store.put_record(_rec("r2", vec=(0.0, 1.0, 0.0)))  # the writer finishes
    ids, _mat, _metas = _file_store(root).matrix()
    assert sorted(ids) == ["r1", "r2"]
    assert (root / "matrix" / "sig.json").exists()


@pytest.mark.parametrize("kind", ["meta", "vectors"])
def test_torn_record_file_is_skipped_not_raised(tmp_path, kind):
    """An empty or truncated file (a pre-atomic writer, a crash) is a miss."""
    root = tmp_path / "corpus"
    store = _file_store(root)
    store.put_record(_rec("r1"))
    store.put_record(_rec("r2"))
    path = root / kind / "r2"
    path.write_bytes(path.read_bytes()[:7])

    ids, _mat, metas = _file_store(root).matrix()
    assert ids == ["r1"] and len(metas) == 1
    if kind == "meta":
        assert _file_store(root).metas()[0] == ["r1"]


def test_record_deleted_mid_read_is_not_a_gap():
    """An id listed but gone by the time it is read was deleted: the read is whole."""

    class VanishingMeta(dict):
        def __iter__(self):
            return iter([*super().__iter__(), "gone"])

    store = CorpusStore(meta=VanishingMeta(), vectors={}, ledger={}, config={})
    store.put_record(_rec("r1"))
    ids, _mat, _metas = store.matrix()
    assert ids == ["r1"]
    assert store._matrix_cache is not None  # complete, so cached
    assert store.metas()[0] == ["r1"]


def test_writes_leave_no_temp_files_and_temp_files_are_not_keys(tmp_path):
    root = tmp_path / "corpus"
    store = _file_store(root)
    store.put_record(_rec("r1"))
    store.put_record(_rec("r1", vec=(0.0, 1.0, 0.0)))  # overwrite
    for kind in ("meta", "vectors"):
        assert sorted(p.name for p in (root / kind).iterdir()) == ["r1"]
    # A temp file left by a writer that crashed before its replace is invisible.
    (root / "meta" / ".r2.deadbeef.tmp").write_bytes(b"{")
    assert list(store.meta) == ["r1"]


def test_absolute_key_is_written_inside_the_root(tmp_path):
    """An artifact id can be an absolute path: it must never replace the root."""
    victim = tmp_path / "victim.md"
    victim.write_text("# my source file")
    root = tmp_path / "store"
    store = _json_store(root)
    store[str(victim)] = {"cites": ["x"]}
    assert victim.read_text() == "# my source file"
    assert store[str(victim)] == {"cites": ["x"]}
    assert [p for p in root.rglob("*") if p.is_file()]  # stored under the root


def test_key_climbing_out_of_the_root_is_refused(tmp_path):
    store = _json_store(tmp_path / "store")
    with pytest.raises(KeyError, match="outside the store"):
        store["a/../../escape"] = {}
    assert not (tmp_path / "escape").exists()


def test_json_store_format_is_unchanged(tmp_path):
    """Corpora written by ``dol.JsonFiles`` (indent=4, UTF-8) read the same."""
    import dol

    old = dol.JsonFiles(str(tmp_path) + "/")
    old["k"] = {"text": "café", "n": [1, 2]}
    new = _json_store(tmp_path)
    assert new["k"] == {"text": "café", "n": [1, 2]}
    raw = (tmp_path / "k").read_bytes()
    new["k"] = {"text": "café", "n": [1, 2]}
    assert json.loads((tmp_path / "k").read_bytes()) == json.loads(raw)


def test_windows_busy_target_retries_then_writes_in_place(tmp_path, monkeypatch):
    """On Windows a reader holding the file makes ``os.replace`` refuse; don't fail."""
    calls = []

    def busy_replace(src, dst):
        calls.append(dst)
        raise PermissionError("target is open in another process")

    monkeypatch.setattr(ir_store, "_IS_WINDOWS", True)
    monkeypatch.setattr(ir_store, "_REPLACE_FIRST_DELAY", 0)
    monkeypatch.setattr(ir_store.os, "replace", busy_replace)
    target = tmp_path / "f"
    ir_store._write_atomically(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert len(calls) == ir_store._REPLACE_ATTEMPTS
    assert [p.name for p in tmp_path.iterdir()] == ["f"]  # temp file cleaned up


def test_permission_error_off_windows_is_raised(tmp_path, monkeypatch):
    def denied(src, dst):
        raise PermissionError("denied")

    monkeypatch.setattr(ir_store, "_IS_WINDOWS", False)
    monkeypatch.setattr(ir_store.os, "replace", denied)
    with pytest.raises(PermissionError):
        ir_store._write_atomically(tmp_path / "f", b"x")
    assert list(tmp_path.iterdir()) == []


# Large records (~80 KB vector, up to ~200 KB of text) keep each write in
# flight long enough that a reader on the old, non-atomic stores reliably hit a
# torn or half-listed record.
_DIM = 20_000

_WRITER = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import numpy as np
    from ir.base import Record
    from ir.store import CorpusStore, _json_store, _ndarray_store

    root = Path(sys.argv[1])
    DIM = int(sys.argv[3])
    store = CorpusStore(
        meta=_json_store(root / "meta"),
        vectors=_ndarray_store(root / "vectors"),
        ledger=_json_store(root / "ledger"),
        config=_json_store(root / "config"),
        packed_dir=root / "matrix",
    )
    first = int(sys.argv[4])  # this writer's ids: r<first> .. r<first + 19>
    rng = np.random.default_rng(first)
    for i in range(int(sys.argv[2])):
        rid = f"r{first + i % 20}"  # new records, then overwrites of existing ones
        store.put_record(Record(
            id=rid, artifact_id="a", surface_kind="document", surface_index=0,
            text="x" * int(rng.integers(1, 200_000)), metadata={},
            vector=rng.random(DIM).astype(np.float32),
        ))
    """
)


@pytest.mark.parametrize("round_", range(3))  # each round alone caught master ~1 in 2
def test_reads_while_another_process_writes_never_raise(tmp_path, round_):
    """The #85 repro: query a corpus while other processes build it."""
    root = tmp_path / "corpus"
    writers = [  # two writers on disjoint ids, as when a build and a maintain overlap
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(root), "150", str(_DIM), str(first)],
            stderr=subprocess.PIPE,
        )
        for first in (0, 20)
    ]
    reads = 0
    try:
        while any(w.poll() is None for w in writers) or reads == 0:
            reader = _file_store(root)
            ids, mat, metas = reader.matrix()
            assert len(ids) == mat.shape[0] == len(metas)
            if len(ids):
                np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1, atol=1e-5)
            assert len(set(_file_store(root).metas()[0])) <= 40
            reads += 1
    finally:
        errs = [w.communicate(timeout=120)[1] for w in writers]
    for w, err in zip(writers, errs, strict=True):
        assert w.returncode == 0, err.decode()
    assert reads > 0
    # Every record the writer wrote is whole on disk. (Whether a *packed* set a
    # reader published mid-build includes them is i2mint/ir#86, below.)
    store = _file_store(root)
    assert sorted(store.meta) == sorted(f"r{i}" for i in range(40))
    for rid in store.meta:
        assert store.get_record(rid).vector.shape == (_DIM,)


@pytest.mark.xfail(
    strict=True,
    reason="i2mint/ir#86: a packed set published mid-build hides later writes",
)
def test_reader_publishing_mid_build_does_not_hide_later_writes(tmp_path):
    root = tmp_path / "corpus"
    writer = _file_store(root)
    writer.put_record(_rec("r1"))
    reader = _file_store(root)
    real_build = reader._build_matrix

    def build_then_writer_continues():
        result = real_build()  # the reader's listing predates r2
        writer.put_record(_rec("r2", vec=(0.0, 1.0, 0.0)))
        return result

    reader._build_matrix = build_then_writer_continues
    reader.matrix()  # publishes a set without r2
    ids, _mat, _metas = _file_store(root).matrix()
    assert sorted(ids) == ["r1", "r2"]
