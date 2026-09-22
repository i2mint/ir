"""Persistence for ``ir`` — the repository layer over ``dol`` key-value views.

A :class:`CorpusStore` bundles three ``MutableMapping`` views, so *where* and
*how* data is persisted is swappable without touching the rest of ``ir``:

- ``meta``   : ``record_id -> dict``  (text + metadata + filter fields), JSON.
- ``vectors``: ``record_id -> ndarray`` (the embedding), numpy bytes.
- ``ledger`` : ``artifact_id -> dict`` (version, embedder id, record ids) —
  drives incremental maintenance.
- ``config`` : ``key -> dict`` (one entry: the corpus build settings).
- ``calibration`` : ``mode -> dict`` (a per-ranking-mode calibrated record, today
  the abstention ``min_score`` floor from :func:`ir.eval.calibrate_min_score`).
  Kept apart from ``config`` on purpose — a calibration is regenerable, derived
  from an eval run, and not part of the corpus's build identity, so it must never
  clobber (or be clobbered by) the build settings.
- ``links`` : ``artifact_id -> {edge_type: [target, ...]}`` (the semantic link
  graph — typed directed edges between artifacts; see :mod:`ir.graph`). Like
  ``calibration`` it is regenerable derived state, kept out of build identity; a
  target is a bare ``artifact_id`` (intra-corpus) or a ``[source, artifact_id]``
  pair (cross-corpus). Optional — an absent view is simply "no edges".

The default factory :meth:`CorpusStore.local` roots all six under
``~/.local/share/ir/corpora/<name>`` via ``dol`` file stores;
:meth:`CorpusStore.memory` gives a dependency-free in-memory store for tests.
Brute-force search reads vectors into a single normalized matrix
(:meth:`matrix`), cached in-process and invalidated on writes.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import os
import uuid
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from .base import Record

logger = logging.getLogger(__name__)

#: Bump when the on-disk packed-matrix layout changes so a stale cache from an
#: older ``ir`` is treated as invalid (rebuilt) rather than mis-read.
_PACKED_FORMAT = 1

#: File in the packed-cache directory that every record write replaces with a
#: fresh random token (i2mint/ir#86). A matrix build notes the token *before*
#: listing records and stamps it into ``sig.json``; a packed set is only loaded
#: while the token is unchanged. Any write that landed after the build started
#: therefore turns the set into a miss, however close together the two were: a
#: token comparison has no clock resolution to fall through.
_WRITE_STAMP_FILE = "write-stamp"

#: Default of ``CorpusStore._save_packed(write_stamp=...)``: "the result was
#: built from the corpus as it is now", i.e. stamp it with the current token.
_CURRENT_STAMP = object()


def _write_durably(path: Path, write) -> None:
    """Create ``path``, let ``write(f)`` fill it, then flush and fsync it."""
    with open(path, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    """Fsync a directory so a rename in it is durable (no-op where unsupported).

    POSIX needs this for ``os.replace`` to survive a power loss; Windows cannot
    open a directory this way (and NTFS journals the rename), so it is skipped.
    """
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _packed_content_sig(ids_json: bytes, metas_json: bytes) -> str:
    """Signature binding a packed matrix to the exact ids/metas written with it.

    Hashes the bytes of ``ids.json`` and ``metas.json`` (length-prefixed, so the
    boundary between them is unambiguous). ``sig.json`` is written last and
    carries this digest, so a ``sig``/``matrix`` pair from one writer cannot
    validate another writer's ids/metas: the four packed files are written
    independently, and a same-length mixture of two concurrent rebuilds passes
    every length check while row *i* no longer belongs to ``ids[i]``.

    >>> _packed_content_sig(b'["a"]', b"[{}]") == _packed_content_sig(
    ...     b'["a"]', b"[{}]"
    ... )
    True
    >>> _packed_content_sig(b'["a"]', b"[{}]") == _packed_content_sig(
    ...     b'["b"]', b"[{}]"
    ... )
    False
    """
    digest = hashlib.sha256()
    digest.update(len(ids_json).to_bytes(8, "big"))
    digest.update(ids_json)
    digest.update(metas_json)
    return digest.hexdigest()


#: What reading one per-record file can raise when another process is writing
#: or deleting that record right now: ``KeyError`` (the file vanished between
#: listing and reading), ``EOFError`` (``np.load`` of an empty/truncated
#: ``.npy``), ``ValueError`` (a torn ``.npy`` body, or a torn JSON meta --
#: ``json.JSONDecodeError`` is a ``ValueError``).
_TORN_RECORD_ERRORS = (KeyError, EOFError, ValueError)


class _IncompleteRead(Exception):
    """A matrix build skipped records that vanished or were torn mid-read.

    Carries the partial ``(ids, matrix, metas)`` so :meth:`CorpusStore.matrix`
    can still answer the query that triggered it, without caching or
    publishing a set it knows is missing records.
    """

    def __init__(self, result, skipped):
        super().__init__(f"{len(skipped)} record(s) changed while being read")
        self.result = result
        self.skipped = skipped


#: How often a per-record ``os.replace`` is retried when the target is held
#: open (a Windows reader in another process), and the first back-off delay in
#: seconds (doubled on each retry). POSIX never needs a retry.
_REPLACE_ATTEMPTS = 6
_REPLACE_FIRST_DELAY = 0.01
_IS_WINDOWS = os.name == "nt"


def _write_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so no reader ever sees a partial file.

    The bytes go to a hidden temp file in the same directory (hidden, so the
    ``dol`` store listing that directory never shows it as a key), which then
    replaces ``path`` in one ``os.replace``: a concurrent reader sees the old
    file or the new one, never a torn mix. This is the same publish step as the
    packed cache's ``sig.json``, minus the fsync -- per-record writes are
    atomic against other processes, not durable against power loss, so a bulk
    build does not pay an fsync per record.

    On Windows ``os.replace`` refuses (``PermissionError``) while another
    process has the target open for reading; it is retried with a short
    back-off, and if the target stays busy the bytes are written in place, as
    before this function existed, rather than failing the write.
    """
    import time

    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        delay = _REPLACE_FIRST_DELAY
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if not _IS_WINDOWS:
                    raise
                if attempt < _REPLACE_ATTEMPTS - 1:
                    time.sleep(delay)
                    delay *= 2
        path.write_bytes(data)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass  # already replaced into ``path`` (the normal case)


class _AtomicFiles(MutableMapping):
    """``relative path -> bytes`` file store whose writes are atomic.

    Reads, listing and deletes go through ``dol.Files`` unchanged; only
    ``__setitem__`` differs, publishing through :func:`_write_atomically` (and
    creating missing sub-directories, as ``dol.mk_dirs_if_missing`` did).
    """

    def __init__(self, rootdir):
        import dol

        self.rootdir = str(rootdir)
        os.makedirs(self.rootdir, exist_ok=True)
        self._files = dol.Files(self.rootdir)

    def __getitem__(self, k):
        return self._files[k]

    def __setitem__(self, k, v):
        # The file path comes from ``dol.Files`` itself (root prefix + key, as a
        # string), so a write lands exactly where reads and deletes look. Never
        # ``Path(root, k)``: pathlib lets an absolute key replace the root, which
        # would write outside the store (an artifact id can be an absolute path).
        path = Path(self._files._id_of_key(k))
        root = os.path.normpath(self.rootdir)
        if os.path.commonpath([root, os.path.normpath(path)]) != root:
            raise KeyError(f"key {k!r} would write outside the store at {root}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(path, v)

    def __delitem__(self, k):
        del self._files[k]

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __contains__(self, k):
        return k in self._files


def _ndarray_store(rootdir) -> MutableMapping[str, np.ndarray]:
    """A file store whose values are float32 ``ndarray``s (atomic writes)."""
    import dol

    files = _AtomicFiles(rootdir)

    def encode(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr, dtype=np.float32), allow_pickle=False)
        return buf.getvalue()

    def decode(data: bytes) -> np.ndarray:
        return np.load(io.BytesIO(data), allow_pickle=False)

    return dol.wrap_kvs(files, obj_of_data=decode, data_of_obj=encode)


def _json_store(rootdir) -> MutableMapping[str, Any]:
    """A file store whose values are JSON objects (atomic writes).

    Same on-disk format as ``dol.JsonFiles`` (UTF-8, ``indent=4``), so existing
    corpora read unchanged.
    """
    import dol

    def encode(obj) -> bytes:
        return json.dumps(obj, indent=4).encode("utf-8")

    return dol.wrap_kvs(
        _AtomicFiles(rootdir), obj_of_data=json.loads, data_of_obj=encode
    )


def _normalized_matrix(ids, rows, metas):
    """``(ids, row-L2-normalized matrix, metas)``; a ``(0, 0)`` matrix when empty."""
    if not ids:
        return ([], np.zeros((0, 0), dtype=np.float32), [])
    mat = np.vstack(rows)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (ids, mat / norms, metas)


class CorpusStore:
    """Repository bundling the meta/vectors/ledger/config views of one corpus."""

    def __init__(
        self,
        meta: MutableMapping[str, Any],
        vectors: MutableMapping[str, np.ndarray],
        ledger: MutableMapping[str, Any],
        config: MutableMapping[str, Any],
        calibration: MutableMapping[str, Any] | None = None,
        links: MutableMapping[str, Any] | None = None,
        *,
        packed_dir: str | Path | None = None,
    ):
        self.meta = meta
        self.vectors = vectors
        self.ledger = ledger
        self.config = config
        # Optional 5th/6th views (default to in-memory dicts so older call sites
        # and tests that construct a store with fewer views keep working — and an
        # absent links view simply means "no edges").
        self.calibration = {} if calibration is None else calibration
        self.links = {} if links is None else links
        self._matrix_cache: tuple | None = None
        # Optional on-disk packed-matrix cache (a single normalized matrix + its
        # ids/metas as three files), so reopening a corpus skips the per-record
        # vector-file storm. ``None`` (e.g. for the in-memory store) keeps the
        # matrix purely in-process, preserving the original behavior. The cache
        # is a *write-invalidated read cache*: any record write clears it.
        self._packed_dir = Path(packed_dir) if packed_dir is not None else None
        self._packed_stale = False
        self._packed_dir_ready = False

    # ----- factories ------------------------------------------------------ #

    @classmethod
    def local(cls, name: str) -> "CorpusStore":
        """File-backed store under ``~/.local/share/ir/corpora/<name>``."""
        from .config import corpus_dir

        root = corpus_dir(name)
        return cls(
            meta=_json_store(root / "meta"),
            vectors=_ndarray_store(root / "vectors"),
            ledger=_json_store(root / "ledger"),
            config=_json_store(root / "config"),
            calibration=_json_store(root / "calibration"),
            links=_json_store(root / "links"),
            packed_dir=root / "matrix",
        )

    @classmethod
    def memory(cls) -> "CorpusStore":
        """In-memory store (no dependencies); ideal for tests."""
        return cls(meta={}, vectors={}, ledger={}, config={})

    # ----- record CRUD ---------------------------------------------------- #

    def put_record(self, record: Record) -> None:
        """Persist *record*'s metadata + vector, invalidating the search matrix.

        The vector is written **before** the meta: record ids are listed from
        the meta view, so a reader in another process that lists an id always
        finds its vector (``delete_record`` removes in the reverse order).
        """
        self.vectors[record.id] = np.asarray(record.vector, dtype=np.float32)
        self.meta[record.id] = {
            "artifact_id": record.artifact_id,
            "surface_kind": record.surface_kind,
            "surface_index": record.surface_index,
            "text": record.text,
            "metadata": dict(record.metadata),
        }
        self._invalidate_matrix()

    def delete_record(self, record_id: str) -> None:
        """Remove a record's metadata + vector; a missing id is tolerated.

        The meta goes first, so the id stops being listed before its vector
        disappears. Each removal tolerates the file being gone already (another
        process may be deleting the same record).
        """
        for view in (self.meta, self.vectors):
            try:
                del view[record_id]
            except KeyError:
                pass
        self._invalidate_matrix()

    def record_ids(self) -> Iterator[str]:
        """Iterate the record ids currently stored."""
        return iter(self.meta)

    def get_record(self, record_id: str) -> Record:
        """Reassemble the :class:`~ir.base.Record` for *record_id* (``KeyError`` if absent)."""
        m = self.meta[record_id]
        return Record(
            id=record_id,
            artifact_id=m["artifact_id"],
            surface_kind=m["surface_kind"],
            surface_index=m["surface_index"],
            text=m["text"],
            vector=np.asarray(self.vectors[record_id], dtype=np.float32),
            metadata=m.get("metadata", {}),
        )

    def __len__(self) -> int:
        """The number of records stored."""
        return len(self.meta)

    # ----- ledger --------------------------------------------------------- #

    def get_ledger_entry(self, key: str) -> dict | None:
        """The ledger entry for *key* (``None`` if absent)."""
        return self.ledger.get(key)

    def set_ledger_entry(self, key: str, entry: Mapping[str, Any]) -> None:
        """Write the ledger *entry* (version / embedder id / record ids) for *key*."""
        self.ledger[key] = dict(entry)

    def delete_ledger_entry(self, key: str) -> None:
        """Remove a ledger entry; a missing key is tolerated."""
        self.ledger.pop(key, None)

    def ledger_items(self) -> Iterator[tuple[str, dict]]:
        """Iterate ``(key, entry)`` ledger pairs (the ledger may be mutated while iterating)."""
        # Materialize to a list so callers may mutate the ledger while iterating.
        return iter(list(self.ledger.items()))

    # ----- config --------------------------------------------------------- #

    def get_config(self) -> dict:
        """The persisted corpus build settings (empty dict if never written)."""
        return dict(self.config.get("config", {}))

    def set_config(self, settings: Mapping[str, Any]) -> None:
        """Persist the corpus build *settings* (name / embedder spec + id)."""
        self.config["config"] = dict(settings)

    def get_maintenance_state(self) -> dict:
        """Background-work bookkeeping (e.g. ``last_maintained``); ``{}`` if unset.

        Kept under a separate ``config``-view key from the build settings: it is
        regenerable scheduler state (when ``ir maintain`` last ran), not part of
        the corpus's build identity, so it must never clobber it.
        """
        return dict(self.config.get("maintenance", {}))

    def set_maintenance_state(self, state: Mapping[str, Any]) -> None:
        """Persist the maintenance bookkeeping for this corpus."""
        self.config["maintenance"] = dict(state)

    # ----- calibration (per-mode) ----------------------------------------- #

    def get_calibration(self, mode: str) -> dict | None:
        """The stored calibration record for ranking ``mode`` (``None`` if absent).

        A deep copy, so a caller cannot mutate the nested ``grid`` back into the
        stored record (in-memory stores share their objects by reference).
        """
        rec = self.calibration.get(mode)
        return copy.deepcopy(rec) if rec is not None else None

    def set_calibration(self, mode: str, record: Mapping[str, Any]) -> None:
        """Persist a calibration ``record`` for ranking ``mode`` (one per mode).

        ``mode`` keys a file in the calibration store, so it must be a non-empty
        string with no path separator (the real modes — ``dense`` / ``lexical`` /
        ``hybrid`` — already satisfy this).
        """
        if not mode or "/" in mode or "\\" in mode:
            raise ValueError(
                f"calibration mode must be a non-empty string without a path "
                f"separator; got {mode!r}"
            )
        self.calibration[mode] = dict(record)

    def calibration_modes(self) -> list[str]:
        """The ranking modes that currently have a stored calibration."""
        return list(self.calibration)

    # ----- links (semantic edge graph) ------------------------------------ #

    def get_links(self, artifact_id: str) -> dict:
        """The outgoing edges of *artifact_id* — ``{edge_type: [target, ...]}``.

        Empty dict when the artifact has no stored edges (or no links view).
        A copy, so a caller cannot mutate the persisted adjacency in place.
        """
        return copy.deepcopy(self.links.get(artifact_id, {}))

    def set_links(self, artifact_id: str, edges: Mapping[str, Any]) -> None:
        """Persist *artifact_id*'s outgoing *edges* (``{edge_type: [target]}``).

        Empty edge-type lists are dropped; an empty result deletes the entry
        (no empty adjacency rows linger). Targets are stored verbatim — a bare
        ``artifact_id`` or a ``[source, artifact_id]`` pair.
        """
        cleaned = {et: list(ts) for et, ts in edges.items() if ts}
        if cleaned:
            self.links[artifact_id] = cleaned
        else:
            self.links.pop(artifact_id, None)

    def delete_links(self, artifact_id: str) -> None:
        """Remove an artifact's edges; a missing entry is tolerated."""
        self.links.pop(artifact_id, None)

    def link_items(self) -> Iterator[tuple[str, dict]]:
        """Iterate ``(artifact_id, {edge_type: [target]})`` adjacency pairs."""
        return iter(list(self.links.items()))

    # ----- search matrix -------------------------------------------------- #

    def matrix(self) -> tuple[list[str], np.ndarray, list[dict]]:
        """Return ``(record_ids, normalized_matrix, metas)`` for brute force.

        Rows are L2-normalized so cosine similarity is a dot product. Empty
        corpora return a ``(0, 0)`` matrix.

        Caching is two-tier: an in-process cache (invalidated on the next write)
        backed, for file-rooted stores, by an on-disk **packed** cache — one
        normalized-matrix ``.npy`` plus its ids/metas, written once and reloaded
        with a single memory-mapped read. The packed cache turns a cold reopen
        from a per-record vector-file storm (thousands of tiny reads) into three
        file reads; it is cleared by a writer's first record write, and every
        record write replaces a *write stamp* that a packed set must match to be
        published or loaded, so a set built before any later write -- by this
        process or another -- is never served (i2mint/ir#86).

        Another process may be writing or deleting records while this one
        rebuilds. A record that vanishes or is only half-written when read is
        left out of the result (it is "not yet written"), with a warning; such
        a partial result is cached in-process but never published as the
        packed set, so a fresh process reads the records again.
        """
        if self._matrix_cache is not None:
            return self._matrix_cache
        packed = self._load_packed()
        if packed is not None:
            self._matrix_cache = packed
            return packed
        # Noted before listing, so any write the build might have missed
        # changes it (put/delete replace it after their record files; ir#86).
        write_stamp = self._read_write_stamp()
        try:
            result = self._build_matrix()
        except _IncompleteRead as incomplete:
            # Kept in-process like any build (the in-process cache never sees
            # other processes' writes anyway), but not published: a record that
            # can't be read -- torn, or damaged for good -- must not become part
            # of the set every other process loads.
            self._matrix_cache = incomplete.result
            return incomplete.result
        self._save_packed(result, write_stamp=write_stamp)
        self._matrix_cache = result
        return result

    def metas(self) -> tuple[list[str], list[dict]]:
        """Return ``(record_ids, metas)`` **without** loading any vectors.

        The vector-free counterpart of :meth:`matrix`, for ranking modes that
        score on text alone (``mode="lexical"``): they need candidate metadata
        (text + filter fields) but never the embedding matrix, so they must not
        pay its I/O. Reuses the in-process or packed cache when present; else
        reads only the ``meta`` view (not ``vectors``), skipping a record that
        vanishes or is half-written mid-read (as :meth:`matrix` does).
        """
        if self._matrix_cache is not None:
            ids, _mat, metas = self._matrix_cache
            return ids, metas
        packed = self._load_packed()
        if packed is not None:
            self._matrix_cache = packed
            return packed[0], packed[2]
        ids: list[str] = []
        metas: list[dict] = []
        for rid in list(self.meta):
            try:
                meta = self.meta[rid]
            except _TORN_RECORD_ERRORS:
                logger.debug("ir: skipped unreadable meta for record %s", rid)
                continue
            ids.append(rid)
            metas.append(meta)
        return ids, metas

    def _build_matrix(self) -> tuple[list[str], np.ndarray, list[dict]]:
        """Build ``(ids, normalized_matrix, metas)`` from the per-record stores.

        One pass over the ids reads each record's meta and vector together
        (the previous implementation iterated the meta view three times).

        Raises :class:`_IncompleteRead` (carrying the partial result) when a
        listed record vanished or could not be decoded because another process
        was writing or deleting it (see ``_TORN_RECORD_ERRORS``).
        """
        ids: list[str] = []
        metas: list[dict] = []
        rows: list[np.ndarray] = []

        def read(rid):
            meta = self.meta[rid]
            return meta, np.asarray(self.vectors[rid], dtype=np.float32)

        def add(rid, meta, row):
            ids.append(rid)
            metas.append(meta)
            rows.append(row)

        retry: list[str] = []
        for rid in list(self.meta):
            try:
                add(rid, *read(rid))
            except _TORN_RECORD_ERRORS:
                retry.append(rid)
        # Second look at what failed: a writer has usually finished by now, and a
        # record whose meta is gone was deleted -- consistent with the rest of
        # the read, so not a gap. Only what still can't be read is missing.
        skipped: list[str] = []
        for rid in retry:
            if rid not in self.meta:
                continue
            try:
                add(rid, *read(rid))
            except _TORN_RECORD_ERRORS:
                skipped.append(rid)
        result = _normalized_matrix(ids, rows, metas)
        if skipped:
            logger.warning(
                "ir: %d record(s) could not be read (being written by another "
                "process, or damaged) and were left out of this search: %s. If "
                "this persists, re-index or delete those records; until then the "
                "packed matrix cache is not written for this corpus.",
                len(skipped),
                ", ".join(skipped[:5]) + (", ..." if len(skipped) > 5 else ""),
            )
            raise _IncompleteRead(result, skipped)
        return result

    # ----- packed-matrix disk cache --------------------------------------- #

    def _invalidate_matrix(self) -> None:
        """Drop the in-process matrix, stamp the write, clear the packed cache once.

        Called on every record write, *after* the record's files are written.
        The on-disk clear happens at most once per rebuild (guarded by
        ``_packed_stale``) so a bulk build's thousands of ``put_record`` calls
        don't each sweep the cache directory. The write stamp is replaced on
        every call: it is what stops another process from publishing, or
        loading, a packed set built before this write (i2mint/ir#86) -- the
        once-per-session clear cannot, since that process may publish after it.
        """
        self._matrix_cache = None
        if self._packed_dir is None:
            return
        self._stamp_write()
        if not self._packed_stale:
            self._clear_packed()
            self._packed_stale = True

    def _write_stamp_path(self) -> Path:
        return self._packed_dir / _WRITE_STAMP_FILE

    def _stamp_write(self) -> None:
        """Replace the write stamp with a fresh token (atomically; one small file).

        If the stamp can't be written, it is removed instead (a value no build
        in flight can hold, unless it began before any stamped write), and the
        packed set with it (``sig.json`` first).
        """
        try:
            if not self._packed_dir_ready:
                self._packed_dir.mkdir(parents=True, exist_ok=True)
                self._packed_dir_ready = True
            _write_atomically(
                self._write_stamp_path(), uuid.uuid4().hex.encode("ascii")
            )
        except OSError as error:
            self._packed_dir_ready = False
            logger.warning(
                "ir: could not update the packed-cache write stamp (%s); "
                "dropping the stamp and the packed cache instead",
                error,
            )
            try:
                self._write_stamp_path().unlink()
            except OSError:
                pass
            self._clear_packed()

    def _read_write_stamp(self) -> str | None:
        """The current write stamp, or ``None`` if no stamped write happened yet."""
        if self._packed_dir is None:
            return None
        try:
            stamp = self._write_stamp_path().read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            stamp = ""
        if not stamp:
            # Unreadable, or empty mid-rewrite (the Windows in-place fallback of
            # ``_write_atomically``): a fresh value that equals nothing, so a
            # load misses and a build does not publish.
            return f"unreadable-{uuid.uuid4().hex}"
        return stamp

    # Legacy (pre-generation) flat file names. A cache written by an older
    # ``ir`` is still read through these; new writes never use them.
    _LEGACY_PACKED_FILES = {
        "matrix": "matrix.npy",
        "ids": "ids.json",
        "metas": "metas.json",
    }

    def _packed_paths(self, generation: str | None = None):
        """Paths of one packed set: the shared ``sig.json`` plus its data files.

        Each writer puts its data files under a fresh *generation* token
        (``matrix-<gen>.npy``, ...), so no data file is ever written by two
        writers, and ``sig.json`` -- replaced atomically, last -- names the one
        generation readers should load. ``generation=None`` gives the legacy
        flat names an older ``ir`` wrote.
        """
        d = self._packed_dir
        if generation is None:
            files = self._LEGACY_PACKED_FILES
        else:
            files = {
                "matrix": f"matrix-{generation}.npy",
                "ids": f"ids-{generation}.json",
                "metas": f"metas-{generation}.json",
            }
        return {"sig": d / "sig.json", **{k: d / v for k, v in files.items()}}

    def _packed_data_files(self) -> list[Path]:
        """Every packed data file on disk (legacy names and all generations)."""
        d = self._packed_dir
        found = [d / name for name in self._LEGACY_PACKED_FILES.values()]
        for pattern in ("matrix-*.npy", "ids-*.json", "metas-*.json", "sig-*.tmp"):
            found.extend(d.glob(pattern))
        return found

    def _clear_packed(self, *, keep: str | None = None) -> None:
        """Remove the packed cache (``sig.json`` first, so it reads as invalid).

        With ``keep``, only ``sig.json`` is kept and only that generation's data
        files survive: this is the post-publish sweep of older generations.
        Best-effort -- a file another process still has mapped may refuse to go
        (Windows); it is then just left for a later sweep.
        """
        if self._packed_dir is None:
            return
        if keep is None:
            try:
                self._packed_paths()["sig"].unlink()
            except OSError:
                pass
        if not self._packed_dir.is_dir():
            return
        kept: set[Path] = set()
        if keep is not None:
            kept.update(self._packed_paths(keep).values())
            try:  # another writer may have published after us: keep its set too
                current = json.loads(self._packed_paths()["sig"].read_text("utf-8"))
                if isinstance(current.get("generation"), str):
                    kept.update(self._packed_paths(current["generation"]).values())
            except (OSError, ValueError):
                pass
        for path in self._packed_data_files():
            if path in kept:
                continue
            try:
                path.unlink()
            except OSError:
                pass

    def _load_packed(self):
        """Load ``(ids, mmap_matrix, metas)`` from the packed cache, or ``None``."""
        if self._packed_dir is None:
            return None
        if self._packed_stale:
            # This process wrote records since it last published a matrix. A
            # packed set on disk now is either absent (we cleared it) or was
            # published by another process that may have built before our
            # writes, so it cannot be trusted to include them: rebuild.
            return None
        sig_path = self._packed_paths()["sig"]
        if not sig_path.exists():
            return None
        try:
            sig = json.loads(sig_path.read_text(encoding="utf-8"))
            if sig.get("format") != _PACKED_FORMAT:
                return None
            generation = sig.get("generation")
            if generation is not None and not (
                isinstance(generation, str) and generation.isalnum()
            ):
                return None
            paths = self._packed_paths(generation)
            mat = np.load(paths["matrix"], mmap_mode="r")
            ids_json = paths["ids"].read_bytes()
            metas_json = paths["metas"].read_bytes()
            ids = json.loads(ids_json)
            metas = json.loads(metas_json)
        except (OSError, ValueError, EOFError):
            # EOFError: np.load of an empty/truncated ``.npy`` (e.g. a crash
            # before the data reached disk). Any unreadable set is a miss.
            return None
        if len(ids) != mat.shape[0] or len(metas) != len(ids):
            return None
        # A cache written before ``sig.json`` carried these fields has neither;
        # keep accepting it on the length checks alone rather than invalidating
        # every existing cache. When they are present they must agree.
        content_sig = sig.get("content_sig")
        if content_sig is not None and content_sig != _packed_content_sig(
            ids_json, metas_json
        ):
            return None
        shape = sig.get("shape")
        if shape is not None and list(mat.shape) != list(shape):
            return None
        # Checked last, after the data files are read: a set built before the
        # latest record write may be missing it (i2mint/ir#86). A sig from an
        # ``ir`` predating the stamp has none, which matches only a corpus no
        # stamping ``ir`` has written to since.
        if sig.get("write_stamp") != self._read_write_stamp():
            return None
        return (ids, mat, metas)

    def _save_packed(
        self,
        result: tuple[list[str], np.ndarray, list[dict]],
        *,
        write_stamp: str | None | object = _CURRENT_STAMP,
    ) -> None:
        """Persist a freshly built matrix to the packed cache (best-effort).

        ``write_stamp`` is the write stamp read *before* the build listed its
        records. It goes into ``sig.json``, and a set whose stamp is no longer
        current is not published at all: a record was written or deleted
        while it was being built, so it may not reflect that write
        (i2mint/ir#86). Loading checks the same stamp again, which covers a
        write that lands between this check and the publish. Left out, the
        current stamp is used: the caller vouches the result is up to date.

        Skips empty corpora. The matrix, ids and metas go to files named by a
        fresh generation token that only this call writes, and ``sig.json``
        (naming that generation, plus a :func:`_packed_content_sig` digest and
        the matrix shape) is published last with an atomic ``os.replace``. So a
        crash mid-write leaves the previous sig (or none) in charge, and two
        concurrent writers can never leave one writer's matrix beside the
        other's ids: a reader follows ``sig.json`` to exactly one writer's
        complete set. Older generations are swept after publishing. Every data
        file is fsynced before the sig that names it is published (and the
        directory after), so a power loss cannot leave a durable ``sig.json``
        pointing at data blocks that never reached the disk.
        """
        if self._packed_dir is None:
            return
        ids, mat, metas = result
        if not ids:
            return
        if write_stamp is _CURRENT_STAMP:
            write_stamp = self._read_write_stamp()
        generation = uuid.uuid4().hex
        try:
            self._packed_dir.mkdir(parents=True, exist_ok=True)
            paths = self._packed_paths(generation)
            arr = np.asarray(mat, dtype=np.float32)
            ids_json = json.dumps(ids).encode("utf-8")
            metas_json = json.dumps(metas).encode("utf-8")
            _write_durably(paths["matrix"], lambda f: np.save(f, arr))
            _write_durably(paths["ids"], lambda f: f.write(ids_json))
            _write_durably(paths["metas"], lambda f: f.write(metas_json))
            sig_json = json.dumps(
                {
                    "format": _PACKED_FORMAT,
                    "count": len(ids),
                    "generation": generation,
                    "content_sig": _packed_content_sig(ids_json, metas_json),
                    "shape": list(arr.shape),
                    "write_stamp": write_stamp,
                }
            ).encode("utf-8")
            sig_tmp = self._packed_dir / f"sig-{generation}.tmp"
            _write_durably(sig_tmp, lambda f: f.write(sig_json))
            if self._read_write_stamp() != write_stamp:
                # Written to while we built: don't replace a possibly-current
                # set with one that may be missing that write.
                for path in (sig_tmp, paths["matrix"], paths["ids"], paths["metas"]):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                return
            os.replace(sig_tmp, paths["sig"])
            _fsync_dir(self._packed_dir)
            self._packed_stale = False
        except OSError:
            # A read cache that can't be written is non-fatal: fall back to the
            # in-process cache (already set by the caller) for this process.
            self._clear_packed()
            return
        self._clear_packed(keep=generation)
