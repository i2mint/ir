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
import io
import os
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

import numpy as np

from .base import Record


def _ndarray_store(rootdir) -> MutableMapping[str, np.ndarray]:
    """A ``dol`` file store whose values are float32 ``ndarray``s."""
    import dol

    rootdir = str(rootdir)
    os.makedirs(rootdir, exist_ok=True)
    files = dol.mk_dirs_if_missing(dol.Files(rootdir))

    def encode(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr, dtype=np.float32), allow_pickle=False)
        return buf.getvalue()

    def decode(data: bytes) -> np.ndarray:
        return np.load(io.BytesIO(data), allow_pickle=False)

    return dol.wrap_kvs(files, obj_of_data=decode, data_of_obj=encode)


def _json_store(rootdir) -> MutableMapping[str, Any]:
    """A ``dol`` file store whose values are JSON objects."""
    import dol

    rootdir = str(rootdir)
    os.makedirs(rootdir, exist_ok=True)
    return dol.mk_dirs_if_missing(dol.JsonFiles(rootdir))


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
        )

    @classmethod
    def memory(cls) -> "CorpusStore":
        """In-memory store (no dependencies); ideal for tests."""
        return cls(meta={}, vectors={}, ledger={}, config={})

    # ----- record CRUD ---------------------------------------------------- #

    def put_record(self, record: Record) -> None:
        """Persist *record*'s metadata + vector, invalidating the search matrix."""
        self.meta[record.id] = {
            "artifact_id": record.artifact_id,
            "surface_kind": record.surface_kind,
            "surface_index": record.surface_index,
            "text": record.text,
            "metadata": dict(record.metadata),
        }
        self.vectors[record.id] = np.asarray(record.vector, dtype=np.float32)
        self._matrix_cache = None

    def delete_record(self, record_id: str) -> None:
        """Remove a record's metadata + vector; a missing id is tolerated."""
        self.meta.pop(record_id, None)
        try:
            del self.vectors[record_id]
        except KeyError:
            pass
        self._matrix_cache = None

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
        corpora return a ``(0, 0)`` matrix. Cached until the next write.
        """
        if self._matrix_cache is not None:
            return self._matrix_cache
        ids = list(self.meta)
        if not ids:
            self._matrix_cache = ([], np.zeros((0, 0), dtype=np.float32), [])
            return self._matrix_cache
        rows = [np.asarray(self.vectors[rid], dtype=np.float32) for rid in ids]
        mat = np.vstack(rows)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        metas = [self.meta[rid] for rid in ids]
        self._matrix_cache = (ids, mat, metas)
        return self._matrix_cache
