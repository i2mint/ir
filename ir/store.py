"""Persistence for ``ir`` — the repository layer over ``dol`` key-value views.

A :class:`CorpusStore` bundles three ``MutableMapping`` views, so *where* and
*how* data is persisted is swappable without touching the rest of ``ir``:

- ``meta``   : ``record_id -> dict``  (text + metadata + filter fields), JSON.
- ``vectors``: ``record_id -> ndarray`` (the embedding), numpy bytes.
- ``ledger`` : ``artifact_id -> dict`` (version, embedder id, record ids) —
  drives incremental maintenance.
- ``config`` : ``key -> dict`` (one entry: the corpus build settings).

The default factory :meth:`CorpusStore.local` roots all four under
``~/.local/share/ir/corpora/<name>`` via ``dol`` file stores;
:meth:`CorpusStore.memory` gives a dependency-free in-memory store for tests.
Brute-force search reads vectors into a single normalized matrix
(:meth:`matrix`), cached in-process and invalidated on writes.
"""

from __future__ import annotations

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
    ):
        self.meta = meta
        self.vectors = vectors
        self.ledger = ledger
        self.config = config
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
        )

    @classmethod
    def memory(cls) -> "CorpusStore":
        """In-memory store (no dependencies); ideal for tests."""
        return cls(meta={}, vectors={}, ledger={}, config={})

    # ----- record CRUD ---------------------------------------------------- #

    def put_record(self, record: Record) -> None:
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
        self.meta.pop(record_id, None)
        try:
            del self.vectors[record_id]
        except KeyError:
            pass
        self._matrix_cache = None

    def record_ids(self) -> Iterator[str]:
        return iter(self.meta)

    def get_record(self, record_id: str) -> Record:
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
        return len(self.meta)

    # ----- ledger --------------------------------------------------------- #

    def get_ledger_entry(self, key: str) -> dict | None:
        return self.ledger.get(key)

    def set_ledger_entry(self, key: str, entry: Mapping[str, Any]) -> None:
        self.ledger[key] = dict(entry)

    def delete_ledger_entry(self, key: str) -> None:
        self.ledger.pop(key, None)

    def ledger_items(self) -> Iterator[tuple[str, dict]]:
        # Materialize to a list so callers may mutate the ledger while iterating.
        return iter(list(self.ledger.items()))

    # ----- config --------------------------------------------------------- #

    def get_config(self) -> dict:
        return dict(self.config.get("config", {}))

    def set_config(self, settings: Mapping[str, Any]) -> None:
        self.config["config"] = dict(settings)

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
