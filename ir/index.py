"""The indexing pipeline and incremental maintenance.

:func:`build` turns a :class:`~ir.sources.CorpusSource` into a queryable
:class:`Corpus`, persisting records through a
:class:`~ir.store.CorpusStore`. It is **incremental and idempotent**: each
artifact's change signal is compared to the ledger, and only new, changed, or
re-modeled artifacts are decomposed and embedded. Artifacts that vanished from
the source are pruned (full-refresh). Re-running ``build`` on an unchanged
source is a near-no-op.

This is the light maintenance path — content-hash CRUD over a flat store, which
is the right tool for the small corpora ``ir`` targets. The heavier
content-addressed artifact-graph (``ef.artifact_graph``) is the documented
upgrade for corpora large enough that recomputing surfaces is the bottleneck.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import Record, ledger_key
from .embed import make_embedder
from .sources import CorpusSource
from .store import CorpusStore


def _strategy_id(strategy) -> str:
    """Stable id for a strategy: class name + its simple parameters.

    Changing the strategy (or its parameters) changes this id, so an unchanged
    corpus rebuilt under a different strategy is correctly re-decomposed rather
    than skipped.
    """
    params = {
        k: v
        for k, v in vars(strategy).items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }
    return f"{type(strategy).__name__}:{json.dumps(params, sort_keys=True)}"


def _embed(emb: Callable, texts: list[str], input_type: str) -> np.ndarray:
    """Call an embedder, tolerating those that don't accept ``input_type``."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        out = emb(texts, input_type=input_type)
    except TypeError:
        out = emb(texts)
    return np.asarray(out, dtype=np.float32)


def _embed_batched(emb, texts, input_type, batch_size):
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(_embed(emb, texts[i : i + batch_size], input_type))
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


@dataclass
class Corpus:
    """A built, queryable corpus: a store plus its embedder."""

    name: str
    store: CorpusStore
    embedder: Callable
    embedder_id: str

    def search(self, query, **kwargs):
        """Search this corpus for *query*.

        ``**kwargs`` (``k`` / ``mode`` / ``filter`` / ``surfaces`` /
        ``per_artifact`` / ...) are forwarded to :func:`ir.retrieve.search`.
        """
        from .retrieve import search

        return search(self, query, **kwargs)

    def __len__(self) -> int:
        return len(self.store)


def build(
    source: CorpusSource,
    *,
    store: CorpusStore | None = None,
    embedder: Any = None,
    full: bool = True,
    batch_size: int = 256,
    edge_extractor: Callable | None = None,
) -> Corpus:
    """Build or incrementally update *source* into a :class:`Corpus`.

    Parameters
    ----------
    store : the persistence backend (default: file-backed under XDG data dir).
    embedder : override the source's embedder spec.
    full : when True (default), prune artifacts no longer in the source.
    batch_size : embedding batch size.
    edge_extractor : an optional :data:`~ir.graph.EdgeExtractor`
        (``(artifact_id, filter_fields) -> {edge_type: [target]}``) that
        populates the corpus's semantic ``links`` graph (see :mod:`ir.graph`;
        pass :func:`ir.default_edge_extractor` for the latent deps/parent
        edges). Ingest is **eager** — edges are (re)written for *every*
        in-scope artifact, a decompose-only pass with no embedding, so the
        graph never goes partially stale — while embedding stays fully
        incremental. Edges are derived state, **not** part of build identity.
        A rebuild *without* an extractor leaves existing edges untouched
        (they are only refreshed by re-running with one, and only cleared per
        artifact by the ``full`` prune below) — so dropping ``edge_extractor``
        does not wipe a graph.
    """
    store = CorpusStore.local(source.name) if store is None else store
    spec = embedder if embedder is not None else source.embedder
    emb, emb_id = make_embedder(spec)
    strat_id = _strategy_id(source.indexing_strategy)

    seen: set[str] = set()
    changed: dict[str, tuple] = {}

    for artifact_id, raw in source.items():
        seen.add(artifact_id)
        version = source.change_signal(artifact_id, raw)
        key = ledger_key(artifact_id)
        prev = store.get_ledger_entry(key)
        unchanged = (
            prev
            and prev.get("version") == version
            and prev.get("embedder_id") == emb_id
            and prev.get("strategy_id") == strat_id
        )
        # Unchanged + no edges to refresh → nothing to do for this artifact.
        if unchanged and edge_extractor is None:
            continue
        meta_extra = source.metadata_of(artifact_id, raw) if source.metadata_of else {}
        plan = source.indexing_strategy.decompose(artifact_id, raw, meta_extra)
        if not unchanged:
            changed[artifact_id] = (key, version, prev, plan)
        if edge_extractor is not None:
            store.set_links(
                artifact_id, edge_extractor(artifact_id, plan.filter_fields)
            )

    # Embed all surfaces of changed artifacts together (cache makes this cheap).
    flat = [
        (artifact_id, i, surface)
        for artifact_id, (_k, _v, _p, plan) in changed.items()
        for i, surface in enumerate(plan.surfaces)
    ]
    vectors = _embed_batched(emb, [s.text for _, _, s in flat], "document", batch_size)

    # Replace records of changed artifacts (delete-then-write).
    for _artifact_id, (_k, _v, prev, _plan) in changed.items():
        if prev:
            for rid in prev.get("record_ids", []):
                store.delete_record(rid)

    record_ids: dict[str, list[str]] = defaultdict(list)
    # strict=True asserts one embedding row per surface (a short/long embedder
    # output is a bug we want to surface, not silently truncate).
    for (artifact_id, i, surface), vec in zip(flat, vectors, strict=True):
        rid = Record.make_id(artifact_id, surface.kind, i)
        plan = changed[artifact_id][3]
        metadata = {**plan.filter_fields, **dict(surface.metadata)}
        store.put_record(
            Record(
                id=rid,
                artifact_id=artifact_id,
                surface_kind=surface.kind,
                surface_index=i,
                text=surface.text,
                vector=vec,
                metadata=metadata,
            )
        )
        record_ids[artifact_id].append(rid)

    for artifact_id, (key, version, _prev, _plan) in changed.items():
        store.set_ledger_entry(
            key,
            {
                "artifact_id": artifact_id,
                "version": version,
                "embedder_id": emb_id,
                "strategy_id": strat_id,
                "record_ids": record_ids.get(artifact_id, []),
            },
        )

    if full:
        for key, entry in store.ledger_items():
            if entry.get("artifact_id") not in seen:
                for rid in entry.get("record_ids", []):
                    store.delete_record(rid)
                store.delete_ledger_entry(key)
                store.delete_links(entry.get("artifact_id", ""))

    store.set_config(
        {
            "name": source.name,
            "embedder_spec": spec if isinstance(spec, str) else "custom",
            "embedder_id": emb_id,
        }
    )
    return Corpus(source.name, store, emb, emb_id)


def open_corpus(name: str, *, embedder: Any = None) -> Corpus:
    """Reopen a previously built corpus by name (resolves its embedder)."""
    store = CorpusStore.local(name)
    cfg = store.get_config()
    spec = embedder if embedder is not None else cfg.get("embedder_spec", "default")
    emb, emb_id = make_embedder(spec)
    return Corpus(name, store, emb, cfg.get("embedder_id", emb_id))
