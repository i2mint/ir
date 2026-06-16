"""Named-corpus registry — persistent, reusable corpus definitions.

A registry entry records *how* to (re)build a corpus — its ``kind`` (a source
preset), parameters, and embedder spec — so a corpus becomes a stable name you
can build once and query across sessions. The registry is a single JSON file
under the config dir (``~/.config/ir/corpora.json``).

Presets map to :class:`~ir.sources.CorpusSource` constructors:

- ``skills``   → :meth:`CorpusSource.from_skills`
- ``packages`` → :meth:`CorpusSource.from_packages`
- ``reports``  → :meth:`CorpusSource.from_md_reports`
- ``sessions`` → :meth:`CorpusSource.from_claude_sessions`
- ``files``    → :meth:`CorpusSource.from_files` (needs ``root``; optional
  ``pattern``)

Unregistered preset names (``skills``/``packages``/``reports``/``sessions``) are
auto-registered with defaults on first use, so ``ir build skills`` just works.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .config import registry_path
from .sources import CorpusSource

PRESETS = ("skills", "packages", "reports", "sessions")


def _load() -> dict[str, Any]:
    path = registry_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save(entries: dict[str, Any]) -> None:
    registry_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def register(
    name: str,
    kind: str,
    *,
    embedder: str = "default",
    strategy: Any = None,
    maintenance: Mapping[str, Any] | None = None,
    storage: Mapping[str, Any] | None = None,
    **params,
) -> dict:
    """Register (or overwrite) a named corpus definition.

    Beyond the v1 ``kind`` / ``embedder`` / ``params``, an entry may now carry
    (all optional, with smart per-kind defaults applied at resolution time — see
    :mod:`ir.policy`):

    - ``strategy`` — an :class:`~ir.strategy.IndexingStrategy` (or a
      ``{"name", "params"}`` spec) persisted so the corpus's *segmentation* is
      stable across rebuilds. ``None`` keeps the preset's default strategy.
    - ``maintenance`` — the background-work policy dict (``reindex`` / ``synopsis``;
      validated here, see :class:`ir.policy.MaintenancePolicy`).
    - ``storage`` — the persistence backend (default ``{"backend": "local"}``).

    Entries written by older ``ir`` (none of these keys) keep working unchanged.
    """
    if kind not in PRESETS and kind != "files":
        raise ValueError(
            f"Unknown corpus kind {kind!r}; use one of {PRESETS} or 'files'."
        )
    from .policy import MaintenancePolicy, resolve_storage
    from .strategy import IndexingStrategy, strategy_to_spec

    entry: dict[str, Any] = {"kind": kind, "embedder": embedder, "params": params}
    if strategy is not None:
        if isinstance(strategy, Mapping):
            entry["strategy"] = dict(strategy)
        elif isinstance(strategy, IndexingStrategy):
            entry["strategy"] = strategy_to_spec(strategy)
        else:
            raise TypeError(
                f"strategy must be an IndexingStrategy or a spec dict, got "
                f"{type(strategy).__name__}"
            )
    if maintenance is not None:
        # Validate (raises on a bad trigger) and store the normalized form.
        entry["maintenance"] = MaintenancePolicy.from_dict(dict(maintenance)).to_dict()
    if storage is not None:
        resolve_storage({"storage": dict(storage)})  # validate backend
        entry["storage"] = dict(storage)

    entries = _load()
    entries[name] = entry
    _save(entries)
    return entries[name]


def registered() -> dict[str, Any]:
    """All registered corpus definitions, keyed by name."""
    return _load()


def get(name: str) -> dict | None:
    """The registry entry for *name*, or ``None``."""
    return _load().get(name)


def unregister(name: str) -> None:
    """Remove *name* from the registry (does not delete built data)."""
    entries = _load()
    entries.pop(name, None)
    _save(entries)


def source_from_entry(name: str, entry: dict) -> CorpusSource:
    """Reconstruct a :class:`CorpusSource` from a registry entry.

    A persisted ``strategy`` spec (registry v2) is reconstructed and passed to
    the preset constructor, so a corpus's segmentation survives across rebuilds.
    A v1 entry (no ``strategy``) passes ``strategy=None`` and keeps the preset's
    default — unchanged behavior.
    """
    from .strategy import strategy_from_spec

    kind = entry["kind"]
    params = dict(entry.get("params", {}))
    embedder = entry.get("embedder", "default")
    strategy = strategy_from_spec(entry.get("strategy"))
    if kind == "skills":
        return CorpusSource.from_skills(name=name, embedder=embedder, strategy=strategy)
    if kind == "packages":
        return CorpusSource.from_packages(
            name=name, embedder=embedder, strategy=strategy
        )
    if kind == "reports":
        return CorpusSource.from_md_reports(
            name=name, embedder=embedder, strategy=strategy
        )
    if kind == "sessions":
        return CorpusSource.from_claude_sessions(
            name=name, embedder=embedder, strategy=strategy, **params
        )
    if kind == "files":
        root = params.pop("root")
        return CorpusSource.from_files(
            root, name=name, embedder=embedder, strategy=strategy, **params
        )
    raise ValueError(f"Unknown corpus kind {kind!r}.")


def policy_for(name: str):
    """The effective :class:`ir.policy.MaintenancePolicy` for corpus *name*.

    Resolves the registered entry's ``maintenance`` over its kind's smart default
    over the global default (see :func:`ir.policy.resolve_policy`). An unregistered
    name resolves to the global default policy.
    """
    from .policy import resolve_policy

    return resolve_policy(get(name))


def source_for(name: str) -> CorpusSource:
    """Resolve *name* to a source, auto-registering a preset if needed."""
    entry = get(name)
    if entry is None:
        if name in PRESETS:
            entry = register(name, name)
        else:
            raise KeyError(
                f"Corpus {name!r} is not registered. Register it with "
                f"`ir register`, or use a preset name: {PRESETS}."
            )
    return source_from_entry(name, entry)


# --------------------------------------------------------------------------- #
# Query-time projection — name -> Retriever (ir_09 §8 / epic #38)
# --------------------------------------------------------------------------- #


def retriever_for(name: str, **search_defaults: Any):
    """A :data:`~ir.retrieve.Retriever` bound to the registered corpus *name*.

    Opens the corpus (it must have been built) and wraps it with
    :func:`ir.as_retriever`; ``search_defaults`` (e.g. ``mode="hybrid"``) bind to
    every call.
    """
    from .index import open_corpus
    from .retrieve import as_retriever

    return as_retriever(open_corpus(name), **search_defaults)


class _RetrieversView(Mapping):
    """A lazy read-projection of the registry into ``name -> Retriever``.

    A strict view over the SSOT (:func:`registered`), **not** a second registry:
    keys are the registered names; ``view[name]`` lazily opens that corpus and
    returns its retriever. So "what can be built" and "what can be queried" can
    never drift, and a corpus is opened only when its key is accessed.
    """

    def __init__(self, **search_defaults: Any):
        self._defaults = search_defaults

    def __getitem__(self, name: str):
        if name not in registered():
            raise KeyError(name)
        return retriever_for(name, **self._defaults)

    def __iter__(self):
        return iter(registered())

    def __len__(self) -> int:
        return len(registered())


def retrievers(**search_defaults: Any) -> Mapping[str, Any]:
    """A lazy ``Mapping[name, Retriever]`` view over the registry (ir_09 §8).

    The query-time projection of the build-recipe registry: each value is a
    ready-to-call :data:`~ir.retrieve.Retriever`. This is the source-registry
    facade an orchestration layer (``raglab``) consumes — it never opens a corpus
    until the key is accessed, and always reflects the current :func:`registered`
    set. ``search_defaults`` apply to every source.
    """
    return _RetrieversView(**search_defaults)
