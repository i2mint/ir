"""Named-corpus registry — persistent, reusable corpus definitions.

A registry entry records *how* to (re)build a corpus — its ``kind`` (a source
preset), parameters, and embedder spec — so a corpus becomes a stable name you
can build once and query across sessions. The registry is a single JSON file
under the config dir (``~/.config/ir/corpora.json``).

Presets map to :class:`~ir.sources.CorpusSource` constructors:

- ``skills``   → :meth:`CorpusSource.from_skills`
- ``packages`` → :meth:`CorpusSource.from_packages`
- ``reports``  → :meth:`CorpusSource.from_md_reports`
- ``files``    → :meth:`CorpusSource.from_files` (needs ``root``; optional
  ``pattern``)

Unregistered preset names (``skills``/``packages``/``reports``) are
auto-registered with defaults on first use, so ``ir build skills`` just works.
"""

from __future__ import annotations

import json
from typing import Any

from .config import registry_path
from .sources import CorpusSource

PRESETS = ("skills", "packages", "reports")


def _load() -> dict[str, Any]:
    path = registry_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save(entries: dict[str, Any]) -> None:
    registry_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def register(name: str, kind: str, *, embedder: str = "default", **params) -> dict:
    """Register (or overwrite) a named corpus definition."""
    if kind not in PRESETS and kind != "files":
        raise ValueError(
            f"Unknown corpus kind {kind!r}; use one of {PRESETS} or 'files'."
        )
    entries = _load()
    entries[name] = {"kind": kind, "embedder": embedder, "params": params}
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
    """Reconstruct a :class:`CorpusSource` from a registry entry."""
    kind = entry["kind"]
    params = dict(entry.get("params", {}))
    embedder = entry.get("embedder", "default")
    if kind == "skills":
        return CorpusSource.from_skills(name=name, embedder=embedder)
    if kind == "packages":
        return CorpusSource.from_packages(name=name, embedder=embedder)
    if kind == "reports":
        return CorpusSource.from_md_reports(name=name, embedder=embedder)
    if kind == "files":
        root = params.pop("root")
        return CorpusSource.from_files(root, name=name, embedder=embedder, **params)
    raise ValueError(f"Unknown corpus kind {kind!r}.")


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
