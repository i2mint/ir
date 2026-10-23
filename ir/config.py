"""Filesystem locations and process-wide defaults for ``ir``.

``ir`` separates three kinds of on-disk state, each under an XDG-standard base
(overridable per-install via ``IR_CONFIG_DIR`` / ``IR_DATA_DIR`` /
``IR_CACHE_DIR``, then ``XDG_CONFIG_HOME`` / ``XDG_DATA_HOME`` /
``XDG_CACHE_HOME``, then ``~/.config`` / ``~/.local/share`` / ``~/.cache``):

- **config** (``~/.config/ir``) — the named-corpus registry and user settings.
- **data** (``~/.local/share/ir``) — durable corpus stores (record metadata,
  vectors, ledgers). The source of truth; losing it means rebuilding the index.
- **cache** (``~/.cache/ir``) — regenerable derived data, chiefly the embedding
  cache keyed by ``(model, content_hash)``.

Every path is a plain :class:`~pathlib.Path` created on demand, so callers can
treat the directories as guaranteed to exist.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

APP_NAME = "ir"

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _ensure(path: Path) -> Path:
    """Create *path* (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base(env_specific: str, xdg_var: str, home_subdir: str) -> Path:
    """Resolve an XDG base dir for ``ir`` from env vars, with a home fallback."""
    specific = os.environ.get(env_specific)
    if specific:
        return Path(specific).expanduser()
    xdg = os.environ.get(xdg_var)
    if xdg:
        return Path(xdg).expanduser() / APP_NAME
    return Path.home() / home_subdir / APP_NAME


def config_dir() -> Path:
    """User configuration directory (default ``~/.config/ir``)."""
    return _ensure(_base("IR_CONFIG_DIR", "XDG_CONFIG_HOME", ".config"))


def data_dir() -> Path:
    """Durable data directory (default ``~/.local/share/ir``)."""
    return _ensure(_base("IR_DATA_DIR", "XDG_DATA_HOME", ".local/share"))


def cache_dir() -> Path:
    """Regenerable cache directory (default ``~/.cache/ir``)."""
    return _ensure(_base("IR_CACHE_DIR", "XDG_CACHE_HOME", ".cache"))


def safe_name(name: str) -> str:
    """Filesystem-safe slug for a corpus/model identifier."""
    return _SAFE.sub("_", name).strip("_") or "unnamed"


def corpus_dir(name: str) -> Path:
    """Durable directory holding one corpus's stores."""
    return _ensure(data_dir() / "corpora" / safe_name(name))


def embeddings_cache_dir(model_id: str) -> Path:
    """Cache directory for one embedding model's vectors."""
    return _ensure(cache_dir() / "embeddings" / safe_name(model_id))


def registry_path() -> Path:
    """JSON file mapping registered corpus names to their build settings."""
    return config_dir() / "corpora.json"


def exclude_path() -> Path:
    """Local list of repo paths never to index (default ``~/.config/ir/exclude``).

    One ``fnmatch`` pattern per line, matched against a repo's path relative to
    the projects folder (``q/some-private-repo``, ``projects/*``) or its absolute
    path; blank lines and ``#`` comments are ignored. It is per-machine config,
    never committed, so a private repo's name stays out of any shared file.
    """
    return config_dir() / "exclude"


def excluded_patterns() -> list[str]:
    """The patterns in :func:`exclude_path`, or ``[]`` when the file is absent."""
    path = exclude_path()
    if not path.is_file():
        return []
    lines = (ln.split("#", 1)[0].strip() for ln in path.read_text().splitlines())
    return [ln for ln in lines if ln]
