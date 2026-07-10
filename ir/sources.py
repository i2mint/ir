"""Defining a corpus source — an abstract strategy plus parameters.

A :class:`CorpusSource` says *what* a corpus is and *how to keep it current*,
independent of how it is indexed or stored:

- ``scope``            — a ``Mapping[id -> raw]`` enumerating the corpus (a dict,
  a ``dol`` file store, or any mapping). This is the "what is in the corpus" slot.
- ``change_signal``    — ``(id, raw) -> version_str``; default is the content
  hash. This is the "what counts as stale" slot, driving incremental reindex.
- ``indexing_strategy``— how a raw item becomes filter fields + surfaces.
- ``embedder``         — embedder spec (default: the decent local model).
- ``metadata_of``      — optional ``(id, raw) -> dict`` of extra filter metadata.

Smart-default constructors cover the common ways to define a source:
:meth:`from_mapping`, :meth:`from_files`, :meth:`from_md_reports`,
:meth:`from_skills`, :meth:`from_packages`.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .strategy import (
    DFLT_CHUNK_MAX_TOKENS,
    Chunked,
    ClaudeTurn,
    IndexingStrategy,
    Package,
    Skill,
    WholeText,
)

ALLCAPS_MD = re.compile(r"^[A-Z0-9_ ]+\.md$")

#: Directory names never descended into when walking a docs tree for reports.
#: Vendored / build / hidden noise — kept as an SSOT so ingestion and any
#: coverage diagnostic exclude exactly the same set (see :func:`_md_in`).
DFLT_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "site-packages",
        ".tox",
        "build",
        "dist",
        ".ipynb_checkpoints",
        ".obsidian",
    }
)

#: Default look-back window (days) for :meth:`CorpusSource.from_claude_sessions`.
DFLT_SESSIONS_SINCE_DAYS = 90


def content_hash_signal(artifact_id: str, raw: Any) -> str:
    """Default change signal: a content hash of the raw payload."""
    import ef

    return ef.content_hash(raw)


@dataclass
class CorpusSource:
    """A corpus definition: scope + change signal + strategy + embedder."""

    name: str
    scope: Mapping[str, Any]
    indexing_strategy: IndexingStrategy = field(default_factory=WholeText)
    change_signal: Callable[[str, Any], str] = content_hash_signal
    embedder: Any = "default"
    metadata_of: Callable[[str, Any], Mapping[str, Any]] | None = None

    def items(self):
        """Iterate ``(artifact_id, raw)`` pairs over the corpus scope."""
        return self.scope.items()

    # ----- smart-default constructors ------------------------------------ #

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        name: str,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """Any mapping ``{id -> raw}`` (dict, ``dol`` store) as a corpus."""
        return cls(
            name=name,
            scope=mapping,
            indexing_strategy=strategy or WholeText(),
            **kwargs,
        )

    @classmethod
    def from_files(
        cls,
        root: str | Path,
        *,
        name: str | None = None,
        pattern: str = r".*\.md$",
        exclude: Callable[[str], bool] | None = None,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """A directory tree of text files as a corpus (lazy ``dol`` scope)."""
        import dol

        root = Path(root).expanduser()
        keep = re.compile(pattern)
        store = dol.TextFiles(str(root))

        def predicate(k: str) -> bool:
            if not keep.search(k):
                return False
            return not (exclude and exclude(k))

        scope = dol.filt_iter(store, filt=predicate)

        def metadata_of(aid, raw):
            return {"path": str(root / aid), "filename": os.path.basename(aid)}

        return cls(
            name=name or root.name,
            scope=scope,
            indexing_strategy=strategy or Chunked(max_tokens=DFLT_CHUNK_MAX_TOKENS),
            metadata_of=metadata_of,
            **kwargs,
        )

    @classmethod
    def from_md_reports(
        cls,
        *,
        name: str = "reports",
        projects_root: str | Path | None = None,
        strategy: IndexingStrategy | None = None,
        recursive: bool = True,
        exclude_dirs: Iterable[str] | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """Markdown reports under projects' ``docs/`` and ``misc/docs/`` trees.

        Walks each ``*/*/docs`` and ``*/*/misc/docs`` folder **recursively**
        (``recursive=True``, default) so reports nested one or more levels deep —
        ``docs/research/…``, ``docs/decisions/…``, ``docs/adr/…`` — are indexed,
        not just files sitting directly in the folder. Pass ``recursive=False``
        for the old shallow (top-level-only) behavior.

        Excludes ALL-CAPS filenames (README/CLAUDE/MEMORY/SKILL...) and any file
        under a vendored/hidden directory (``exclude_dirs``, default
        :data:`DFLT_EXCLUDE_DIRS`; a dotted directory is always skipped). Each
        record is a project-tagged document; ids are paths relative to the
        projects root.
        """
        root = Path(projects_root or _projects_root())
        exclude = DFLT_EXCLUDE_DIRS if exclude_dirs is None else frozenset(exclude_dirs)
        scope: dict[str, dict] = {}
        meta: dict[str, dict] = {}
        for path in _iter_md_reports(root, recursive=recursive, exclude_dirs=exclude):
            # POSIX-normalize the id so it's stable across platforms (forward
            # slashes on Windows too); a no-op on posix systems.
            rel = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            scope[rel] = {"text": text}
            meta[rel] = {
                "project": _project_of(rel),
                "path": str(path),
                "filename": path.name,
            }

        return cls(
            name=name,
            scope=scope,
            indexing_strategy=strategy or Chunked(max_tokens=DFLT_CHUNK_MAX_TOKENS),
            metadata_of=lambda aid, raw: meta.get(aid, {}),
            **kwargs,
        )

    @classmethod
    def from_skills(
        cls,
        *,
        name: str = "skills",
        filter: Any = None,
        fetcher: Callable[[], list] | None = None,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """The agent-skills corpus, via ``priv.skills_index``.

        ``fetcher`` overrides the source of skill records (each a mapping with
        ``name``/``description``/``parent``) — inject a test double to avoid the
        ``priv`` dependency.
        """
        if fetcher is not None:
            records = list(fetcher())
        else:
            from priv.skills_index import skills_index as _skills_index

            records = _skills_index(filter=filter, egress="raw")
        # Preserve same-named skills from different packages (collision-safe ids).
        scope: dict[str, dict] = {}
        for r in records:
            key = r["name"]
            if key in scope:
                key = f"{r['name']}@{r.get('parent')}"
            scope[key] = r

        def metadata_of(aid, raw):
            return {
                "parent": raw.get("parent"),
                "skill_path": raw.get("skill_path"),
                "base_path": raw.get("base_path"),
            }

        return cls(
            name=name,
            scope=scope,
            indexing_strategy=strategy or Skill(),
            metadata_of=metadata_of,
            **kwargs,
        )

    @classmethod
    def from_claude_sessions(
        cls,
        *,
        name: str = "sessions",
        since: float | None = DFLT_SESSIONS_SINCE_DAYS,
        projects: Any = None,
        include_full: bool = False,
        include_session_title: bool = True,
        max_sessions: int | None = None,
        root: str | Path | None = None,
        fetcher: Callable[[], list] | None = None,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """The user's Claude Code session transcripts as a corpus (turn pairs).

        Each artifact is one user→assistant turn pair; the default
        :class:`~ir.strategy.ClaudeTurn` strategy indexes the user prompt and the
        assistant's end-of-turn summary as separate surfaces (target either with
        ``surfaces={"user_prompt"}`` / ``{"assistant_summary"}``). ``include_full``
        adds the full assistant text surface (off by default — the summary is the
        signal). ``include_session_title`` (default on) also indexes one record per
        session whose surface is the session's persisted custom/AI title — a cheap
        "what was this session about" surface. Scope defaults to the last ``since``
        days (a full-history build is heavy); narrow with ``projects`` (a cwd
        substring or list) and ``max_sessions``.

        ``fetcher`` overrides the record source (each a mapping with
        ``user_prompt`` / ``assistant_summary`` / ... ) — inject a test double to
        avoid the ``priv`` dependency. Otherwise records come from
        :func:`priv.claude_transcripts.turn_pair_records`.
        """
        if fetcher is not None:
            records = list(fetcher())
        else:
            from priv.claude_transcripts import turn_pair_records

            records = list(
                turn_pair_records(
                    root=root,
                    since=since,
                    projects=projects,
                    max_sessions=max_sessions,
                    include_session_title=include_session_title,
                )
            )
        # Collision-safe scope: ids are session_id:user_uuid (already unique); a
        # rare missing id falls back to enumeration so no pair is silently dropped.
        scope: dict[str, dict] = {}
        for i, r in enumerate(records):
            key = r.get("id") or f"turn_{i}"
            scope[key] = r

        return cls(
            name=name,
            scope=scope,
            indexing_strategy=strategy or ClaudeTurn(include_full=include_full),
            **kwargs,
        )

    @classmethod
    def from_packages(
        cls,
        *,
        name: str = "packages",
        manifest: str | Path | None = None,
        readme_chars: int = 20000,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """The local package ecosystem, scanned from the ``.pth`` manifest."""
        scope = _scan_packages(manifest, readme_chars=readme_chars)

        def metadata_of(aid, raw):
            return {"path": raw.get("path")}

        return cls(
            name=name,
            scope=scope,
            indexing_strategy=strategy or Package(max_tokens=DFLT_CHUNK_MAX_TOKENS),
            metadata_of=metadata_of,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Source-specific scope helpers
# --------------------------------------------------------------------------- #


def _projects_root() -> Path:
    """Locate the projects folder (``$PP``), preferring ``priv.config``.

    Falls back to the ``$PP`` environment variable when ``priv.config`` is not
    installed. Only :class:`ImportError` triggers the fallback — a
    ``priv.config`` that *is* installed but raises is allowed to propagate, so a
    misconfigured projects folder fails loudly instead of being silently masked
    as "priv not installed".
    """
    try:
        from priv.config import projects_folder
    except ImportError:
        pass
    else:
        return Path(projects_folder())
    pp = os.environ.get("PP")
    if pp:
        return Path(pp).expanduser()
    raise RuntimeError(
        "Cannot locate the projects folder; set $PP or install priv.config."
    )


#: The two per-project doc-tree roots reports are drawn from (relative globs).
REPORT_DOC_GLOBS = ("*/*/docs", "*/*/misc/docs")


def report_exclude_reason(
    path: Path, folder: Path, *, exclude_dirs: Iterable[str] = DFLT_EXCLUDE_DIRS
) -> str | None:
    """Why a ``*.md`` under *folder* is excluded from the reports corpus.

    The **SSOT** for "does this report file get indexed" — consumed by both the
    ingestion walk (:func:`_md_in`) and the coverage diagnostic
    (:func:`ir.coverage.reports_coverage`) so the two can never drift. Returns
    ``None`` when the file *should* be indexed, else a short reason tag:
    ``"allcaps"`` (a README/CLAUDE/… metadata file) or ``"excluded_dir"`` (under a
    vendored/hidden subtree).
    """
    if ALLCAPS_MD.match(path.name):
        return "allcaps"
    exclude = (
        exclude_dirs if isinstance(exclude_dirs, frozenset) else frozenset(exclude_dirs)
    )
    rel_dirs = path.relative_to(folder).parts[:-1]
    if any(part in exclude or part.startswith(".") for part in rel_dirs):
        return "excluded_dir"
    return None


def iter_report_doc_folders(root: Path):
    """Yield each existing ``*/*/docs`` and ``*/*/misc/docs`` folder under *root*."""
    for pattern in REPORT_DOC_GLOBS:
        for folder in root.glob(pattern):
            if folder.is_dir():
                yield folder


def _iter_md_reports(
    root: Path,
    *,
    recursive: bool = True,
    exclude_dirs: Iterable[str] = DFLT_EXCLUDE_DIRS,
):
    """Yield non-ALLCAPS ``*.md`` files under projects' docs/ and misc/docs/.

    Each ``*/*/docs`` and ``*/*/misc/docs`` folder is walked recursively by
    default (``recursive=True``); ``exclude_dirs`` (plus any dotted directory)
    prunes vendored/hidden subtrees.
    """
    for folder in iter_report_doc_folders(root):
        yield from _md_in(folder, recursive=recursive, exclude_dirs=exclude_dirs)


def _md_in(
    folder: Path,
    *,
    recursive: bool = True,
    exclude_dirs: Iterable[str] = DFLT_EXCLUDE_DIRS,
):
    """Yield report ``*.md`` files in *folder*.

    Recurses into subdirectories when *recursive* (default), skipping ALL-CAPS
    filenames and any file living under an *exclude_dirs* or dotted directory
    (per :func:`report_exclude_reason`, the shared inclusion SSOT).
    """
    if not folder.is_dir():
        return
    exclude = frozenset(exclude_dirs)
    walk = folder.rglob if recursive else folder.glob
    for f in walk("*.md"):
        if not f.is_file():
            continue
        if report_exclude_reason(f, folder, exclude_dirs=exclude) is None:
            yield f


def _project_of(rel_path: str) -> str:
    """``i/dol/docs/x.md`` -> ``dol`` (the package folder name)."""
    parts = Path(rel_path).parts
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "")


def _manifest_paths(manifest: str | Path | None) -> list[Path]:
    """Absolute package directories listed in the ``.pth`` manifest."""
    manifest = manifest or os.environ.get("PTH_FILEPATH")
    if not manifest:
        raise RuntimeError("No manifest given and $PTH_FILEPATH is unset.")
    paths = []
    for line in Path(manifest).read_text().splitlines():
        line = line.strip()
        if line.startswith("/") and Path(line).is_dir():
            paths.append(Path(line))
    return paths


def _scan_packages(manifest, *, readme_chars: int) -> dict[str, dict]:
    """Build ``{package_name: {name, description, readme, owner, deps, path}}``."""
    import tomllib

    scope: dict[str, dict] = {}
    for path in _manifest_paths(manifest):
        name = path.name
        description, deps = "", []
        pyproject = path / "pyproject.toml"
        if pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                proj = data.get("project", {})
                description = proj.get("description", "") or ""
                deps = list(proj.get("dependencies", []) or [])
            except (tomllib.TOMLDecodeError, OSError, ValueError) as e:
                # Warn rather than silently thinning the index (matching the
                # warn-on-degradation discipline used across ir).
                warnings.warn(
                    f"Could not parse {pyproject} for package {name!r} ({e}); "
                    f"indexing it with an empty description and no deps.",
                    stacklevel=2,
                )
        readme = ""
        for cand in ("README.md", "README.rst", "README.txt"):
            rp = path / cand
            if rp.is_file():
                readme = rp.read_text(encoding="utf-8", errors="ignore")[:readme_chars]
                break
        scope[name] = {
            "name": name,
            "description": description,
            "readme": readme,
            "owner": "ours",
            "deps": deps,
            "path": str(path),
        }
    return scope
