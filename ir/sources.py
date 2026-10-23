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
:meth:`from_skills`, :meth:`from_packages`, :meth:`from_records`,
:meth:`from_mcp_tools`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import excluded_patterns
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


def resolve_fetcher(fetcher: Any) -> Callable[[], list]:
    """Resolve a fetcher spec to a zero-argument callable returning records.

    A callable passes through; a **string** is an import reference —
    ``"package.module:attribute"`` (preferred) or ``"package.module.attribute"``
    — imported and returned. The string form is what makes a fetcher-backed
    corpus *persistable*: a registry entry is JSON, so a corpus whose records
    come from another package (``ir.register("mine", "records",
    fetcher="mypkg.recall:turn_records")``) can only survive a rebuild if the
    seam accepts a name rather than an object.

    >>> resolve_fetcher(lambda: [])()
    []
    >>> resolve_fetcher("ir.sources:DFLT_EXCLUDE_DIRS") is DFLT_EXCLUDE_DIRS
    True
    """
    if callable(fetcher):
        return fetcher
    if not isinstance(fetcher, str) or not fetcher.strip():
        raise TypeError(
            f"fetcher must be a callable or a 'module:attr' string, got {fetcher!r}"
        )
    import importlib

    ref = fetcher.strip()
    module_name, _, attr = ref.partition(":")
    if not attr:  # dotted form: the last segment is the attribute
        module_name, _, attr = ref.rpartition(".")
    if not module_name or not attr:
        raise ValueError(
            f"fetcher reference {fetcher!r} is not of the form 'module:attr'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"fetcher reference {fetcher!r}: cannot import {module_name!r} ({e})."
        ) from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise AttributeError(
            f"fetcher reference {fetcher!r}: {module_name!r} has no {attr!r}."
        ) from e


#: Fields a tool record may carry its parameter schema under, in lookup order:
#: fastmcp's ``parameters``, the MCP wire format's ``inputSchema``, and the
#: Anthropic API's ``input_schema``.
MCP_TOOL_PARAM_KEYS = ("parameters", "inputSchema", "input_schema")


def _field(record: Any, key: str) -> Any:
    """``record[key]`` for a mapping, ``record.key`` for any other object."""
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def mcp_tool_record(tool: Any) -> dict:
    """Normalize one duck-typed tool to ``{name, description, parameters}``.

    ``tool`` is a mapping or an object (a fastmcp or ``mcp.types`` tool, say)
    with a ``name``, an optional ``description``, and an optional parameter
    schema under any of :data:`MCP_TOOL_PARAM_KEYS`. Every other field is
    dropped, so it neither reaches the index nor invalidates it.

    >>> mcp_tool_record({"name": "ping", "inputSchema": {"type": "object"}})
    {'name': 'ping', 'description': '', 'parameters': {'type': 'object'}}
    """
    name = _field(tool, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tool record has no name: {tool!r}")
    parameters = next(
        (p for key in MCP_TOOL_PARAM_KEYS if (p := _field(tool, key)) is not None),
        {},
    )
    return {
        "name": name,
        "description": str(_field(tool, "description") or ""),
        "parameters": parameters,
    }


def mcp_tool_signal(artifact_id: str, raw: Any) -> str:
    """Change signal for a tool: SHA-256 over ``(name, description, parameters)``.

    The parameter schema takes part in the hash even though it is never
    embedded, so a schema change re-embeds the tool on the next build (ir_02
    §6). The schema is serialized as canonical JSON (sorted keys), so key order
    does not count as a change. ``raw`` is a tool record in any form
    :func:`mcp_tool_record` accepts.

    >>> len(mcp_tool_signal("ping", {"name": "ping"}))
    64
    """
    tool = mcp_tool_record(raw)
    canonical = json.dumps(
        [tool["name"], tool["description"], tool["parameters"]],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _named(tool: Any, name: str) -> dict:
    """A record of *tool*'s indexed fields under *name* (any object or mapping)."""
    record = {"name": name, "description": _field(tool, "description")}
    for key in MCP_TOOL_PARAM_KEYS:
        value = _field(tool, key)
        if value is not None:
            record[key] = value
    return record


def _param_names(parameters: Any) -> list[str]:
    """The top-level argument names of a JSON-Schema ``parameters`` object."""
    props = parameters.get("properties") if isinstance(parameters, Mapping) else None
    return [str(k) for k in props] if isinstance(props, Mapping) else []


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
    def from_records(
        cls,
        *,
        name: str,
        fetcher: Any,
        strategy: IndexingStrategy | None = None,
        metadata_keys: Iterable[str] | None = None,
        id_key: str = "id",
        text_key: str = "text",
        **kwargs,
    ) -> "CorpusSource":
        """Records produced by a **named callable** as a corpus.

        The generic, *persistable* counterpart of :meth:`from_mapping`: where
        ``from_mapping`` takes an in-process object (so a registry entry cannot
        name it), this takes a ``fetcher`` — a callable **or** a
        ``"module:attr"`` import reference (see :func:`resolve_fetcher`) — so a
        corpus owned by another package round-trips through
        ``~/.config/ir/corpora.json`` and rebuilds in a fresh process.

        Each record is a mapping; its ``id_key`` is the artifact id (falling back
        to ``<name>_<i>`` so no record is silently dropped). ``metadata_keys``
        names the record fields lifted into the strategy's **filter fields** —
        the JSON-friendly stand-in for a ``metadata_of`` callable, which no
        registry entry could carry. The default strategy chunks ``text_key``.

        >>> src = CorpusSource.from_records(
        ...     name="notes",
        ...     fetcher=lambda: [{"id": "a", "text": "hi", "project": "p"}],
        ...     metadata_keys=["project"],
        ... )
        >>> src.scope["a"]["text"], src.metadata_of("a", src.scope["a"])
        ('hi', {'project': 'p'})
        """
        records = list(resolve_fetcher(fetcher)())
        keys = tuple(metadata_keys or ())
        scope: dict[str, Any] = {}
        for i, record in enumerate(records):
            key = (record.get(id_key) if isinstance(record, Mapping) else None) or (
                f"{name}_{i}"
            )
            scope[str(key)] = record

        def metadata_of(aid, raw):
            if not keys or not isinstance(raw, Mapping):
                return {}
            return {k: raw[k] for k in keys if k in raw}

        return cls(
            name=name,
            scope=scope,
            indexing_strategy=strategy
            or Chunked(text_key=text_key, max_tokens=DFLT_CHUNK_MAX_TOKENS),
            metadata_of=metadata_of,
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

        Walks each ``*/*/docs`` and ``*/*/misc/docs`` folder, and those of
        pocket repos (``i/_zodals/<repo>/docs``, see :func:`iter_pocket_repos`), **recursively**
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
                "project": _project_of(rel, root),
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
        fetcher: Callable[[], list] | str | None = None,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """The agent-skills corpus, via ``priv.skills_index``.

        ``fetcher`` overrides the source of skill records (each a mapping with
        ``name``/``description``/``parent``) — a callable (inject a test double
        to avoid the ``priv`` dependency) or a ``"module:attr"`` reference
        (:func:`resolve_fetcher`), which is the form a registry entry can carry.
        """
        if fetcher is not None:
            records = list(resolve_fetcher(fetcher)())
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
    def from_mcp_tools(
        cls,
        tools: Any,
        *,
        name: str = "tools",
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """A catalogue of MCP tools as a corpus, for tool search.

        Each tool is indexed on ``name + description`` only, through the
        :class:`~ir.strategy.Skill` strategy: its parameter schema never enters
        the embedded text (ir_02 §4, "the body carries no retrieval
        responsibility"). The schema still drives invalidation, through
        :func:`mcp_tool_signal`, so a changed schema re-embeds the tool. Its
        top-level argument names ride along as the ``param_names`` metadata of
        every hit, so a caller can see what a tool takes without loading it.

        ``tools`` is an iterable of tool records, a ``{name: tool}`` mapping (the
        key names a record that has no ``name``; a record's own name wins), or
        a fetcher (a zero-argument callable or a ``"module:attr"`` reference,
        see :func:`resolve_fetcher`). A record is duck-typed (see
        :func:`mcp_tool_record`), so ``ir`` takes no MCP dependency; an adapter
        for a specific client lives with the consumer. Tool names are the
        artifact ids and must be unique: namespace them (``"server.tool"``)
        before merging several servers' catalogues.

        >>> src = CorpusSource.from_mcp_tools(
        ...     [{"name": "ping", "description": "Check the server is up"}]
        ... )
        >>> plan = src.indexing_strategy.decompose("ping", src.scope["ping"])
        >>> print(plan.surfaces[0].text)
        ping
        <BLANKLINE>
        Check the server is up
        """
        if isinstance(tools, str) or callable(tools):
            tools = resolve_fetcher(tools)()
        if isinstance(tools, Mapping):
            # The key names a tool whose record carries no name of its own.
            tools = [
                tool if _field(tool, "name") else _named(tool, key)
                for key, tool in tools.items()
            ]
        scope: dict[str, dict] = {}
        for tool in tools:
            record = mcp_tool_record(tool)
            if record["name"] in scope:
                raise ValueError(
                    f"duplicate tool name {record['name']!r}: tool names are the "
                    f"artifact ids; namespace them (e.g. 'server.tool') first."
                )
            scope[record["name"]] = record

        def metadata_of(aid, raw):
            return {"param_names": _param_names(raw.get("parameters"))}

        return cls(
            name=name,
            scope=scope,
            indexing_strategy=strategy or Skill(),
            change_signal=kwargs.pop("change_signal", mcp_tool_signal),
            metadata_of=kwargs.pop("metadata_of", metadata_of),
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
        fetcher: Callable[[], list] | str | None = None,
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
        ``user_prompt`` / ``assistant_summary`` / ... ) — a callable (inject a
        test double to avoid the ``priv`` dependency) or a ``"module:attr"``
        reference (:func:`resolve_fetcher`), the form a registry entry can carry,
        so another package can own the transcripts this corpus indexes. Otherwise
        records come from :func:`priv.claude_transcripts.turn_pair_records`.
        """
        if fetcher is not None:
            records = list(resolve_fetcher(fetcher)())
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
        repos: Callable[[], Iterable] | str | None = None,
        exclude: Iterable[str] | None = None,
        readme_chars: int = 20000,
        strategy: IndexingStrategy | None = None,
        **kwargs,
    ) -> "CorpusSource":
        """The local repos: the ``.pth`` manifest plus every fleet repo on disk.

        ``repos`` overrides where the repo directories come from — a
        zero-argument callable yielding directory paths, or a ``"module:attr"``
        reference (:func:`resolve_fetcher`). The default is
        :func:`package_repo_dirs`: manifest packages first, then the repos
        ``priv.layout.scan_fleet`` finds under the letter dirs (JS/TS apps,
        data repos), so a repo is findable whether or not it is a Python package.

        ``exclude`` is a list of ``fnmatch`` patterns for repos never to read
        (see :func:`ir.config.exclude_path`); the default is that local file's
        patterns, applied to every directory whatever ``repos`` yields.
        """
        if repos is not None:
            dirs = [Path(d) for d in resolve_fetcher(repos)()]
        else:
            dirs = package_repo_dirs(manifest)
        patterns = excluded_patterns() if exclude is None else list(exclude)
        dirs = [d for d in dirs if not _is_excluded(d, patterns)]
        scope = _scan_packages(dirs, readme_chars=readme_chars)

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


#: The doc-tree roots inside a repo (relative to the repo folder).
REPO_DOC_SUBDIRS = ("docs", "misc/docs")


def iter_pocket_repos(root: Path):
    """Yield repos one level deeper than usual: ``<letter>/<pocket>/<repo>``.

    A *pocket* is a ``*/*`` folder that is not itself a git repo but holds repos
    (``i/_zodals/zodal-groups``, ``i/o/<repo>``, ``i/dols/<repo>``). A child counts
    as a repo only when its ``.git`` is a **directory**: a ``.git`` *file* marks a
    worktree or submodule, whose docs duplicate a checkout indexed elsewhere.
    """
    for pocket in root.glob("*/*"):
        if not pocket.is_dir() or (pocket / ".git").exists():
            continue
        for repo in pocket.iterdir():
            if (repo / ".git").is_dir():
                yield repo


def iter_report_doc_folders(root: Path):
    """Yield each existing report doc folder under *root*.

    ``*/*/docs`` and ``*/*/misc/docs`` (a repo at ``<letter>/<repo>``), plus the
    same two folders of every pocket repo (:func:`iter_pocket_repos`). The single
    enumeration both ingestion and :func:`ir.coverage.reports_coverage` use.
    """
    for pattern in REPORT_DOC_GLOBS:
        for folder in root.glob(pattern):
            if folder.is_dir():
                yield folder
    for repo in iter_pocket_repos(root):
        for sub in REPO_DOC_SUBDIRS:
            folder = repo / sub
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


def _project_of(rel_path: str, root: Path | None = None) -> str:
    """``i/dol/docs/x.md`` -> ``dol`` (the package folder name).

    With *root*, a pocket repo resolves to the repo, not the pocket:
    ``i/_zodals/zodal-groups/docs/x.md`` -> ``zodal-groups``.
    """
    parts = Path(rel_path).parts
    if (
        root is not None
        and len(parts) >= 4
        and not (root / parts[0] / parts[1] / ".git").exists()
        and (root / parts[0] / parts[1] / parts[2] / ".git").is_dir()
    ):
        return parts[2]
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


def _fleet_repo_dirs() -> list[Path]:
    """Every repo checkout under the letter dirs, or ``[]`` when ``priv`` is absent.

    ``priv.layout.scan_fleet`` is the fleet's SSOT for what is on disk (it already
    drops symlinked aliases); linked git worktrees are dropped here, being the
    same checkout as their main clone under another name.
    """
    try:
        from priv.layout import scan_fleet
    except ImportError:
        return []
    repos, _ = scan_fleet()
    return [Path(r.path) for r in repos if not (Path(r.path) / ".git").is_file()]


def _is_excluded(path: Path, patterns: Iterable[str]) -> bool:
    """Does ``path`` (absolute, or relative to the projects folder) match a pattern?"""
    patterns = list(patterns)
    if not patterns:
        return False
    candidates = [path.as_posix()]
    try:
        candidates.append(
            path.resolve().relative_to(_projects_root().resolve()).as_posix()
        )
    except (ValueError, RuntimeError):
        pass
    return any(fnmatch.fnmatch(c, p) for c in candidates for p in patterns)


def package_repo_dirs(manifest: str | Path | None = None) -> list[Path]:
    """Manifest package directories, then fleet repos the manifest does not name."""
    dirs = _manifest_paths(manifest)
    seen = {d.resolve() for d in dirs}
    for d in _fleet_repo_dirs():
        if d.resolve() not in seen:
            seen.add(d.resolve())
            dirs.append(d)
    return dirs


def _package_json_description(path: Path) -> str:
    """The ``description`` of ``path/package.json``, or ``""``."""
    pj = path / "package.json"
    if not pj.is_file():
        return ""
    try:
        return json.loads(pj.read_text(encoding="utf-8")).get("description") or ""
    except (json.JSONDecodeError, OSError, ValueError, AttributeError):
        return ""


def _scan_packages(dirs: Iterable[Path], *, readme_chars: int) -> dict[str, dict]:
    """Build ``{package_name: {name, description, readme, owner, deps, path}}``.

    The key is the directory name; a second directory with the same name (e.g.
    ``t/x`` and ``tt/x``) is keyed ``name@parent`` rather than overwriting it.
    """
    import tomllib

    scope: dict[str, dict] = {}
    for path in dirs:
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
        description = description or _package_json_description(path)
        readme = ""
        for cand in ("README.md", "README.rst", "README.txt"):
            rp = path / cand
            if rp.is_file():
                readme = rp.read_text(encoding="utf-8", errors="ignore")[:readme_chars]
                break
        key = name if name not in scope else f"{name}@{path.parent.name}"
        scope[key] = {
            "name": name,
            "description": description,
            "readme": readme,
            "owner": "ours",
            "deps": deps,
            "path": str(path),
        }
    return scope
