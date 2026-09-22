# ir.sources

Defining a corpus source — an abstract strategy plus parameters.

A [`CorpusSource`](#ir.sources.CorpusSource) says *what* a corpus is and *how to keep it current*,
independent of how it is indexed or stored:

- `scope`            — a `Mapping[id -> raw]` enumerating the corpus (a dict,
  a `dol` file store, or any mapping). This is the “what is in the corpus” slot.
- `change_signal`    — `(id, raw) -> version_str`; default is the content
  hash. This is the “what counts as stale” slot, driving incremental reindex.
- `indexing_strategy`— how a raw item becomes filter fields + surfaces.
- `embedder`         — embedder spec (default: the decent local model).
- `metadata_of`      — optional `(id, raw) -> dict` of extra filter metadata.

Smart-default constructors cover the common ways to define a source:
`from_mapping()`, `from_files()`, `from_md_reports()`,
`from_skills()`, `from_packages()`, `from_records()`.

### Module Attributes

| [`DFLT_EXCLUDE_DIRS`](#ir.sources.DFLT_EXCLUDE_DIRS)        | Directory names never descended into when walking a docs tree for reports.                                                |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_SESSIONS_SINCE_DAYS`](#ir.sources.DFLT_SESSIONS_SINCE_DAYS) | Default look-back window (days) for [`CorpusSource.from_claude_sessions()`](#ir.sources.CorpusSource.from_claude_sessions). |
| [`REPORT_DOC_GLOBS`](#ir.sources.REPORT_DOC_GLOBS)         | The two per-project doc-tree roots reports are drawn from (relative globs).                                               |

### Functions

| [`content_hash_signal`](#ir.sources.content_hash_signal)(artifact_id, raw)          | Default change signal: a content hash of the raw payload.               |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`iter_report_doc_folders`](#ir.sources.iter_report_doc_folders)(root)                  | Yield each existing `*/*/docs` and `*/*/misc/docs` folder under *root*. |
| [`report_exclude_reason`](#ir.sources.report_exclude_reason)(path, folder, \*[, ...]) | Why a `*.md` under *folder* is excluded from the reports corpus.        |
| [`resolve_fetcher`](#ir.sources.resolve_fetcher)(fetcher)                       | Resolve a fetcher spec to a zero-argument callable returning records.   |

### Classes

| [`CorpusSource`](#ir.sources.CorpusSource)(name, scope[, ...])   | A corpus definition: scope + change signal + strategy + embedder.   |
|-------------------------------------------------------------------------------------|---------------------------------------------------------------------|

### *class* ir.sources.CorpusSource(name, scope, indexing_strategy=<factory>, change_signal=<function content_hash_signal>, embedder='default', metadata_of=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A corpus definition: scope + change signal + strategy + embedder.

#### change_signal(raw)

Default change signal: a content hash of the raw payload.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### *classmethod* from_claude_sessions(, name='sessions', since=90, projects=None, include_full=False, include_session_title=True, max_sessions=None, root=None, fetcher=None, strategy=None, \*\*kwargs)

The user’s Claude Code session transcripts as a corpus (turn pairs).

Each artifact is one user→assistant turn pair; the default
[`ClaudeTurn`](ir.strategy.html.md#ir.strategy.ClaudeTurn) strategy indexes the user prompt and the
assistant’s end-of-turn summary as separate surfaces (target either with
`surfaces={"user_prompt"}` / `{"assistant_summary"}`). `include_full`
adds the full assistant text surface (off by default — the summary is the
signal). `include_session_title` (default on) also indexes one record per
session whose surface is the session’s persisted custom/AI title — a cheap
“what was this session about” surface. Scope defaults to the last `since`
days (a full-history build is heavy); narrow with `projects` (a cwd
substring or list) and `max_sessions`.

`fetcher` overrides the record source (each a mapping with
`user_prompt` / `assistant_summary` / … ) — a callable (inject a
test double to avoid the `priv` dependency) or a `"module:attr"`
reference ([`resolve_fetcher()`](#ir.sources.resolve_fetcher)), the form a registry entry can carry,
so another package can own the transcripts this corpus indexes. Otherwise
records come from `priv.claude_transcripts.turn_pair_records()`.

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

#### *classmethod* from_files(root, , name=None, pattern='.\*\\\\\\\\.md$', exclude=None, strategy=None, \*\*kwargs)

A directory tree of text files as a corpus (lazy `dol` scope).

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

#### *classmethod* from_mapping(mapping, , name, strategy=None, \*\*kwargs)

Any mapping `{id -> raw}` (dict, `dol` store) as a corpus.

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

#### *classmethod* from_md_reports(, name='reports', projects_root=None, strategy=None, recursive=True, exclude_dirs=None, \*\*kwargs)

Markdown reports under projects’ `docs/` and `misc/docs/` trees.

Walks each `*/*/docs` and `*/*/misc/docs` folder **recursively**
(`recursive=True`, default) so reports nested one or more levels deep —
`docs/research/…`, `docs/decisions/…`, `docs/adr/…` — are indexed,
not just files sitting directly in the folder. Pass `recursive=False`
for the old shallow (top-level-only) behavior.

Excludes ALL-CAPS filenames (README/CLAUDE/MEMORY/SKILL…) and any file
under a vendored/hidden directory (`exclude_dirs`, default
[`DFLT_EXCLUDE_DIRS`](#ir.sources.DFLT_EXCLUDE_DIRS); a dotted directory is always skipped). Each
record is a project-tagged document; ids are paths relative to the
projects root.

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

#### *classmethod* from_packages(, name='packages', manifest=None, readme_chars=20000, strategy=None, \*\*kwargs)

The local package ecosystem, scanned from the `.pth` manifest.

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

#### *classmethod* from_records(, name, fetcher, strategy=None, metadata_keys=None, id_key='id', text_key='text', \*\*kwargs)

Records produced by a **named callable** as a corpus.

The generic, *persistable* counterpart of [`from_mapping()`](#ir.sources.CorpusSource.from_mapping): where
`from_mapping` takes an in-process object (so a registry entry cannot
name it), this takes a `fetcher` — a callable **or** a
`"module:attr"` import reference (see [`resolve_fetcher()`](#ir.sources.resolve_fetcher)) — so a
corpus owned by another package round-trips through
`~/.config/ir/corpora.json` and rebuilds in a fresh process.

Each record is a mapping; its `id_key` is the artifact id (falling back
to `<name>_<i>` so no record is silently dropped). `metadata_keys`
names the record fields lifted into the strategy’s **filter fields** —
the JSON-friendly stand-in for a `metadata_of` callable, which no
registry entry could carry. The default strategy chunks `text_key`.

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

```pycon
>>> src = CorpusSource.from_records(
...     name="notes",
...     fetcher=lambda: [{"id": "a", "text": "hi", "project": "p"}],
...     metadata_keys=["project"],
... )
>>> src.scope["a"]["text"], src.metadata_of("a", src.scope["a"])
('hi', {'project': 'p'})
```

#### *classmethod* from_skills(, name='skills', filter=None, fetcher=None, strategy=None, \*\*kwargs)

The agent-skills corpus, via `priv.skills_index`.

`fetcher` overrides the source of skill records (each a mapping with
`name`/`description`/`parent`) — a callable (inject a test double
to avoid the `priv` dependency) or a `"module:attr"` reference
([`resolve_fetcher()`](#ir.sources.resolve_fetcher)), which is the form a registry entry can carry.

* **Return type:**
  [`CorpusSource`](#ir.sources.CorpusSource)

#### items()

Iterate `(artifact_id, raw)` pairs over the corpus scope.

### ir.sources.DFLT_EXCLUDE_DIRS *= frozenset({'.git', '.hg', '.ipynb_checkpoints', '.obsidian', '.svn', '.tox', '.venv', '_\_pycache_\_', 'build', 'dist', 'node_modules', 'site-packages', 'venv'})*

Directory names never descended into when walking a docs tree for reports.
Vendored / build / hidden noise — kept as an SSOT so ingestion and any
coverage diagnostic exclude exactly the same set (see `_md_in()`).

### ir.sources.DFLT_SESSIONS_SINCE_DAYS *= 90*

Default look-back window (days) for [`CorpusSource.from_claude_sessions()`](#ir.sources.CorpusSource.from_claude_sessions).

### ir.sources.REPORT_DOC_GLOBS *= ('\*/\*/docs', '\*/\*/misc/docs')*

The two per-project doc-tree roots reports are drawn from (relative globs).

### ir.sources.content_hash_signal(artifact_id, raw)

Default change signal: a content hash of the raw payload.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.sources.iter_report_doc_folders(root)

Yield each existing `*/*/docs` and `*/*/misc/docs` folder under *root*.

### ir.sources.report_exclude_reason(path, folder, , exclude_dirs=frozenset({'.git', '.hg', '.ipynb_checkpoints', '.obsidian', '.svn', '.tox', '.venv', '_\_pycache_\_', 'build', 'dist', 'node_modules', 'site-packages', 'venv'}))

Why a `*.md` under *folder* is excluded from the reports corpus.

The **SSOT** for “does this report file get indexed” — consumed by both the
ingestion walk (`_md_in()`) and the coverage diagnostic
([`ir.coverage.reports_coverage()`](ir.coverage.html.md#ir.coverage.reports_coverage)) so the two can never drift. Returns
`None` when the file *should* be indexed, else a short reason tag:
`"allcaps"` (a README/CLAUDE/… metadata file) or `"excluded_dir"` (under a
vendored/hidden subtree).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.sources.resolve_fetcher(fetcher)

Resolve a fetcher spec to a zero-argument callable returning records.

A callable passes through; a **string** is an import reference —
`"package.module:attribute"` (preferred) or `"package.module.attribute"`
— imported and returned. The string form is what makes a fetcher-backed
corpus *persistable*: a registry entry is JSON, so a corpus whose records
come from another package (`ir.register("mine", "records",
fetcher="mypkg.recall:turn_records")`) can only survive a rebuild if the
seam accepts a name rather than an object.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]

```pycon
>>> resolve_fetcher(lambda: [])()
[]
>>> resolve_fetcher("ir.sources:DFLT_EXCLUDE_DIRS") is DFLT_EXCLUDE_DIRS
True
```
