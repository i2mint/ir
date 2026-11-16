# ir.grep

The grep Retriever leaf: an index-free first stage over a corpus’s raw files.

Embedding indexes find paraphrases but miss the exact things a keyword search is for
(a name, an identifier, an error string) and need a build step that may be stale. This
leaf reads the files where they lie: it runs a regex (`rg` when available, else
Python `re`) over the files under some roots and returns
[`SearchHit`](ir.base.html.md#ir.base.SearchHit)s, so it can be fused with a dense or BM25 ranking like
any other Retriever ([`ir.retrieve.Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever)).

Python’s `re` is the authority: `pattern` is a Python regex, lines are split on
`\n` only (a trailing `\r` is ignored), and counts and snippets come from Python,
so scores never depend on whether `rg` is installed. `rg` is only a prefilter that
skips files with no possible match, and it is used **only when that is provably safe**:
an ASCII *literal* pattern (no regex metacharacters, see `_rg_can_prefilter()`),
since `rg` and Python read `\w`, `\s`, `.`, character classes and Unicode case
folding differently and a prefilter that disagrees would silently drop files. Any
other pattern is read in Python (`use_rg=True` then raises).

Scoring is explicit and tunable: a file’s score is its **matching-line count**, times a
per-file weight (`weight_of`: e.g. a title or a recent folder counts more), times an
optional recency decay (`recency_half_life_days`: a file’s weight halves for each
half-life of age). Metadata filters are pushed into the scan: `metadata_of(path)`
derives a file’s metadata and a Mongo-style `filter` (the same one
[`ir.search()`](ir.html.md#ir.search) takes) drops files **before** they are read.

Hit metadata carries only the id-relative `path` and `filename` (never an absolute
local path, which would leak into anything published) unless `absolute_paths=True`.

It returns hits, never a loop and never a plan; whether to grep or to use the index is
the caller’s (or the agent layer’s) choice.

```pycon
>>> import tempfile, pathlib
>>> with tempfile.TemporaryDirectory() as d:
...     _ = pathlib.Path(d, "a.md").write_text("Acme invoice\nnothing\nacme again")
...     _ = pathlib.Path(d, "b.md").write_text("unrelated")
...     hits = grep_hits("acme", roots=d, use_rg=False)
>>> [(h.artifact_id, h.score, h.metadata["n_matches"]) for h in hits]
[('a.md', 2.0, 2)]
```

### Module Attributes

| [`DFLT_RG_BATCH`](#ir.grep.DFLT_RG_BATCH)      | How many files one `rg` call is handed (keeps argv under the OS limit).                 |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`DFLT_SNIPPET_CHARS`](#ir.grep.DFLT_SNIPPET_CHARS) | Longest snippet kept per hit (the first matching line, trimmed).                        |
| [`DFLT_EXCLUDE_DIRS`](#ir.grep.DFLT_EXCLUDE_DIRS)  | Directories never descended into (VCS and tooling noise; override with `exclude_dirs`). |
| [`GREP_KIND`](#ir.grep.GREP_KIND)          | The `surface_kind` of a grep hit.                                                       |

### Functions

| [`grep_hits`](#ir.grep.grep_hits)(pattern, \*, roots[, glob, k, ...])   | The top-`k` files under `roots` whose lines match `pattern`, as hits.                                                                           |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [`make_grep_retriever`](#ir.grep.make_grep_retriever)(roots, \*\*grep_defaults)   | Bind roots (and defaults) to the uniform [`ir.retrieve.Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) contract. |

### ir.grep.DFLT_EXCLUDE_DIRS *= ('.git', 'node_modules', '_\_pycache_\_', '.venv')*

Directories never descended into (VCS and tooling noise; override with `exclude_dirs`).

### ir.grep.DFLT_RG_BATCH *= 500*

How many files one `rg` call is handed (keeps argv under the OS limit).

### ir.grep.DFLT_SNIPPET_CHARS *= 200*

Longest snippet kept per hit (the first matching line, trimmed).

### ir.grep.GREP_KIND *= 'grep'*

The `surface_kind` of a grep hit.

### ir.grep.grep_hits(pattern, , roots, glob='\*', k=10, ignore_case=True, fixed=False, filter=None, metadata_of=None, weight_of=None, recency_half_life_days=None, id_of=None, use_rg=None, rg_batch=500, exclude_dirs=('.git', 'node_modules', '_\_pycache_\_', '.venv'), absolute_paths=False, snippet_chars=200, source=None, now=None)

The top-`k` files under `roots` whose lines match `pattern`, as hits.

`pattern` is a Python regex (`fixed=True`: a literal). Files are those whose
root-relative posix path matches the fnmatch `glob` (`*` crosses `/`, so
`"*.md"` means every markdown file below the root); a root that is a file
stands for itself. `filter` is applied to `metadata_of(path)` *before* any file
is read (a filter without `metadata_of` sees `{"path", "filename"}`, the path
relative to its root). `id_of(root, path)` names the artifact (default: the
relative path); two files with the same id raise `ValueError` (several roots with
a common relative path: pass `id_of`). `use_rg`: `None` = prefilter with
`rg` if installed, `True` = require it (`RuntimeError` otherwise), `False` =
Python only. The score is described in the module docstring.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]

### ir.grep.make_grep_retriever(roots, \*\*grep_defaults)

Bind roots (and defaults) to the uniform [`ir.retrieve.Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) contract.

The result is `retrieve(pattern, **overrides) -> list[SearchHit]` (`roots` may
be overridden per call too), the same shape as [`ir.as_retriever()`](ir.html.md#ir.as_retriever), so a grep
leaf can sit in a `Mapping[name, Retriever]` registry beside index-backed
corpora. Overrides are the keywords of [`grep_hits()`](#ir.grep.grep_hits); an unknown one
(`mode=`) is a `TypeError`, not silently dropped.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]
