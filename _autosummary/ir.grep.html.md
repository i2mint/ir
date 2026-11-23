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

Scoring is explicit and tunable: a file’s score is its **matching-line count** (or, with
`score_line`, the sum of each matching line’s score: a heading or a title field can
count more than body text, see [`markdown_heading_score()`](#ir.grep.markdown_heading_score) and
[`json_field_score()`](#ir.grep.json_field_score)), times a per-file weight (`weight_of`: e.g. a title or a recent folder counts more), times an
optional recency decay (`recency_half_life_days`: a file’s weight halves for each
half-life of age). Metadata filters are pushed into the scan: `metadata_of(path)`
derives a file’s metadata and a Mongo-style `filter` (the same one
[`ir.search()`](ir.html.md#ir.search) takes) drops files **before** they are read.

The files are either everything under `roots` (filtered by `glob`) or, with
`paths`, an explicit `{artifact_id: path}` mapping: [`corpus_paths()`](#ir.grep.corpus_paths) reads one
from a built corpus, which is how `ir.search(corpus, pattern, mode="grep")` greps the
files a corpus indexed, under the corpus’s own artifact ids.

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
| [`LineScore`](#ir.grep.LineScore)          | the score one matching line adds to its file.                                           |

### Functions

| [`corpus_paths`](#ir.grep.corpus_paths)(corpus)                              | `{artifact_id: file}` for the artifacts of a built corpus that are files.                                                                       |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [`grep_hits`](#ir.grep.grep_hits)(pattern, \*[, roots, paths, glob, ...]) | The top-`k` files under `roots` whose lines match `pattern`, as hits.                                                                           |
| [`json_field_score`](#ir.grep.json_field_score)(weights, \*[, default, ...])     | A `score_line` seam for JSONL records: field weights, for title-over-body.                                                                      |
| [`make_grep_retriever`](#ir.grep.make_grep_retriever)(roots, \*\*grep_defaults)     | Bind roots (and defaults) to the uniform [`ir.retrieve.Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) contract. |
| [`markdown_heading_score`](#ir.grep.markdown_heading_score)([heading, body])           | A `score_line` seam: a match on a markdown heading line counts `heading`.                                                                       |
| [`recency_weight`](#ir.grep.recency_weight)(half_life_days, \*[, now])         | A composable `weight_of` seam: a file's weight halves per `half_life_days` of age.                                                              |

### ir.grep.DFLT_EXCLUDE_DIRS *= ('.git', 'node_modules', '_\_pycache_\_', '.venv')*

Directories never descended into (VCS and tooling noise; override with `exclude_dirs`).

### ir.grep.DFLT_RG_BATCH *= 500*

How many files one `rg` call is handed (keeps argv under the OS limit).

### ir.grep.DFLT_SNIPPET_CHARS *= 200*

Longest snippet kept per hit (the first matching line, trimmed).

### ir.grep.GREP_KIND *= 'grep'*

The `surface_kind` of a grep hit.

### ir.grep.LineScore

the score one matching line adds to its file.

* **Type:**
  `score_line(line, rx) -> float`

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Pattern`](https://docs.python.org/3/library/re.html#re.Pattern)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### ir.grep.corpus_paths(corpus)

`{artifact_id: file}` for the artifacts of a built corpus that are files.

Read from the metadata the corpus stored at build time (the `path` that
file-backed sources such as `from_files` and `from_md_reports` record), so
no source is walked again; the files themselves are read live by
[`grep_hits()`](#ir.grep.grep_hits). Artifacts without a stored path (records, packages, skills)
are left out.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]

### ir.grep.grep_hits(pattern, , roots=None, paths=None, glob='\*', k=10, ignore_case=True, fixed=False, filter=None, metadata_of=None, weight_of=None, score_line=None, recency_half_life_days=None, id_of=None, use_rg=None, rg_batch=500, exclude_dirs=('.git', 'node_modules', '_\_pycache_\_', '.venv'), absolute_paths=False, snippet_chars=200, source=None, now=None)

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

`paths` replaces `roots`: an explicit `{artifact_id: path}` mapping (e.g.
[`corpus_paths()`](#ir.grep.corpus_paths)), its keys used as ids and matched against `glob`; a path
that is not a file any more is skipped. Give exactly one of `roots` and `paths`.
`score_line(line, rx)` scores each matching line (default 1, so the score is
the count); see [`markdown_heading_score()`](#ir.grep.markdown_heading_score) and [`json_field_score()`](#ir.grep.json_field_score).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]

### ir.grep.json_field_score(weights, , default=1.0, other_lines=1.0)

A `score_line` seam for JSONL records: field weights, for title-over-body.

Each line is read as one JSON object; the line scores the sum of `weights[f]`
over the top-level fields `f` whose (string) value matches, `default` for a
matching field not in `weights`. A match only in a key, or in a non-string
value, scores 0. A line that is not a JSON object scores `other_lines`.

```pycon
>>> score = json_field_score({"title": 3.0, "body": 1.0})
>>> rx = re.compile("acme", re.I)
>>> score('{"title": "Acme", "body": "acme again"}', rx)
4.0
>>> score('{"title": "Other", "body": "about acme"}', rx)
1.0
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Pattern`](https://docs.python.org/3/library/re.html#re.Pattern)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### ir.grep.make_grep_retriever(roots, \*\*grep_defaults)

Bind roots (and defaults) to the uniform [`ir.retrieve.Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) contract.

The result is `retrieve(pattern, **overrides) -> list[SearchHit]` (`roots` may
be overridden per call too), the same shape as [`ir.as_retriever()`](ir.html.md#ir.as_retriever), so a grep
leaf can sit in a `Mapping[name, Retriever]` registry beside index-backed
corpora. Overrides are the keywords of [`grep_hits()`](#ir.grep.grep_hits); an unknown one
(`mode=`) is a `TypeError`, not silently dropped.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]

### ir.grep.markdown_heading_score(heading=3.0, , body=1.0)

A `score_line` seam: a match on a markdown heading line counts `heading`.

Any other matching line counts `body`, so a document whose title names the
pattern outranks one that only mentions it in passing.

```pycon
>>> score = markdown_heading_score(3.0)
>>> rx = re.compile("acme", re.I)
>>> score("# Acme invoices", rx), score("paid acme today", rx)
(3.0, 1.0)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Pattern`](https://docs.python.org/3/library/re.html#re.Pattern)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### ir.grep.recency_weight(half_life_days, , now=None)

A composable `weight_of` seam: a file’s weight halves per `half_life_days` of age.

`now` pins the clock (default: the time of each call). Multiply it into your own
`weight_of` to combine it with other weights; it is what
`recency_half_life_days` uses.

```pycon
>>> import os, tempfile, time
>>> from pathlib import Path
>>> with tempfile.TemporaryDirectory() as d:
...     f = Path(d, "a.md"); _ = f.write_text("x")
...     os.utime(f, (0, 0))
...     recency_weight(10, now=10 * _SECONDS_PER_DAY)(f)
0.5
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]
