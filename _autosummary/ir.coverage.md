# ir.coverage

Corpus-coverage diagnostic — what’s on disk vs. what’s actually indexed.

Retrieval can only return what was indexed, and an ingestion gap is *silent*:
a whole folder of reports can be missing from the index with no error (a
non-recursive glob once dropped every doc in a `research/` / `decisions/`
subtree — 29% of the corpus — invisibly). Back-translation evals cannot catch
this: their gold is drawn from what is *already* indexed, so an un-indexed doc
can never lower a metric. Coverage therefore needs its own check.

[`reports_coverage()`](#ir.coverage.reports_coverage) **independently** walks the reports doc-trees on disk
(the raw `*/*/docs` and `*/*/misc/docs` files, before any filtering),
classifies each file with the same inclusion SSOT the ingestion walk uses
([`ir.sources.report_exclude_reason()`](ir.sources.md#ir.sources.report_exclude_reason)), and diffs the *should-be-indexed*
set against the built corpus’s ledger — surfacing exactly the on-disk-but-absent
reports that a search would silently miss.

### Functions

| [`reports_coverage`](#ir.coverage.reports_coverage)([name, projects_root, ...])   | Diff the reports docs tree on disk against the built `name` corpus.   |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|

### Classes

| [`CoverageReport`](#ir.coverage.CoverageReport)(name, on_disk, indexed, excluded)   | Disk-vs-index coverage for a filesystem-backed corpus.   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------|

### *class* ir.coverage.CoverageReport(name, on_disk, indexed, excluded, missing=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Disk-vs-index coverage for a filesystem-backed corpus.

#### *property* coverage_ratio *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Fraction of should-index files that are actually indexed (1.0 = full).

#### *property* should_index *: [int](https://docs.python.org/3/builtins/functions.html#int)*

On-disk files that pass the inclusion rule (indexed + missing).

### ir.coverage.reports_coverage(name='reports', , projects_root=None, indexed_ids=None, exclude_dirs=frozenset({'.git', '.hg', '.ipynb_checkpoints', '.obsidian', '.svn', '.tox', '.venv', '_\_pycache_\_', 'build', 'dist', 'node_modules', 'site-packages', 'venv'}))

Diff the reports docs tree on disk against the built `name` corpus.

Walks every `*/*/docs` and `*/*/misc/docs` folder recursively (raw, no
filtering), classifies each `*.md` with the shared inclusion SSOT, and
reports how many *includable* reports are indexed vs. silently missing. A
`coverage_ratio` below 1.0 means on-disk reports are absent from the index —
rebuild with `ir build <name>`.

`indexed_ids` (the built corpus’s artifact ids) is read from the registered
`name` corpus by default; pass it explicitly (with `projects_root`) to
diff an arbitrary tree against an arbitrary index without opening a corpus.

* **Return type:**
  [`CoverageReport`](#ir.coverage.CoverageReport)
