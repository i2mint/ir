# ir

`ir` — an information-retrieval substrate for agentic systems.

One uniform “find the relevant things in this corpus” contract that scales from
an ad-hoc search over an ephemeral list to a maintained search engine. Retrieval
is the core; selection/expansion/reranking/generation are layered on top.

Quick start:

```default
import ir

# Define a corpus source (abstract strategy + parameters, smart defaults):
source = ir.CorpusSource.from_md_reports()          # project docs/ reports
corpus = ir.build(source)                            # index (incremental)
hits = ir.search(corpus, "how do I deploy the app")  # ranked SearchHits

# Light, dependency-free embedding for fast tests:
corpus = ir.build(source, embedder="light")
```

A corpus source is defined by a `scope` (what is in the corpus), a
`change_signal` (what counts as stale), an `indexing_strategy` (how a raw
item becomes filter fields + embeddable surfaces), and an `embedder`. The
default embedder is a decent *local* model (`all-MiniLM-L6-v2`); `"light"`
selects a numpy-only hashing embedder. Data persists under XDG dirs through a
`dol` repository layer.

`ir` is a retrieval **substrate, not a search agent** — it has no agent of its
own. [`ir.discover()`](#ir.discover) is a deterministic single-shot `retrieve → select →
disclose` pipeline: the query is embedded verbatim (no LLM query rewriting) and
committed by a score rule (no LLM reviewing results), with no planner and no
*back-edge* (evaluate → reformulate → search again). The agent layer — planner,
LLM query formulator, and the evaluator→reformulate loop — lives in **raglab**
([https://github.com/thorwhalen/raglab](https://github.com/thorwhalen/raglab)), which is built *on top of* `ir`:
`raglab` imports `ir`; `ir` never imports `raglab`. To drive an agentic
search from Claude Code instead of a coded agent, see the `ir-search` skill and
the reviewing search subagent shipped under `.claude/`.

### Functions

| [`search_tool`](#ir.search_tool)(query, \*, corpus[, k, mode, filter])   | Search a named `ir` corpus and return a JSON-serializable result dict.                                                                     |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [`make_search`](#ir.make_search)(corpus, \*[, name, description, ...])   | Return a **corpus-bound** `search(query, k=...) -> dict` tool.                                                                             |
| [`with_synopsis`](#ir.with_synopsis)(strategy, \*[, synthesize, ...])      | Wrap *strategy* to add one LLM-derived `synopsis` surface per artifact.                                                                    |
| [`make_llm_synthesizer`](#ir.make_llm_synthesizer)(\*[, summarize, prompt, ...])  | An LLM-backed `Synthesizer` ([`Artifact`](ir.base.html.md#ir.base.Artifact) → synopsis).                       |
| [`build`](#ir.build)(source, \*[, store, embedder, full, ...])     | Build or incrementally update *source* into a [`Corpus`](#ir.Corpus).                                     |
| [`open_corpus`](#ir.open_corpus)(name, \*[, embedder])                   | Reopen a previously built corpus by name.                                                                                                  |
| [`reports_coverage`](#ir.reports_coverage)([name, projects_root, ...])        | Diff the reports docs tree on disk against the built `name` corpus.                                                                        |
| [`search`](#ir.search)(corpus, query, \*\*kwargs)                   | Search a [`Corpus`](ir.index.html.md#ir.index.Corpus) (or a corpus *name*, reopened lazily).                    |
| [`as_retriever`](#ir.as_retriever)(corpus_or_name, \*\*search_defaults)   | Bind ONE corpus to the uniform `Retriever` contract.                                                                                       |
| [`retrievers`](#ir.retrievers)(\*\*search_defaults)                     | A lazy `Mapping[name, Retriever]` view over the registry (ir_09 §8).                                                                       |
| [`retriever_for`](#ir.retriever_for)(name, \*\*search_defaults)            | A [`Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) bound to the registered corpus *name*.                  |
| [`fuse_hits`](#ir.fuse_hits)(hits_by_source, \*[, rrf_k, ...])         | Merge per-source ranked hit lists into one ranking — by rank, not score.                                                                   |
| [`records_for_artifact`](#ir.records_for_artifact)(store_or_corpus, ...[, ...])   | All stored records of *artifact_id*, ordered by `surface_index`.                                                                           |
| [`default_edge_extractor`](#ir.default_edge_extractor)(artifact_id, ...)            | Edges latent in the standard filter fields: `deps` → REF, `parent` → PARENT.                                                               |
| [`canonical_node_id`](#ir.canonical_node_id)(target, \*, source)               | Canonicalize a neighbor *target* to a `(source, artifact_id)` node id.                                                                     |
| [`traverse`](#ir.traverse)(query, store, \*, policy[, ...])           | Walk *store* from *query* under *policy*, returning the top-*k* hits.                                                                      |
| [`collapsed_tree_policy`](#ir.collapsed_tree_policy)(\*[, summary_kinds, ...])     | The pure-vector summary-routing / collapsed-tree [`WalkPolicy`](#ir.WalkPolicy).                              |
| [`expand`](#ir.expand)(hit, corpus, \*[, policy])                   | Expand *hit* into a [`Passage`](#ir.Passage) of its neighborhood in *corpus*.                              |
| [`sentence_window_policy`](#ir.sentence_window_policy)([k])                         | ±\*k\* same-kind neighbors around the seed (NEXT/PREV expansion).                                                                          |
| [`parent_policy`](#ir.parent_policy)()                                     | The whole artifact (small-to-big): every stored surface, plan order.                                                                       |
| [`tag_source`](#ir.tag_source)(hits, source)                            | Stamp *source* on every hit that doesn't already carry one.                                                                                |
| [`make_llm_formulator`](#ir.make_llm_formulator)(\*[, rewriter, prompt, ...])    | An LLM-backed `Formulator` (rewrite / expand / multi-query).                                                                               |
| [`select`](#ir.select)(hits, \*[, strategy, max_k, rel, ...])       | Commit to a subset of ranked `hits` — the selection stage.                                                                                 |
| [`disclose`](#ir.disclose)(selection, \*[, level, loader, ...])       | Reveal the payload of each selected hit at `level` — append-only, pure.                                                                    |
| [`discover`](#ir.discover)(corpus, query, \*[, k, mode, ...])         | Find and commit to the capabilities for `query` — the one search tool.                                                                     |
| [`register`](#ir.register)(name, kind, \*[, embedder, ...])           | Register (or overwrite) a named corpus definition.                                                                                         |
| [`corpora`](#ir.corpora)()                                           | All registered corpus definitions, keyed by name.                                                                                          |
| [`build_corpus`](#ir.build_corpus)(name, \*\*kwargs)                      | Build (or update) a registered/preset corpus by *name*; returns a [`Corpus`](ir.index.html.md#ir.index.Corpus). |
| [`maintain`](#ir.maintain)([name, all, now, dry_run])                 | Run due background work for one corpus (*name*) or every registered one (*all*).                                                           |
| [`maintain_corpus`](#ir.maintain_corpus)(name, \*[, now, dry_run, full])     | Do the due background work for one corpus (idempotent).                                                                                    |
| [`resolve_policy`](#ir.resolve_policy)(entry)                               | The effective policy for a registry `entry`: entry over kind over global.                                                                  |
| [`default_policy_for_kind`](#ir.default_policy_for_kind)(kind)                       | The smart-default [`MaintenancePolicy`](#ir.MaintenancePolicy) for a corpus `kind`.                                  |
| [`policy_for`](#ir.policy_for)(name)                                    | The effective [`ir.policy.MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy) for corpus *name*.  |

### Classes

| [`Artifact`](#ir.Artifact)(id, raw[, metadata])                     | A logical corpus item before decomposition into surfaces.                                                                    |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`Surface`](#ir.Surface)(artifact_id, kind, text[, ...])           | One embeddable unit derived from an artifact.                                                                                |
| [`Record`](#ir.Record)(id, artifact_id, surface_kind, ...[, ...]) | A stored, embedded surface — one row of the index.                                                                           |
| [`SearchHit`](#ir.SearchHit)(artifact_id, surface_kind, score, text) | A scored record returned by retrieval (higher score = closer).                                                               |
| [`IndexPlan`](#ir.IndexPlan)([filter_fields, surfaces])              | An [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy)'s output for one artifact. |
| [`IndexingStrategy`](#ir.IndexingStrategy)(\*args, \*\*kwargs)              | Decompose one artifact into filter fields + embeddable surfaces.                                                             |
| [`WholeText`](#ir.WholeText)(\*[, text_key, kind])                   | One surface = the entire text.                                                                                               |
| [`Chunked`](#ir.Chunked)(\*[, chunk_size, overlap, text_key, ...]) | Split the artifact's text into overlapping chunk surfaces.                                                                   |
| [`Skill`](#ir.Skill)()                                           | Capability strategy: embed `name + description` only.                                                                        |
| [`Package`](#ir.Package)(\*[, chunk_size, overlap, ...])           | Package strategy: `name + description` surface plus README chunks.                                                           |
| [`ClaudeTurn`](#ir.ClaudeTurn)(\*[, include_full])                    | Index a Claude Code session turn-pair: user prompt + assistant summary.                                                      |
| [`CorpusSource`](#ir.CorpusSource)(name, scope[, ...])                  | A corpus definition: scope + change signal + strategy + embedder.                                                            |
| [`CorpusStore`](#ir.CorpusStore)(meta, vectors, ledger, config[, ...]) | Repository bundling the meta/vectors/ledger/config views of one corpus.                                                      |
| [`Corpus`](#ir.Corpus)(name, store, embedder, embedder_id)        | A built, queryable corpus: a store plus its embedder.                                                                        |
| [`CoverageReport`](#ir.CoverageReport)(name, on_disk, indexed, excluded)  | Disk-vs-index coverage for a filesystem-backed corpus.                                                                       |
| [`GraphStore`](#ir.GraphStore)(\*args, \*\*kwargs)                    | Structural contract a traversal operator binds to — node + neighbors.                                                        |
| [`CorpusGraph`](#ir.CorpusGraph)(store_or_corpus)                      | A [`GraphStore`](#ir.GraphStore) over one corpus — artifact nodes, `links` edges.               |
| [`WalkPolicy`](#ir.WalkPolicy)(\*args, \*\*kwargs)                    | The pluggable strategy of a walk — graph semantics, not safety.                                                              |
| [`WalkState`](#ir.WalkState)(query, max_depth, budget[, ...])        | The operator-owned state of one [`traverse()`](#ir.traverse) call — the safety home.          |
| [`Passage`](#ir.Passage)(artifact_id, surface_kind, score, text)   | An expanded hit: the seed's identity + the stitched neighborhood text.                                                       |
| [`Selection`](#ir.Selection)(selected, candidates, abstained, ...)   | A selector's commitment: the chosen subset of a ranked candidate list.                                                       |
| [`Disclosure`](#ir.Disclosure)(artifact_id, level, name, score, ...)  | The progressively-disclosed payload for one selected artifact.                                                               |
| [`DiscoveryResult`](#ir.DiscoveryResult)(query, mode, strategy, ...)       | The result of [`discover()`](#ir.discover) — retrieve → select → (optional) disclose.         |
| [`MaintenanceResult`](#ir.MaintenanceResult)(name, ran, reason[, ...])       | What [`maintain_corpus()`](#ir.maintain_corpus) did (or would do) for one corpus.                    |
| [`MaintenancePolicy`](#ir.MaintenancePolicy)([reindex, synopsis])            | The background-work policy for one corpus (reindex + synopsis).                                                              |
| [`ReindexPolicy`](#ir.ReindexPolicy)([on, every_hours])                  | When to (incrementally) rebuild a corpus.                                                                                    |
| [`SynopsisPolicy`](#ir.SynopsisPolicy)([enabled, scope, ...])             | Whether/when to attach (expensive, LLM-generated) synopses.                                                                  |

### *class* ir.Artifact(id, raw, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A logical corpus item before decomposition into surfaces.

### *class* ir.Chunked(, chunk_size=1200, overlap=200, text_key=None, kind='chunk', max_tokens=None, token_model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Split the artifact’s text into overlapping chunk surfaces.

By default chunks are packed to ~\`\`chunk_size\`\` **characters**. Set
`max_tokens` to chunk by **tokens** instead — each chunk is bounded to
`max_tokens` tokens of `token_model` (default: the default embedding
model), so no chunk overflows the embedder’s sequence limit and is silently
truncated. Token mode degrades to char mode with a warning if the tokenizer
is unavailable. The char `chunk_size` / `overlap` still set the overlap
*ratio* used in token mode.

### *class* ir.ClaudeTurn(, include_full=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Index a Claude Code session turn-pair: user prompt + assistant summary.

Two surfaces by default — `user_prompt` (what the human asked) and
`assistant_summary` (the assistant’s *final* end-of-turn text, the
highest-signal “here’s what I did”; the deliberation before it is mostly
noise) — so a query can target either side via `surfaces={"user_prompt"}` /
`{"assistant_summary"}`. With `include_full=True` a third
`assistant_full` surface (all the turn’s assistant natural-language text)
trades noise for recall — off by default. Session / project / time / model /
tool-use become hard-filter fields.

The raw artifact is a turn-pair record (see
`priv.claude_transcripts.turn_pair_records()`): a mapping with
`user_prompt` / `assistant_summary` / `assistant_full` plus metadata.

### *class* ir.Corpus(name, store, embedder, embedder_id)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A built, queryable corpus: a store plus its embedder.

#### search(query, \*\*kwargs)

Search this corpus for *query*.

`**kwargs` (`k` / `mode` / `filter` / `surfaces` /
`per_artifact` / …) are forwarded to [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search).

### *class* ir.CorpusGraph(store_or_corpus)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A [`GraphStore`](#ir.GraphStore) over one corpus — artifact nodes, `links` edges.

`node_id` is an `artifact_id`. `graph[aid]` is the artifact’s stored
records (its scorable surfaces, in plan order); `graph.neighbors(aid,
edge_type=...)` reads the corpus store’s `links` view. Single-corpus, so
it resolves **intra-corpus** targets; cross-corpus `[source, artifact_id]`
targets are returned by [`neighbors()`](#ir.CorpusGraph.neighbors) verbatim but `__getitem__` only
dereferences ids in *this* corpus (federated traversal is a follow-up).

#### edge_types(node_id)

The edge types present on *node_id* (`[]` if none).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### neighbors(node_id, , edge_type=None)

Outgoing neighbor ids of *node_id*, optionally of one *edge_type*.

Returns target ids in stored form (a bare `artifact_id`, whose source
is *this* graph’s [`source`](#ir.CorpusGraph.source); or a `[source, artifact_id]` list
for a cross-corpus edge), de-duplicated with first-seen order
preserved. An artifact with no edges — or a store without a links view —
yields `[]`. Pass a canonical `(source, artifact_id)` to
[`canonical_node_id()`](#ir.canonical_node_id) for a traversal’s visited-set.

*node_id* is an intra-corpus `artifact_id` (a `str`); a cross-corpus
target fetched from another graph is out of contract here (it has no
edges *in this corpus*).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

#### source

The corpus name, when known — the `source` half of this graph’s
node identities (`None` for a bare store).

### *class* ir.CorpusSource(name, scope, indexing_strategy=<factory>, change_signal=<function content_hash_signal>, embedder='default', metadata_of=None)

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
reference (`resolve_fetcher()`), the form a registry entry can carry,
so another package can own the transcripts this corpus indexes. Otherwise
records come from `priv.claude_transcripts.turn_pair_records()`.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

#### *classmethod* from_files(root, , name=None, pattern='.\*\\\\\\\\.md$', exclude=None, strategy=None, \*\*kwargs)

A directory tree of text files as a corpus (lazy `dol` scope).

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

#### *classmethod* from_mapping(mapping, , name, strategy=None, \*\*kwargs)

Any mapping `{id -> raw}` (dict, `dol` store) as a corpus.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

#### *classmethod* from_md_reports(, name='reports', projects_root=None, strategy=None, recursive=True, exclude_dirs=None, \*\*kwargs)

Markdown reports under projects’ `docs/` and `misc/docs/` trees.

Walks each `*/*/docs` and `*/*/misc/docs` folder **recursively**
(`recursive=True`, default) so reports nested one or more levels deep —
`docs/research/…`, `docs/decisions/…`, `docs/adr/…` — are indexed,
not just files sitting directly in the folder. Pass `recursive=False`
for the old shallow (top-level-only) behavior.

Excludes ALL-CAPS filenames (README/CLAUDE/MEMORY/SKILL…) and any file
under a vendored/hidden directory (`exclude_dirs`, default
`DFLT_EXCLUDE_DIRS`; a dotted directory is always skipped). Each
record is a project-tagged document; ids are paths relative to the
projects root.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

#### *classmethod* from_packages(, name='packages', manifest=None, readme_chars=20000, strategy=None, \*\*kwargs)

The local package ecosystem, scanned from the `.pth` manifest.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

#### *classmethod* from_records(, name, fetcher, strategy=None, metadata_keys=None, id_key='id', text_key='text', \*\*kwargs)

Records produced by a **named callable** as a corpus.

The generic, *persistable* counterpart of [`from_mapping()`](#ir.CorpusSource.from_mapping): where
`from_mapping` takes an in-process object (so a registry entry cannot
name it), this takes a `fetcher` — a callable **or** a
`"module:attr"` import reference (see `resolve_fetcher()`) — so a
corpus owned by another package round-trips through
`~/.config/ir/corpora.json` and rebuilds in a fresh process.

Each record is a mapping; its `id_key` is the artifact id (falling back
to `<name>_<i>` so no record is silently dropped). `metadata_keys`
names the record fields lifted into the strategy’s **filter fields** —
the JSON-friendly stand-in for a `metadata_of` callable, which no
registry entry could carry. The default strategy chunks `text_key`.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

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
(`resolve_fetcher()`), which is the form a registry entry can carry.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

#### items()

Iterate `(artifact_id, raw)` pairs over the corpus scope.

### *class* ir.CorpusStore(meta, vectors, ledger, config, calibration=None, links=None, , packed_dir=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Repository bundling the meta/vectors/ledger/config views of one corpus.

#### calibration_modes()

The ranking modes that currently have a stored calibration.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### delete_ledger_entry(key)

Remove a ledger entry; a missing key is tolerated.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete_links(artifact_id)

Remove an artifact’s edges; a missing entry is tolerated.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete_record(record_id)

Remove a record’s metadata + vector; a missing id is tolerated.

The meta goes first, so the id stops being listed before its vector
disappears. Each removal tolerates the file being gone already (another
process may be deleting the same record).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_calibration(mode)

The stored calibration record for ranking `mode` (`None` if absent).

A deep copy, so a caller cannot mutate the nested `grid` back into the
stored record (in-memory stores share their objects by reference).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_config()

The persisted corpus build settings (empty dict if never written).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### get_ledger_entry(key)

The ledger entry for *key* (`None` if absent).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_links(artifact_id)

The outgoing edges of *artifact_id* — `{edge_type: [target, ...]}`.

Empty dict when the artifact has no stored edges (or no links view).
A copy, so a caller cannot mutate the persisted adjacency in place.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### get_maintenance_state()

Background-work bookkeeping (e.g. `last_maintained`); `{}` if unset.

Kept under a separate `config`-view key from the build settings: it is
regenerable scheduler state (when `ir maintain` last ran), not part of
the corpus’s build identity, so it must never clobber it.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### get_record(record_id)

Reassemble the [`Record`](ir.base.html.md#ir.base.Record) for *record_id* (`KeyError` if absent).

* **Return type:**
  [`Record`](ir.base.html.md#ir.base.Record)

#### ledger_items()

Iterate `(key, entry)` ledger pairs (the ledger may be mutated while iterating).

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### link_items()

Iterate `(artifact_id, {edge_type: [target]})` adjacency pairs.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### *classmethod* local(name)

File-backed store under `~/.local/share/ir/corpora/<name>`.

* **Return type:**
  [`CorpusStore`](ir.store.html.md#ir.store.CorpusStore)

#### matrix()

Return `(record_ids, normalized_matrix, metas)` for brute force.

Rows are L2-normalized so cosine similarity is a dot product. Empty
corpora return a `(0, 0)` matrix.

Caching is two-tier: an in-process cache (invalidated on the next write)
backed, for file-rooted stores, by an on-disk **packed** cache — one
normalized-matrix `.npy` plus its ids/metas, written once and reloaded
with a single memory-mapped read. The packed cache turns a cold reopen
from a per-record vector-file storm (thousands of tiny reads) into three
file reads; it is cleared by any record write, so it never goes stale.

Another process may be writing or deleting records while this one
rebuilds. A record that vanishes or is only half-written when read is
left out of the result (it is “not yet written”), with a warning; such
a partial result is cached in-process but never published as the
packed set, so a fresh process reads the records again.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `ndarray`, [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### *classmethod* memory()

In-memory store (no dependencies); ideal for tests.

* **Return type:**
  [`CorpusStore`](ir.store.html.md#ir.store.CorpusStore)

#### metas()

Return `(record_ids, metas)` **without** loading any vectors.

The vector-free counterpart of [`matrix()`](#ir.CorpusStore.matrix), for ranking modes that
score on text alone (`mode="lexical"`): they need candidate metadata
(text + filter fields) but never the embedding matrix, so they must not
pay its I/O. Reuses the in-process or packed cache when present; else
reads only the `meta` view (not `vectors`), skipping a record that
vanishes or is half-written mid-read (as [`matrix()`](#ir.CorpusStore.matrix) does).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### put_record(record)

Persist *record*’s metadata + vector, invalidating the search matrix.

The vector is written **before** the meta: record ids are listed from
the meta view, so a reader in another process that lists an id always
finds its vector (`delete_record` removes in the reverse order).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### record_ids()

Iterate the record ids currently stored.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### set_calibration(mode, record)

Persist a calibration `record` for ranking `mode` (one per mode).

`mode` keys a file in the calibration store, so it must be a non-empty
string with no path separator (the real modes — `dense` / `lexical` /
`hybrid` — already satisfy this).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_config(settings)

Persist the corpus build *settings* (name / embedder spec + id).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_ledger_entry(key, entry)

Write the ledger *entry* (version / embedder id / record ids) for *key*.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_links(artifact_id, edges)

Persist *artifact_id*’s outgoing *edges* (`{edge_type: [target]}`).

Empty edge-type lists are dropped; an empty result deletes the entry
(no empty adjacency rows linger). Targets are stored verbatim — a bare
`artifact_id` or a `[source, artifact_id]` pair.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_maintenance_state(state)

Persist the maintenance bookkeeping for this corpus.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* ir.CoverageReport(name, on_disk, indexed, excluded, missing=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Disk-vs-index coverage for a filesystem-backed corpus.

#### *property* coverage_ratio *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Fraction of should-index files that are actually indexed (1.0 = full).

#### *property* should_index *: [int](https://docs.python.org/3/builtins/functions.html#int)*

On-disk files that pass the inclusion rule (indexed + missing).

### *class* ir.Disclosure(artifact_id, level, name, score, summary, body=None, pointer=None, metadata=<factory>, source=None, passage=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The progressively-disclosed payload for one selected artifact.

#### artifact_id

the artifact this payload belongs to.

#### level

how much was loaded — `"metadata"` (no I/O), `"body"` (the
pointer’s full text), or `"bundled"` (body + extras).

#### name

a display name (the `name` filter field, else the id).

#### score

the selecting hit’s score.

#### summary

the matched surface text — always present, always cheap.

#### body

the full payload (SKILL.md / file text); `None` below
`"body"` level or when the pointer could not be read.

#### pointer

the source pointer (`skill_path` / `path`) — the “package
pointer” an agent follows to act; `None` if the hit has none.

#### metadata

the hit’s filter metadata, plus a `disclosure` note when a
pointer was present but unreadable (stale/moved/deleted), and an
`expansion` note when expansion was requested but not possible
for this hit.

#### source

the corpus/source name the selecting hit came from (`None`
when unattributed) — the attribution a federated caller needs to
tell two same-id artifacts from different corpora apart.

#### passage

the expanded neighborhood text (`disclose(..., expand=...)`)
— the mid-granularity payload between `summary` (the matched
surface) and `body` (the pointer’s full text); `None` when
expansion was not requested or not possible. Assembled from the
corpus’s *stored records* (see [`ir.expand`](#ir.expand)), unlike `body`,
which dereferences the pointer to an external resource.

#### to_dict()

JSON-serializable form (score cast to `float`).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.DiscoveryResult(query, mode, strategy, disclose_level, results, abstained, reason, n_retrieved, signals=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of [`discover()`](#ir.discover) — retrieve → select → (optional) disclose.

The qh-exposable payload: [`to_dict()`](#ir.DiscoveryResult.to_dict) is fully JSON-serializable (lists
of dicts, floats, strings, bools — no numpy, no objects), so a FastAPI
facade can return it directly.

#### *property* ids *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

The committed artifact ids, best-first.

#### to_dict()

JSON-serializable result for the qh / HTTP surface.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.GraphStore(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Structural contract a traversal operator binds to — node + neighbors.

Deliberately minimal (two methods) so it is satisfied by ir’s
[`CorpusGraph`](#ir.CorpusGraph) *and* by any external graph: `__getitem__` resolves
a node id to its scorable payload, `neighbors` lists adjacent node ids
(optionally of one edge type). Granularity-agnostic on purpose — an
artifact graph and a surface-level tree are both 

```
``
```

GraphStore\`\`s.

`runtime_checkable` makes `isinstance(x, GraphStore)` a *structural*
check on attribute **names** only (not signatures) — enough to tell a
conforming adapter from an arbitrary object, but it cannot validate that
`neighbors` takes the right arguments; treat it as a smoke check.

### *class* ir.IndexPlan(filter_fields=<factory>, surfaces=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy)’s output for one artifact.

### *class* ir.IndexingStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Decompose one artifact into filter fields + embeddable surfaces.

### *class* ir.MaintenancePolicy(reindex=<factory>, synopsis=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The background-work policy for one corpus (reindex + synopsis).

#### merged(override)

Layer an `override` dict on top of this policy (entry over defaults).

* **Return type:**
  [`MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy)

### *class* ir.MaintenanceResult(name, ran, reason, reindex=False, synopsis=False, records=None, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What [`maintain_corpus()`](#ir.maintain_corpus) did (or would do) for one corpus.

### *class* ir.Package(, chunk_size=1500, overlap=200, embed_deps=False, deps_template=None, max_tokens=None, token_model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Package strategy: `name + description` surface plus README chunks.

Filter fields capture ownership (ours vs third-party), name, deps. AI
synopsis / problem-class surfaces are a documented extension point.

With `embed_deps=True`, `decompose` additionally emits one
`Surface(kind="deps", granularity="field")` whose text is a prefix-form
serialization of the **bare** dependency names (`deps_template`, default
`_default_deps_text()`) — so a query for a domain matches a package by the
libraries it depends on (e.g. `sentence-transformers` -> embeddings,
`networkx` -> graphs), and the BM25 leg picks up exact dep-token matches. The
deps bag is kept **separate from prose** (its own surface) so a rare library
name is not diluted, and deps remain a filter field regardless. `embed_deps`
defaults `False` (today’s behavior); it folds into the strategy id, so
toggling it re-decomposes incrementally. The deps surface is appended **last**,
leaving the `description` (position 0) and `readme_chunk` indices unchanged.

Surface indexing: the `description` surface (kept whenever *name* or
*description* is non-empty) occupies plan position 0, so `readme_chunk`
*j* is stored with
`Record.surface_index == j + 1` while its surface metadata says
`chunk_index == j` — `surface_index` is plan-global, `chunk_index`
per-kind (see [`ir.base.Record.make_id()`](ir.base.html.md#ir.base.Record.make_id)). Never derive sibling record
ids from `chunk_index`; use the ledger
([`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact)). `n_chunks` is stamped on
readme chunks at decompose time, but corpora built before the stamp keep
records without it until the artifact re-indexes (content / embedder /
strategy change) — read it with `metadata.get("n_chunks")`.

### *class* ir.Passage(artifact_id, surface_kind, score, text, record_ids=(), source=None, surface_index=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An expanded hit: the seed’s identity + the stitched neighborhood text.

`artifact_id` / `surface_kind` / `score` / `source` /
`surface_index` are the **seed hit’s** — expansion never disturbs hit
identity `(source, artifact_id)` or scores. `text` is the assembled
neighborhood (overlap-deduped, plan order) and `record_ids` the ordered
stored segments it was stitched from (empty when expansion degraded to the
seed’s own text).

#### to_dict()

JSON-serializable form (`score` cast to `float`, ids as a list).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.Record(id, artifact_id, surface_kind, surface_index, text, vector, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A stored, embedded surface — one row of the index.

#### *static* make_id(artifact_id, surface_kind, surface_index)

Deterministic storage id for a surface of an artifact.

`surface_index` is the surface’s **plan-global** position — its
enumeration index across *all* surfaces of the artifact’s
[`IndexPlan`](#ir.IndexPlan), regardless of kind — as assigned by
[`ir.index.build()`](ir.index.html.md#ir.index.build). On multi-kind strategies it therefore differs
from per-kind counters like `metadata["chunk_index"]` (e.g.
[`Package`](ir.strategy.html.md#ir.strategy.Package): the `description` surface takes
position 0, shifting `readme_chunk` *j* to `surface_index` `j+1`
— and the offset is plan-dependent, since empty surfaces are dropped).

Ids of already-built corpora are a stability contract: never re-derive
a sibling’s id from a per-kind index — address siblings through the
ledger via [`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* ir.ReindexPolicy(on='source-change', every_hours=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

When to (incrementally) rebuild a corpus.

### *class* ir.SearchHit(artifact_id, surface_kind, score, text, metadata=<factory>, source=None, surface_index=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A scored record returned by retrieval (higher score = closer).

Maps onto ir_09’s `Result`: `text` is the snippet, `score` the rank
score, `metadata` the meta, and [`pointer`](#ir.SearchHit.pointer) the key into a resource
store (ir_09 §5). [`to_dict()`](#ir.SearchHit.to_dict) is the serialization-clean form for a
cross-process / subagent boundary (no numpy scalars leak).

`source` is the corpus/source name the hit came from (`None` when
unattributed — e.g. an ad-hoc corpus without a name). It is a first-class
field, not a metadata key, because `metadata` is the strategy-owned
hard-filter namespace and provenance is structural: artifact identity is
only unique *within* a source, so any cross-source operation keys on
`(source, artifact_id)` (see `best_per_artifact()`).

`surface_index` is the stored `Record.surface_index` of the hit’s
surface — its plan-global position among the artifact’s surfaces — so a
hit can name *which* surface of its artifact it is (the prerequisite for
sibling addressing and context expansion). `None` when unknown (e.g. a
hand-built hit). It is **not** the per-kind `metadata["chunk_index"]`;
see [`Record.make_id()`](#ir.Record.make_id) for why the two differ on multi-kind
strategies.

#### *property* pointer *: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The disclosure pointer on this hit, if any (see `POINTER_KEYS`).

#### to_dict()

JSON-serializable form (`score` cast to a Python `float`).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.Selection(selected, candidates, abstained, reason, signals=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A selector’s commitment: the chosen subset of a ranked candidate list.

#### selected

the committed hits, best-first (empty iff `abstained`).

#### candidates

the full ranked input, kept for provenance / audit.

#### abstained

True iff the selector committed to nothing by policy.

#### reason

which rule ended the commit (e.g. `"rel_threshold"`,
`"score_gap"`, `"max_k"`, `"abstain:below_floor"`).

#### signals

concrete, defined numbers behind the decision (`top_score`,
`n_candidates`, `n_selected`, `min_ratio`) — the auditable
replacement for an opaque “confidence” float.

#### *property* selected_ids *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

The committed artifact ids, best-first.

#### *property* sufficient *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

A model-free sufficiency *hint* for an agent’s Evaluator (ir_09 §3).

`True` when this selection committed to at least one item (i.e. did not
abstain). It is a **signal, not a directive**: the re-query / `refinement`
decision and the loop belong to the agent layer (the back-edge, ir_09 §4)
— `ir` derives this from its own outcome and never acts on it.

#### to_dict()

JSON-serializable form (scores cast to `float`).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.Skill

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Capability strategy: embed `name + description` only.

The body (SKILL.md) is loaded post-selection and is *not* indexed; name and
parent are filter fields.

### *class* ir.Surface(artifact_id, kind, text, granularity='document', metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One embeddable unit derived from an artifact.

`kind` names the surface type (e.g. `"description"`, `"synopsis"`,
`"problem_class"`, `"chunk"`) so a query can match the *right part* of
an artifact. `granularity` is a coarse hint (`"document"` / `"chunk"`
/ `"field"`). `metadata` is surface-local (e.g. chunk offsets).

### *class* ir.SynopsisPolicy(enabled=False, scope='recent', window_days=30, downtime_hours=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Whether/when to attach (expensive, LLM-generated) synopses.

`downtime_hours` is a `[start, end)` pair of local-clock hours
(wrapping past midnight is allowed, e.g. `(22, 6)`); `None` means
“any time”. `scope="recent"` limits synthesis to artifacts whose
timestamp is within `window_days` (corpora that expose a time signal);
`scope="all"` synthesizes every artifact (bounded by incrementality).

### *class* ir.WalkPolicy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The pluggable strategy of a walk — graph semantics, not safety.

`seed` produces the initial frontier; `score` ranks a node against the
query; `select` chooses which scored frontier nodes to commit/expand this
step (beam/greedy — default: all, best-first); `expand` yields a node’s
neighbors; `node_id` is the hashable visited-set key; `stop` is the
injected sufficiency check; `to_hit` materializes a committed node as a
[`SearchHit`](ir.base.html.md#ir.base.SearchHit) — or `None` for a *router-only* node (a
summary that routes but is not itself a result).

### *class* ir.WalkState(query, max_depth, budget, visited=<factory>, results=<factory>, cache=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The operator-owned state of one [`traverse()`](#ir.traverse) call — the safety home.

`visited` (node ids already committed), `budget`, and `max_depth` are
the structural safety primitives the operator enforces; `results` are the
emitted hits; `cache` is scratch space a policy may use (e.g. to embed the
query once). A policy reads this but the *operator* enforces the bounds —
a policy cannot opt out of termination.

### *class* ir.WholeText(, text_key=None, kind='document')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One surface = the entire text. Sensible default for a naive corpus.

### ir.as_retriever(corpus_or_name, \*\*search_defaults)

Bind ONE corpus to the uniform `Retriever` contract.

Returns `retrieve(query, **overrides) -> list[SearchHit]` that calls
[`search()`](#ir.search) with `search_defaults` (a per-call kwarg overrides a bound
default). A corpus *name* is resolved once via [`ir.open_corpus()`](#ir.open_corpus); pass
an open [`Corpus`](ir.index.html.md#ir.index.Corpus) to skip that. The returned callable carries
the bound corpus on `.corpus` for introspection.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]

```pycon
>>> retr = as_retriever(corpus, mode="hybrid", k=20)
>>> hits = retr("how do I deploy the app")
>>> hits = retr("deploy", filter={"owner": "me"})
```

### ir.build(source, , store=None, embedder=None, full=True, batch_size=256, edge_extractor=None)

Build or incrementally update *source* into a [`Corpus`](#ir.Corpus).

* **Parameters:**
  * **store** ([`CorpusStore`](ir.store.html.md#ir.store.CorpusStore) | [`None`](https://docs.python.org/3/builtins/constants.html#None))
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any))
  * **full** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool))
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int))
  * **edge_extractor** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – (`(artifact_id, filter_fields) -> {edge_type: [target]}`) that
    populates the corpus’s semantic `links` graph (see [`ir.graph`](ir.graph.html.md#module-ir.graph);
    pass [`ir.default_edge_extractor()`](#ir.default_edge_extractor) for the latent deps/parent
    edges). Ingest is **eager** — edges are (re)written for *every*
    in-scope artifact, a decompose-only pass with no embedding, so the
    graph never goes partially stale — while embedding stays fully
    incremental. Edges are derived state, **not** part of build identity.
    A rebuild *without* an extractor leaves existing edges untouched
    (they are only refreshed by re-running with one, and only cleared per
    artifact by the `full` prune below) — so dropping `edge_extractor`
    does not wipe a graph.
* **Return type:**
  [`Corpus`](ir.index.html.md#ir.index.Corpus)

### ir.build_corpus(name, \*\*kwargs)

Build (or update) a registered/preset corpus by *name*; returns a [`Corpus`](ir.index.html.md#ir.index.Corpus).

`**kwargs` are forwarded to [`ir.build()`](#ir.build) — notably `store`,
`embedder` (e.g. `"light"` for the numpy-only hashing embedder),
`full` (prune artifacts no longer in the source), and `batch_size`.

### ir.canonical_node_id(target, , source)

Canonicalize a neighbor *target* to a `(source, artifact_id)` node id.

The repo’s node identity is `(source, artifact_id)` — the key a traversal
visited-set must use so the same id in two corpora stays two nodes.
[`CorpusGraph.neighbors()`](#ir.CorpusGraph.neighbors) returns targets in stored form: a bare
`artifact_id` (implicitly in *source*, the graph it came from) or a
`[source, artifact_id]` cross-corpus pair. This resolves either to the
canonical tuple.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> canonical_node_id("dol", source="packages")
('packages', 'dol')
>>> canonical_node_id(["skills", "deploy"], source="packages")
('skills', 'deploy')
```

### ir.collapsed_tree_policy(, summary_kinds=('description', 'synopsis', 'capability', 'document'), leaf_kinds=('chunk', 'readme_chunk'), seed_k=10)

The pure-vector summary-routing / collapsed-tree [`WalkPolicy`](#ir.WalkPolicy).

Seeds on the top *seed_k* matches among `summary_kinds` surfaces and
descends to each routed artifact’s `leaf_kinds` surfaces (the emitted
results), scored by cosine to the query. No LLM in the loop. A summary
surface is a *router* (suppressed from results) only when its artifact has
leaf surfaces; on a single-surface corpus (WholeText `document`, Skill
`capability`) the summaries are leaf-less and emitted directly, so the
walk degrades to flat-over-summaries instead of returning nothing.

The defaults keep `document` / `capability` in `summary_kinds` \*on
purpose\* — that is what lets a WholeText / Skill corpus seed at all; the
structural router check (above) is what keeps those seeds from being
silently swallowed.

* **Return type:**
  [`WalkPolicy`](#ir.WalkPolicy)

```pycon
>>> hits = traverse(q, corpus, policy=collapsed_tree_policy())
```

### ir.corpora()

All registered corpus definitions, keyed by name.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.default_edge_extractor(artifact_id, filter_fields)

Edges latent in the standard filter fields: `deps` → REF, `parent` → PARENT.

- `deps` (Package) → `REF` edges to each dependency’s bare name
  (version specifiers / extras / markers stripped). Self-edges and blanks
  are dropped.
- `parent` (Skill) → a single `PARENT` edge.

A package whose `deps` name other packages in the same corpus gets
intra-corpus REF edges; third-party deps become REF edges to ids not in the
corpus (harmless — [`CorpusGraph.neighbors()`](#ir.CorpusGraph.neighbors) lists them, and a
traversal simply finds no node to expand).

Self-edges are dropped case-insensitively (`_dep_name` lower-cases, so a
package `"AA"` depending on `"aa"` is recognized as a self-reference).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]

### ir.default_policy_for_kind(kind)

The smart-default [`MaintenancePolicy`](#ir.MaintenancePolicy) for a corpus `kind`.

* **Return type:**
  [`MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy)

### ir.disclose(selection, , level='body', loader=None, store=None, expand=None, corpus=None)

Reveal the payload of each selected hit at `level` — append-only, pure.

* **Parameters:**
  * **selection** ([`Selection`](#ir.Selection)) – a committed [`Selection`](#ir.Selection).
  * **level** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"metadata"` (no I/O — summary + pointer only), `"body"` (load
    the pointer’s full text), or `"bundled"` (body + extras; today the
    same as `"body"`, reserved for bundled scripts/references).
  * **loader** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – override the body resolver — `metadata -> str | None`. The
    default reads the `skill_path` / `path` pointer from disk and
    tolerates a missing target (returns `None`, never raises).
  * **store** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – a `ResourceStore` (`pointer -> payload` `Mapping`) to
    dereference instead of disk — ir_09 §5 pointer-passing over a `dol`
    store / URL map / blob storage. Mutually exclusive with `loader`.
  * **expand** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]]]) – a `NeighborhoodPolicy` to also stitch each
    hit’s neighborhood from the corpus’s stored records into
    [`Disclosure.passage`](#ir.Disclosure.passage) (see [`ir.expand`](#ir.expand)). Orthogonal to
    `level`, which governs *pointer* payloads: e.g.
    `level="metadata", expand=sentence_window_policy()` reads no
    pointer at all but still returns mid-granularity passages.
    Requires `corpus=`.
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – where `expand` finds each hit’s stored siblings — a
    [`Corpus`](ir.index.html.md#ir.index.Corpus) / [`CorpusStore`](ir.store.html.md#ir.store.CorpusStore) / name,
    or, for cross-source selections, a `{source_name: corpus}`
    `Mapping` resolved per hit via `hit.source`. Only meaningful
    with `expand=`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Disclosure`](#ir.Disclosure)]
* **Returns:**
  one [`Disclosure`](#ir.Disclosure) per selected hit, best-first. This is a pure
  read: the [`Selection`](#ir.Selection) and its hits are never mutated, so a caller
  can disclose append-only without disturbing a cached ranked prefix.

### ir.discover(corpus, query, , k=10, mode='hybrid', strategy='conservative', disclose_level='metadata', filter=None, surfaces=None, max_k=3, rel=0.9, gap_ratio=0.5, min_score=None, merge='rrf', merge_weights=None, merge_rrf_k=None, loader=None, store=None, expand=None, \*\*search_kw)

Find and commit to the capabilities for `query` — the one search tool.

Retrieves `k` candidates, commits to a distractor-robust subset, and
(optionally) discloses each committed item’s payload. This is the single
agent-callable surface the capability-discovery research argues for: one
tool that returns *few, high-precision* answers rather than a long candidate
list the model must then filter under context rot.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a built [`Corpus`](ir.index.html.md#ir.index.Corpus), or a registered corpus
    **name** (resolved with [`ir.open_corpus()`](#ir.open_corpus)). Pass a *name* for
    the qh / HTTP surface — it is the JSON-friendly form. Pass a
    **list/tuple of names (or Corpus objects)** for single-shot
    *federated* discovery across several corpora: each is searched,
    per-source abstention floors gate before any merging, and the
    survivors are rank-fused (see `merge`). The caller names the
    sources explicitly; ir never chooses the set (source planning is
    the agent layer’s job, ir_09 §3).
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the user intent.
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – candidate depth retrieved before selection. Federated: `k`
    candidates are retrieved *per source*, and the fused ranking is
    also truncated to `k` before selection.
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode — `"hybrid"` (default; `ir`’s strongest overall),
    `"dense"`, or `"lexical"`.
  * **strategy** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]]) – selection strategy (see [`select()`](#ir.select)).
  * **disclose_level** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"metadata"` (default; cheap, no body I/O), `"body"`,
    or `"bundled"`.
  * **filter** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – retrieval constraints (forwarded to
    [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search)).
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – retrieval constraints (forwarded to
    [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search)).
  * **max_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – selection parameters (see
    [`select()`](#ir.select)). `min_score="auto"` loads the floor calibrated for
    this `(corpus, mode)` by [`ir.eval.calibrate_min_score()`](ir.eval.html.md#ir.eval.calibrate_min_score) and
    persisted on the corpus — the opt-in that turns on absolute
    abstention; it falls back to no floor (with a warning) when no
    calibration is stored or it is stale (a different embedder).
    **Federated**: floors are per-(corpus, mode, embedder), so a single
    number cannot apply across corpora — pass `"auto"` (each source’s
    own calibrated floor), a `{name: floor_or_"auto"}` mapping, or
    `None`; a bare float raises. Floors gate each source on its own
    raw scores *before* fusion; the fused ranking is never floored
    (rank-fused scores are ordinal — ir_07/ir_08).
  * **rel** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – selection parameters (see
    [`select()`](#ir.select)). `min_score="auto"` loads the floor calibrated for
    this `(corpus, mode)` by [`ir.eval.calibrate_min_score()`](ir.eval.html.md#ir.eval.calibrate_min_score) and
    persisted on the corpus — the opt-in that turns on absolute
    abstention; it falls back to no floor (with a warning) when no
    calibration is stored or it is stale (a different embedder).
    **Federated**: floors are per-(corpus, mode, embedder), so a single
    number cannot apply across corpora — pass `"auto"` (each source’s
    own calibrated floor), a `{name: floor_or_"auto"}` mapping, or
    `None`; a bare float raises. Floors gate each source on its own
    raw scores *before* fusion; the fused ranking is never floored
    (rank-fused scores are ordinal — ir_07/ir_08).
  * **gap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – selection parameters (see
    [`select()`](#ir.select)). `min_score="auto"` loads the floor calibrated for
    this `(corpus, mode)` by [`ir.eval.calibrate_min_score()`](ir.eval.html.md#ir.eval.calibrate_min_score) and
    persisted on the corpus — the opt-in that turns on absolute
    abstention; it falls back to no floor (with a warning) when no
    calibration is stored or it is stale (a different embedder).
    **Federated**: floors are per-(corpus, mode, embedder), so a single
    number cannot apply across corpora — pass `"auto"` (each source’s
    own calibrated floor), a `{name: floor_or_"auto"}` mapping, or
    `None`; a bare float raises. Floors gate each source on its own
    raw scores *before* fusion; the fused ranking is never floored
    (rank-fused scores are ordinal — ir_07/ir_08).
  * **min_score** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – selection parameters (see
    [`select()`](#ir.select)). `min_score="auto"` loads the floor calibrated for
    this `(corpus, mode)` by [`ir.eval.calibrate_min_score()`](ir.eval.html.md#ir.eval.calibrate_min_score) and
    persisted on the corpus — the opt-in that turns on absolute
    abstention; it falls back to no floor (with a warning) when no
    calibration is stored or it is stale (a different embedder).
    **Federated**: floors are per-(corpus, mode, embedder), so a single
    number cannot apply across corpora — pass `"auto"` (each source’s
    own calibrated floor), a `{name: floor_or_"auto"}` mapping, or
    `None`; a bare float raises. Floors gate each source on its own
    raw scores *before* fusion; the fused ranking is never floored
    (rank-fused scores are ordinal — ir_07/ir_08).
  * **merge** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – federated only — how the per-source rankings combine: `"rrf"`
    (default; rank-based, scale-free — see
    [`ir.retrieve.fuse_hits()`](ir.retrieve.html.md#ir.retrieve.fuse_hits)), `"score"` (raw-score merge,
    valid only when all corpora share an embedder — verified, raises
    on mismatch), or a callable `{name: hits} -> hits`.
  * **merge_weights** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – federated only — per-source trust weights for
    `merge="rrf"` (default 1.0 each).
  * **merge_rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – federated only — the cross-source RRF rank constant
    (default: [`DFLT_RRF_K`](ir.retrieve.html.md#ir.retrieve.DFLT_RRF_K); distinct from the
    within-corpus hybrid `rrf_k` in `search_kw`).
  * **loader** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – optional body resolver for disclosure (see [`disclose()`](#ir.disclose)).
  * **expand** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]]]) – a `NeighborhoodPolicy` — also stitch each
    committed hit’s neighborhood from its corpus’s stored records into
    [`Disclosure.passage`](#ir.Disclosure.passage) (retrieval-time context expansion, see
    [`ir.expand`](#ir.expand)). Works at any `disclose_level`; the federated
    form resolves each hit’s corpus via its `source`.
  * **\*\*search_kw** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – any other [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search) keyword (`rrf_k`,
    `rerank`, `bm25`, …).
* **Return type:**
  [`DiscoveryResult`](#ir.DiscoveryResult)
* **Returns:**
  a [`DiscoveryResult`](#ir.DiscoveryResult) (`.to_dict()` for JSON / qh). Federated
  results add `signals["per_source"]` (per-corpus `n_retrieved` /
  `top_score` / `floor` / `abstained`) and each disclosure carries
  its `source`.

### ir.expand(hit, corpus, , policy=None)

Expand *hit* into a [`Passage`](#ir.Passage) of its neighborhood in *corpus*.

Fetches the hit’s sibling records through the ledger
([`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact)), asks *policy* which to keep,
and stitches them in plan order with overlap-aware dedupe. The default
policy is a ±:data:`DFLT_WINDOW` sentence window; pass
[`parent_policy()`](#ir.parent_policy) for the whole artifact, or any
`NeighborhoodPolicy`.

* **Parameters:**
  * **hit** ([`SearchHit`](ir.base.html.md#ir.base.SearchHit)) – the seed [`SearchHit`](ir.base.html.md#ir.base.SearchHit) (its identity and score pass
    through to the [`Passage`](#ir.Passage) untouched).
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a [`Corpus`](ir.index.html.md#ir.index.Corpus), [`CorpusStore`](ir.store.html.md#ir.store.CorpusStore),
    or corpus name — whatever [`records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact)
    accepts. Must be the corpus the hit came from.
  * **policy** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]]]) – which siblings make up the neighborhood (default: sentence
    window). A policy that selects nothing degrades the passage to the
    hit’s own text (`record_ids=()`) rather than returning nothing.
* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – the corpus has no ledger entry for the hit’s artifact
        ([`ir.retrieve.NoLedgerEntry`](ir.retrieve.html.md#ir.retrieve.NoLedgerEntry)), or the ledger is stale —
        an entry listing records missing from the store.
  * **SeedNotFound** – the default window policy cannot find the seed among
        its artifact’s stored records (stale hit / wrong corpus).
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – the policy returned records that are not siblings of the
        hit’s artifact (operator-enforced safety), or the seed hit lacks
        `surface_index` (hand-built hit) under the default window
        policy — use [`parent_policy()`](#ir.parent_policy), which needs no seed position.
* **Return type:**
  [`Passage`](#ir.Passage)

### ir.fuse_hits(hits_by_source, , rrf_k=60, weights=None, identity=None, k=None)

Merge per-source ranked hit lists into one ranking — by rank, not score.

The cross-source counterpart of the within-corpus hybrid fusion: scores
from different (corpus, mode, embedder) tuples live on incommensurable
scales (ir_07: “a different model re-scales everything”), so \*\*raw scores
never cross the source boundary\*\* — within each source they order and
dedup that source’s hits (one scale, sound), and across sources only ranks
interact, via weighted Reciprocal Rank Fusion: each hit contributes
`weights[source] / (rrf_k + rank)`.

* **Parameters:**
  * **hits_by_source** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]) – `{source_name: ranked hits}`. Hits without a
    `source` are stamped with their mapping key (existing tags win,
    so one corpus bound under two keys still counts as one source).
    A `None` key is the *untagged pseudo-source*: its hits fuse as
    one rank group and stay unattributed (`source=None`). Within
    each list, duplicate artifacts — and, when `identity` is given,
    identity-duplicates — collapse to their best raw score before
    ranking, so a multi-query / multi-round pool can never
    double-count one artifact’s RRF mass.
  * **rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – the RRF rank constant (standard default 60).
  * **weights** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional per-source trust dial (default 1.0 each) — a
    source’s contribution scales linearly, no score comparability
    needed. Keys naming sources absent from `hits_by_source` are
    ignored (a per-round pool may legitimately lack a configured
    source); callers with a closed source set should validate keys
    upfront, as federated [`ir.discover()`](#ir.discover) does.
  * **identity** (`Union`[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – how cross-source duplicates merge — see `Identity`.
    Default `None`: never; each `(source, artifact_id)` stays a
    distinct result.
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – truncate the fused ranking to this many hits.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]
* **Returns:**
  the fused hits, best-first. Each carries the fused score in `score`
  and keeps its pre-fusion magnitude as `metadata["source_score"]` (+
  `"source_rank"`), so downstream consumers (abstention gates, LLM
  judges) never lose the per-source signal. When an `identity` merge
  combined several sources’ hits, `metadata["fused_sources"]` lists
  them and the representative hit is the one with the best rank.
  **Single-source input passes through with raw scores** (RRF of one
  list is that list’s order — same convention as the hybrid fusion’s
  single-channel fallback), so the fused-score rescaling only happens
  when there is genuinely something to fuse. The post-fusion `score`
  is ordinal: valid for ordering and relative cuts, meaningless against
  absolute floors — apply calibrated `min_score` floors per source,
  *before* fusing (see ir_07/ir_08 and `ir.discover`’s federated form).

### ir.maintain(name=None, , all=False, now=None, dry_run=False)

Run due background work for one corpus (*name*) or every registered one (*all*).

With neither, defaults to all registered corpora. Returns one
[`MaintenanceResult`](#ir.MaintenanceResult) per corpus considered.

**A named corpus raises; a sweep records.** Asking for one corpus by name is a
direct request, and swallowing its error would turn a typo’d name into a
result that reads as success to any caller written before `error` existed.
A sweep is the unattended case, where the opposite is true: one corpus
failing must not leave every corpus after it stale with nothing to say so.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`MaintenanceResult`](ir.maintenance.html.md#ir.maintenance.MaintenanceResult)]

### ir.maintain_corpus(name, , now=None, dry_run=False, full=True)

Do the due background work for one corpus (idempotent).

Reads the corpus’s resolved policy and its `last_maintained` time, decides
whether a (synopsis-aware) reindex is due *and* permitted now, and — unless
`dry_run` — runs the incremental build and records the run.

* **Return type:**
  [`MaintenanceResult`](ir.maintenance.html.md#ir.maintenance.MaintenanceResult)

### ir.make_llm_formulator(, rewriter=None, prompt='Rewrite the search query into {n} short, diverse alternative search queries that would retrieve the same target documents: fix typos, expand jargon, and add synonyms, but keep each a terse search phrase. One query per line, no numbering.\\\\n\\\\nQuery: {query}', n=3, fallback=None, \*\*prompt_function_kwargs)

An LLM-backed `Formulator` (rewrite / expand / multi-query).

`rewriter` is an injectable `query -> str | [str, ...]` callable (a test
double, or your own router); when omitted it is built lazily on `aix`
(`aix.prompt_func`), so importing this module stays offline. `n` is the
multi-query fan-out width. Any error or empty reply falls back to `fallback`
(default: `identity_formulator()`).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.make_llm_synthesizer(, summarize=None, prompt='Write a concise synopsis (2-4 sentences) of the document below: what it is about and what questions it answers, so that a search over synopses can route to it. Output only the synopsis, no preamble.\\\\n\\\\nDocument:\\\\n{text}', model=None, synthesizer_id=None, text_key=None, \*\*prompt_function_kwargs)

An LLM-backed `Synthesizer` ([`Artifact`](ir.base.html.md#ir.base.Artifact) → synopsis).

`summarize` is an injectable `text -> str` callable (a test double, or
your own summarizer); when omitted it is built lazily on `aix`
(`aix.prompt_func`) on the **first** synthesis and reused — so importing
this module, and even constructing the synthesizer, stays offline. The
artifact’s text is extracted with [`ir.strategy.text_of()`](ir.strategy.html.md#ir.strategy.text_of) using
`text_key` — which [`with_synopsis()`](#ir.with_synopsis) threads from the inner strategy, so
the synopsis summarizes the *same* field the strategy indexes. An empty text,
or any synthesis error, yields `""` (the surface is then skipped, never a
fabricated summary).

The returned callable carries a `synthesizer_id` attribute (default
`"aix:{model}:{sha(prompt)[:12]}"`) that [`with_synopsis()`](#ir.with_synopsis) reads into the
corpus’s `strategy_id` for staleness — a prompt or model change re-synthesizes.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Artifact`](ir.base.html.md#ir.base.Artifact)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.make_search(corpus, , name=None, description=None, k=8, mode='hybrid')

Return a **corpus-bound** `search(query, k=...) -> dict` tool.

The returned function exposes only `query` (and `k`) — the corpus is fixed
— so a connector built over it surfaces exactly one corpus and nothing else.
Its `__name__` / `__doc__` are set so an MCP/agent host shows a clean tool
name and description. Use this when wiring a single-corpus connector; use
[`search()`](#ir.search) when the caller should choose the corpus.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ir.open_corpus(name, , embedder=None)

Reopen a previously built corpus by name.

The embedding model is **lazily** resolved (see `_LazyEmbedder`): the
returned corpus knows its `embedder_id` from stored config immediately, but
only loads the model when a dense/hybrid query actually embeds. So `ir ls`,
`ir info`, and lexical-only search open a corpus without the model-load
cost. Pass `embedder=` to override the stored spec.

* **Return type:**
  [`Corpus`](ir.index.html.md#ir.index.Corpus)

### ir.parent_policy()

The whole artifact (small-to-big): every stored surface, plan order.

The mid-granularity analogue of `disclose(level="body")` — but assembled
from the *indexed* segments rather than dereferencing the pointer, so it
works for corpora whose artifacts have no on-disk body.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]]

### ir.policy_for(name)

The effective [`ir.policy.MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy) for corpus *name*.

Resolves the registered entry’s `maintenance` over its kind’s smart default
over the global default (see [`ir.policy.resolve_policy()`](ir.policy.html.md#ir.policy.resolve_policy)). An unregistered
name resolves to the global default policy.

### ir.records_for_artifact(store_or_corpus, artifact_id, , surface_kind=None)

All stored records of *artifact_id*, ordered by `surface_index`.

The sibling-addressing primitive beneath retrieval-time context expansion:
a [`SearchHit`](ir.base.html.md#ir.base.SearchHit) names its artifact (and, via
`surface_index`, which surface of it matched);
this returns every surface of that artifact, in plan order, so an expansion
policy can stitch neighbors / parents around the hit.

Resolution is **ledger-backed only**: the artifact’s ledger entry lists its
`record_ids`. Record ids are never re-derived from a per-kind index like
`metadata["chunk_index"]` — on multi-kind strategies that index differs
from the plan-global `surface_index` baked into the id (see
[`ir.base.Record.make_id()`](ir.base.html.md#ir.base.Record.make_id)), so derivation would fetch wrong or missing
siblings.

* **Parameters:**
  * **store_or_corpus** – a [`CorpusStore`](ir.store.html.md#ir.store.CorpusStore), anything carrying
    one as `.store` (e.g. a [`Corpus`](ir.index.html.md#ir.index.Corpus)), or a corpus
    *name* — resolved straight to its local store: sibling lookup
    never embeds, so unlike [`ir.open_corpus()`](#ir.open_corpus) no embedder is
    loaded.
  * **artifact_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the artifact whose surfaces to fetch.
  * **surface_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict to one surface kind (e.g. `"readme_chunk"`);
    a known artifact with no surfaces of that kind yields `[]`.
* **Raises:**
  * [**NoLedgerEntry**](ir.retrieve.html.md#ir.retrieve.NoLedgerEntry) – the ledger has no entry for *artifact_id* (an unknown
        artifact, or a corpus built without [`ir.index.build()`](ir.index.html.md#ir.index.build)’s
        ledger bookkeeping). A `KeyError` subclass.
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – an entry exists but lists a record missing from the store
        (a stale ledger: interrupted build or out-of-band
        `delete_record`) — data corruption, named in the message.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Record`](ir.base.html.md#ir.base.Record)]

### ir.register(name, kind, , embedder='default', strategy=None, maintenance=None, storage=None, \*\*params)

Register (or overwrite) a named corpus definition.

Beyond the v1 `kind` / `embedder` / `params`, an entry may now carry
(all optional, with smart per-kind defaults applied at resolution time — see
[`ir.policy`](ir.policy.html.md#module-ir.policy)):

- `strategy` — an [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy) (or a
  `{"name", "params"}` spec) persisted so the corpus’s *segmentation* is
  stable across rebuilds. `None` keeps the preset’s default strategy.
- `maintenance` — the background-work policy dict (`reindex` / `synopsis`;
  validated here, see [`ir.policy.MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy)).
- `storage` — the persistence backend (default `{"backend": "local"}`).

Entries written by older `ir` (none of these keys) keep working unchanged.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.reports_coverage(name='reports', , projects_root=None, indexed_ids=None, exclude_dirs=frozenset({'.git', '.hg', '.ipynb_checkpoints', '.obsidian', '.svn', '.tox', '.venv', '_\_pycache_\_', 'build', 'dist', 'node_modules', 'site-packages', 'venv'}))

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
  [`CoverageReport`](ir.coverage.html.md#ir.coverage.CoverageReport)

### ir.resolve_policy(entry)

The effective policy for a registry `entry`: entry over kind over global.

A v1 entry (no `maintenance` key) resolves to its kind’s smart default, so
existing corpora gain a sensible policy without a migration.

* **Return type:**
  [`MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy)

### ir.retriever_for(name, \*\*search_defaults)

A [`Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) bound to the registered corpus *name*.

Opens the corpus (it must have been built) and wraps it with
[`ir.as_retriever()`](#ir.as_retriever); `search_defaults` (e.g. `mode="hybrid"`) bind to
every call.

### ir.retrievers(\*\*search_defaults)

A lazy `Mapping[name, Retriever]` view over the registry (ir_09 §8).

The query-time projection of the build-recipe registry: each value is a
ready-to-call [`Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever). This is the source-registry
facade an orchestration layer (`raglab`) consumes — it never opens a corpus
until the key is accessed, and always reflects the current `registered()`
set. `search_defaults` apply to every source.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.search(corpus, query, \*\*kwargs)

Search a [`Corpus`](ir.index.html.md#ir.index.Corpus) (or a corpus *name*, reopened lazily).

Thin facade over [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search); `**kwargs` are forwarded to
it — the useful ones are `k` (how many hits), `mode` (`"dense"` /
`"lexical"` / `"hybrid"`), `filter` (a `vd` Mongo-style metadata
filter), `surfaces` (restrict to surface kinds), and `per_artifact`
(collapse to the best surface per artifact). See [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search)
for the full signature and defaults.

### ir.search_tool(query, , corpus, k=8, mode='hybrid', filter=None)

Search a named `ir` corpus and return a JSON-serializable result dict.

Thin agent-callable wrapper over [`ir.discover()`](#ir.discover) — returns its
`.to_dict()` (committed results, scores, disclosures), fit to hand straight
back from an MCP tool or HTTP endpoint. The *corpus* is a parameter, so this
single function serves any corpus.

* **Parameters:**
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the natural-language query.
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a registered corpus **name** (str), a list of names (federated
    search), or a built [`ir.Corpus`](#ir.Corpus).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum number of results.
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"dense"` | `"lexical"` | `"hybrid"`.
  * **filter** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `vd` Mongo-style metadata filter (hard pre-filter).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.select(hits, , strategy='conservative', max_k=3, rel=0.9, gap_ratio=0.5, min_score=None)

Commit to a subset of ranked `hits` — the selection stage.

* **Parameters:**
  * **hits** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]) – ranked [`SearchHit`](ir.base.html.md#ir.base.SearchHit)s (best first), as returned by
    [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search).
  * **strategy** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]]) – `"conservative"` (the default distractor-robust commit),
    one of `"top_k"` / `"abs_threshold"` / `"rel_threshold"` /
    `"score_gap"`, or any `Selector` callable (`hits ->
    subset`) — e.g. one built by `make_llm_selector()`.
  * **max_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – never commit to more than this many (caps distractor exposure).
  * **rel** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – relative-to-top keep threshold for `"conservative"` / the ratio
    for `"rel_threshold"`.
  * **gap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – score-gap elbow ratio — used by the `"score_gap"` strategy
    only (`"conservative"` deliberately uses `rel` alone, not an
    elbow; see this module’s docstring).
  * **min_score** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional absolute floor; with `"conservative"` the selector
    abstains when even the top hit falls below it (also usable as the
    `"abs_threshold"` floor).
* **Return type:**
  [`Selection`](#ir.Selection)
* **Returns:**
  a [`Selection`](#ir.Selection). `abstained` is True iff `selected` is empty.

### ir.sentence_window_policy(k=1)

±\*k\* same-kind neighbors around the seed (NEXT/PREV expansion).

The window runs over the artifact’s surfaces of the **seed’s kind only**,
in plan order — a `readme_chunk` window never swallows the
`description` surface. `k=0` selects just the seed’s own record.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Record`](ir.base.html.md#ir.base.Record)]]

### ir.tag_source(hits, source)

Stamp *source* on every hit that doesn’t already carry one.

Existing tags win: a hit already attributed to a corpus keeps that
attribution (so re-tagging under a different registry key cannot
double-count one corpus as two sources). A `None` source is the
untagged pseudo-source — hits pass through unattributed.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]

### ir.traverse(query, store, , policy, max_depth=2, node_budget=64, k=10)

Walk *store* from *query* under *policy*, returning the top-*k* hits.

The loop — *score the frontier → select → commit → expand* — is the
operator’s; the **safety primitives are non-negotiable and enforced here**:
a node id is committed at most once (the visited-set), expansion stops at
`max_depth`, and no more than `node_budget` nodes are ever committed.
A policy whose `expand` cycles forever and whose `stop` never fires
still terminates.

* **Parameters:**
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the user intent.
  * **store** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – passed to *policy* verbatim — a [`Corpus`](ir.index.html.md#ir.index.Corpus) for
    [`collapsed_tree_policy()`](#ir.collapsed_tree_policy), a [`CorpusGraph`](ir.graph.html.md#ir.graph.CorpusGraph) for
    an artifact-link policy. The operator never inspects it.
  * **policy** ([`WalkPolicy`](#ir.WalkPolicy)) – the [`WalkPolicy`](#ir.WalkPolicy) (e.g. [`collapsed_tree_policy()`](#ir.collapsed_tree_policy)).
  * **max_depth** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum expansion depth from a seed (safety).
  * **node_budget** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum nodes committed (safety).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of hits to return.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]
* **Returns:**
  the committed hits, best-first, top-*k* — each a
  [`SearchHit`](ir.base.html.md#ir.base.SearchHit) with `metadata["walk_depth"]` / `["seed"]`.

### ir.with_synopsis(strategy, , synthesize=None, synthesizer_id=None, synopsis_kind='synopsis')

Wrap *strategy* to add one LLM-derived `synopsis` surface per artifact.

* **Parameters:**
  * **strategy** ([`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy)) – the inner [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy) (`Chunked`,
    `Package`, …). Its surfaces are kept; the synopsis is prepended.
  * **synthesize** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Artifact`](ir.base.html.md#ir.base.Artifact)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – an injectable `Artifact -> str` (test double / custom
    summarizer). Omitted → [`make_llm_synthesizer()`](#ir.make_llm_synthesizer) (lazy `aix`).
  * **synthesizer_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – explicit identity stamp for staleness (recommended when
    injecting an unnamed callable / lambda). Omitted → the synthesizer’s
    own `synthesizer_id` / `__qualname__`.
  * **synopsis_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the surface kind (default `"synopsis"`, a summary kind).
* **Return type:**
  [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy)
* **Returns:**
  an [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy) usable anywhere a strategy is —
  `ir.CorpusSource.from_mapping(docs, name=..., strategy=with_synopsis(...))`.

```pycon
>>> strat = with_synopsis(Chunked(), synthesize=lambda a: "a summary")
```

### Modules

| [`base`](ir.base.html.md#module-ir.base)               | Core data model for `ir`.                                                                                              |
|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [`cli`](ir.cli.html.md#module-ir.cli)                 | Command-line surface for `ir` (dispatched by cw).                                                                      |
| [`config`](ir.config.html.md#module-ir.config)           | Filesystem locations and process-wide defaults for `ir`.                                                               |
| [`coverage`](ir.coverage.html.md#module-ir.coverage)       | Corpus-coverage diagnostic — what's on disk vs. what's actually indexed.                                               |
| [`embed`](ir.embed.html.md#module-ir.embed)             | Embedder resolution for `ir` — a decent local default, a light fallback.                                               |
| [`eval`](ir.eval.html.md#module-ir.eval)               | Capability-discovery evaluation — measuring how well a corpus is *found*.                                              |
| [`eval_gen`](ir.eval_gen.html.md#module-ir.eval_gen)       | LLM-backed generation of evaluation cases for [`ir.eval`](ir.eval.html.md#module-ir.eval). |
| [`formulate`](ir.formulate.html.md#module-ir.formulate)     | Query formulation — the Formulator seam (ir_09 §3).                                                                    |
| [`graph`](ir.graph.html.md#module-ir.graph)             | The semantic link graph — typed edges between artifacts (report 12).                                                   |
| [`index`](ir.index.html.md#module-ir.index)             | The indexing pipeline and incremental maintenance.                                                                     |
| [`maintenance`](ir.maintenance.html.md#module-ir.maintenance) | Idempotent background-work runner — `ir maintain` (issue #58).                                                         |
| [`policy`](ir.policy.html.md#module-ir.policy)           | Per-corpus policy — how a corpus is segmented/stored and what background work it gets.                                 |
| [`registry`](ir.registry.html.md#module-ir.registry)       | Named-corpus registry — persistent, reusable corpus definitions.                                                       |
| [`retrieve`](ir.retrieve.html.md#module-ir.retrieve)       | Retrieval — hard metadata filtering + dense / lexical / hybrid ranking.                                                |
| [`schedule`](ir.schedule.html.md#module-ir.schedule)       | Install and operate the OS job that runs `ir maintain` — `ir schedule` (issue #75).                                    |
| [`sources`](ir.sources.html.md#module-ir.sources)         | Defining a corpus source — an abstract strategy plus parameters.                                                       |
| [`store`](ir.store.html.md#module-ir.store)             | Persistence for `ir` — the repository layer over `dol` key-value views.                                                |
| [`strategy`](ir.strategy.html.md#module-ir.strategy)       | Indexing strategies — the "what do we index?" seam.                                                                    |
| [`synopsis`](ir.synopsis.html.md#module-ir.synopsis)       | Synopsis surfaces — LLM-derived summaries as an indexed surface (report 12).                                           |
| [`tools`](ir.tools.html.md#module-ir.tools)             | Agent-callable tool surface over `ir` — plain functions returning JSON-ready dicts, deliberately MCP/HTTP-agnostic.    |
