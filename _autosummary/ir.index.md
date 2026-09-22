# ir.index

The indexing pipeline and incremental maintenance.

[`build()`](#ir.index.build) turns a [`CorpusSource`](ir.sources.md#ir.sources.CorpusSource) into a queryable
[`Corpus`](#ir.index.Corpus), persisting records through a
[`CorpusStore`](ir.store.md#ir.store.CorpusStore). It is **incremental and idempotent**: each
artifact’s change signal is compared to the ledger, and only new, changed, or
re-modeled artifacts are decomposed and embedded. Artifacts that vanished from
the source are pruned (full-refresh). Re-running `build` on an unchanged
source is a near-no-op.

This is the light maintenance path — content-hash CRUD over a flat store, which
is the right tool for the small corpora `ir` targets. The heavier
content-addressed artifact-graph (`ef.artifact_graph`) is the documented
upgrade for corpora large enough that recomputing surfaces is the bottleneck.

### Functions

| [`build`](#ir.index.build)(source, \*[, store, embedder, full, ...])   | Build or incrementally update *source* into a [`Corpus`](#ir.index.Corpus).   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`open_corpus`](#ir.index.open_corpus)(name, \*[, embedder])                 | Reopen a previously built corpus by name.                                                                |

### Classes

| [`Corpus`](#ir.index.Corpus)(name, store, embedder, embedder_id)   | A built, queryable corpus: a store plus its embedder.   |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------|

### *class* ir.index.Corpus(name, store, embedder, embedder_id)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A built, queryable corpus: a store plus its embedder.

#### search(query, \*\*kwargs)

Search this corpus for *query*.

`**kwargs` (`k` / `mode` / `filter` / `surfaces` /
`per_artifact` / …) are forwarded to [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search).

### ir.index.build(source, , store=None, embedder=None, full=True, batch_size=256, edge_extractor=None)

Build or incrementally update *source* into a [`Corpus`](#ir.index.Corpus).

* **Parameters:**
  * **store** ([`CorpusStore`](ir.store.md#ir.store.CorpusStore) | [`None`](https://docs.python.org/3/builtins/constants.html#None))
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any))
  * **full** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool))
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int))
  * **edge_extractor** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – (`(artifact_id, filter_fields) -> {edge_type: [target]}`) that
    populates the corpus’s semantic `links` graph (see [`ir.graph`](ir.graph.md#module-ir.graph);
    pass [`ir.default_edge_extractor()`](ir.md#ir.default_edge_extractor) for the latent deps/parent
    edges). Ingest is **eager** — edges are (re)written for *every*
    in-scope artifact, a decompose-only pass with no embedding, so the
    graph never goes partially stale — while embedding stays fully
    incremental. Edges are derived state, **not** part of build identity.
    A rebuild *without* an extractor leaves existing edges untouched
    (they are only refreshed by re-running with one, and only cleared per
    artifact by the `full` prune below) — so dropping `edge_extractor`
    does not wipe a graph.
* **Return type:**
  [`Corpus`](#ir.index.Corpus)

### ir.index.open_corpus(name, , embedder=None)

Reopen a previously built corpus by name.

The embedding model is **lazily** resolved (see `_LazyEmbedder`): the
returned corpus knows its `embedder_id` from stored config immediately, but
only loads the model when a dense/hybrid query actually embeds. So `ir ls`,
`ir info`, and lexical-only search open a corpus without the model-load
cost. Pass `embedder=` to override the stored spec.

* **Return type:**
  [`Corpus`](#ir.index.Corpus)
