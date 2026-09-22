# ir.synopsis

Synopsis surfaces — LLM-derived summaries as an indexed surface (report 12).

The document-summary-index / collapsed-tree *fuel* (ADR #43): run a summarizer
over each artifact at **build time** to produce a short *synopsis*, index it as a
`"synopsis"` surface, and let the collapsed-tree policy ([`ir.traverse()`](ir.md#ir.traverse))
route a synopsis match down to that artifact’s chunks. Build-time cost, ≈free at
query time — and incremental, so only new / changed artifacts are re-synthesized.

[`with_synopsis()`](#ir.synopsis.with_synopsis) wraps *any* [`IndexingStrategy`](ir.strategy.md#ir.strategy.IndexingStrategy) and adds
one synopsis surface per artifact:

```default
strat  = ir.with_synopsis(ir.Chunked(), synthesize=my_summarizer)
corpus = ir.build(ir.CorpusSource.from_mapping(docs, name="d", strategy=strat))
hits   = ir.traverse(q, corpus, policy=ir.collapsed_tree_policy())  # routes via synopsis
```

The synopsis is **prepended** (plan position 0) so it is the *first* summary
surface — hence the collapsed-tree *router* (on `with_synopsis(Package())` the
synopsis, not the terse `description`, routes). An empty synopsis is dropped, so
a synth that returns `""` simply leaves the artifact with its other surfaces.

`synthesize: Callable[[Artifact], str]` is **injectable** (a test double, or
your own summarizer); omitted, it is built lazily on `aix` (the
multi-provider LLM facade) via [`make_llm_synthesizer()`](#ir.synopsis.make_llm_synthesizer) (the `make_llm_*`
idiom — `import ir` stays offline, `aix` is imported only on the first
synthesis).

**Staleness.** The wrapper exposes its identity as scalar attributes
(`synthesizer_id`, `synopsis_kind`) and holds the inner strategy, so
`ir.index._strategy_id()` (which recurses into nested strategies) folds both
the inner strategy’s parameters *and* the synthesizer identity into the corpus’s
`strategy_id`. A prompt / model change — or an inner-strategy change — therefore
re-synthesizes exactly the affected artifacts, the same way an `embedder_id`
change does; no silent staleness. (An injected *unnamed* synthesizer — a lambda
or a local closure — has no stable identity to fold in, so `with_synopsis`
**warns** and disables its staleness tracking unless you pass `synthesizer_id=`.)

**Routing needs no edges.** collapsed-tree descends synopsis→chunks *within* an
artifact via [`ir.retrieve.records_for_artifact()`](ir.retrieve.md#ir.retrieve.records_for_artifact) (shared `artifact_id`), so
synopsis routing works with no entries in the [`ir.graph`](ir.graph.md#module-ir.graph) `links` view —
that view models *cross-artifact* edges (REF / PARENT between artifacts), a
different grain than surface→surface within one artifact.

\*\*Caveat — `edge_extractor`.\*\* Eager edge ingest (`build(edge_extractor=...)`)
calls `decompose` for *every* artifact each build, and synthesis lives in
`decompose`; so combining [`with_synopsis()`](#ir.synopsis.with_synopsis) with an `edge_extractor`
re-runs synthesis on every rebuild. The common path — synopsis routing with no
`edge_extractor` — stays fully incremental.

### Module Attributes

| [`Synthesizer`](#ir.synopsis.Synthesizer)     | an [`Artifact`](ir.base.md#ir.base.Artifact) → its synopsis text (`""` to skip — no synopsis surface is added for that artifact).   |
|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`SYNOPSIS_KIND`](#ir.synopsis.SYNOPSIS_KIND)   | The surface kind a synopsis is indexed under — a member of `ir.traverse.DFLT_SUMMARY_KINDS`, so collapsed-tree routes from it.                                       |
| [`SYNOPSIS_PROMPT`](#ir.synopsis.SYNOPSIS_PROMPT) | Default prompt for [`make_llm_synthesizer()`](#ir.synopsis.make_llm_synthesizer) (a routing-oriented summary).                                             |

### Functions

| [`make_llm_synthesizer`](#ir.synopsis.make_llm_synthesizer)(\*[, summarize, prompt, ...])   | An LLM-backed [`Synthesizer`](#ir.synopsis.Synthesizer) ([`Artifact`](ir.base.md#ir.base.Artifact) → synopsis).   |
|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`with_synopsis`](#ir.synopsis.with_synopsis)(strategy, \*[, synthesize, ...])       | Wrap *strategy* to add one LLM-derived `synopsis` surface per artifact.                                                                                               |

### ir.synopsis.SYNOPSIS_KIND *= 'synopsis'*

The surface kind a synopsis is indexed under — a member of
`ir.traverse.DFLT_SUMMARY_KINDS`, so collapsed-tree routes from it.

### ir.synopsis.SYNOPSIS_PROMPT *= 'Write a concise synopsis (2-4 sentences) of the document below: what it is about and what questions it answers, so that a search over synopses can route to it. Output only the synopsis, no preamble.\\n\\nDocument:\\n{text}'*

Default prompt for [`make_llm_synthesizer()`](#ir.synopsis.make_llm_synthesizer) (a routing-oriented summary).

### ir.synopsis.Synthesizer

an [`Artifact`](ir.base.md#ir.base.Artifact) → its synopsis text (`""` to
skip — no synopsis surface is added for that artifact).

* **Type:**
  A synthesizer

alias of `Callable`[[[`Artifact`](ir.base.md#ir.base.Artifact)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.synopsis.make_llm_synthesizer(, summarize=None, prompt='Write a concise synopsis (2-4 sentences) of the document below: what it is about and what questions it answers, so that a search over synopses can route to it. Output only the synopsis, no preamble.\\\\n\\\\nDocument:\\\\n{text}', model=None, synthesizer_id=None, text_key=None, \*\*prompt_function_kwargs)

An LLM-backed [`Synthesizer`](#ir.synopsis.Synthesizer) ([`Artifact`](ir.base.md#ir.base.Artifact) → synopsis).

`summarize` is an injectable `text -> str` callable (a test double, or
your own summarizer); when omitted it is built lazily on `aix`
(`aix.prompt_func`) on the **first** synthesis and reused — so importing
this module, and even constructing the synthesizer, stays offline. The
artifact’s text is extracted with [`ir.strategy.text_of()`](ir.strategy.md#ir.strategy.text_of) using
`text_key` — which [`with_synopsis()`](#ir.synopsis.with_synopsis) threads from the inner strategy, so
the synopsis summarizes the *same* field the strategy indexes. An empty text,
or any synthesis error, yields `""` (the surface is then skipped, never a
fabricated summary).

The returned callable carries a `synthesizer_id` attribute (default
`"aix:{model}:{sha(prompt)[:12]}"`) that [`with_synopsis()`](#ir.synopsis.with_synopsis) reads into the
corpus’s `strategy_id` for staleness — a prompt or model change re-synthesizes.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Artifact`](ir.base.md#ir.base.Artifact)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.synopsis.with_synopsis(strategy, , synthesize=None, synthesizer_id=None, synopsis_kind='synopsis')

Wrap *strategy* to add one LLM-derived `synopsis` surface per artifact.

* **Parameters:**
  * **strategy** ([`IndexingStrategy`](ir.strategy.md#ir.strategy.IndexingStrategy)) – the inner [`IndexingStrategy`](ir.strategy.md#ir.strategy.IndexingStrategy) (`Chunked`,
    `Package`, …). Its surfaces are kept; the synopsis is prepended.
  * **synthesize** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Artifact`](ir.base.md#ir.base.Artifact)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – an injectable `Artifact -> str` (test double / custom
    summarizer). Omitted → [`make_llm_synthesizer()`](#ir.synopsis.make_llm_synthesizer) (lazy `aix`).
  * **synthesizer_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – explicit identity stamp for staleness (recommended when
    injecting an unnamed callable / lambda). Omitted → the synthesizer’s
    own `synthesizer_id` / `__qualname__`.
  * **synopsis_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the surface kind (default `"synopsis"`, a summary kind).
* **Return type:**
  [`IndexingStrategy`](ir.strategy.md#ir.strategy.IndexingStrategy)
* **Returns:**
  an [`IndexingStrategy`](ir.strategy.md#ir.strategy.IndexingStrategy) usable anywhere a strategy is —
  `ir.CorpusSource.from_mapping(docs, name=..., strategy=with_synopsis(...))`.

```pycon
>>> strat = with_synopsis(Chunked(), synthesize=lambda a: "a summary")
```
