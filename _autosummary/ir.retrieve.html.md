# ir.retrieve

Retrieval — hard metadata filtering + dense / lexical / hybrid ranking.

[`search()`](#ir.retrieve.search) embeds the query, applies a hard metadata filter to narrow the
candidate set (the `vd` Mongo-style filter language — ownership, name, tags),
then ranks the survivors. Three ranking modes share that one filtered candidate
set:

- `"dense"` (default) — exact brute-force cosine over the embedding matrix.
- `"lexical"` — Okapi BM25 over the candidates’ text (`vd.bm25_lexical_search`).
- `"hybrid"` — dense and lexical fused, either by Reciprocal Rank Fusion
  (`fusion="rrf"`, the default `vd.reciprocal_rank_fusion` rank-based fuse
  that sidesteps the cosine/BM25 score-scale mismatch) or by a
  magnitude-preserving convex blend (`fusion="blend"`) that keeps the dense
  cosine’s absolute scale for better abstention separability (see ir_08).

Hybrid matters for short, identifier-heavy capability text (skill / package /
tool names), the regime where dense-only retrieval fails silently on exact
identifiers and rare terms. An optional `rerank` hook (any
`ef.Reranker`) re-scores the top fused candidates with a cross-encoder;
it defaults to `None` so retrieval stays offline and API-free out of the box.

All ranking is exact brute force — correct and instant at `ir`’s corpus
sizes — and surface-hits collapse to the best surface per artifact. Lexical and
fusion reuse `vd`; if `vd` is unavailable, hybrid degrades to dense and
lexical returns no results (both with a warning), so a missing optional dep
never hard-fails a search.

Beyond the within-corpus channel fusion above, [`fuse_hits()`](#ir.retrieve.fuse_hits) merges ranked
hit lists from **different sources** (corpora / embedders / modes) by weighted
Reciprocal Rank Fusion — raw scores never cross a source boundary, only ranks
do. It is the shared cross-source merge primitive consumed by federated
[`ir.discover()`](ir.html.md#ir.discover) and by an orchestration layer’s fan-in reranker (ir_09 §3).
Every search hit carries its corpus name as `source`
and its surface’s plan position as `surface_index`.

[`records_for_artifact()`](#ir.retrieve.records_for_artifact) is the hit-operation beneath retrieval-time
context expansion: given a hit’s `artifact_id`, it returns *all* of that
artifact’s stored records (its sibling surfaces), ordered — resolved through
the ledger, never by re-deriving record ids. The expansion operator itself
(`ir.expand.expand()` — retrieve → expand → rerank) builds on it.

### Module Attributes

| [`MODES`](#ir.retrieve.MODES)            | Ranking modes accepted by [`search()`](#ir.retrieve.search).                                                                                                                                                                                                                                                                                                         |
|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`FUSIONS`](#ir.retrieve.FUSIONS)          | Hybrid fusion methods accepted by [`search()`](#ir.retrieve.search) (`mode="hybrid"`).                                                                                                                                                                                                                                                                               |
| [`DFLT_RRF_K`](#ir.retrieve.DFLT_RRF_K)       | RRF rank constant — the `k` of `1 / (k + rank)` (standard default 60).                                                                                                                                                                                                                                                                                                                       |
| [`DFLT_BLEND_ALPHA`](#ir.retrieve.DFLT_BLEND_ALPHA) | Dense weight in the `"blend"` fusion convex combination (`1-alpha` on the bounded lexical term).                                                                                                                                                                                                                                                                                             |
| [`DFLT_BM25_SAT_K`](#ir.retrieve.DFLT_BM25_SAT_K)  | `bm25 -> bm25/(bm25+k)`, a bounded squash to `[0, 1)` that needs no *per-query* normalization (which would erase the absolute-magnitude signal abstention calibration depends on).                                                                                                                                                                                                           |
| [`Identity`](#ir.retrieve.Identity)         | `None` (default — hits from different sources never merge; same id, different corpus = different artifact), the string `"pointer"` (merge hits whose [`pointer`](ir.base.html.md#ir.base.SearchHit.pointer) match — opt-in, for corpora that genuinely index the same files), or any `hit -> hashable` callable (a falsy key falls back to per-source identity). |
| [`Retriever`](#ir.retrieve.Retriever)        | `(query, **overrides) -> list[SearchHit]` — ir_09's Retriever leaf (one query, one corpus) as a swappable callable.                                                                                                                                                                                                                                                                          |

### Functions

| [`as_retriever`](#ir.retrieve.as_retriever)(corpus_or_name, \*\*search_defaults)   | Bind ONE corpus to the uniform [`Retriever`](#ir.retrieve.Retriever) contract.          |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [`fuse_hits`](#ir.retrieve.fuse_hits)(hits_by_source, \*[, rrf_k, ...])         | Merge per-source ranked hit lists into one ranking — by rank, not score.                                     |
| [`records_for_artifact`](#ir.retrieve.records_for_artifact)(store_or_corpus, ...[, ...])   | All stored records of *artifact_id*, ordered by `surface_index`.                                             |
| [`search`](#ir.retrieve.search)(corpus, query, \*[, k, filter, ...])         | Return the top-*k* [`SearchHit`](ir.base.html.md#ir.base.SearchHit) for *query*. |

### Exceptions

| [`NoLedgerEntry`](#ir.retrieve.NoLedgerEntry)   | The corpus has no ledger entry for the requested artifact.   |
|------------------------------------------------------------------|--------------------------------------------------------------|

### ir.retrieve.DFLT_BLEND_ALPHA *= 0.5*

Dense weight in the `"blend"` fusion convex combination (`1-alpha` on the
bounded lexical term). `0.5` weighs the two equally.

### ir.retrieve.DFLT_BM25_SAT_K *= 8.0*

`bm25 -> bm25/(bm25+k)`, a
bounded squash to `[0, 1)` that needs no *per-query* normalization (which
would erase the absolute-magnitude signal abstention calibration depends on).

* **Type:**
  BM25 saturation constant for `"blend"` fusion

### ir.retrieve.DFLT_RRF_K *= 60*

RRF rank constant — the `k` of `1 / (k + rank)` (standard default 60).

### ir.retrieve.FUSIONS *= ('rrf', 'blend')*

Hybrid fusion methods accepted by [`search()`](#ir.retrieve.search) (`mode="hybrid"`).
`"rrf"` is the rank-based default; `"blend"` preserves score magnitude
(see `_blend_fuse()` and ir_08) for better abstention separability.

### ir.retrieve.Identity

`None`
(default — hits from different sources never merge; same id, different
corpus = different artifact), the string `"pointer"` (merge hits whose
[`pointer`](ir.base.html.md#ir.base.SearchHit.pointer) match — opt-in, for corpora that
genuinely index the same files), or any `hit -> hashable` callable (a
falsy key falls back to per-source identity).

* **Type:**
  How cross-source duplicates are detected in [`fuse_hits()`](#ir.retrieve.fuse_hits)

alias of `Callable`[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.retrieve.MODES *= ('dense', 'lexical', 'hybrid')*

Ranking modes accepted by [`search()`](#ir.retrieve.search).

### *exception* ir.retrieve.NoLedgerEntry

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

The corpus has no ledger entry for the requested artifact.

The *benign* miss (unknown artifact, or a corpus built without ledger
bookkeeping) — distinct from a stale/torn ledger, which raises a plain
`KeyError` describing the corruption. A `KeyError` subclass so
existing `except KeyError` callers keep working; the disclosure seam
catches this one specifically and tolerates it per hit.

### ir.retrieve.Retriever

`(query, **overrides) -> list[SearchHit]` — ir_09’s Retriever
leaf (one query, one corpus) as a swappable callable. [`as_retriever()`](#ir.retrieve.as_retriever)
binds one corpus to this contract so an orchestration layer (e.g. `raglab`)
can register an ir corpus as one source key without importing ir internals.

* **Type:**
  A retriever

alias of `Callable`[[…], `list[SearchHit]`]

### ir.retrieve.as_retriever(corpus_or_name, \*\*search_defaults)

Bind ONE corpus to the uniform [`Retriever`](#ir.retrieve.Retriever) contract.

Returns `retrieve(query, **overrides) -> list[SearchHit]` that calls
[`search()`](#ir.retrieve.search) with `search_defaults` (a per-call kwarg overrides a bound
default). A corpus *name* is resolved once via [`ir.open_corpus()`](ir.html.md#ir.open_corpus); pass
an open [`Corpus`](ir.index.html.md#ir.index.Corpus) to skip that. The returned callable carries
the bound corpus on `.corpus` for introspection.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]]

```pycon
>>> retr = as_retriever(corpus, mode="hybrid", k=20)
>>> hits = retr("how do I deploy the app")
>>> hits = retr("deploy", filter={"owner": "me"})
```

### ir.retrieve.fuse_hits(hits_by_source, , rrf_k=60, weights=None, identity=None, k=None)

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
    upfront, as federated [`ir.discover()`](ir.html.md#ir.discover) does.
  * **identity** (`Union`[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – how cross-source duplicates merge — see [`Identity`](#ir.retrieve.Identity).
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

### ir.retrieve.records_for_artifact(store_or_corpus, artifact_id, , surface_kind=None)

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
    never embeds, so unlike [`ir.open_corpus()`](ir.html.md#ir.open_corpus) no embedder is
    loaded.
  * **artifact_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the artifact whose surfaces to fetch.
  * **surface_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict to one surface kind (e.g. `"readme_chunk"`);
    a known artifact with no surfaces of that kind yields `[]`.
* **Raises:**
  * [**NoLedgerEntry**](#ir.retrieve.NoLedgerEntry) – the ledger has no entry for *artifact_id* (an unknown
        artifact, or a corpus built without [`ir.index.build()`](ir.index.html.md#ir.index.build)’s
        ledger bookkeeping). A `KeyError` subclass.
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – an entry exists but lists a record missing from the store
        (a stale ledger: interrupted build or out-of-band
        `delete_record`) — data corruption, named in the message.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Record`](ir.base.html.md#ir.base.Record)]

### ir.retrieve.search(corpus, query, , k=10, filter=None, surfaces=None, per_artifact=True, mode='dense', fusion='rrf', rrf_k=60, alpha=0.5, bm25_sat_k=8.0, fetch_k=None, rerank=None, bm25=None, formulate=None)

Return the top-*k* [`SearchHit`](ir.base.html.md#ir.base.SearchHit) for *query*.

* **Parameters:**
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int))
  * **filter** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None))
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None))
  * **per_artifact** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool))
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – (dense + BM25 fused). Hybrid is the strongest default for short,
    identifier-heavy text; `"dense"` is the historical behavior and is
    kept as the default for backward compatibility.
  * **fusion** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – rank-based Reciprocal Rank Fusion) or `"blend"` (magnitude-preserving
    convex blend; better abstention separability — see ir_08). Ignored for
    non-hybrid modes.
  * **rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int))
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float))
  * **bm25_sat_k** ([`float`](https://docs.python.org/3/builtins/functions.html#float))
  * **fetch_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – (default `max(k*5, 50)` when collapsing per artifact, else `k`).
  * **rerank** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – that re-scores the ranked candidates after fusion. The reranker sees
    the full candidate list (up to `fetch_k` items) and may reorder it.
    Default `None` keeps retrieval offline (no model download / API call).
  * **bm25** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – (e.g. `{"k1": 1.5, "b": 0.75}`).
  * **formulate** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – [str, …]\`\`) applied *before* retrieval — rewrite / expand / HyDE
    (ir_09 §3). Identity by default (embed the query verbatim). When it
    returns several queries, ir runs each and fuses the results (best surface
    per artifact). See [`ir.make_llm_formulator()`](ir.html.md#ir.make_llm_formulator).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ir.base.html.md#ir.base.SearchHit)]
