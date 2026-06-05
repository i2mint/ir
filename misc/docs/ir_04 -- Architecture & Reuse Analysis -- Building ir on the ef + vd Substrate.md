# ir_04 — Architecture & Reuse Analysis: Building `ir` on the `ef` + `vd` Substrate

> **Purpose.** This document does for `ir` what the user explicitly asked: lay out the
> vision, then map every piece of it against what the ecosystem (`ef`, `vd`, and friends)
> *already provides*, so `ir` is built by **composition, not reinvention**. The headline
> finding is that the lower ~70–80% of `ir`'s stated substrate already exists and is
> mature — most of it in `ef` (indexing/maintenance/retrieval) and `vd` (store +
> metadata filtering). `ir`'s genuine, non-redundant work is a comparatively thin (but
> conceptually hard) band at the top: **multi-surface artifact indexing**, a **selection
> stage**, **corpus-source adapters** for the first concrete corpora, an **agent-callable
> tool surface**, and a **capability-discovery evaluation harness**.
>
> Code is referenced repo-relative under `$PP` (the projects folder): `t/ef/…`, `i/vd/…`,
> etc. Companion research lives alongside this file: `ir_01` (progressive capability
> discovery), `ir_02` (indexing/embedding strategy), `ir_03` (evaluation).

---

## 0. Executive summary

| Layer of the `ir` vision | Who already provides it | `ir`'s job |
|---|---|---|
| Corpus abstraction (any store as a corpus) | **`ef.corpus`** (`Corpus = MutableMapping`, `as_corpus`) + **`dol`** | Reuse wholesale |
| Change detection / staleness | **`ef`** (`content_hash`, `ChangeDetectingCorpus`, `diagnostics`, `refresh`) | Reuse wholesale; supply per-source signals |
| Corpus maintenance as a repeatable process | **`ef.artifact_graph`** (content-addressed producer DAG, cascade invalidation, config branching) | Reuse wholesale — this *is* the "maintenance contract" |
| Segmentation | **`ef.segmenters`** + **`imbed`** | Reuse |
| Embedding (facade + adapters + wrappers) | **`ef.embedders`** (OpenAI, Voyage, Cohere, SBERT, Gemini, HTTP, hashing) | Reuse |
| Vector store (16 backends) | **`vd`** | Reuse wholesale |
| Metadata filtering (9 Mongo-style ops, `$and/$or/$not/$in/$exists`) | **`vd.filters`** | Reuse wholesale — this is the "hard filter" half of the vision |
| Hybrid retrieval (BM25 + dense + RRF), multi-query | **`vd.search`** | Reuse; extend for scale/large lexical |
| Reranking | **`ef.reranking`** (protocol + cross-encoder) | Reuse; add instruction-steered rerankers |
| Retrieval evaluation (BEIR/MTEB-shaped) | **`ef.evaluation`** | Reuse for IR metrics; **build** capability-discovery eval on top |
| HTTP / agent-callable surface | **`qh`** (+ `ef.service` pattern) | Compose |
| LLM-authored synopsis / problem-class tags | **`oa`** (`prompt_function`) | Compose |
| Corpus sources (skills, packages, GitHub, docs) | **`priv.skills_index`, `projreg`, `hubcap`, `contaix`** | Compose as scope/change-detection adapters |
| **Multi-surface artifact indexing** (one artifact → many heterogeneous filterable+embeddable units) | *nobody — partial in `ef` multi-config* | **BUILD (core seam)** |
| **Selection stage** (distractor-robust commit to a subset) | *nobody* | **BUILD** |
| **Capability-artifact model** (tool / skill / subagent as indexable+executable) | *nobody* | **BUILD** |
| **Agentic "one search tool" + progressive payload disclosure** | *nobody (defer-loading is host-side)* | **BUILD (thin)** |

**The single most important strategic question this analysis surfaces** (see §8): `ef`'s
own one-line self-description — *"a facade for semantic-embeddings user journeys … `ef` is
**not** RAG — it returns ranked segments; bring your own LLM"* — is **almost verbatim the
`ir` vision's** "retrieval is the core competency … generation/reranking/citation are
layered on top." `ef` and `ir` are positioned on the same spectrum. `ir` must be defined as
a **distinct layer above `ef`**, not a parallel reimplementation of it. The rest of this
document assumes that boundary and draws it precisely.

---

## 1. The vision, restated — and where it already lives

The `ir` brief makes five load-bearing claims. Each maps onto existing machinery:

1. **"A uniform retrieval contract across the whole scale spectrum"** — from ad-hoc `find`
   over an ephemeral list to a millions-of-docs search engine, same facade.
   → This is exactly `ef`'s **progressive-disclosure facade**: the light path
   `ingest([...]) → SearchableCorpus` (one line, hashing embedder, in-memory `vd`) and the
   heavy path `SourceManager(corpus, …).materialize().search()` share one surface
   (`t/ef/ef/source_manager.py`). The store seam (`_open_store`, `t/ef/ef/source_manager.py:956`)
   already swaps in-memory ↔ SQLite ↔ pgvector ↔ dedicated vector DB without changing the
   caller's contract — precisely the vision's "any backing store, any retrieval strategy,
   caller's contract unchanged."

2. **"Extensible into RAG without being one by default"** — retrieval core; generation
   layered on.
   → Verbatim `ef`'s stance. `ef.search` returns `SearchHit`s; `ef.retrieve` returns plain
   `Segment`s as the RAG plug-in surface (`t/ef/ef/source_manager.py:816`). Generation is
   deliberately out of scope.

3. **"Corpus maintenance as a defined, repeatable process,"** with two pluggable slots —
   **scope** (what's in the corpus) and **change detection** (what's stale).
   → `ef` already implements both halves of this and more:
   - *Change detection*: `content_hash` (SHA-256 over normalized content, text-only for
     mappings so metadata edits don't churn — `t/ef/ef/corpus.py:109`) and
     `ChangeDetectingCorpus` (fires `added/modified/deleted` events only on real hash
     change — `t/ef/ef/corpus.py`).
   - *Staleness diagnosis*: four conditions — **orphan / missing / stale / misconfigured** —
     computed read-only by `diagnose()` (`t/ef/ef/diagnostics.py`).
   - *Repeatable refresh*: four modes — `none / incremental / full / scoped_full` — via a
     **pure** `plan_refresh()` → `RefreshPlan` and `refresh_on_change()` handler
     (`t/ef/ef/refresh.py`).
   - *Cascade invalidation + config branching as one operation*: the **artifact graph**
     (§2.3). This is the deepest part of the vision's "maintenance contract," and it is
     already built.
   What `ef` does **not** prescribe is the *concrete* scope (globs vs API queries vs GitHub)
   and *concrete* change signal (mtime vs ETag vs version) per source — exactly the
   "abstract slots, concrete-per-source" the vision wants. Those concrete definitions are
   where `ir` plugs in existing source packages (§5).

4. **"Good retrieval is not all embeddings — it is also classic metadata filtering,"** with
   hard filters (ownership, package name, domain) constraining the candidate set *before/
   alongside* semantic ranking.
   → `vd.filters` is a complete Mongo-style filter language — `$eq/$ne/$gt/$gte/$lt/$lte/
   $in/$nin/$exists` plus `$and/$or/$not` — validated per-backend and pushed down to native
   filters where supported (`i/vd/vd/filters.py`, `i/vd/vd/base.py:578`). `ef` already
   threads a `filter=` straight through search to `vd`
   (`t/ef/ef/source_manager.py:260`). The "hard filter, not fuzzy match" requirement is a
   solved problem at the `vd` layer.

5. **"Deciding what to index is a problem in its own right — not one `ir` should solve
   internally; expose the seams."**
   → This is the **one place the existing substrate is genuinely insufficient** and is the
   heart of `ir`'s new work. See §4.

**Conclusion of §1:** four of the five pillars are substantially built. The fifth — the
indexing-strategy seam (artifact decomposition into filterable fields + multi-granularity
embeddable surfaces) — is `ir`'s defining contribution, together with the agent-facing
selection/disclosure/eval band that the three `ir_0x` research docs specify.

---

## 2. What `ef` already gives `ir` (the indexing / maintenance / retrieval spine)

`ef` is a layered facade: **corpus (L0) → segment (L2) → embed (L3) → index in `vd` (L4) →
derive/explore (L5)**, with a cross-cutting content-addressed **artifact graph** and a
**diagnose/refresh** maintenance loop. Reuse verdicts below are "wholesale" unless noted.

### 2.1 Corpus + change detection (`ef.corpus`, `ef.hashing`)
- `Corpus = MutableMapping[source_id, str | Mapping]`; `as_corpus(None | mapping | dir-path
  | iterable)` is the DI seam. A corpus is *any* `dol` store — RAM, filesystem, S3, an API
  view. **Reuse wholesale.**
- `content_hash(source, content_keys=…)` — normalized (NFC, `\n`, no BOM) SHA-256; for
  mappings hashes `text` only unless `content_keys` opts metadata in. This is the
  idempotency primitive every maintenance decision rests on. **Reuse wholesale.**
- `ChangeDetectingCorpus(on_change=…)` wraps any corpus, emitting `ChangeEvent(source_id,
  kind, old_hash, new_hash)` and `CorpusDiff`. **Reuse wholesale.** (Out-of-band edits —
  a file changed on disk — are caught by `SourceManager.scan()`, which re-hashes the
  corpus.)

### 2.2 Segment / embed facades
- `Segment` TypedDict + `SegmentRecord` (canonical interchange: `text, id, metadata,
  parent_id, index, start, end, tokens`; `PROMOTED_METADATA_KEYS` lifts `source,
  source_type, tokenizer, token_count, embedding_model, page, license, ingestion_run_id`
  into the stored doc). **Reuse wholesale** — this is the data model `ir` records flow
  through.
- `Segmenter` protocol + `RecursiveCharacterSegmenter`, `line_segmenter`, `with_overlap`,
  `hierarchical`, `materialise`. **Reuse** (and/or `imbed`'s `fixed_step_chunker`).
- `Embedder` protocol (`Iterable[str] -> ndarray(n, dim)`), `as_embedder` DI seam,
  `HashingEmbedder` (numpy-only default), adapters (`openai_/voyage_/cohere_/
  sentence_transformers_/gemini_/http_embedder`), wrappers (`Cached/Retrying/Multi/
  Normalizing`). Carries `InputType ∈ {query, document, classification, clustering}` for
  asymmetric embedding. **Reuse wholesale.** Note `ir_02`'s recommendation of
  *instruction-tuned* embedders and per-query instruction steering maps onto extending the
  `InputType`/adapter mechanism, not replacing it.

### 2.3 The artifact graph — the maintenance engine (`ef.artifact_graph`)
This is the single most valuable reuse for `ir`. Every produced artifact is addressed by its
**recipe**:
```
artifact_id = H(op_key, op_version, input_ids, params)
```
so two pipeline configs that share an upstream step share that step's artifact — *segment
once, embed once*, no matter how many configs branch through. Operations:
- `materialize(id)` — lazy backward compute (cache hit if present), recursing into inputs;
- `mark_stale(leaf)` — forward cascade invalidation when a source changes;
- `delete_cascade(id)` — forward delete of an artifact and everything depending only on it;
- `freshness(id) → materialized | stale | unknown`.

Stores (`store / producers / edges`) are injectable `MutableMapping`s, so the graph persists
to any `dol` backend (SQLite for ≫10⁶ nodes) with **no DB dependency baked into `ef`**.
`ProducerSpec` uses *string* op keys (e.g. `"embed:openai:text-embedding-3-large@1024"`)
resolved through an `ops` registry at materialize time — so the spec is serializable and
branchable. **Reuse wholesale.** The vision's "keep the index (ledger, embeddings, derived
metadata) in sync with a living, mutable corpus" is this graph plus diagnose/refresh.

### 2.4 Retrieval surface
- `SourceManager` holds corpus + graph + N named configs (each a `(segmenter, embedder)`
  pipeline → its own `vd` collection); `SearchableCorpus` is the light wrapper.
- `search(query, *, config, limit, filter) → list[SearchHit(segment, score, source_id)]`;
  `retrieve(...) → Segment[]`. Every stored doc carries `source_id, source_hash,
  config_hash` provenance (`t/ef/ef/source_manager.py:232`).
- `ef.reranking`: `Reranker` protocol `(query, segments) → scores`, `rerank()`,
  `with_reranker(retriever, reranker, fetch_k=50)` decorator, `cross_encoder_reranker`.
  **Reuse**; `ir`/`ir_02` add instruction-following rerankers (Voyage rerank-2.5,
  Qwen3-Reranker) behind the same protocol.
- `ef.exploration` (`project/cluster/label_clusters/explore`) and `ef.evaluation`
  (BEIR/MTEB retrieval metrics, Ragas bridge) — reuse as needed.

### 2.5 What `ef` does **not** have (relevant to `ir`)
- **Multi-surface indexing of a single structured artifact.** `ef` multi-config gives you
  *multiple embedders over the same text* and segment→chunk decomposition. It does **not**
  model "one artifact (a package) → a *heterogeneous* set of sub-surfaces (name field,
  short description, AI synopsis, per-module text, problem-class list) where some are
  **filterable metadata** and some are **separately embedded vectors** with their own
  granularity and kind." (§4.)
- **A selection stage.** `ef` returns ranked hits; it never commits to a distractor-robust
  subset. (§3, `ir_01`/`ir_03`.)
- **Lexical/hybrid** lives in `vd`, not `ef` (so `ir` reaches for `vd.hybrid_search`).
- **Agent orchestration, answer synthesis, the "one search tool" disclosure protocol** —
  deliberately out of scope for `ef`; partly `ir`'s job, partly the host's.

---

## 3. What `vd` already gives `ir` (the store + filter substrate)

`vd` is a production-grade vector-DB facade. For `ir` it supplies the **entire** storage and
metadata-filtering layer:
- **Contract**: `Document(id, text, vector, metadata)`, `Vector`, `SearchResult(id, text,
  score, metadata)`, higher-is-better score normalization across `cosine/dot/l2`. Collections
  are `MutableMapping[str, Document] + search()`; adapters implement six raw primitives
  (`_write/_read/_drop/_keys/_count/_query`) — `i/vd/vd/base.py`.
- **Metadata filtering**: the 9-operator Mongo-style language with `$and/$or/$not`, validated
  per backend (`supported_filter_operators`) and translated to native filters; pure-Python
  `matches_filter` is the reference semantics — `i/vd/vd/filters.py`. **This is the vision's
  "hard filter, not fuzzy match" requirement, already solved.**
- **Search modes** (`i/vd/vd/search.py`): dense kNN; `hybrid_search` = dense + **BM25** fused
  with **Reciprocal Rank Fusion** (native where a backend supports it, client-side fallback
  otherwise); `multi_query_search` (`interleave/concatenate/union/best`);
  `search_similar_to_document`; standalone `reciprocal_rank_fusion` and `deduplicate_results`.
- **16 backends** via `@register_backend`: `memory` (brute-force reference), `chroma`,
  `lancedb`, `sqlite_vec`, `duckdb`, `faiss`, `qdrant`, `weaviate`, `milvus`, `redis`,
  `elasticsearch`, `mongodb`, `pgvector`, `pinecone`, `turbopuffer`. Plus `vd.recommend_backend(...)`
  to pick by scale/latency/deployment, `check_requirements`/`setup_guide`, JSON/JSONL/dir
  import-export, cross-backend `migrate_collection`, analytics (`collection_stats,
  find_duplicates, find_outliers`), `chunk_text`/`chunk_documents`/`clean_text`,
  `TimeIndexedCollection`, and async wrappers.

**`vd` gaps relevant to `ir`:** the client-side BM25 fallback is O(N) (fine < ~100k docs;
push lexical into Elasticsearch/pgvector-FTS or a native-hybrid backend at scale); **no
first-class sparse-vector type** (BM25 is post-hoc, not a stored sparse vector); **no
late-interaction/ColBERT** multi-vector-per-doc; **no reranking layer** (that's `ef`'s). Per
`ir_02`, none of these are urgent: document expansion and instruction-tuned embedders +
rerank dominate the accuracy budget; ColBERT was *not* justified on tool-retrieval
benchmarks.

---

## 4. The core new work: the "what do we index?" seam (IndexingStrategy)

This is where `ir` earns its existence. The vision is precise: **a package is not one
document** — it is a hierarchy of indexable surfaces with two fundamentally different roles:

- **Structured metadata for *filtering*, not embedding** — `name`, ownership (ours vs.
  blessed-third-party), domain/problem-class tags, maturity, dependencies, license. "Ours
  vs. third-party" is a *hard filter*; package `name` is a *first-class filterable field*.
- **Embeddable representations at multiple granularities and of multiple *kinds*** — a short
  canonical description; a longer AI-authored synopsis; and per-package *sub-surfaces*:
  individual modules, distinct functionalities, the **problem classes** a package addresses,
  how-to material. Each may warrant its **own** embedded vector so a query matches the *right
  part* of a package — and so the "class of solution" goal (find the package to *extend* even
  when the exact function is absent) can match the **problem-classes surface** specifically.

### 4.1 Why `ef`'s model doesn't cover this directly
`ef`'s atom is a `Segment` of one source's `text`, and multi-config means multiple embedders
over that text. The vision needs a different decomposition: **one logical artifact → a set of
*typed units*, each either a filter-field or an embeddable-surface, each with its own metadata
and (for surfaces) its own embedding config.** A package's `name` is not a chunk of text to
embed — it's a filter key. Its "problem-classes" surface is a distinct embedded space from its
"how-to" surface. This is a richer structure than "segment a string into overlapping chunks."

### 4.2 The seam `ir` should expose
`ir`'s responsibility (per the vision) is **not** to decide what to index, but to expose the
**three-part seam** and ship sensible defaults:

```
(a) DECOMPOSE  artifact  ->  { filter_fields: Mapping[str, Filterable],
                               surfaces: list[Surface] }      # Surface = (kind, granularity, text, metadata)
(b) INDEX      each surface -> embed (its own ef config) ; each filter_field -> vd metadata
(c) RETRIEVE   combine vd metadata-filter (hard)  with  multi-surface semantic ranking (soft)
```

Concretely:
- **`IndexingStrategy` protocol** — `decompose(artifact) -> ArtifactIndexPlan`. The default
  strategy treats an artifact as `ef` does today (one text → segments, one embedder); a
  *package-aware* strategy emits the filter-fields + multi-surface plan above. This is the
  "pluggable extension, not a fork of the core" the vision demands.
- **Surface → `ef` artifact-graph node.** Each surface's embedding is just another
  `ProducerSpec` in the existing graph, so multi-surface indexing inherits cascade
  invalidation and shared-artifact dedup *for free*. A surface that is itself AI-generated
  (the synopsis, the problem-class tags) is an **op in the graph** (`op="synthesize:synopsis@1"`,
  backed by `oa.prompt_function`) — so when the source changes, the synopsis is recomputed and
  re-embedded by the same `mark_stale` cascade. This is a clean, powerful fit.
- **Filter-fields → `vd` document metadata**, queried with `vd.filters`. "Ours vs.
  third-party," `name`, domain tags become `{$eq}`/`{$in}` constraints applied *before/
  alongside* ranking.
- **Cross-surface fusion.** A query may run against several surfaces (description, synopsis,
  problem-classes) and fuse via RRF (`vd.reciprocal_rank_fusion`), then rerank
  (`ef.with_reranker`). The "match the right part of a package" goal becomes per-surface
  retrieval + fusion.

### 4.3 Defaults vs. overrides (progressive disclosure at the indexing layer)
- **Default (naive corpus):** one surface = the doc text; minimal metadata. Works out of the
  box — identical to `ef` today.
- **Override (package-aware):** the rich decomposition, with AI-authored surfaces produced by
  graph ops. `ir` ships the *protocol* and a couple of reference strategies; the sophisticated
  package indexer is a plug-in.

This seam is the through-line that unifies all three of `ir`'s first corpora (§5): each is
"a maintained corpus + an `IndexingStrategy` + filter-fields + surfaces + a retrieve/select
pipeline," differing only in concrete decomposition and change signal.

---

## 5. Corpus-source adapters: the first corpora already have sources

`ir`'s "scope" and "change-detection" slots do not need to be built from scratch for the
first targets — existing packages already enumerate these corpora and produce records with
staleness signals. `ir` wraps them as scope/change-detection adapters and adds only the
index→retrieve→select layer.

| `ir` corpus | Scope (enumerate) | Change signal | Artifact source | Existing package |
|---|---|---|---|---|
| **A. Capability discovery — skills** | `skills_index.collect_skills()` sweeps `~/.claude/skills` + every `.pth` base's `.claude/skills/` | `refresh=True` re-scan; file mtime | `{name, description, skill_path, parent, base_path}` from SKILL.md frontmatter | **`priv.skills_index`** |
| **A. Capability discovery — MCP tools / subagents** | *no registry yet* — extend the skills_index sweep pattern | n/a yet | tool = typed callable + schema; subagent = Agent Card | **build (thin) atop `priv` pattern**; `ir_01`/`ir_02` |
| **B. Research / dev-context docs** | `projreg.docs.DocStore` nested FS store (`docs/<proj>/readme|issues|discussions`) | SHA-256 on write (skips unchanged); `fetched_at` | markdown + `{source, fetched_at, sha256, bytes}` | **`projreg`** (+ `contaix` to materialize new docs) |
| **B. Dev artifacts — issues/PRs/discussions/commits** | `projreg.gh_sync` → `gh_cache/<owner>/<repo>/…`; or live `hubcap.RepoReader` | `last_sync` + GitHub `.updated_at` | raw JSON + rendered markdown | **`projreg.gh_sync` / `hubcap`** |
| **B. Code context** | `contaix.code_aggregate()` over dir / GitHub / package name | FS mtime / GitHub timestamp | `Mapping[filename, code_str]` → markdown | **`contaix`** |
| **C. Preferred ecosystem — our ~200 packages** | `projreg.ledger.load_ledger()` (`ProjectRecord`, 14 fields) + `priv.dep_graph` | `diff_ledgers(old,new)`; build-config mtime | `ProjectRecord{name, path, description, version, keywords, urls, deps, github_repo, …}` | **`projreg` + `priv.dep_graph`** |
| **C. Preferred ecosystem — blessed third-party** | *curated list* — small authored corpus | manual / version | `{name, problem_class, rationale}` | **build (tiny)** |
| **AI-authored surfaces** (synopsis, problem-class tags) | n/a (generative) | upstream source hash (graph op) | LLM output, cached | **`oa.prompt_function`** (as `ef` graph ops) |
| **External fetch caching / staleness** | n/a | FS mtime | cached bytes | **`graze`** |

**Notably, `projreg` already contains a working `search.py`** (BM25 + optional `oa`
embeddings cached by SHA-256, returning `SearchHit(score, project, doc_type, snippet)`). It is
a *proto-`ir`* for corpus C. `ir` should **absorb/generalize** that search rather than leave a
second retrieval implementation drifting — treat `projreg.search` as a reference to subsume,
with `projreg` becoming a *source* (ledger + docs + gh_sync) under `ir`'s unified retrieval.

---

## 6. Proposed `ir` shape and package boundary

```
                    ┌─────────────────────────────────────────────────────────┐
   agent  ──tool──▶ │  ir.tool  (the ONE search-and-select tool; qh-exposable) │
                    └───────────────┬─────────────────────────────────────────┘
                                    │  retrieve()  +  select()
        ┌───────────────────────────┴───────────────────────────┐
        │  ir.select   (NEW: distractor-robust commit to subset; │
        │              progressive payload disclosure)           │
        └───────────────────────────┬───────────────────────────┘
                                    │  candidates
        ┌───────────────────────────┴───────────────────────────┐
        │  ir.retrieve  (compose: vd hard-filter  +  multi-      │
        │               surface dense + BM25 RRF  +  ef rerank)  │
        └───────────────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │  ir.index  (NEW seam: IndexingStrategy.decompose →     │
        │            filter_fields + surfaces; surfaces & AI-    │
        │            authored fields are ef artifact-graph ops)  │
        └───────────────────────────┬───────────────────────────┘
                                    │
   ┌────────────────────────────────┴────────────────────────────────┐
   │  REUSED SUBSTRATE                                                 │
   │   ef:  corpus · change-detect · artifact-graph · segment · embed │
   │        · diagnose/refresh · rerank · evaluate                    │
   │   vd:  store (16 backends) · metadata filters · hybrid · RRF      │
   │   sources: priv.skills_index · projreg · hubcap · contaix        │
   │   helpers: oa (LLM ops) · graze (cache) · ju (schemas) · qh (HTTP)│
   │            · meshed (DAG wiring) · imbed (chunk/cluster/viz)      │
   └──────────────────────────────────────────────────────────────────┘
```

**`ir`'s own modules (the genuinely-new band):**
1. `ir.artifact` — the capability/resource artifact model (tool / skill / subagent /
   package / doc), polymorphic: a shared `(name, description)` retrieval key + a typed
   **executable/loadable payload** (schema injection, SKILL.md body load, subagent
   delegation, package pointer). From `ir_01`/`ir_02`.
2. `ir.index` — the **`IndexingStrategy` seam** (§4): `decompose → {filter_fields,
   surfaces}`; default + package-aware strategies; surfaces/AI-fields realized as `ef`
   artifact-graph ops. **This is the keystone.**
3. `ir.retrieve` — a thin composer over `vd` (hard filter), multi-surface dense + BM25 +
   RRF, and `ef` rerank. Mostly wiring.
4. `ir.select` — **new**: the selection stage. Distractor-robust subset commitment +
   progressive disclosure of heavy payloads (defer-load; append-only to protect prompt
   cache, per `ir_01`).
5. `ir.sources` — scope/change-detection adapters wrapping `priv.skills_index`, `projreg`,
   `hubcap`, `contaix` (§5).
6. `ir.tool` — the single agent-callable surface (`qh.mk_app`-exposable), with the
   `ef.service` stateless-handle-registry pattern.
7. `ir.eval` — the capability-discovery harness (§7), building on `ef.evaluation`.

**Dependency-wise**, `ir` declares `ef` and `vd` (and pulls `oa`/`qh`/`projreg`/`hubcap`/
`contaix`/`priv` as the source/feature extras). It must **not** re-vendor any of their
internals.

---

## 7. Evaluation: build the capability-discovery harness on `ef.evaluation`

`ef.evaluation` already gives BEIR/MTEB-shaped retrieval metrics (recall@k, NDCG@k) and a
Ragas bridge. `ir_03` specifies what `ir` must add on top, and it is genuinely new:
- **Stage-decoupled metrics**: retrieval (recall@k/NDCG@k, reuse `ef`) **vs.** *conditional
  selection accuracy* `P(correct | gold retrieved)` **vs.** parameter-fill (deepdiff) **vs.**
  end-task — never conflated.
- **Distractor-robustness curve**: one gold + N−1 distractors, sweep N ∈ {1,4,…,128}, plot
  accuracy. This is the metric that exposes the "43% → 2% as catalog grows" collapse the whole
  project exists to prevent.
- **Failure-mode taxonomy** (query-planning miss, retrieval miss, selector false-negative,
  over-selection, namespace confusion, hallucination, parameterization, ordering, abstention).
- **Harness**: `ir_03` endorses **Inspect AI** (MIT) as backbone, **deepdiff** for structural
  param scoring, **back-translation + function-masking** for synthetic cases, and **ToolRet**
  framing for the retrieval slice. These are external; adopt behind `ir.eval` interfaces.

---

## 8. Consolidation, risks, and the `ir`-vs-`ef` boundary

**(R1) `ir` vs `ef` overlap — the #1 decision.** `ef` and `ir` share positioning almost word
for word. If the boundary is not drawn deliberately, `ir` will reinvent `ef`. The boundary
this analysis recommends:
- `ef` = **substrate**: corpus, content-addressed maintenance, segment/embed, store-wiring,
  generic retrieval, generic retrieval-eval. Stays domain-agnostic.
- `ir` = **agentic IR layer**: heterogeneous *capability/resource* artifacts, the
  multi-surface **IndexingStrategy** seam, **selection** + progressive disclosure, **source
  adapters** for the concrete corpora, the **one-tool** agent surface, and **capability-
  discovery eval**.
  Anything `ir` builds that turns out to be domain-agnostic (e.g. multi-surface indexing as a
  general capability) is a candidate to **push *down* into `ef`** later — but it should
  incubate in `ir` first. *Decision needed from the user (see §9).*

**(R2) `raglab` and `srag` are parallel/legacy.** `raglab` (`a/raglab`) is a CRUD-over-typed-
RAG-resources skeleton that currently only exports a `LazyAccessor`; `srag` (`a/srag`) is an
explicitly experimental LangChain-coupled prototype (`Raglab2.ask`). Both predate the
`ef`+`vd` substrate and the `ir` framing. **Recommendation:** treat them as superseded — mine
any worthwhile ideas, then let `ir` be the definitive substrate rather than maintaining three
RAG-ish efforts. Do not build `ir` on either. *Flagging, not deciding.*

**(R3) `projreg.search` is a second retrieval implementation.** It works and is the proto-`ir`
for corpus C. Leaving it independent guarantees drift. **Recommendation:** make `projreg` a
*source* (ledger/docs/gh_sync) and route its search through `ir`, deprecating the standalone
scorer once `ir` covers it.

**(R4) `chromadol` is now redundant for `ir`.** `vd` already wraps Chroma (and 15 others) with
a uniform contract; `ir` should target `vd`, not `chromadol` directly. (`chromadol` remains
fine as a standalone DOL.)

**(R5) `imbed_data_prep` is an empty placeholder** — not a building block today.

**(R6) Scale honesty.** `vd`'s client-side BM25 is O(N) (<~100k docs); above that, lean on a
native-hybrid backend or push lexical into ES/pgvector-FTS. `ir` should pick the backend via
`vd.recommend_backend` per corpus and *log* when it falls back to brute-force lexical (no
silent caps).

**(R7) Foundation-package opportunities to surface (not silently change):** multi-surface
indexing arguably belongs in `ef` long-term; a first-class **sparse-vector** type and a
**reranking** seam arguably belong in `vd`. These are ecosystem improvements to raise with the
user per the "report bugs/improvements in local packages" rule — not to slip in unannounced.

---

## 9. Open decisions for the user

1. **`ir` ⟂ `ef` boundary (R1).** Confirm `ir` = agentic-IR layer *above* `ef`, with `ef`/`vd`
   as hard dependencies and no internal re-vendoring. (Recommended.) Or do you envision `ir`
   absorbing parts of `ef`?
2. **Legacy consolidation (R2/R3).** OK to formally mark `raglab`/`srag` superseded and turn
   `projreg.search` into an `ir` source rather than a parallel searcher?
3. **First corpus to ship.** The three `ir_0x` docs and the substrate readiness both point at
   **capability discovery (skills first)** as the cheapest end-to-end slice (source already
   exists in `priv.skills_index`; small N; clean eval via `ir_03`). Confirm that's target #1,
   with **preferred-ecosystem (our packages)** — the richest test of the multi-surface seam —
   as #2.
4. **Where multi-surface indexing lives.** Incubate the `IndexingStrategy` seam in `ir`
   (recommended), with a documented path to graduate it into `ef` if it proves general?
5. **Embedder/reranker defaults.** Adopt `ir_02`'s production stack (instruction-tuned
   embedder + Voyage rerank-2.5 / Qwen3-Reranker behind `ef`'s protocols) as `ir` defaults,
   keeping the hashing embedder as the zero-dependency light-path default?

---

## 10. References

**Companion design docs (this folder):**
- `ir_01 — Progressive Capability Discovery for AI Agents — Give the Agent One Search Tool`
- `ir_02 — Indexing & Embedding Strategy for Agentic Capability Artifacts`
- `ir_03 — Evaluating the Capability-Discovery Layer of Agentic AI`

**Ecosystem code (repo-relative under `$PP`):**
- `t/ef` — Embedding Flow. Key: `ef/corpus.py`, `ef/segments.py`, `ef/segmenters.py`,
  `ef/embedders.py`, `ef/embedder_adapters.py`, `ef/artifact_graph.py`, `ef/source_manager.py`,
  `ef/diagnostics.py`, `ef/refresh.py`, `ef/reranking.py`, `ef/evaluation.py`, `ef/service.py`.
- `i/vd` — vector-DB facade. Key: `vd/base.py`, `vd/filters.py`, `vd/search.py`,
  `vd/providers.py`, `vd/backends/`.
- `t/imbed` — chunking, clustering, planar embeddings, cluster labeling.
- `t/priv` — `priv/skills_index.py`, `priv/dep_graph.py`, `priv/contexts.py`,
  `priv/pypi_alignment.py`.
- `projreg` — `read.py` (`ProjectRecord`), `ledger.py`, `docs.py` (`DocStore`), `gh_sync.py`,
  `search.py`.
- `t/hubcap` — GitHub facade (`RepoReader`, `repo_slurp.py`).
- `t/contaix` — `code.py` (`code_aggregate`), `web.py`, `markdown.py`.
- `t/oa` — `prompt_function`, embeddings (LLM-authored surfaces).
- `t/graze` — disk cache for external fetches (staleness).
- `i/ju` — schema/contract layer; `i/qh` — HTTP/agent surface; `i/meshed` — DAG composition.

**External work (from `ir_01`–`ir_03`, for the agentic-IR layer):**
- ToolRet — tool-retrieval benchmark, arXiv 2503.01763.
- Tool-DE (LLM document expansion) — arXiv 2510.22670.
- RAG-MCP — arXiv 2505.03275. ScaleMCP (content-hash CRUD sync) — arXiv 2505.06416.
- Re-Invoke (synthetic queries) — arXiv 2408.01875.
- Anthropic Tool Search Tool (2025-11). Inspect AI (MIT). BEIR / MTEB. deepdiff.

---

*Authored as the architecture/reuse analysis for `ir`. The operative instruction throughout:
**compose `ef` + `vd` + existing source packages; build only the agentic-IR band on top.***
