# ir

**An information-retrieval substrate for agentic systems** — one uniform
"find the relevant things in this corpus" contract that scales from an ad-hoc
search over an ephemeral list to a maintained capability-discovery engine.

Give an agent *one* search tool, not fifty tool schemas. `ir` retrieves
candidates, **commits to a small high-precision subset** (the distractor problem
is the central selection risk — fewer, better candidates beat more), and
**discloses** each committed item's payload only when asked.

```python
import ir

# Define a corpus, build the index (incremental), then discover:
source = ir.CorpusSource.from_skills()       # or from_packages(), from_md_reports(),
                                              # from_claude_sessions(), from_files(...)
corpus = ir.build(source)                     # embed + persist under XDG dirs
result = ir.discover(corpus, "how do I deploy the app to the server")

for item in result.results:
    print(item.score, item.name)              # the committed few (or result.abstained)
print(result.to_dict())                       # JSON-serializable (qh / HTTP ready)
```

## Install

```bash
pip install ir
```

`ir` is **light by default** — `numpy` + [`dol`](https://github.com/i2mint/dol)
for storage, plus [`ef`](https://github.com/thorwhalen/ef) /
[`vd`](https://github.com/i2mint/vd) for embedding and lexical/hybrid retrieval.
Python ≥ 3.10.

Notes for the default (semantic) path:

- The default embedder is `all-MiniLM-L6-v2` (384-dim, via
  `sentence-transformers`), **downloaded on first build** (needs network) and
  cached under `~/.cache/ir`. For a fast, offline, dependency-light run — tests,
  CI, quick experiments — pass `embedder="light"` (a numpy-only hashing
  embedder): `ir.build(source, embedder="light")`.
- `ir` sets `USE_TF=0` on import so `transformers` does not pull in TensorFlow
  (which crashes on some numpy ABIs); import `ir` before anything that imports
  `transformers`.
- Case generation (`ir.eval_gen`) and the optional LLM selector need an LLM via
  [`oa`](https://github.com/thorwhalen/oa) — install the extra,
  `pip install "ir[llm]"`. Scoring and evaluation themselves stay offline.

## The pipeline

`ir` is a six-stage pipeline, each stage a small, swappable seam:

| Stage | Entry point | What it does |
|-------|-------------|--------------|
| **source** | `CorpusSource` | what is in the corpus + what counts as stale |
| **index** | `ir.build` | decompose artifacts into embeddable *surfaces*, embed, persist (incremental, idempotent) |
| **retrieve** | `ir.search` | hard metadata filter + `dense` / `lexical` / `hybrid` ranking |
| **expand** | `ir.expand` | stitch a hit's stored neighborhood (sentence window / whole artifact) into the passage downstream reads — opt-in |
| **select** | `ir.select` | commit to a distractor-robust subset, or abstain |
| **disclose** | `ir.disclose` | load the heavy payload (SKILL.md body, package pointer, file text) for committed items — append-only |

`ir.discover` chains retrieve → select → disclose into the single agent-callable
(and `qh`-exposable) tool. Pass a **list** of corpus names for single-shot
*federated* discovery across several corpora:

```python
ir.discover(["skills", "packages"], "deploy the app")    # fan-out → fuse → select
ir.discover(["skills", "packages"], q, min_score="auto") # gate each source on its own floor
```

Each source is searched and gated on its **own** calibrated abstention floor
*before* any merging; the survivors then rank-fuse (weighted RRF via
`ir.fuse_hits`) — raw scores never cross a source boundary, because scores from
different corpora / embedders / modes live on incommensurable scales. Every hit
carries its corpus name as `hit.source`, so same-id artifacts from different
corpora stay distinct, attributable results. The caller names the sources;
`ir` never chooses the set (source planning belongs to the agent layer —
see [`raglab`](https://github.com/thorwhalen/raglab)).

### Retrieve

```python
hits = ir.search(corpus, "deploy app", mode="hybrid")   # dense | lexical | hybrid (RRF)
```

Dense is exact brute-force cosine; `lexical` is Okapi BM25; `hybrid` fuses both
by Reciprocal Rank Fusion (the strongest default for short, identifier-heavy
capability text). Lexical/hybrid reuse [`vd`](https://github.com/i2mint/vd);
dense needs only numpy.

Hybrid has a second fusion, `fusion="blend"` — a magnitude-preserving score
blend instead of rank-RRF. RRF discards score magnitude, which is exactly what
abstention calibration needs, so `blend` separates in-scope from out-of-scope
queries far better (and even beats dense); the tradeoff is lower lexical recall
on terse corpora, so RRF stays the default. Use `blend` when abstention matters
— see [`ir_08`](misc/docs/ir_08%20--%20Magnitude-Preserving%20Hybrid%20Fusion%20--%20Trading%20rank-RRF%20for%20abstention%20separability.md).

### Expand

```python
passage = ir.expand(hit, corpus)                             # ±1 chunk window (default)
passage = ir.expand(hit, corpus, policy=ir.parent_policy())  # the whole artifact
```

The matched chunk is rarely the unit you want to read, and the whole document
is usually too much. `ir.expand` stitches a hit's *stored sibling segments*
into a mid-granularity `Passage` (retrieve → expand → rerank), with chunker
overlap deduped and the hit's identity and score untouched. Policies are
injectable (`NeighborhoodPolicy`); `sentence_window_policy(k)` and
`parent_policy()` ship. It also works through the disclosure seam:

```python
ir.disclose(sel, expand=ir.sentence_window_policy(2), corpus=corpus)
ir.discover("skills", q, expand=ir.sentence_window_policy())  # passages on committed items
```

Each `Disclosure` then carries the stitched text as `.passage` (additive;
`summary` stays the matched surface, `body` stays the pointer's payload).

### Select

```python
sel = ir.select(hits)                      # conservative default: stay within rel of top, cap at max_k
sel = ir.select(hits, min_score=0.4)       # opt in to abstention ("nothing applies")
sel = ir.select(hits, strategy="score_gap")  # elbow cut, or "top_k" / "rel_threshold" / a callable
```

The abstention floor is mode-specific (dense cosine, BM25, and RRF live on
different scales), so rather than guess `min_score`, **calibrate** it from a case
file and let `discover` load it:

```python
ev.calibrate_min_score(corpus, cases, mode="dense", persist=True)  # learn + store the floor
ir.discover(corpus, query, mode="dense", min_score="auto")         # abstain by the calibrated floor
```

Calibration separates in-scope from out-of-scope query top-scores and picks the
floor that best splits them — see
[`ir_07`](misc/docs/ir_07%20--%20Min-Score%20Calibration%20--%20Abstention%20floors%20from%20score-distribution%20separability.md);
it works best on `dense` / `lexical` (hybrid's RRF scores barely separate).
`min_score` defaults to `None` (never abstain), so abstention stays fully opt-in.

The conservative defaults (`max_k=3`, `rel=0.9`) are tuned, not guessed — see
[`ir_06`](misc/docs/ir_06%20--%20Selector%20Tuning%20--%20Picking%20conservative-selector%20defaults%20from%20the%20data.md);
re-tune for your own corpus with `ev.sweep_selector` / `ir sweep-select`.

Selection is *relative* (ratios to the top score), so one selector works across
`dense` / `hybrid` / `lexical` whose absolute scales differ by orders of
magnitude. The result carries auditable `signals` and a `reason` — no opaque
"confidence" float. An optional LLM selector (`make_llm_selector`, lazy on
[`oa`](https://github.com/thorwhalen/oa), injectable for tests) falls back to the
heuristic on any failure.

### Disclose

```python
payloads = ir.disclose(sel, level="body")  # "metadata" (no I/O) | "body" | "bundled"
```

Disclosure is a *pure* read that follows the pointer already stored on each hit
(`skill_path` / `path`); it never mutates the ranked hits and tolerates a stale
pointer. Keeping the agent's context append-only (to protect the prompt cache)
is then the caller's discipline — `ir` hands back additive payloads.

### Graph & traverse (opt-in)

Artifacts refer to each other — a package depends on packages, a skill has a
parent. `ir` models those as a **semantic link graph**: a typed-edge `links`
view on the store, populated at build time by an `EdgeExtractor`.

```python
corpus = ir.build(source, edge_extractor=ir.default_edge_extractor)  # deps→REF, parent→PARENT
graph  = ir.CorpusGraph(corpus)
graph.neighbors("contaix", edge_type="REF")     # the package's dependencies
```

`ir.traverse` walks that structure at query time under a pluggable `WalkPolicy`
(*score frontier → select → expand → stop*). **Safety is the operator's**: a
visited-set, depth cap, and node budget live in `traverse` itself, so even a
cyclic graph and a never-stopping policy terminate. The shipped
`collapsed_tree_policy` is pure-vector — it routes a query that matches an
artifact's *summary* down to that artifact's best *chunk*:

```python
hits = ir.traverse(query, corpus, policy=ir.collapsed_tree_policy())
```

The *summary* a query routes on can be an artifact's own short field — or an
**LLM-authored synopsis**. `ir.with_synopsis` wraps any indexing strategy to add
one `synopsis` surface per artifact at build time (the document-summary-index
pattern: build-time cost, ≈free at query time), and that synopsis becomes the
collapsed-tree router:

```python
strat  = ir.with_synopsis(ir.Chunked(), synthesize=my_summarizer)  # or default (lazy oa)
corpus = ir.build(ir.CorpusSource.from_mapping(docs, name="d", strategy=strat))
hits   = ir.traverse(q, corpus, policy=ir.collapsed_tree_policy())  # routes via the synopsis
```

`synthesize` is injectable (a test double or your own summarizer); omitted, it is
built lazily on [`oa`](https://github.com/thorwhalen/oa) so `import ir` stays
offline. Synopses are derived state with a stamped synthesizer identity, so a
prompt/model change re-synthesizes only the affected artifacts on the next
incremental `build` — no silent staleness.

**Flat top-k stays the default** — `traverse` is opt-in, and a policy earns its
keep only by beating flat+rerank on your eval set (a strong flat retriever wins
simple lookup; graph methods cost far more). Results are ordinary `SearchHit`s
with additive `metadata["walk_depth"]` / `["seed"]` provenance, so `select` /
`disclose` compose unchanged. This is the **semantic link graph** (cyclic,
query-time) — distinct from `ef.artifact_graph` (the acyclic build-time
derivation DAG).

## Evaluation

`ir.eval` scores discovery quality offline (reusing
[`ef`](https://github.com/thorwhalen/ef)'s retrieval metrics):

```python
from ir import eval as ev

cases = ev.load_cases("skills_eval.jsonl")               # query + gold artifact_id(s)
ev.evaluate_discovery(corpus, cases, mode="hybrid")      # recall@k / NDCG@k / MRR / MAP + failure taxonomy
ev.evaluate_selection(corpus, cases, strategy="conservative")  # conditional commit rate + selection P/R/F1
ev.sweep_selector(corpus, cases)                         # tune max_k × rel; .best() / .frontier() / .table()
ev.distractor_robustness_curve(source.scope, probes)     # accuracy vs catalog size
```

`evaluate_selection`'s headline is the **conditional commit rate** — the
selection decision *isolated* from retrieval (did the selector keep the gold,
*given* retrieval surfaced it?). `sweep_selector` scores a whole `max_k × rel`
grid against the cases off **one** retrieval pass, so the selector defaults can
be read off the data (`.best()`) rather than guessed. Generate cases by
back-translation with `ir.eval_gen` (needs an LLM; scoring stays offline).

## CLI

```bash
ir build skills                          # build/update a preset corpus
ir search skills "deploy the app"        # rank candidates (retrieval only)
ir discover skills "deploy the app"      # retrieve -> select
ir discover skills "deploy the app" --disclose       # + load bodies
ir discover skills "deploy the app" --min-score auto # + calibrated abstention
ir build sessions                        # index recent Claude Code sessions (turn pairs)
ir search sessions "numpy abi error" --mode lexical   # find past sessions
ir ls                                    # list corpora + record counts
ir info skills                           # config, stats, policy, calibrated floors
ir maintain --all                        # run due background work (idempotent; cron-friendly)
ir register notes files --root ~/notes --pattern '.*\.md$'  # register a custom corpus
ir rm notes                              # unregister (keeps built data)
ir eval-gen skills skills_eval.jsonl     # generate eval cases (needs oa/LLM)
ir eval skills skills_eval.jsonl         # score retrieval on a case file
ir eval-select skills skills_eval.jsonl  # score the selection stage
ir sweep-select skills skills_eval.jsonl # tune the selector (max_k × rel) on your corpus
ir calibrate-min-score skills skills_eval.jsonl --persist  # calibrate the abstention floor
```

## Design

The design is grounded in a set of capability-discovery research reports and
eval-run findings under `misc/docs/` (`ir_01`–`ir_08`): the single-search-tool
pattern, indexing & embedding strategy, evaluation, the `ef` + `vd` reuse
analysis, the dense-vs-lexical-vs-hybrid eval, selector tuning, abstention-floor
calibration, and magnitude-preserving fusion. `ir` is light by default (numpy /
`dol`) and reuses the ecosystem (`ef`, `vd`, `oa`) only where it composes cleanly.
