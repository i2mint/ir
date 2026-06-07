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
source = ir.CorpusSource.from_skills()       # or from_packages(), from_md_reports(), from_files(...)
corpus = ir.build(source)                     # embed + persist under XDG dirs
result = ir.discover(corpus, "how do I deploy the app to the server")

for item in result.results:
    print(item.score, item.name)              # the committed few (or result.abstained)
print(result.to_dict())                       # JSON-serializable (qh / HTTP ready)
```

## The pipeline

`ir` is a five-stage pipeline, each stage a small, swappable seam:

| Stage | Entry point | What it does |
|-------|-------------|--------------|
| **source** | `CorpusSource` | what is in the corpus + what counts as stale |
| **index** | `ir.build` | decompose artifacts into embeddable *surfaces*, embed, persist (incremental, idempotent) |
| **retrieve** | `ir.search` | hard metadata filter + `dense` / `lexical` / `hybrid` ranking |
| **select** | `ir.select` | commit to a distractor-robust subset, or abstain |
| **disclose** | `ir.disclose` | load the heavy payload (SKILL.md body, package pointer, file text) for committed items — append-only |

`ir.discover` chains retrieve → select → disclose into the single agent-callable
(and `qh`-exposable) tool.

### Retrieve

```python
hits = ir.search(corpus, "deploy app", mode="hybrid")   # dense | lexical | hybrid (RRF)
```

Dense is exact brute-force cosine; `lexical` is Okapi BM25; `hybrid` fuses both
by Reciprocal Rank Fusion (the strongest default for short, identifier-heavy
capability text). Lexical/hybrid reuse [`vd`](https://github.com/i2mint/vd);
dense needs only numpy.

### Select

```python
sel = ir.select(hits)                      # conservative default: stay within rel of top, cap at max_k
sel = ir.select(hits, min_score=0.4)       # opt in to abstention ("nothing applies")
sel = ir.select(hits, strategy="score_gap")  # elbow cut, or "top_k" / "rel_threshold" / a callable
```

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
ir discover skills "deploy the app"      # retrieve -> select
ir discover skills "deploy the app" --disclose   # + load bodies
ir eval-select skills skills_eval.jsonl  # score the selection stage
ir sweep-select skills skills_eval.jsonl # tune the selector (max_k × rel) on your corpus
ir ls                                    # list corpora
```

## Design

The design is grounded in a set of capability-discovery research reports under
`misc/docs/` (`ir_01`–`ir_05`): the single-search-tool pattern, indexing &
embedding strategy, evaluation, the `ef` + `vd` reuse analysis, and a dense-vs-
lexical-vs-hybrid eval run. `ir` is light by default (numpy / `dol`) and reuses
the ecosystem (`ef`, `vd`, `oa`) only where it composes cleanly.
