# ir_05 — Capability-Discovery Eval Run: Findings

**Dense vs Lexical vs Hybrid retrieval, measured with the `ir.eval` harness over three real corpora**

*Run date: 2026-06-06 · Machine-local (Mac) · Embedder: `sentence-transformers/all-MiniLM-L6-v2` · Generation model: `gpt-4o-mini` (via `oa`)*

This is the **results companion** to `ir_03` (the eval *design*). It reports the
first full measurement run of the capability-discovery eval harness
(`ir.eval` + `ir.eval_gen`, shipped in ir 0.1.3/0.1.4) and settles the open
question carried in the eval-harness handoff.

---

## TL;DR

1. **The preliminary "dense ≫ hybrid" finding (n=8) does not generalize — it
   reverses.** On the full skills set (n=776 back-translated, name-masked
   queries), **hybrid beats dense by a statistically significant margin under
   every scoring lens** we tried (Wilcoxon p between 9×10⁻⁵ and 7×10⁻⁶). The
   n=8 sample was simply too small and unrepresentative.

2. **The best ranking mode is corpus-dependent, and the deciding factor is how
   keyword-rich the artifact descriptions are.**
   - **skills** (verbose "Use when…" descriptions) → **hybrid > dense ≈ lexical**
   - **reports** (markdown doc chunks, text-rich) → **hybrid > dense > lexical**
   - **packages** (terse one-line descriptions) → **dense > hybrid > lexical**

   Where descriptions carry real lexical content, BM25 has signal even after the
   gold name is masked out, and RRF fusion lifts hybrid to the top. Where
   descriptions are terse, BM25 is mostly noise and drags hybrid below pure dense.

3. **`hybrid` is the safest default.** It wins outright on 2 of 3 corpora and is
   never catastrophic; pure `lexical` is never best and is fragile on terse
   corpora (NDCG@10 0.31 on packages).

4. **Two infrastructure findings fell out of the run** (details in §6):
   - `ir.retrieve` rebuilds the BM25 index from scratch on **every** query, so
     lexical/hybrid are O(corpus × queries) and do **not** scale — a full
     lexical/hybrid pass over the 16,836-record `reports` corpus did not finish
     in 10 min of CPU. A persisted/cached BM25 index is needed for large corpora.
   - The skills corpus contains **27% near-duplicate artifacts** (the same skill
     indexed globally and again under a package, e.g. `acquire-references` and
     `acquire-references@citeget`), which silently deflates single-gold scores.

---

## 1. What was measured

For each corpus we generated an eval set by **back-translation** (`ir eval-gen`):
for every artifact, `gpt-4o-mini` was asked for `k=5` natural user requests that
the artifact should answer, *after* the artifact's name was masked out of its
description (so a lexical retriever can't trivially match on the name). A ~15%
slice of out-of-scope ("abstention") intents was added. Each artifact id is the
free ground-truth gold label.

| corpus | artifacts indexed | cases generated | gold | abstention |
|---|---|---|---|---|
| **skills** | 157 (full) | 913 | 776 | 137 |
| **packages** | 120 (sample of 1,508) | 685 | 582 | 103 |
| **reports** | 120 (sample of 16,836) | 705 | 600 | 105 |

Scoring (`ir eval`) is offline and deterministic: each query is retrieved once
per mode, scored with `ef.evaluation`'s NDCG/recall/MRR/MAP primitives plus the
`ir` failure taxonomy (`hit_rank_1` / `surfaced_low_rank` / `retrieval_miss`).
All corpora were built with the production `all-MiniLM-L6-v2` embedder.

> **Sampling caveat.** packages/reports were capped at 120 artifacts (`--max-artifacts`,
> sorted-id order for skills/packages; the reports sample spans many top-level path
> prefixes, so it is not degenerate). The reports *scoring* was further restricted
> to a random 100-gold sample because of the BM25 scaling issue (§6.1).

---

## 2. Headline results (NDCG@10)

| corpus / lens | n_gold | dense | lexical | hybrid | winner |
|---|---|---|---|---|---|
| **skills** — strict | 776 | 0.659 | 0.641 | **0.694** | hybrid |
| **skills** — near-dup expanded | 776 | 0.705 | 0.679 | **0.744** | hybrid |
| **skills** — judge-clean | 704 | 0.728 | 0.709 | **0.772** | hybrid |
| **skills** — judge-graded | 719 | 0.733 | 0.684 | **0.765** | hybrid |
| **packages** — strict | 582 | **0.583** | 0.305 | 0.489 | dense |
| **reports** — strict (n=100) | 100 | 0.431 | 0.367 | **0.477** | hybrid |

"Lenses" are four increasingly-fair definitions of the gold set for skills (§4).
The mode **ordering is identical across all four skills lenses** — the result is
not an artifact of how we define correctness.

---

## 3. Is hybrid > dense real? (significance)

Paired per-query NDCG@10, Wilcoxon signed-rank test + paired bootstrap 95% CI on
the mean difference, for the skills corpus:

| lens | hybrid − dense | 95% CI | Wilcoxon p | hybrid − lexical | dense − lexical |
|---|---|---|---|---|---|
| strict (776) | **+0.036** | [+0.015, +0.056] | 8.9×10⁻⁵ ✓ | +0.053 (3.8×10⁻⁷ ✓) | +0.018 (n.s.) |
| near-dup (776) | **+0.038** | [+0.017, +0.060] | 5.1×10⁻⁵ ✓ | +0.065 (3.5×10⁻¹¹ ✓) | +0.027 (n.s.) |
| judge-clean (704) | **+0.044** | [+0.022, +0.066] | 6.6×10⁻⁶ ✓ | +0.063 (6.8×10⁻¹⁰ ✓) | +0.019 (n.s.) |
| judge-graded (719) | **+0.032** | [+0.013, +0.051] | 3.5×10⁻⁵ ✓ | +0.082 (1.5×10⁻²⁰ ✓) | +0.050 (5.4×10⁻⁴ ✓) |

- **hybrid > dense: significant in all four lenses.**
- **hybrid > lexical: significant in all four lenses** (and the strongest effect).
- **dense ≈ lexical: not significant** in three of four lenses — on skills,
  pure lexical retrieval is statistically *tied* with pure dense. (Only under the
  judge-graded lens, whose expanded gold rewards the semantic matching dense does
  best, does dense pull significantly ahead of lexical.)

The effect sizes are modest (~3–4 NDCG points) but consistent and significant —
the kind of robust, replicated delta the n=8 preliminary run could not see.

---

## 4. The four skills lenses (and why they exist)

Single-gold back-translation is a *proxy*: a query may legitimately be answered
by more than the one artifact it was generated from. We measured under four
progressively-fairer gold definitions and confirmed the conclusion survives all:

1. **strict** — gold = the single source artifact. Lower bound.
2. **near-dup expanded** — gold = the artifact's base-name equivalence class.
   The skills corpus indexes the *same* skill twice (globally and under a
   package): **21 groups, 42 of 157 artifacts (27%)**. A query from
   `acquire-references` is equally answered by `acquire-references@citeget`, but
   strict gold lists only one. Expanding to the equivalence class lifts NDCG@1 by
   ~0.11 (e.g. dense 0.459 → 0.566) — i.e. ~27% of "rank-1 misses" were really
   the sibling at rank 1.
3. **judge-clean** — an LLM judge (32-agent fan-out, gpt-4o-mini) audited every
   gold case against the full 157-skill catalog and classified the probe:
   **704 good / 42 trivial / 15 misrouted / 15 ambiguous**. judge-clean keeps the
   704 "good" probes only (gold = near-dup expanded). Measures retrieval on fair,
   unambiguous probes — generation quality is high (90.7% good).
4. **judge-graded** — keeps good + ambiguous probes and expands gold to
   near-dup ∪ judge-confirmed *also-relevant* skills. The judge flagged
   **additional** genuinely-relevant skills for **319 of 776 (41%)** cases
   (e.g. CI-planning queries → `ci-advisor` *and* `ci-setup`), so single-gold
   scoring understates true relevance for a large fraction of probes.

---

## 5. Why the ordering flips between corpora

The mechanism is **lexical signal density in the masked description**:

- **skills** descriptions are long, imperative "Use when… / Triggers on…" prose
  packed with domain keywords. Even with the skill's own name masked out, the
  back-translated query shares many content words with the description, so BM25
  is strong (lexical NDCG@10 0.64, ≈ dense). Hybrid then fuses two strong, partly
  independent signals → best.
- **reports** are markdown doc chunks — also text-rich, so the same logic holds
  (hybrid > dense > lexical).
- **packages** descriptions are terse single lines (often just a noun phrase).
  After masking there is little lexical overlap with a paraphrased query, so BM25
  is mostly noise (lexical NDCG@10 0.31) and **drags hybrid 0.10 below pure
  dense**. Semantic (dense) matching is the only thing that works on terse text.

**Takeaway:** the more your capability descriptions read like keyword-rich usage
docs, the more lexical/hybrid helps. For terse catalogs, lean dense.

---

## 6. Infrastructure findings (surfaced, not yet fixed)

### 6.1 BM25 is rebuilt on every query — lexical/hybrid don't scale
`ir.retrieve._lexical_ranked` constructs the candidate collection and calls
`vd.bm25_lexical_search` **per query**, with no persisted/cached index. Cost is
O(corpus_size × n_queries). For skills (157) and packages (1,508) this is
tolerable; for the **reports corpus (16,836 records)** a full lexical+hybrid pass
over 600 queries did **not finish in >10 min of 100%-CPU** and had to be killed
and down-sampled to 100 queries. **Opportunity (ir/vd):** build/persist the BM25
index once per corpus (alongside the dense matrix) and reuse it across queries —
this is the single biggest blocker to running `ir` at corpus scale.

### 6.2 Near-duplicate artifacts in the skills corpus
27% of skills are indexed twice (global + package-scoped) with near-identical
descriptions. This silently deflates single-gold scores and adds noise to any A/B.
**Opportunity (ir):** either dedupe at index time, or support **gold equivalence
classes** in `ir.eval` (so a measurement can treat known duplicates as one). The
near-dup lens here was a measurement-time script (`expand_gold.py`), not a library
feature.

### 6.3 `ef` private metric registry
`ir.eval.evaluate_discovery` re-declares a `name→fn` map that mirrors
`ef.evaluation._RETRIEVAL_METRICS` (private). **Opportunity (ef):** expose a
public `RETRIEVAL_METRICS` registry so `ir` stops mirroring it. (Already noted in
the handoff.)

### 6.4 `distractor_robustness_curve` reloads the embedder per build
With `embedder="default"`, `distractor_robustness_curve` calls `build(...)` per
sub-corpus, and each `build` re-instantiates the embedder from the string spec —
reloading the MiniLM/torch model thousands of times. The MiniLM curve above was
only feasible after pre-building the embedder once and passing the **object**.
**Opportunity (ir):** have the curve accept (or internally build once) a resolved
embedder and thread it through, so the production curve is practical out of the box.

---

## 7. Distractor-robustness curve (RAG-MCP framing)

Accuracy@1 as the catalog grows from 1 (gold alone) to the full 157 skills,
averaged over random distractor draws — *the* needle-in-a-haystack diagnostic.

**Production (MiniLM) embedder** — 20 probes, 5 trials, k=1:

| N | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 157 |
|---|---|---|---|---|---|---|---|---|
| dense | 1.00 | 0.92 | 0.84 | 0.72 | 0.64 | 0.54 | 0.49 | 0.45 |
| lexical | 1.00 | 0.78 | 0.65 | 0.62 | 0.57 | 0.47 | 0.39 | 0.35 |
| hybrid | 1.00 | 0.87 | 0.83 | **0.77** | **0.69** | 0.56 | 0.45 | 0.40 |

With the real embedder, **dense and hybrid both degrade gracefully and stay
close** (hybrid edges ahead in the mid-range, N=16–32), while **lexical degrades
fastest** — the production ordering, matching §2/§3. (Only 20 probes, so the
dense/hybrid crossover at large N is within sampling noise; the full-corpus
N=157 recall@1 from the n=776 scoring run is dense 0.459 / lexical 0.469 /
hybrid 0.495 — hybrid ahead, as elsewhere.) **The takeaway is the steady decay:
even the best mode loses ~half its rank-1 accuracy by N≈64, which is the case for
a selection/filtering stage (issue #11) as the catalog grows.**

> The MiniLM curve required pre-building the embedder once and passing the object;
> `distractor_robustness_curve` with a bare `embedder="default"` string reloads the
> model on every sub-corpus build and is impractically slow (opportunity 6.4).

**Offline (hashing/"light") embedder** — 60 probes, 12 trials, k=1:

| N | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 157 |
|---|---|---|---|---|---|---|---|---|
| dense | 1.00 | 0.65 | 0.52 | 0.41 | 0.33 | 0.29 | 0.24 | 0.23 |
| lexical | 1.00 | 0.80 | 0.75 | 0.66 | 0.61 | 0.52 | 0.44 | 0.40 |
| hybrid | 1.00 | 0.70 | 0.63 | 0.50 | 0.43 | 0.35 | 0.28 | 0.25 |

> The *hashing* embedder is a deliberately weak offline stand-in (no model
> download), so its "dense" leg is poor and lexical dominates — **the opposite of
> the MiniLM ordering**. The point of this curve is twofold: (a) the *shape* —
> accuracy decays steadily with catalog size for every mode, so selection/filtering
> (issue #11) matters more as the catalog grows; and (b) the dense/lexical balance
> is **embedder-quality-dependent**, reinforcing §5.

---

## 8. Reproducing this run

```bash
# generate (needs oa/LLM; writes a corpus-signature header)
ir eval-gen skills   skills_eval.jsonl   --k 5
ir eval-gen packages packages_eval.jsonl --k 5 --max-artifacts 120
ir eval-gen reports  reports_eval.jsonl  --k 5 --max-artifacts 120

# score (offline) — one mode at a time, or use score_sweep.py for a table
ir eval skills skills_eval.jsonl --mode dense   --k 10
ir eval skills skills_eval.jsonl --mode hybrid  --k 10
```

The measurement scripts (`score_sweep.py`, `sig_test.py`, `expand_gold.py`,
`build_judge_lenses.py`, `distractor_curve.py`) and the judge-audit workflow
(`judge_audit_workflow.js`) used for this run live alongside the generated case
files (machine-local, not committed — the corpora are machine-specific; the
`corpus_signature` in each case file's `__meta__` anchors it to the snapshot it
was generated against).

---

## REFERENCES

1. Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*. 2009;3(4):333–389. [link](https://doi.org/10.1561/1500000019)
2. Cormack GV, Clarke CLA, Büttcher S. Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. *SIGIR '09*. 2009. [link](https://doi.org/10.1145/1571941.1572114)
3. Thakur N, Reimers N, Rücklé A, Srivastava A, Gurevych I. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS Datasets & Benchmarks*. 2021. [link](https://arxiv.org/abs/2104.08663)
4. Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP*. 2019. [link](https://arxiv.org/abs/1908.10084)
5. Gan T, et al. RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation. 2025. [link](https://arxiv.org/abs/2505.03275)
6. Wilcoxon F. Individual Comparisons by Ranking Methods. *Biometrics Bulletin*. 1945;1(6):80–83. [link](https://doi.org/10.2307/3001968)
