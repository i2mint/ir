# ir_06 — Selector Tuning: Picking the conservative-selector defaults from the data

**Sweeping the commit knobs (`max_k` × `rel`) of `ir.select`'s conservative selector against three real corpora, in two ranking modes**

*Run date: 2026-06-07 · Machine-local (Mac) · Embedder: `sentence-transformers/all-MiniLM-L6-v2` · Harness: `ir.eval.sweep_selector` (shipped in ir 0.1.8)*

This is the results companion to the **selection stage** (`ir.select`, shipped in
ir 0.1.7, design in `ir_01` §3). It settles the open question carried in that
stage's handoff: the conservative selector's defaults — `max_k=5`, `rel=0.6` —
were documented as *starting points, not tuned optima*. This run tunes them.

---

## TL;DR

1. **The shipped defaults (`max_k=5`, `rel=0.6`) are dominated in all six
   corpus × mode combinations.** They commit large, distractor-padded sets
   (mean 4–5 items, precision 0.20–0.33) for no recall benefit a much tighter
   setting doesn't also get.

2. **Selection F1 peaks at *tight* settings everywhere: small `max_k` (1–3) with
   the strictest band swept, `rel≈0.9`.** Versus the current default this roughly
   **halves the committed-set size and improves both precision and F1**
   across every corpus and both modes.

3. **The "informative" relative-score band for MiniLM/RRF is ≈0.85–0.95, not
   0.6.** Genuine near-ties sit within ~10–15% of the top score; everything below
   ~0.8×top is mostly distractor. `rel=0.6` is loose enough to rake those in. This
   is a transferable fact about the score geometry, and it validates the design
   choice to compare *relatively* — the band just needs to be strict.

4. **A single mode-agnostic default holds.** The optimum is the same shape
   (small `max_k`, `rel≈0.9`) under dense and hybrid on all three corpora, which
   is exactly what the relative-score selector was designed to deliver.

5. **The genuine trade-off is precision vs. commit-rate, not precision vs.
   recall-of-multi-gold** (the cases are mostly single-gold). Tightening to
   `rel=0.9` drops the *conditional commit rate* (P(commit ≥1 gold | gold
   retrieved)) by ~10–15 points — i.e. it occasionally drops a gold that sat at
   rank 2–3 below the band — in exchange for the large precision/size win. Which
   way to lean is a product call; see §4.

---

## 1. What was measured

The conservative selector keeps the top hit and admits each next hit only while
its score stays `≥ rel × top_score`, capped at `max_k` (`ir/select.py`). Two
knobs, swept as a grid:

- `max_k ∈ {1, 2, 3, 5, 8}`
- `rel ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`
- `min_score`: left off (`None`) — an absolute floor is mode-specific and
  orthogonal to the relative band being tuned here.

For each of the 35 cells, `ir.eval.sweep_selector` scores the commit with the
exact `evaluate_selection` metric (selection quality **conditioned on retrieval
having surfaced the gold**, so retrieval misses are never charged to the
selector). Retrieval is run **once per case** and the candidates reused across
the whole grid, so a 35-cell sweep costs one retrieval pass, not 35.

Corpora and case files (the judge-clean sets from the `ir_05` run, `k=10`
candidate window):

| corpus | embedder | cases | gold | case file |
|---|---|---|---|---|
| skills | MiniLM | 704 | 704 | `skills_eval_judgeclean.jsonl` |
| packages | MiniLM | 685 | 582 | `packages_eval.jsonl` |
| reports | MiniLM | 120 (sample) | 100 | `reports_eval_sample.jsonl` |

Each was swept in `dense` and `hybrid` mode (6 runs total).

**Metrics** (all conditioned on `n_gold_retrieved`, the gold-was-surfaced
denominator):
- `cond_commit` — P(selector commits ≥1 gold | gold retrieved). The
  retrieval-isolated "did we keep the right thing" rate.
- `precision` — `|selected ∩ gold| / |selected|` (distractor padding).
- `recall` — `|selected ∩ gold| / |gold ∩ retrieved|`.
- `F1` — per-case harmonic mean (the headline scalar tuned against).
- `sel_size` — mean committed-set size (distractor exposure / cost).

---

## 2. Results — F1-best vs. the shipped default

For each corpus × mode: the F1-optimal cell, and the **shipped default cell
(`max_k=5, rel=0.6`)** for comparison. (Full 35-row grids are in
`~/Downloads/ir_eval/sweep_selector_results.txt`.)

| corpus / mode | best cell | F1 | P | R | cond_commit | size | | default (5, 0.6) F1 | P | size |
|---|---|---|---|---|---|---|---|---|---|---|
| skills / dense  | **max_k=3, rel=0.9** | 0.676 | 0.635 | 0.766 | 0.777 | 1.84 | | 0.459 | 0.332 | 4.24 |
| skills / hybrid | **max_k=1**          | 0.630 | 0.709 | 0.591 | 0.709 | 1.00 | | 0.386 | 0.249 | 5.00 |
| packages / dense  | **max_k=1**        | 0.599 | 0.599 | 0.599 | 0.599 | 1.00 | | 0.352 | 0.236 | 4.56 |
| packages / hybrid | **max_k=2, rel=0.9** | 0.449 | 0.395 | 0.558 | 0.558 | 1.60 | | (loose cells, F1 ≤ 0.29) |
| reports / dense  | **max_k=3, rel=0.9** | 0.395 | 0.331 | 0.571 | 0.571 | 2.19 | | 0.259 | 0.157 | 4.93 |
| reports / hybrid | **max_k=1**         | 0.500 | 0.500 | 0.500 | 0.500 | 1.00 | | (loose cells, F1 ≤ 0.35) |

The pattern is consistent: the optimum is `rel=0.9` (the strictest band swept)
with small `max_k`. Two sub-regimes:

- **Rank-1 retrieval is strong** (skills/hybrid, packages, reports/hybrid) →
  `max_k=1` (just commit the top) is at or near the F1-peak.
- **Gold often sits at rank 2–3** (skills/dense, reports/dense) → a small
  `max_k=3` with strict `rel=0.9` wins, recovering near-tie golds without
  pulling in the distractor tail.

### Picking one mode-agnostic default

Mean F1 across all six combos for the leading candidates:

| candidate | mean F1 | notes |
|---|---|---|
| **`max_k=3, rel=0.9`** | **0.520** | wins skills/dense + reports/dense; close 2nd elsewhere; a little headroom for genuine multi-gold |
| `max_k=2, rel=0.9` | 0.530 | best mean; caps at 2 even when 3 genuine near-ties exist |
| `max_k=1` (any rel)  | 0.496 | best where rank-1 is strong; worst where gold sits at rank 2–3 (reports/dense F1 0.30) |
| `max_k=3, rel=0.8` | 0.474 | higher commit-rate, lower precision/F1 |
| **`max_k=5, rel=0.6` (current)** | **~0.39** | dominated everywhere |

`max_k=2` and `max_k=3` at `rel=0.9` are within 0.01 of each other on mean F1;
`max_k=3` gives a little more headroom for genuinely multi-capability queries
(the cases here are mostly single-gold, which under-counts that benefit) at
negligible cost on these sets.

---

## 3. Why `rel=0.6` is the wrong band

`rel` is a fraction of the *top* score, and the score geometry of both MiniLM
cosine and RRF-fused hybrid is such that a genuine second-best near-tie lands at
~0.85–0.95 × top, while the distractor tail starts around ~0.8 × top and below.
So:

- `rel=0.9` admits only true near-ties → precision stays high, size stays ~1–2.
- `rel=0.6` reaches deep into the distractor tail → with `max_k≥3` it commits
  3–5 items at precision 0.2–0.4.

The relative-to-top design is sound (the same band transfers across dense/RRF
scales, confirmed by the mode-agnostic optimum); the shipped *value* was simply
too loose.

---

## 4. The trade-off, stated honestly

F1 weights precision and recall equally and points unambiguously at `rel=0.9`.
But the underlying choice for an agent-facing tool is **precision vs. conditional
commit-rate**:

- A **padded commit** (low precision, the current default) = the agent gets the
  right capability *plus* 2–4 distractors it must ignore — a *soft* failure
  (context cost, possible wrong-tool selection; the central risk in `ir_01` §3).
- A **dropped gold** (lower commit-rate, the tight setting) = for ~10–15% more
  of cases the agent gets *nothing* committed even though retrieval surfaced the
  gold — a *hard* failure for that query.

Concretely, on skills/dense, moving (5, 0.6) → (3, 0.9) takes precision
0.33 → 0.64 and size 4.24 → 1.84, but conditional commit-rate 0.91 → 0.78.

Either way the current default is dominated — every tighter cell beats it on F1
*and* size. The remaining question is only *how* tight, and that is the call
recorded in §5.

---

## 5. Recommendation & decision

**Empirical recommendation:** change the conservative-selector defaults to
**`max_k=3`, `rel=0.9`** (was `max_k=5`, `rel=0.6`). It is at/near the F1 peak on
every corpus and mode, roughly halves committed-set size, and keeps the
mode-agnostic design intact.

> **Decision (2026-06-07, confirmed):** adopted **`max_k=3`, `rel=0.9`**
> (`DFLT_MAX_K` / `DFLT_REL_THRESHOLD` in `ir/select.py`, mirrored by
> `evaluate_selection` and the `eval-select` CLI). Shipped in ir 0.1.8. The
> precision-leaning option was chosen because for an agent-facing capability
> surface a padded commit (distractors the model must filter) is the costlier
> failure mode at scale (`ir_01` §3).

If a more recall-leaning default is preferred (keep more borderline golds at the
cost of larger, less precise commits), `max_k=3, rel=0.8` is the balanced
alternative (mean F1 0.474, commit-rate ~5–10 points higher than rel=0.9).

`min_score` stays `None` by default (an absolute floor is mode-specific; the
relative band does the work). Callers wanting true abstention still supply a
calibrated floor or the LLM selector, exactly as before.

---

## 6. Caveats & reproduction

- **Single-gold dominance.** The eval sets are mostly single-gold, which
  *under*-counts the benefit of `max_k>1` (a query with 3 genuine co-relevant
  capabilities can't reward a larger cap if the gold is one id). The chosen
  `max_k=3` deliberately leaves that headroom.
- **Sample reports set.** Reports was tuned on the n=100 sample (the full
  16,836-record corpus is slow under hybrid); the skills/packages sets are full.
- **Machine-specific numbers.** The corpora and case files are live and local,
  so the *numbers* are not reproducible in CI — but the **harness**
  (`ir.eval.sweep_selector`) and the **chosen defaults** are committed. Re-tuning
  on a new corpus is one command: `ir sweep-select <corpus> <cases.jsonl>`.
- **Reproduce this run:** `USE_TF=0 python ~/Downloads/ir_eval/sweep_selector_run.py`
  (script not committed; corpora built per `ir_05`).
