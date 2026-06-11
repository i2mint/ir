# ir_07 — Min-Score Calibration: Abstention floors from score-distribution separability

**Calibrating the absolute `min_score` floor that lets the conservative selector *abstain* ("nothing applies"), by separating in-scope from out-of-scope query top-scores — across three real corpora, in all three ranking modes**

*Run date: 2026-06-10 · Machine-local (Mac) · Embedder: `sentence-transformers/all-MiniLM-L6-v2` · Harness: `ir.eval.calibrate_min_score` (shipped in ir 0.1.9)*

This is the results companion to **selector tuning** (`ir_06`, ir 0.1.8). That run
tuned the *relative* commit knobs (`max_k`, `rel`); it deliberately left
`min_score=None`, because a relative selector cannot tell "all candidates
irrelevant" from "all relevant" (`ir.select` module docstring; `ir_01` §3). This
run settles the deferred follow-up: calibrating the **absolute** floor that turns
relative selection into one that can also *abstain*.

---

## TL;DR

1. **Abstention calibration is a 1-D separability problem, not a selector sweep.**
   In-scope queries (a gold artifact exists *and* was retrieved) produce a HIGH
   top-score; out-of-scope queries (no artifact applies) produce a LOWER one (the
   best of a bad lot). The floor that best separates the two distributions is the
   abstention floor. Optimizing `sweep_selector(objective="abstention_accuracy")`
   instead is degenerate — floor → ∞ abstains on everything for a free 100%.

2. **A calibrated floor is a *useful but imperfect* filter on real corpora.**
   With the balanced objective (Youden's J), dense and lexical reach
   **J ≈ 0.65–0.90** — catching ~85–91% of out-of-scope queries while keeping
   ~85–91% of in-scope ones. None of the nine (corpus × mode) distributions is
   *perfectly* separable: the score ranges overlap, but the **central masses
   separate** (e.g. skills/dense: in-scope median 0.457 vs out-of-scope 0.210).

3. **Lexical (BM25) separates best, dense is close, hybrid is worst.**

   | corpus | dense J | lexical J | hybrid J |
   |--------|--------:|----------:|---------:|
   | skills   | 0.764 | **0.835** | 0.610 |
   | packages | 0.651 | **0.794** | 0.342 |
   | reports  | 0.814 | **0.900** | 0.326 |

4. **Hybrid is the *best ranking* mode but the *worst abstention* mode — and the
   reason is structural.** RRF (`1/(k+rank)`) is **rank-based**: it discards score
   *magnitude*. A top-ranked-but-irrelevant result earns the same RRF value
   (~0.0328) as a top-ranked-relevant one, so the in-scope and out-of-scope
   top-score distributions both collapse into the tiny ~0.0164–0.0328 band and
   barely separate (skills/hybrid: in-scope median 0.0325 vs out-of-scope 0.0301).
   The very fusion that makes hybrid *rank* well makes it nearly unable to express
   "everything here is weak."

5. **Guidance: calibrate and run abstention on `dense` or `lexical`, not
   `hybrid`** — even though `hybrid` is `discover`'s default *ranking* mode. (A
   future refinement, not built here: gate abstention on the dense sub-score even
   when ranking with hybrid — see *Open follow-ups*.)

6. **Numbers are machine-local; the harness and wiring are committed.** The floor
   is mode-, corpus- and embedder-specific, so it cannot be a committed global
   constant — it is *calibrated* per `(corpus, mode)` and persisted on the corpus
   (XDG data dir), then loaded by `discover(min_score="auto")`. Recalibrate any
   corpus with `ir calibrate-min-score <corpus> <cases.jsonl> --persist`.

---

## 1. The problem: relative selection can't abstain

`ir.select`'s conservative selector compares scores **relatively** (ratios to the
top score), which is why one selector works across `dense` cosine, `hybrid` RRF
and `lexical` BM25 despite their scales differing by orders of magnitude. But a
ratio test is blind to absolute level: a query whose best hit is genuinely
irrelevant still has *a* top score, and the tail still falls off below it, so it
looks exactly like a query with a strong, relevant top. Telling the two apart —
the **abstention** decision, "nothing here applies" — needs an *absolute* floor:
abstain when even the top hit scores below `min_score`.

That floor is **not** a global constant. Three things move it:

- **mode** — dense cosine ∈ ~[0, 1], hybrid RRF ∈ ~[0, 0.033], lexical BM25 ∈
  ~[0, 50+]; the scales differ by orders of magnitude;
- **corpus** — denser/larger corpora shift the typical top-score band;
- **embedder** — a different model re-scales everything.

So the floor must be **calibrated** from data, per `(corpus, mode)`, and treated
as machine-local derived state — never a committed literal.

## 2. Method: separability, not a sweep

Calibration is a one-dimensional threshold-classification problem. Treat
`top_score ≥ floor → commit` as a binary classifier over two labelled query sets:

- **in-scope** (label *commit*): a gold-bearing case whose gold actually reached
  the `k` retrieved candidates. Cases whose gold was *not* retrieved are a
  **retrieval** miss, not an abstention signal, and are **excluded** — the same
  `n_gold_retrieved` conditioning `evaluate_selection` already uses, so the floor
  is never blamed for retrieval's recall gap.
- **out-of-scope** (label *abstain*): an empty-`gold` abstention case. (Generate
  these with `ir.eval_gen` — it adds an abstention slice; this module consumes a
  mixed case set, it does not synthesize negatives.)

For each candidate floor `f`:

- **sensitivity** (TPR) = fraction of in-scope queries kept (`top ≥ f`);
- **specificity** (TNR) = fraction of out-of-scope queries abstained (`top < f`);
- pick `f` maximizing `w·sensitivity + (1−w)·specificity` (`w = 0.5` default,
  whose argmax is **Youden's J** `= sens + spec − 1` and balanced accuracy).

The candidate floors are the **midpoints between adjacent observed top-scores**
(plus a commit-all floor below the minimum and an abstain-all floor above the
maximum). The optimal 1-D separator always lies between two adjacent observed
values, so this grid is *exact* — there is no resolution to tune. The chosen
midpoint also maximizes the margin to both clusters. Retrieval is paid **once
per case** and reused across the whole floor grid (verified by call-count test).

Empty retrievals get `top_score = −∞` (they already commit nothing, so they
abstain at any finite floor). Ties break toward the **higher** floor — the
precision-leaning choice (`ir_01` §3: a padded commit costs an agent more than a
dropped gold), and the more robust separator when the gap above the out-of-scope
scores is wide.

## 3. Results (three real corpora, `k=10`, balanced `w=0.5`)

Top-score distributions are `[min · median · max]`. `J` is Youden's J; `npos` /
`nneg` are the in-scope / out-of-scope counts after retrieval-miss exclusion.

### skills (157 records · 913 cases · 137 abstention)

| mode | floor | sens | spec | J | in-scope top | out-of-scope top |
|------|------:|-----:|-----:|----:|------|------|
| dense   | 0.340 | 0.852 | 0.912 | 0.764 | 0.187 · 0.457 · 0.822 | 0.093 · 0.210 · 0.538 |
| lexical | 11.09 | 0.872 | 0.964 | **0.835** | 6.72 · 15.45 · 35.74 | 5.22 · 7.27 · 13.51 |
| hybrid  | 0.0314 | 0.843 | 0.766 | 0.610 | 0.0273 · 0.0325 · 0.0328 | 0.0265 · 0.0301 · 0.0328 |

### packages (1 508 records · 685 cases · 103 abstention)

| mode | floor | sens | spec | J | in-scope top | out-of-scope top |
|------|------:|-----:|-----:|----:|------|------|
| dense   | 0.366 | 0.893 | 0.757 | 0.651 | 0.239 · 0.469 · 0.761 | 0.184 · 0.286 · 0.600 |
| lexical | 15.79 | 0.901 | 0.893 | **0.794** | 12.0 · 20.4 · 53.8 | 6.92 · 12.5 · 21.5 |
| hybrid  | 0.0306 | 0.633 | 0.709 | 0.342 | 0.0164 · 0.0313 · 0.0328 | 0.0164 · 0.0290 · 0.0328 |

### reports (16 836 records · 120 cases · 20 abstention)

| mode | floor | sens | spec | J | in-scope top | out-of-scope top |
|------|------:|-----:|-----:|----:|------|------|
| dense   | 0.423 | 0.914 | 0.900 | 0.814 | 0.298 · 0.526 · 0.707 | 0.203 · 0.341 · 0.493 |
| lexical | 20.98 | 0.900 | 1.000 | **0.900** | 17.8 · 26.5 · 41.0 | 11.9 · 16.0 · 20.5 |
| hybrid  | 0.0302 | 0.576 | 0.750 | 0.326 | 0.0164 · 0.0307 · 0.0328 | 0.0164 · 0.0290 · 0.0325 |

**Reading the tables.** On dense and lexical the medians are cleanly apart and the
floor sits between them, giving a filter that is right ~85–96% of the time on each
side. On hybrid both classes are crushed into the discrete RRF band (the maxima
are all exactly `2/61 ≈ 0.0328`, a rank-1-in-both-lists tie), the medians nearly
touch, and J collapses. Lexical wins everywhere because an out-of-scope query
shares few *rare* terms with any document, so BM25 — which rewards rare-term
matches — leaves a wide low-score gap that abstention can exploit.

## 4. Decision

> **2026-06-10.** Ship `calibrate_min_score` + `MinScoreCalibration` +
> `discover(min_score="auto")` + the `ir calibrate-min-score` CLI, with the
> **balanced** objective (`sensitivity_weight = 0.5`) as the default and a knob to
> lean precision (lower `w` → abstain more readily). **Recommend `dense` or
> `lexical` for abstention**, and document that `hybrid`'s rank-fusion makes its
> absolute floor weak. Do **not** auto-persist any floor — `discover` abstains by
> absolute score only on the explicit `min_score="auto"` opt-in, and only after a
> calibration has been persisted; the default (`min_score=None`) is unchanged, so
> nothing about existing `discover` behaviour shifts.

Rationale: the floor is a genuinely useful safety filter on the two modes where it
works, the opt-in keeps it from surprising existing callers, and persisting it on
the corpus (not in the build `config`) keeps a regenerable, machine-local artifact
out of the corpus's build identity. The `embedder_id` is stamped into the record;
`discover(min_score="auto")` ignores a floor whose embedder no longer matches the
live corpus (a stale floor after a rebuild) and warns rather than abstaining on a
mis-scaled value.

## 5. How to use it

```bash
# 1. Generate cases WITH an abstention slice (needs oa/LLM):
ir eval-gen skills skills_eval.jsonl --abstention-frac 0.15

# 2. Calibrate + persist a floor (use dense or lexical, not hybrid):
ir calibrate-min-score skills skills_eval.jsonl --mode dense --persist

# 3. Discover with absolute abstention turned on:
ir discover skills "bake a sourdough loaf" --mode dense --min-score auto
#   → (abstained: abstain:below_floor; 10 candidates retrieved)
ir info skills        # shows the stored min_score floors per mode
```

```python
import ir
from ir import eval as ev

corpus = ir.open_corpus("skills")
cases = ev.load_cases("skills_eval.jsonl")        # gold-bearing + abstention
calib = ev.calibrate_min_score(corpus, cases, mode="dense", persist=True)
print(calib)                                       # floor, sens/spec/J, distributions

ir.discover(corpus, "bake a sourdough loaf", mode="dense", min_score="auto").abstained
# True  — the calibrated floor catches the out-of-scope query
```

## 6. Open follow-ups (noted, not built)

- **Hybrid abstention via the dense sub-score.** Because RRF discards magnitude,
  the clean fix for abstaining *while ranking hybrid* is to gate abstention on the
  candidates' **dense** scores (which carry magnitude) and only then fuse for
  *ordering*. This is an `ir.retrieve` / `ir.select` enhancement: expose the dense
  sub-score on a hybrid `SearchHit`, or let `discover` consult a dense floor
  before committing a hybrid ranking. (`ir.retrieve._rrf_fuse` also returns *raw
  dense* scores when the lexical list is empty, a second, smaller scale
  inconsistency — both point at the same place.)
- **Per-mode `min_score` in the sweep.** `sweep_selector` already accepts a
  `min_score_grid`; a joint `max_k × rel × min_score` sweep that scores abstention
  and selection together (a single Pareto view) would unify `ir_06` and `ir_07`.
- **A committed, reproducible real fixture.** The numbers here are machine-local
  (live corpora). Freezing a small, embedder-pinned fixture would make a real
  calibration run reproducible in CI (carried as a deferred item in #12).

---

*Harness `ir.eval.calibrate_min_score`; run script (not committed)
`~/Downloads/ir_eval/mincalib_run.py`, raw output
`~/Downloads/ir_eval/mincalib_results.txt`. Re-tune any corpus with
`ir calibrate-min-score <corpus> <cases.jsonl> --persist`.*
