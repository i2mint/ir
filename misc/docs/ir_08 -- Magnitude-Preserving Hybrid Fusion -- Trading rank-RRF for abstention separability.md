# ir_08 — Magnitude-Preserving Hybrid Fusion

**Trading rank-based RRF for abstention separability (and when it's worth it)**

*2026-06-11. Findings from a real-corpora eval comparing the default Reciprocal
Rank Fusion (RRF) against a new magnitude-preserving "blend" fusion for hybrid
retrieval. Companion to [`ir_07`](ir_07%20--%20Min-Score%20Calibration%20--%20Abstention%20floors%20from%20score-distribution%20separability.md)
(abstention calibration) and issues [#30](https://github.com/i2mint/ir/issues/30),
[#28](https://github.com/i2mint/ir/issues/28).*

---

## 1. The problem: RRF discards the magnitude abstention needs

`ir_07` established that **abstention** ("nothing in this corpus applies") is a
1-D separability problem: in-scope queries should produce *high* top scores,
out-of-scope queries *low* ones, and the calibrated floor is whatever best
separates the two distributions. It also flagged that **hybrid was the worst
mode** for this (Youden's J ≈ 0.34–0.61, vs 0.65–0.88 for dense).

The cause is structural. Hybrid fuses dense + BM25 with **Reciprocal Rank
Fusion**: `score = Σ 1/(k + rank)`. RRF is purely *rank-based* — it throws away
the score magnitudes it fuses. A query whose top hit is overwhelmingly relevant
(cosine 0.9) and one whose top hit is barely relevant (cosine 0.2) both produce
the *same* fused top score ≈ `2/(60+1) ≈ 0.0328`, because both hits are rank 1.
The absolute signal that distinguishes in-scope from out-of-scope is gone.

This is also why the much larger **`vd.Collection` storage refactor was deferred
([#28](https://github.com/i2mint/ir/issues/28))**: `vd.hybrid_search`'s fusion
fallback uses the *same* rank-based RRF, so unifying storage behind it would
reproduce the identical weakness. The fix is not a storage change — it is a
fusion change, and it lives in `ir.retrieve` alone.

## 2. The fix: a magnitude-preserving convex blend

`ir.retrieve.search(mode="hybrid", fusion="blend")` fuses by

```
fused(doc) = alpha · cosine(doc) + (1 − alpha) · bm25(doc) / (bm25(doc) + k_sat)
```

with defaults `alpha = 0.5`, `k_sat = 8.0`. Two design choices matter:

1. **Keep the dense cosine raw.** Cosine is already on a fixed `[−1, 1]` scale,
   so its magnitude *is* the in-scope/out-of-scope signal. An irrelevant query
   scores low across the board; the blend preserves that.
2. **Squash BM25 with a *fixed* constant, not per-query min-max.** BM25 has no
   fixed upper bound, so it must be scaled to be commensurable with cosine. The
   tempting choice — min-max normalize per query — is exactly wrong for
   abstention: it rescales *every* query's best hit to 1.0, re-introducing the
   same magnitude loss RRF has. A fixed saturating squash `b/(b+k_sat)` keeps a
   no-lexical-match query's contribution near zero.

The blend degrades exactly like RRF when `vd` is unavailable (falls back to the
dense ranking). Default fusion stays `"rrf"` — this is purely opt-in.

## 3. Method

Three real corpora, each with an LLM-generated case set carrying gold-bearing
*and* abstention (empty-gold) cases:

| Corpus | artifacts | gold cases | abstention cases |
|--------|-----------|-----------|------------------|
| skills | ~157 | 662–695 | 137 |
| packages | ~231 | 313–431 | 103 |
| reports | ~98 | 386–451 | 105 |

Embedder: `all-MiniLM-L6-v2` (the real default, not the test hashing embedder).
For each `(corpus, config)` we run `ir.eval.calibrate_min_score` (abstention) and
`ir.eval.evaluate_discovery` (retrieval quality). Abstention is scored by
**Youden's J** = sensitivity + specificity − 1 (balanced separability of the
calibrated floor); retrieval by NDCG@10 / recall@k / MRR over the gold cases.

## 4. Results

### 4.1 Abstention separability (Youden's J — higher is better)

| Corpus | dense | hybrid/rrf | **hybrid/blend** (α=0.5,k=8) | blend α=0.7,k=8 | blend α=0.7,k=4 |
|--------|------:|-----------:|-----------:|-----------:|-----------:|
| skills | 0.764 | 0.610 | **0.840** | 0.831 | 0.839 |
| packages | 0.651 | 0.342 | **0.798** | 0.749 | 0.742 |
| reports | 0.883 | 0.453 | **0.928** | 0.889 | 0.867 |

**Blend recovers everything RRF lost and then some — it beats even dense in all
three corpora.** RRF is the worst mode everywhere (the ir_07 finding,
reconfirmed). The default `α=0.5, k_sat=8` was the best blend setting tested.

### 4.2 Retrieval quality (NDCG@10 / recall@10 — higher is better)

| Corpus | metric | dense | hybrid/rrf | hybrid/blend |
|--------|--------|------:|-----------:|-----------:|
| skills | NDCG@10 | 0.659 | 0.694 | **0.713** |
| skills | recall@10 | 0.853 | 0.880 | **0.885** |
| packages | NDCG@10 | **0.583** | 0.489 | 0.437 |
| packages | recall@10 | **0.740** | 0.703 | 0.538 |
| reports | NDCG@10 | 0.500 | **0.537** | 0.494 |
| reports | recall@10 | 0.730 | **0.745** | 0.643 |

**Here the result is a genuine tradeoff.** Blend wins retrieval on the text-rich
`skills` corpus but *loses recall* on `packages` and `reports` — sharply so at
recall@10 (packages 0.538 vs 0.740 dense). The mechanism is the mirror image of
its abstention strength: by weighting the dense *magnitude*, blend does not lift
a gold doc that only BM25 found (low cosine) into the top-10 the way RRF's
rank-based boost does. RRF trades abstention for lexical recall; blend trades
lexical recall for abstention. (Unit test `test_blend_does_not_push_rare_
identifier_below_dense` pins this behavior.)

## 5. Decision

**RRF stays the default hybrid fusion. Blend ships as opt-in.**

- Changing the default would be backward-incompatible (it shifts the hybrid
  score scale, invalidating any persisted hybrid calibration) *and* the data
  does not support a universal switch — blend hurts recall on terse,
  identifier-heavy corpora (packages) where lexical-only matches carry weight.
- Blend is the right choice **when abstention matters** — when the agent must
  reliably answer "nothing here applies" rather than commit to a weak top hit —
  especially on text-rich corpora (skills, reports prose). This updates ir_07's
  guidance "don't calibrate hybrid": with `fusion="blend"`, hybrid abstention
  calibrates *better than dense*.
- `alpha` is the recall↔abstention dial: lower it toward lexical to recover
  recall, raise it toward dense for sharper abstention. `0.5` balances.

## 6. Practical guidance

```python
# Recall-first hybrid (default): RRF lifts lexical-only matches best.
ir.search(corpus, q, mode="hybrid")  # fusion="rrf"

# Abstention-first hybrid: blend keeps the magnitude calibration needs.
ev.calibrate_min_score(corpus, cases, mode="hybrid", fusion="blend", persist=True)
ir.discover(corpus, q, mode="hybrid", fusion="blend", min_score="auto")
```

A persisted calibration is per-`(corpus, mode)`; calibrate under the *same*
fusion you will query with, since the two fusions live on different score
scales.

## 7. Open threads

- **Per-corpus default fusion.** A registry could remember the better fusion per
  corpus (skills→blend, packages→rrf) the way calibration is already persisted.
- **`alpha` auto-tuning.** `sweep_selector`-style grid over `alpha` against a
  case set, optimizing a chosen blend of recall and abstention-J.
- Both are deferred until there is a concrete consumer.

---

## REFERENCES

1. [`ir_07` — Min-Score Calibration](ir_07%20--%20Min-Score%20Calibration%20--%20Abstention%20floors%20from%20score-distribution%20separability.md): abstention as 1-D separability; the original hybrid-is-worst finding.
2. [`ir_04` — Architecture & Reuse Analysis](ir_04%20--%20Architecture%20%26%20Reuse%20Analysis%20--%20Building%20ir%20on%20the%20ef%20%2B%20vd%20Substrate.md): the `vd` reuse stance.
3. ir issue [#28](https://github.com/i2mint/ir/issues/28): why the `vd.Collection` storage refactor was deferred (RRF magnitude loss survives it).
4. ir issue [#30](https://github.com/i2mint/ir/issues/30): this work item.
5. Cormack, Clarke & Büttcher (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR — the rank-based fusion whose magnitude loss this doc addresses.
