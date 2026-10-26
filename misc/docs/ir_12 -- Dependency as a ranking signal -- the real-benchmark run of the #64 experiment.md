# ir_12 -- Dependency as a ranking signal -- the real-benchmark run of the #64 experiment

Date: 2026-10-26. Issue: [i2mint/ir#64](https://github.com/i2mint/ir/issues/64). Code: `misc/evals/dep_signal_real.py`. The synthetic companion run is `ir_11`.

## What this is

The run that `ir_11` left owed: the arms of #64 scored against the graded gold of the private benchmark (`thorwhalen/ir-eval-data`: `package_relevance_cases.jsonl` and `named_sets_official.json`). This page carries **aggregates only**. The per-package ranks (which name private packages) are in the benchmark repo as `dep_signal_report.json`.

Setup: the 228 labelled packages that have a local checkout (one of the 229 has none), read with the default `Package` strategy, `default_edge_extractor` and `all-MiniLM-L6-v2`, built in memory. The theme probes are the benchmark's own. The seeds and `relevant_libs` are the issue's lists (with `sklearn` written as its distribution name, `scikit-learn`), passed by the caller as `ir` requires. Rankings run to the full corpus depth. The cross-encoder is ef's default (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Recall@20 counts `core` and `strong` as relevant. nDCG@20 uses the benchmark's graded gains.

The baseline here is a little below the frozen June baseline (`baseline_report.json`: nDCG@20 0.80 / 0.74), because the READMEs have changed since then. Every arm below is compared with the baseline of this run, on the same corpus.

## Results

### embeddings (5 hard positives, 14 distractors)

| arm | nDCG@20 | recall@20 | hard positives in top-20 | distractor fp_rate@20 | distractors in top-10 | uses-tools in top-20 |
|---|---|---|---|---|---|---|
| baseline (hybrid) | 0.77 | 0.42 | 1 | 0.14 | 0 | 1 |
| + reverse-dep walk (RRF) | 0.88 | 0.48 | 2 | 0.07 | 1 | 2 |
| + dep_boost w=0.005 | 0.83 | 0.45 | 1 | 0.07 | 0 | 1 |
| + dep_boost w=0.01 | 0.85 | 0.48 | 2 | 0.14 | 0 | 0 |
| + dep_boost w=0.02 | 0.90 | 0.52 | 2 | 0.07 | 1 | 1 |
| + dep_boost w=0.05 | 0.90 | 0.52 | 2 | 0.07 | 1 | 1 |
| cross-encoder | 0.89 | 0.52 | 3 | 0.00 | 0 | 1 |
| cross-encoder + dep_boost w=0.1 | 0.89 | 0.52 | 3 | 0.07 | 1 | 2 |
| cross-encoder + dep_boost w=0.3 | 0.90 | 0.52 | 3 | 0.07 | 1 | 2 |

### graphs (6 hard positives, 9 distractors)

| arm | nDCG@20 | recall@20 | hard positives in top-20 | distractor fp_rate@20 | distractors in top-10 | uses-tools in top-20 |
|---|---|---|---|---|---|---|
| baseline (hybrid) | 0.67 | 0.46 | 1 | 0.11 | 0 | 0 |
| + reverse-dep walk (RRF) | 0.73 | 0.42 | 3 | 0.00 | 0 | 1 |
| + dep_boost w=0.005 | 0.75 | 0.50 | 1 | 0.11 | 0 | 0 |
| + dep_boost w=0.01 | 0.82 | 0.58 | 1 | 0.00 | 0 | 0 |
| + dep_boost w=0.02 | 0.80 | 0.54 | 3 | 0.00 | 0 | 0 |
| + dep_boost w=0.05 | 0.73 | 0.46 | 3 | 0.00 | 0 | 1 |
| cross-encoder | 0.63 | 0.38 | 1 | 0.22 | 0 | 1 |
| cross-encoder + dep_boost w=0.1 | 0.71 | 0.46 | 3 | 0.00 | 0 | 1 |
| cross-encoder + dep_boost w=0.3 | 0.74 | 0.46 | 3 | 0.00 | 0 | 1 |

Fan-in histogram (number of dependents: number of dependency targets): {1: 147, 2: 40, 3: 8, 4: 9, 5: 3, 6: 3, 7: 2, 8: 2, 9: 1, 10: 1, 12: 2, 21: 2, 22: 2, 26: 1, 29: 1, 35: 1, 73: 1}. Of the 18 seed libraries, the most depended-on has 10 dependents in the corpus. Four have none.

## Reading

- **Both mechanisms help on real data, and neither breaks the guardrail.** At `w=0.02` the boost lifts nDCG@20 from 0.77 to 0.90 (embeddings) and from 0.67 to 0.80 (graphs), and recall@20 from 0.42 to 0.52 and from 0.46 to 0.54. The walk, fused by RRF, gives 0.88 and 0.73. The distractor FP rate falls or stays the same in every dependency arm (embeddings 0.14 → 0.07, graphs 0.11 → 0.00).
- **The dependency signal finds every hard positive it can reach.** A hard positive is reachable only if it depends on a seed library. Five of the eleven do (two for embeddings, three for graphs), and the walk and the `w=0.02` boost put all five in the top 20. The other six depend on no seed library: three declare no dependencies at all, and three depend only on libraries the seed lists do not name (a vector-DB client, for example). No dependency signal can reach those.
- **So the issue's hard-positive target (at least 6 of its 8 named in the top 20) cannot be met with its seed lists.** Only 4 of those 8 depend on a seed library, and all 4 make the top 20. Raising the ceiling means choosing better seeds (adding the vector-DB clients to the embeddings seeds, for example). That choice belongs to the caller, which in the design is raglab's Planner (`thorwhalen/raglab#2`), not ir.
- **The dependency signal complements the cross-encoder, as the issue expected.** On graphs the cross-encoder alone is the worst arm (nDCG 0.63, FP rate 0.22, up from 0.11): it is fooled by prose like "DAG". Adding `dep_boost` on the cross-encoder's scale recovers it (0.74, FP rate 0.00) and finds 3 hard positives instead of 1. On embeddings the cross-encoder is strong on its own (3 hard positives, FP rate 0.00). There the boost adds nothing to nDCG and lets one distractor back into the top 10.
- **The weight is a dial, as in `ir_11`.** On the hybrid (RRF) scale, 0.01-0.02 is the useful range. At 0.05 graphs drops back (nDCG 0.73): the boost now outweighs the text score, so any package with a seed dependency outranks a better prose match. This supports `dependency_evidence_boost` having no default weight.
- **The fan-in prior is not degenerate, but it is thin.** Most dependency targets have one dependent. The seed libraries have between 0 and 10. A centrality prior over this corpus would mostly separate the few widely shared internal libraries from everything else.

## What this settles, and what it leaves

The experiment of #64 has now run on the real labels. Both mechanisms improve ir's single-shot search when the caller supplies seeds. The remaining gap is seed coverage, which is a planning concern and outside ir. The harness gap this run exposed (`ir.eval.compare_indexings` can compare search settings but not a fused walk arm, so this script scores the arms itself) is tracked as its own issue.
