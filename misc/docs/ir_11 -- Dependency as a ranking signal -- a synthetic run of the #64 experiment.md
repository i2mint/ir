# ir_11 -- Dependency as a ranking signal -- a synthetic run of the #64 experiment

Date: 2026-10-20. Issue: [i2mint/ir#64](https://github.com/i2mint/ir/issues/64). Code: `misc/evals/dep_signal_synthetic.py`; pinned by `tests/test_dep_signal_synthetic_eval.py`.

## What this is, and what it is not

The two mechanisms of #64 (the reverse-dependency walk `ir.reverse_dependency_policy` and the ranking-time `search(..., dep_boost=ir.dependency_evidence_boost(...))`) landed in #108. The experiment the issue specifies needs the private graded labels of `thorwhalen/ir-eval-data`, which are not on the machine this ran on. So this run uses a **synthetic** package corpus with known relevance, built to the failure modes the issue describes. It checks that the mechanisms do what they claim and shows how the boost weight trades recall against false positives. It does **not** measure the size of the effect on the real fleet; that run is still owed (the issue keeps its `needs-local` label).

## The corpus

Per theme (graphs, embeddings), generated from a seed: 10 *prose positives* (theme words in the description, theme libraries in `deps`, relevance 2), 8 *hard positives* (theme libraries in `deps`, prose drawn only from off-theme words, relevance 1: the "uses-tools" false negatives of #61), 8 *distractors* (theme words, unrelated deps: the `au` case), 8 *incidental users* (off-theme prose and relevance 0, but one theme library in `deps`, like `torch` in an audio package: the false positives a dependency signal creates), plus 150 background packages. Light embedder, in-memory store, `mode="hybrid"`. The seeds / `relevant_libs` are the theme's four libraries, passed by the caller as `ir` requires. Means over 5 corpora (seeds 0-4).

## Results

### graphs

| arm | nDCG@20 | recall@20 | hard in top-20 | distractors in top-10 | incidental in top-20 |
|---|---|---|---|---|---|
| baseline (hybrid) | 0.57 | 0.34 | 0.00 | 3.80 | 0.60 |
| + reverse-dep walk (RRF) | 0.72 | 0.58 | 3.00 | 0.80 | 2.60 |
| + dep_boost w=0.005 | 0.62 | 0.39 | 0.80 | 3.40 | 2.40 |
| + dep_boost w=0.01 | 0.68 | 0.51 | 2.40 | 3.20 | 5.20 |
| + dep_boost w=0.02 | 0.72 | 0.59 | 3.60 | 1.00 | 7.20 |
| + dep_boost w=0.05 | 0.77 | 0.69 | 5.20 | 0.00 | 7.60 |

### embeddings

| arm | nDCG@20 | recall@20 | hard in top-20 | distractors in top-10 | incidental in top-20 |
|---|---|---|---|---|---|
| baseline (hybrid) | 0.61 | 0.44 | 0.00 | 3.80 | 0.20 |
| + reverse-dep walk (RRF) | 0.87 | 0.72 | 3.20 | 0.00 | 1.80 |
| + dep_boost w=0.005 | 0.82 | 0.66 | 2.60 | 2.20 | 0.80 |
| + dep_boost w=0.01 | 0.87 | 0.76 | 4.00 | 2.00 | 1.00 |
| + dep_boost w=0.02 | 0.94 | 0.89 | 6.00 | 1.00 | 1.00 |
| + dep_boost w=0.05 | 0.97 | 0.93 | 6.80 | 0.00 | 3.20 |

Fan-in of the first corpus (dependents: how many dependencies): {5: 1, 7: 1, 9: 2, 10: 2, 11: 2, 50: 1, 52: 1, 56: 1, 57: 1, 58: 2, 59: 1, 61: 2, 70: 1}

## Reading

- The flat hybrid search finds **no** hard positive in the top 20 (by construction: their prose never names the theme), and lets about 4 prose look-alikes into the top 10.
- The **reverse-dependency walk**, fused by RRF, lifts nDCG@20 by 0.14-0.26 and finds about 3 of the 8 hard positives, while pushing the look-alikes out of the top 10 (3.8 → 0.8 and 3.8 → 0).
- The **boost** is monotone in its weight: more hard positives and fewer look-alikes, but also more incidental users. At w=0.02 it finds 3.6 (graphs) and 6.0 (embeddings) of the 8 hard positives, and also brings 7.2 of the 8 incidental users into the graphs top 20 (1.0 for embeddings). The weight is a dial between "uses the library" and "is about the topic", which is why `dependency_evidence_boost` has no default weight; the walk, fused by rank, admits fewer incidental users at a similar gain.
- The fan-in prior is not degenerate: the theme libraries have 5-11 dependents, the common third-party ones 50-70.

## What remains

The real-data run of the issue (nDCG@20 / recall@20 per theme against the all-MiniLM hybrid baseline, the 8 named hard positives, the named distractor guardrail, an `ef.Reranker` comparison arm) on the private labels, from a machine that has them.
