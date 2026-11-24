# ir_12 -- Dependency as a ranking signal -- the real-benchmark run of the #64 experiment

Date: 2026-10-26. Issue: [i2mint/ir#64](https://github.com/i2mint/ir/issues/64). Code: `misc/evals/dep_signal_real.py`. The synthetic companion run is `ir_11`.

## What this is

This is the run that `ir_11` left owed: the arms of #64, scored against the graded gold of the private benchmark (`thorwhalen/ir-eval-data`: `package_relevance_cases.jsonl` and `named_sets_official.json`). This page carries **aggregates only**. The per-package ranks, which name private packages, are in the benchmark repo as `dep_signal_report.md` / `.json`.

**Setup.** The corpus is the 228 labelled packages that have a local checkout (one of the 229 has none, and is dropped from the cases). It is built in memory with the default `Package` strategy, `default_edge_extractor` and `all-MiniLM-L6-v2`. For the three names with two checkouts, the first in manifest order is indexed.

The probes are the benchmark's own theme probes. The seeds and `relevant_libs` are the issue's lists, with `sklearn` written as its distribution name, `scikit-learn`; the caller passes them, as `ir` requires. Rankings run to the full corpus depth. The cross-encoder is ef's default, `cross-encoder/ms-marco-MiniLM-L-6-v2`, and the script fails rather than silently skipping it.

**Metrics.** Recall@20 counts `core` and `strong` as relevant. nDCG@20 uses the benchmark's graded gains. "Of them not seeds" counts the hard positives in the top 20 that are not themselves seed libraries (the walk commits a seed at once).

**This run needed an ir fix first.** ir used to read a package's deps only from `pyproject.toml`, and 50 of the 228 packages declare theirs only in `setup.cfg`. #123 / #124 (an adversarial review of this run found the gap) made the packages source fall back to `setup.cfg`. The numbers below are after that fix.

**Compare arms within this run only.** The baseline is below the frozen June baseline in `baseline_report.json` (nDCG@20 0.80 / 0.74). The two runs index different corpora: the June one was the shared `packages` corpus, this one is the labelled subset with one checkout missing, and it now includes the `setup.cfg` deps and descriptions.

## Results
### embeddings (5 hard positives, 14 distractors)

| arm | nDCG@20 | recall@20 | hard positives in top-20 | of them not seeds | distractor fp_rate@20 | distractors in top-10 | uses-tools in top-20 |
|---|---|---|---|---|---|---|---|
| baseline (hybrid) | 0.77 | 0.42 | 1 | 0 | 0.14 | 0 | 1 |
| + reverse-dep walk (RRF) | 0.89 | 0.52 | 3 | 2 | 0.07 | 0 | 1 |
| + dep_boost w=0.005 | 0.83 | 0.45 | 2 | 1 | 0.07 | 0 | 1 |
| + dep_boost w=0.01 | 0.87 | 0.48 | 3 | 2 | 0.07 | 0 | 1 |
| + dep_boost w=0.02 | 0.86 | 0.48 | 3 | 2 | 0.07 | 0 | 1 |
| + dep_boost w=0.05 | 0.81 | 0.42 | 3 | 2 | 0.07 | 0 | 2 |
| cross-encoder | 0.89 | 0.52 | 3 | 2 | 0.00 | 0 | 1 |
| cross-encoder + dep_boost w=0.01 | 0.83 | 0.45 | 3 | 2 | 0.07 | 0 | 1 |
| cross-encoder + dep_boost w=0.03 | 0.81 | 0.42 | 3 | 2 | 0.07 | 0 | 2 |
| cross-encoder + dep_boost w=0.1 | 0.81 | 0.42 | 3 | 2 | 0.07 | 0 | 2 |

### graphs (6 hard positives, 9 distractors)

| arm | nDCG@20 | recall@20 | hard positives in top-20 | of them not seeds | distractor fp_rate@20 | distractors in top-10 | uses-tools in top-20 |
|---|---|---|---|---|---|---|---|
| baseline (hybrid) | 0.67 | 0.46 | 1 | 1 | 0.11 | 0 | 0 |
| + reverse-dep walk (RRF) | 0.70 | 0.38 | 3 | 3 | 0.11 | 1 | 1 |
| + dep_boost w=0.005 | 0.72 | 0.46 | 1 | 1 | 0.11 | 0 | 0 |
| + dep_boost w=0.01 | 0.82 | 0.58 | 1 | 1 | 0.11 | 0 | 0 |
| + dep_boost w=0.02 | 0.78 | 0.50 | 3 | 3 | 0.11 | 0 | 0 |
| + dep_boost w=0.05 | 0.69 | 0.38 | 3 | 3 | 0.11 | 1 | 1 |
| cross-encoder | 0.63 | 0.38 | 1 | 1 | 0.22 | 0 | 1 |
| cross-encoder + dep_boost w=0.01 | 0.71 | 0.46 | 3 | 3 | 0.11 | 0 | 1 |
| cross-encoder + dep_boost w=0.03 | 0.65 | 0.38 | 3 | 3 | 0.11 | 0 | 1 |
| cross-encoder + dep_boost w=0.1 | 0.65 | 0.38 | 3 | 3 | 0.11 | 0 | 1 |

Fan-in histogram (number of dependents: number of dependency targets): {1: 159, 2: 57, 3: 15, 4: 6, 5: 7, 6: 2, 7: 3, 8: 1, 9: 2, 10: 2, 12: 4, 14: 1, 18: 1, 19: 1, 22: 1, 23: 1, 28: 1, 33: 1, 37: 1, 44: 1, 54: 1, 94: 1}

Seed libraries: of the 18, the most depended-on has 19 dependents in the corpus, and four have none.

## Reading

- **Both mechanisms help, but the effect is smaller than on the synthetic corpus.**
  - Embeddings: the walk (fused by RRF) lifts nDCG@20 from 0.77 to 0.89 and recall@20 from 0.42 to 0.52. The boost peaks at 0.87 (`w=0.01`).
  - Graphs: the boost at `w=0.02` lifts nDCG@20 from 0.67 to 0.78 and finds 3 hard positives instead of 1. `w=0.01` gives a higher nDCG (0.82) but finds no extra hard positive. The walk gives 0.70, but it lowers recall@20 (0.46 to 0.38) and brings a distractor into the top 10.
- **The dependency signal finds every hard positive it can reach.** A hard positive is reachable only if it depends on a seed library. Six of the eleven do: three per theme, and one of them is itself a seed. In both themes, the walk and the `w>=0.01` boost (`w>=0.02` for graphs) put all six in the top 20. The other five all declare dependencies, but on none of the seed libraries (a vector-DB client, for example).
- **So the issue's hard-positive target (at least 6 of its 8 named in the top 20) cannot be met with its seed lists.** Only 4 of those 8 depend on a seed library, and all 4 reach the top 20. Raising the ceiling means choosing better seeds. That choice belongs to the caller, in the design raglab's Planner (`thorwhalen/raglab#2`), not ir.
- **The guardrail holds on the rate, but not on the issue's premise.** The distractor FP rate@20 never rises above the baseline: embeddings falls from 0.14 to 0.07, graphs stays at 0.11. However, the issue assumed distractors "don't depend on seed libs", and that is not true of two of the 23 named distractors: an audio package that depends on `torch` (one of the issue's seven guardrail distractors), and a graphs distractor that depends on `meshed`. The dependency arms lift both of them. On graphs the walk brings that distractor into the top 10, and the boost (rank 11 at `w=0.02`) nearly does too. On embeddings no distractor enters the top 10 in any arm. A dependency is evidence of use, not of topic, and seed lists built from general-purpose libraries (`torch`, `scikit-learn`) carry that cost.
- **The cross-encoder comparison is split by theme.**
  - Graphs: the cross-encoder alone is the worst arm (nDCG 0.63, FP rate 0.22): prose like "DAG" fools it. Adding `dep_boost` on its scale (`w=0.01`) recovers part of the loss (0.71, FP rate 0.11, 3 hard positives instead of 1), so there the two signals are complementary.
  - Embeddings: the cross-encoder alone is already as good as the walk (0.89, FP rate 0.00), and every boost weight makes it worse (0.81-0.83).
  - Larger weights saturate quickly on the cross-encoder's scale: 0.03 and 0.1 give identical metrics on embeddings and near-identical ones on graphs.
- **The weight is a dial, as in `ir_11`.** On the hybrid (RRF) scale the useful range is 0.01-0.02. At 0.05 both themes fall back (0.81 / 0.69), because any package with a seed dependency then outranks a better prose match. That supports `dependency_evidence_boost` having no default weight.
- **The fan-in prior is not degenerate, but it is thin.** Most dependency targets have one or two dependents. The seed libraries have between 0 and 19.

## What this settles, and what it leaves

The experiment of #64 has now run on the real labels. Both mechanisms improve ir's single-shot search when the caller supplies seeds. Neither raises the distractor rate, and together they reach every hard positive that the seeds make reachable. What limits the effect is seed choice: seeds that are too narrow miss hard positives, and seeds that are too general promote distractors. That choice is a planning concern, outside ir.

Two pieces of follow-up work are tracked as their own issues: the harness gap (`ir.eval.compare_indexings` cannot score a fused walk arm, so this script scores the arms itself), #121; and a reusable project-metadata reader, #125.

## Addendum, 2026-11-23: the cross-encoder A/B/C of #104

The same benchmark, corpus and probes, with ef's cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a precision stage at `fetch_k=50`: **A** hybrid, **B** A reranked on the matched surface, **C** A reranked on the whole artifact (`rerank_text="artifact"`: description, README chunks and deps joined; the model reads its first 512 tokens). Script: `misc/evals/rerank_abc.py`. The success rule was set in #104 before the run: C cuts the named-distractor FP-rate@20 by at least 40% against A, with hard-positive recall@20 at least A's.

### embeddings (5 hard positives, 14 distractors)

| arm | nDCG@20 | P@10 | distractor FP-rate@20 | hard-positive recall@20 |
|---|---|---|---|---|
| A hybrid | 0.751 | 0.90 | 0.14 | 0.20 |
| B + cross-encoder (surface) | 0.775 | 1.00 | 0.14 | 0.20 |
| C + cross-encoder (artifact) | 0.803 | 1.00 | 0.14 | 0.20 |

FP-rate cut, C vs A: 0%; success rule met: **no**

### graphs (6 hard positives, 9 distractors)

| arm | nDCG@20 | P@10 | distractor FP-rate@20 | hard-positive recall@20 |
|---|---|---|---|---|
| A hybrid | 0.681 | 0.70 | 0.33 | 0.17 |
| B + cross-encoder (surface) | 0.642 | 0.70 | 0.33 | 0.17 |
| C + cross-encoder (artifact) | 0.646 | 0.70 | 0.33 | 0.17 |

FP-rate cut, C vs A: 0%; success rule met: **no**

**The rule is not met on either theme.** Reranking does not move a single named distractor out of the top 20 (FP-rate cut 0%), and the hard positives the hybrid misses stay missed: a reranker can only reorder the 50 candidates it is shown, and these hard positives are not among them (they are the thin-description packages of ir_11 and the dependency-only ones of #64, whose fix is the dependency signal above, not precision). What it does change is the order inside the top 20: on embeddings, reading the whole artifact lifts nDCG@20 by 0.05 and puts only relevant packages in the top 10; on graphs it lowers nDCG@20 by 0.035.

Latency was timed too, but the machine was loaded by other jobs and the numbers were not coherent (the reranked arms ran faster than the plain one on one theme), so none are reported. The earlier measure from #103 stands: the cross-encoder adds about one forward pass over `fetch_k` passages per query.

So `rerank="cross-encoder"` stays opt-in: it is a top-10 ordering stage for text-rich themes, not the distractor filter #65 hoped for. No floor is calibrated for it on the Package corpus, since it does not become the default.

## Addendum, 2026-11-23: the deps-surface weight sweep of #68

Two in-memory corpora over the same labelled packages, identical except `Package(embed_deps=...)`; on the one with a `deps` surface, hybrid search with `surface_weights={"deps": w}`, ranked 100 deep. Script: `misc/evals/deps_weight_sweep.py`. #68's rule: the `w` that maximizes hard-positive recall@20 without raising the distractor FP-rate@20 above the no-deps baseline.

### embeddings (5 hard positives, 14 distractors)

| arm | nDCG@20 | recall@20 | hard-positive recall@20 | distractor FP-rate@20 |
|---|---|---|---|---|
| no deps surface | 0.841 | 0.45 | 0.40 | 0.07 |
| deps, w=1.0 | 0.840 | 0.45 | 0.40 | 0.07 |
| deps, w=1.5 | 0.866 | 0.48 | 0.40 | 0.07 |
| deps, w=2.0 | 0.845 | 0.45 | 0.40 | 0.07 |
| deps, w=3.0 | 0.715 | 0.45 | 0.40 | 0.00 |

Best by #68's rule: **deps, w=1.0**

### graphs (6 hard positives, 9 distractors)

| arm | nDCG@20 | recall@20 | hard-positive recall@20 | distractor FP-rate@20 |
|---|---|---|---|---|
| no deps surface | 0.709 | 0.46 | 0.17 | 0.33 |
| deps, w=1.0 | 0.723 | 0.50 | 0.17 | 0.22 |
| deps, w=1.5 | 0.777 | 0.58 | 0.33 | 0.11 |
| deps, w=2.0 | 0.777 | 0.54 | 0.50 | 0.11 |
| deps, w=3.0 | 0.706 | 0.50 | 0.67 | 0.11 |

Best by #68's rule: **deps, w=3.0**

**Per theme the rule picks different weights**: on embeddings the deps surface never moves the hard positives (so the rule falls to the first tie, w=1), and on graphs recall climbs with the weight, to 0.67 at w=3. One weight has to serve both. **w=1.5 and w=2.0 are each at least as good as the no-deps baseline on every metric of both themes**, and w=2.0 is chosen for its higher graphs hard-positive recall: embeddings flat (nDCG 0.845 against 0.841), graphs hard-positive recall 0.17 to 0.50, distractor FP-rate 0.33 to 0.11, nDCG +0.07. w=3 raises graphs hard-positive recall to 0.67 (4 of the 6) at a cost of 0.13 of embeddings nDCG.

**Decision: no default changes here.** The evidence is a clean dominance but on 11 hard positives and 23 distractors across two themes, and the weight is a search-level argument: making `{"deps": 2.0}` the global default would put every corpus's scores on a weighted scale and skip its calibrated `min_score="auto"` floor. The recommendation for package corpora is `Package(embed_deps=True)` with `surface_weights={"deps": 2.0}`, and the way to ship it as a default is per-corpus search settings with their own floor: https://github.com/i2mint/ir/issues/183.

## Addendum, 2026-11-23: conditional synopsis on the real benchmark (#99)

Three corpora over the same labelled packages, identical except strategy: **a** `Package()`, **b** `with_synopsis(Package())`, **c** `with_synopsis(Package(), when=make_thin_description_predicate(min_chars=m))` for m in 20, 40 (the default), 80, 160. Synopses from `TOPICAL_SYNOPSIS_PROMPT` through `aix`'s `claude-cli/haiku` backend (the Claude subscription), computed once and shared by b and c through a counting synthesizer, so the arms differ only in which packages get one. Script: `misc/evals/synopsis_conditional.py`. "Thin" in the metrics is the benchmark's own `thin_description` label on packages labelled core or strong; the nDCG is over the well-described packages only.

### embeddings (16 thin-description relevant packages)

| arm | thin recall@10 | thin mean rank | nDCG@20 (well-described) | LLM calls (% of b) |
|---|---|---|---|---|
| a Package() | 0.19 | 39.9 | 0.696 | 0 (0%) |
| b synopsis for all | 0.31 | 27.8 | 0.711 | 228 (100%) |
| c synopsis when thin (min_chars=20) | 0.19 | 29.2 | 0.697 | 17 (7%) |
| c synopsis when thin (min_chars=40) | 0.25 | 29.1 | 0.695 | 100 (44%) |
| c synopsis when thin (min_chars=80) | 0.31 | 27.2 | 0.728 | 192 (84%) |
| c synopsis when thin (min_chars=160) | 0.31 | 27.8 | 0.711 | 226 (99%) |

### graphs (11 thin-description relevant packages)

| arm | thin recall@10 | thin mean rank | nDCG@20 (well-described) | LLM calls (% of b) |
|---|---|---|---|---|
| a Package() | 0.18 | 49.5 | 0.578 | 0 (0%) |
| b synopsis for all | 0.27 | 31.8 | 0.673 | 228 (100%) |
| c synopsis when thin (min_chars=20) | 0.18 | 50.4 | 0.578 | 17 (7%) |
| c synopsis when thin (min_chars=40) | 0.18 | 36.9 | 0.665 | 100 (44%) |
| c synopsis when thin (min_chars=80) | 0.18 | 32.9 | 0.671 | 192 (84%) |
| c synopsis when thin (min_chars=160) | 0.27 | 31.8 | 0.673 | 226 (99%) |

**A synopsis for every package helps, on both themes and beyond the thin packages**: thin-package recall@10 rises 0.19 to 0.31 (embeddings) and 0.18 to 0.27 (graphs), their mean rank improves by 12 and 18 places, and nDCG@20 on the *well-described* packages rises too (+0.015 and +0.095).

**The conditional gate does not meet #99's target.** The default `min_chars=40` already synthesizes for 44% of the packages, not 15% or fewer; the gate that does stay under 15% (`min_chars=20`, 7%) buys nothing on either theme. Description length is not what picks out the packages a synopsis helps: the benchmark's thin-description packages include many whose pyproject description is short but not empty, and the gain on well-described packages says the synopsis adds a topical summary even where the description is fine.

**Decision**: no default changes. `min_chars` stays 40 (no value in the sweep saves calls without losing most of the gain). When a corpus can afford one LLM call per artifact (228 here, on a subscription), `with_synopsis` on everything is the better configuration; `when=` remains for corpora where it cannot, with this caveat.

