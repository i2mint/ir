# ir.eval

Capability-discovery evaluation — measuring how well a corpus is *found*.

`ir.eval` answers the question that the rest of `ir` raises: \*does retrieval
actually surface the right capability?\* It is a thin, deterministic scoring layer
over the substrate already in place — `ir.retrieve.search` for ranking and
`ef.evaluation` for the metric mathematics — plus the genuinely new pieces a
*capability-discovery* eval needs that generic retrieval evaluation does not.

## Design stance (retrieval-centric, reuse `ef`)

`ir`’s corpora are **documents** — skills, packages, reports — each a
name + description (+ body), joined on a stable `artifact_id`. They are *not*
a typed function-calling registry with argument signatures. So the eval that
fits is **retrieval of the right artifact**, not argument-dict scoring:

- **recall@k / NDCG@k / MRR / MAP** — computed by `ef.evaluation`’s pure,
  tested primitives (`ef.evaluation.recall_at_k()`, … ) and driver
  (`ef.evaluation.evaluate_retrieval()`). `ir.eval` does not reimplement
  them; it adapts an `ir` corpus to `ef`’s retriever contract and reads back
  a `ef.evaluation.RetrievalEvalReport`.
- **distractor-robustness curve** — retrieval accuracy as the catalog grows
  (1 gold + N−1 distractors). The single most diagnostic capability-discovery
  metric, and one `ef` does not provide.
- **failure-mode taxonomy** — every gold case is classed `hit_rank_1` /
  `surfaced_low_rank` / `retrieval_miss` so a headline number decomposes
  into *why* it is what it is.
- **abstention** — an optional score-threshold proxy for “no artifact applies”
  cases. True abstention is a *selection* concern (a selector commits or
  refuses); here it is a retrieval-side diagnostic, clearly labelled as such.

The unit of evaluation is a [`DiscoveryCase`](#ir.eval.DiscoveryCase): an intent (`query`) and
the `gold` `artifact_id`s that should answer it (empty `gold` = an
abstention case). Cases are plain data — [`save_cases()`](#ir.eval.save_cases) / [`load_cases()`](#ir.eval.load_cases)
round-trip them as JSONL so an evaluation set is a committable, reproducible
fixture (the corpora themselves are machine-specific and live, so freezing the
cases is what makes a run repeatable; [`validate_cases()`](#ir.eval.validate_cases) flags gold ids that
have since drifted out of the corpus).

Case *generation* (back-translation from a corpus with name-masking) needs an
LLM and is deliberately **out of scope here** — this module scores a given set
of cases offline, with no network and no model download. (Dense scoring needs
only numpy; `lexical` / `hybrid` ranking additionally need the optional
`vd` dependency — a hybrid run that silently fell back to dense because `vd`
was absent is flagged in the report.)

Quick start:

```default
import ir
from ir import eval as ev

corpus = ir.build(ir.CorpusSource.from_skills())     # or ir.open_corpus("skills")
cases = ev.load_cases("skills_eval.jsonl")
report = ev.evaluate_discovery(corpus, cases, mode="hybrid")
print(report)                                        # NDCG@10 + taxonomy

# Just the BEIR-shaped metrics (pure ef), e.g. to A/B dense vs hybrid:
ev.retrieval_report(corpus, cases, mode="dense").primary    # mean NDCG@10
ev.retrieval_report(corpus, cases, mode="hybrid").primary
```

### Module Attributes

| [`DFLT_K_VALUES`](#ir.eval.DFLT_K_VALUES)                 | Rank cutoffs reported by default.                                                                                                                                                                                                                      |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_METRICS`](#ir.eval.DFLT_METRICS)                  | Metrics reported by default — the `ef.evaluation` retrieval metric names.                                                                                                                                                                              |
| [`DFLT_MODE`](#ir.eval.DFLT_MODE)                     | Default ranking mode for the eval adapter (hybrid is `ir`'s strongest).                                                                                                                                                                                |
| [`DFLT_DISTRACTOR_SIZES`](#ir.eval.DFLT_DISTRACTOR_SIZES)         | Catalog sizes swept by [`distractor_robustness_curve()`](#ir.eval.distractor_robustness_curve) (RAG-MCP-style).                                                                                                                                 |
| [`DFLT_CURVE_MODE`](#ir.eval.DFLT_CURVE_MODE)               | Defaults for [`distractor_robustness_curve()`](#ir.eval.distractor_robustness_curve) — deliberately *not* `DFLT_MODE`: the curve builds throwaway in-memory sub-corpora, so it favours the fast, always-offline dense + hashing path.           |
| [`DFLT_SWEEP_MAX_K`](#ir.eval.DFLT_SWEEP_MAX_K)              | `max_k` values swept by [`sweep_selector()`](#ir.eval.sweep_selector) (the commit-size cap).                                                                                                                                       |
| [`DFLT_SWEEP_REL`](#ir.eval.DFLT_SWEEP_REL)                | `rel` values swept by [`sweep_selector()`](#ir.eval.sweep_selector) (relative-to-top keep band).                                                                                                                                   |
| [`DFLT_SWEEP_MIN_SCORE`](#ir.eval.DFLT_SWEEP_MIN_SCORE)          | `min_score` floors swept by [`sweep_selector()`](#ir.eval.sweep_selector).                                                                                                                                                         |
| [`DFLT_SWEEP_OBJECTIVE`](#ir.eval.DFLT_SWEEP_OBJECTIVE)          | Default objective [`SelectionSweep.best()`](#ir.eval.SelectionSweep.best) optimizes — the standard precision/recall balance.                                                                                                            |
| [`SWEEP_METRICS`](#ir.eval.SWEEP_METRICS)                 | The scalar selection metrics a sweep can optimize / rank by.                                                                                                                                                                                           |
| [`DFLT_CALIB_SENSITIVITY_WEIGHT`](#ir.eval.DFLT_CALIB_SENSITIVITY_WEIGHT) | Weight on *sensitivity* (catch true positives) vs *specificity* (reject out-of-scope queries) when [`calibrate_min_score()`](#ir.eval.calibrate_min_score) picks the abstention floor: `objective = w·sensitivity + (1−w)·specificity`. |
| [`RELEVANCE_LEVELS`](#ir.eval.RELEVANCE_LEVELS)              | Graded relevance levels for the package-relevance harness, weakest -> strongest.                                                                                                                                                                       |
| [`LEVEL_GAINS`](#ir.eval.LEVEL_GAINS)                   | Graded relevance gains per level — fed straight to `ef.evaluation`'s graded nDCG.                                                                                                                                                                      |

### Functions

| [`as_doc_retriever`](#ir.eval.as_doc_retriever)(corpus, \*[, mode, surfaces])     | Adapt an `ir` corpus to `ef`'s retriever contract.                                                                                                             |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`calibrate_min_score`](#ir.eval.calibrate_min_score)(corpus, cases, \*[, ...])      | Calibrate the absolute abstention `min_score` floor for one `mode`.                                                                                            |
| [`compare_indexings`](#ir.eval.compare_indexings)(corpora, cases, \*, themes, ...) | A/B (or N-way) regression gate over indexing / embedder configurations.                                                                                        |
| [`corpus_artifact_ids`](#ir.eval.corpus_artifact_ids)(corpus)                        | The set of `artifact_id`s present in a built corpus.                                                                                                           |
| [`derive_named_sets`](#ir.eval.derive_named_sets)(cases, theme, \*, ...[, gains])  | Derive the [`NamedSets`](#ir.eval.NamedSets) for `theme` deterministically from labels.                                               |
| [`distractor_curve_from_cases`](#ir.eval.distractor_curve_from_cases)(source, cases, ...)    | Convenience: a distractor curve over a source's scope from single-gold cases.                                                                                  |
| [`distractor_robustness_curve`](#ir.eval.distractor_robustness_curve)(scope, probes, \*)     | Retrieval accuracy as the catalog grows — the needle-in-a-haystack curve.                                                                                      |
| [`evaluate_discovery`](#ir.eval.evaluate_discovery)(corpus, cases, \*[, ...])       | Score a corpus against `cases` — metrics, failure taxonomy, abstention.                                                                                        |
| [`evaluate_named_sets`](#ir.eval.evaluate_named_sets)(corpus, named_sets[, ...])     | Score one theme's named distractor / hard-positive sets with a single probe.                                                                                   |
| [`evaluate_selection`](#ir.eval.evaluate_selection)(corpus, cases, \*[, ...])       | Score a selector against `cases` — selection quality, isolated.                                                                                                |
| [`fp_rate_on_distractors`](#ir.eval.fp_rate_on_distractors)(ranking, ...)               | Fraction of named distractors that appear in the top-`k` of `ranking`.                                                                                         |
| [`level_histogram`](#ir.eval.level_histogram)(cases, theme)                      | Count of artifacts at each level for `theme` (every level represented).                                                                                        |
| [`load_calibration`](#ir.eval.load_calibration)(corpus, mode)                     | Load a persisted [`MinScoreCalibration`](#ir.eval.MinScoreCalibration) for `(corpus, mode)`.                                                    |
| [`load_cases`](#ir.eval.load_cases)(path)                                   | Read [`DiscoveryCase`](#ir.eval.DiscoveryCase)s from a JSONL file (skips a `__meta__` header).                                            |
| [`load_package_cases`](#ir.eval.load_package_cases)(path)                           | Read [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)s from JSONL (skips a `__meta__` header).                                            |
| [`read_package_meta`](#ir.eval.read_package_meta)(path)                            | Return the `__meta__` header dict from a package-cases JSONL (`{}` if absent).                                                                                 |
| [`recall_on_hard_positives`](#ir.eval.recall_on_hard_positives)(ranking, ...)             | Fraction of named hard-positives in the top-`k` (duplicate ids counted once).                                                                                  |
| [`retrieval_report`](#ir.eval.retrieval_report)(corpus, cases, \*[, ...])         | Score a corpus's retrieval against the cases — the pure-`ef` path.                                                                                             |
| [`save_cases`](#ir.eval.save_cases)(cases, path, \*[, meta])                | Write `cases` to a JSONL file (one case per line).                                                                                                             |
| [`save_package_cases`](#ir.eval.save_package_cases)(cases, path, \*[, meta])        | Write [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)s to JSONL (mirrors [`save_cases()`](#ir.eval.save_cases)). |
| [`sweep_selector`](#ir.eval.sweep_selector)(corpus, cases, \*[, strategy, ...]) | Tune the selector: score a grid of commit knobs against the cases.                                                                                             |
| [`to_graded_qrels`](#ir.eval.to_graded_qrels)(cases, theme, \*[, probe, ...])    | Build `ef`'s **graded** `(queries, qrels)` for one `theme`.                                                                                                    |
| [`to_qrels`](#ir.eval.to_qrels)(cases, \*[, grade])                       | Build `ef`'s `(queries, qrels)` from the gold-bearing cases.                                                                                                   |
| [`validate_cases`](#ir.eval.validate_cases)(corpus, cases)                      | Gold ids a case references that are absent from the corpus.                                                                                                    |

### Classes

| [`ComparisonReport`](#ir.eval.ComparisonReport)(k, themes, labels, ...)          | The outcome of [`compare_indexings()`](#ir.eval.compare_indexings) — an N-way indexing/embedder bake-off.   |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`DiscoveryCase`](#ir.eval.DiscoveryCase)(query[, gold, corpus, ...])         | One discovery probe: an intent and the artifact(s) that should answer it.                                                    |
| [`DiscoveryReport`](#ir.eval.DiscoveryReport)(retrieval, failure_classes, ...)  | The outcome of [`evaluate_discovery()`](#ir.eval.evaluate_discovery).                                        |
| [`MinScoreCalibration`](#ir.eval.MinScoreCalibration)(corpus_name, mode, k, ...)    | A calibrated absolute abstention floor for one `(corpus, mode)`.                                                             |
| [`NamedSetReport`](#ir.eval.NamedSetReport)(theme, k, fp_rate, ...[, ...])     | The outcome of [`evaluate_named_sets()`](#ir.eval.evaluate_named_sets) for one theme.                         |
| [`NamedSets`](#ir.eval.NamedSets)(theme[, distractors, hard_positives])   | The two named diagnostic sets for one theme.                                                                                 |
| [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)(artifact_id[, labels, ...])  | One artifact's graded relevance to one or more themes.                                                                       |
| [`SelectionGridPoint`](#ir.eval.SelectionGridPoint)(max_k, rel, min_score, report) | One cell of a selector sweep: a parameter setting and the report it scored.                                                  |
| [`SelectionReport`](#ir.eval.SelectionReport)(selection_precision, ...[, ...])  | The outcome of [`evaluate_selection()`](#ir.eval.evaluate_selection) — selection quality, isolated.          |
| [`SelectionSweep`](#ir.eval.SelectionSweep)(points, objective, strategy, ...)  | The outcome of [`sweep_selector()`](#ir.eval.sweep_selector) — a grid of selector settings scored.       |

### *class* ir.eval.ComparisonReport(k, themes, labels, baseline, metrics, deltas)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`compare_indexings()`](#ir.eval.compare_indexings) — an N-way indexing/embedder bake-off.

#### k

the rank cutoff all `@k` metrics use.

### themes / labels

the themes scored and the corpus labels compared.

#### baseline

the label every [`regressions()`](#ir.eval.ComparisonReport.regressions) check compares against
(the first entry of `corpora`).

#### metrics

`{label: {theme: {"ndcg", "fp_rate"?, "hard_positive_recall"?}}}`.

#### deltas

`{theme: {artifact_id: {"role", "by_label": {label: {"rank",
"score"}}}}}` for every named FP/FN id — the per-package effect.

#### regressions(, threshold=0, baseline=None)

Named packages that got WORSE than `baseline` — drives a pytest gate.

For a `hard_positive`, worse = its rank dropped (grew larger) by more
than `threshold` positions. For a `distractor`, worse = its rank rose
(grew smaller, more prominent) by more than `threshold`. A package
absent from a ranking is treated as rank `inf` (worst), so a vanished
hard-positive and a newly-appearing distractor both register. Returns one
dict per regressing `(theme, id, label)`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### to_dict()

JSON-serializable form — the qh / HTTP surface.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.eval.DFLT_CALIB_SENSITIVITY_WEIGHT *= 0.5*

Weight on *sensitivity* (catch true positives) vs *specificity* (reject
out-of-scope queries) when [`calibrate_min_score()`](#ir.eval.calibrate_min_score) picks the abstention
floor: `objective = w·sensitivity + (1−w)·specificity`. `0.5` is balanced
(its argmax matches Youden’s J / balanced accuracy). Lower it to lean
*precision* — abstain more readily — per ir_01 §3’s stance that a padded
commit is the costlier failure for an agent surface than a dropped gold.

### ir.eval.DFLT_CURVE_MODE *= 'dense'*

Defaults for [`distractor_robustness_curve()`](#ir.eval.distractor_robustness_curve) — deliberately *not*
`DFLT_MODE`: the curve builds throwaway in-memory sub-corpora, so it favours
the fast, always-offline dense + hashing path. Pass `mode="hybrid"` /
`embedder="default"` to measure the production configuration instead.

### ir.eval.DFLT_DISTRACTOR_SIZES *= (1, 4, 8, 16, 32, 64, 128)*

Catalog sizes swept by [`distractor_robustness_curve()`](#ir.eval.distractor_robustness_curve) (RAG-MCP-style).

### ir.eval.DFLT_K_VALUES *= (1, 5, 10)*

Rank cutoffs reported by default.

### ir.eval.DFLT_METRICS *= ('ndcg', 'recall', 'precision', 'mrr', 'map')*

Metrics reported by default — the `ef.evaluation` retrieval metric names.

### ir.eval.DFLT_MODE *= 'hybrid'*

Default ranking mode for the eval adapter (hybrid is `ir`’s strongest).

### ir.eval.DFLT_SWEEP_MAX_K *= (1, 2, 3, 5, 8)*

`max_k` values swept by [`sweep_selector()`](#ir.eval.sweep_selector) (the commit-size cap).

### ir.eval.DFLT_SWEEP_MIN_SCORE *= (None,)*

`min_score` floors swept by [`sweep_selector()`](#ir.eval.sweep_selector). Default `(None,)` —
the absolute floor is mode-specific (cosine / RRF / BM25 scales differ), so
it is left off unless the caller supplies calibrated values to sweep.

### ir.eval.DFLT_SWEEP_OBJECTIVE *= 'selection_f1'*

Default objective [`SelectionSweep.best()`](#ir.eval.SelectionSweep.best) optimizes — the standard
precision/recall balance. `mean_selected_size` is the only metric minimized;
every other is maximized (see `_SWEEP_MINIMIZE`).

### ir.eval.DFLT_SWEEP_REL *= (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)*

`rel` values swept by [`sweep_selector()`](#ir.eval.sweep_selector) (relative-to-top keep band).

### *class* ir.eval.DiscoveryCase(query, gold=(), corpus=None, source_id=None, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One discovery probe: an intent and the artifact(s) that should answer it.

#### query

the user intent / search query.

#### gold

the `artifact_id`s that correctly answer `query` (one for a
single-answer case, several when capabilities overlap). An **empty**
tuple marks an *abstention* case — no artifact applies.

#### corpus

optional name of the corpus the case targets (provenance / for
multi-corpus case files).

#### source_id

optional `artifact_id` the query was generated from
(back-translation provenance).

#### metadata

free-form per-case metadata (difficulty, generator, …).

#### *classmethod* from_dict(d)

Inverse of [`to_dict()`](#ir.eval.DiscoveryCase.to_dict); `gold` may be a string or a list.

* **Return type:**
  [`DiscoveryCase`](#ir.eval.DiscoveryCase)

#### *property* gold_is_none *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

True when no artifact should match — an abstention case.

#### to_dict()

JSON-serializable form (omitting empty optional fields).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.eval.DiscoveryReport(retrieval, failure_classes, n_cases, n_gold, n_abstention, primary_k=10, abstention_accuracy=None, mode='hybrid', vd_degraded=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_discovery()`](#ir.eval.evaluate_discovery).

Bundles the standard retrieval metrics with the capability-discovery extras:

#### retrieval

the `ef.evaluation.RetrievalEvalReport` over the
gold-bearing cases.

#### failure_classes

counts per failure class (`hit_rank_1` /
`surfaced_low_rank` / `retrieval_miss` for gold cases;
`abstention_ok` / `false_action` / `abstention_unscored` for
abstention cases).

### n_cases / n_gold / n_abstention

case counts.

#### primary_k

the cutoff `primary` reports at (default 10).

#### abstention_accuracy

accuracy on the abstention slice when an
`abstain_threshold` was supplied, else `None`.

#### mode

the ranking mode the report was produced with — recorded so a
number is never mistaken for a different configuration’s.

#### vd_degraded

True when `mode` was `lexical` / `hybrid` but `vd`
was unavailable, so ranking silently fell back to dense (the scores
are *not* the requested configuration’s).

#### *property* primary *: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The headline number — mean NDCG@\`\`primary_k\`\` (`None` if not computed).

### ir.eval.LEVEL_GAINS *= {'core': 3.0, 'none': 0.0, 'strong': 2.0, 'tangential': 0.0, 'uses-tools': 1.0}*

Graded relevance gains per level — fed straight to `ef.evaluation`’s graded
nDCG. `uses-tools` is a *weak* positive (a package that merely uses the
relevant tooling); `tangential`/`none` are non-relevant (gain `0`). These
gains are the knob that lets nDCG rank a core-first ordering strictly above a
uses-tools-first one at identical recall.

### *class* ir.eval.MinScoreCalibration(corpus_name, mode, k, min_score, sensitivity, specificity, youden_j, balanced_accuracy, separable, sensitivity_weight, n_positive, n_abstention, n_retrieval_miss, positive_scores, abstention_scores, grid, reason, embedder_id=None, strategy='conservative', vd_degraded=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A calibrated absolute abstention floor for one `(corpus, mode)`.

Produced by [`calibrate_min_score()`](#ir.eval.calibrate_min_score). `min_score` is the floor to pass to
`ir.select.select()` / [`ir.discover()`](ir.md#ir.discover) (`discover(min_score="auto")`
loads it): the conservative selector abstains when even the top hit scores
below it. `None` means no floor could be calibrated (one of the two query
classes was missing — see [`reason`](#ir.eval.MinScoreCalibration.reason)); the selector then never abstains
by absolute score, exactly as before.

The quality numbers treat `top_score ≥ min_score → commit` as a binary
classifier over in-scope (label *commit*) vs out-of-scope (label *abstain*)
queries:

#### min_score

the calibrated floor (`None` if not calibratable).

#### sensitivity

TPR — fraction of in-scope queries kept (`top ≥ floor`).

#### specificity

TNR — fraction of out-of-scope queries abstained
(`top < floor`).

#### youden_j

`sensitivity + specificity − 1` (0 = no separation, 1 = perfect).

#### balanced_accuracy

`(sensitivity + specificity) / 2`.

#### separable

whether the two score distributions are perfectly separable
(some floor reaches `youden_j == 1`).

#### sensitivity_weight

the `w` used to pick the floor (see
[`DFLT_CALIB_SENSITIVITY_WEIGHT`](#ir.eval.DFLT_CALIB_SENSITIVITY_WEIGHT)).

#### n_positive

in-scope cases used (gold-bearing **and** gold-retrieved).

#### n_abstention

out-of-scope (empty-gold) cases used.

#### n_retrieval_miss

gold-bearing cases whose gold never reached the `k`
candidates — excluded (a *retrieval* failure, not an abstention
signal), mirroring [`evaluate_selection()`](#ir.eval.evaluate_selection)’s conditioning.

### positive_scores / abstention_scores

`{min, median, max}` summaries of
each class’s top-score distribution (the auditable evidence the floor
sits between them).

#### grid

every evaluated floor, `{min_score, sensitivity, specificity,
youden_j, objective}` — the full sweep behind the choice.

#### reason

why `min_score` is what it is (`"calibrated"`,
`"no_positive_cases"`, `"no_abstention_cases"`, `"no_cases"`).

### corpus_name / mode / k / embedder_id

the configuration scored — and the
`embedder_id` stamp lets a consumer detect a stale floor after a
rebuild with a different embedder.

#### vd_degraded

True when `mode` was `lexical` / `hybrid` but `vd`
was unavailable, so ranking fell back to dense and the floor is on the
*dense* scale, not the requested mode’s — recalibrate once `vd` is
installed.

#### *classmethod* from_dict(d)

Inverse of [`to_dict()`](#ir.eval.MinScoreCalibration.to_dict) (tolerant of a persisted record’s defaults).

* **Return type:**
  [`MinScoreCalibration`](#ir.eval.MinScoreCalibration)

#### save(corpus)

Persist this calibration on `corpus` (keyed by mode); returns self.

Stored in the corpus’s `calibration` view (not its build `config`), so
it survives reopen and is loaded by `discover(min_score="auto")`. The
floor is machine-local (derived from a live corpus + embedder) and is
never committed to source control.

* **Return type:**
  [`MinScoreCalibration`](#ir.eval.MinScoreCalibration)

#### to_dict()

JSON-serializable form (for persistence and the qh / HTTP surface).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.eval.NamedSetReport(theme, k, fp_rate, hard_positive_recall, distractors_seen=(), hard_positives_missed=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_named_sets()`](#ir.eval.evaluate_named_sets) for one theme.

### theme / k

the theme and the rank cutoff scored.

#### fp_rate

distractor false-positive rate at `k` (lower is better).

#### hard_positive_recall

recall over the hard-positive set at `k` (higher
is better).

#### distractors_seen

the distractor ids actually in the top-`k` — so the
`fp_rate` is auditable back to specific packages.

#### hard_positives_missed

the hard-positive ids NOT in the top-`k`.

#### to_dict()

JSON-serializable form.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.eval.NamedSets(theme, distractors=(), hard_positives=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The two named diagnostic sets for one theme.

#### theme

the theme these sets diagnose.

#### distractors

ids that are NOT relevant (`none`/`tangential`) yet rank
prominently — every appearance in the top-`k` is a false positive.

#### hard_positives

ids that ARE relevant (`core`/`strong`) but hard to
rank (e.g. thin-description packages) — recall on these is the
headline gap a fix must close.

#### *classmethod* from_dict(d)

Inverse of [`to_dict()`](#ir.eval.NamedSets.to_dict).

* **Return type:**
  [`NamedSets`](#ir.eval.NamedSets)

#### to_dict()

JSON-serializable form.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.eval.PackageRelevanceCase(artifact_id, labels=<factory>, evidence=<factory>, observed=<factory>, thin_description=False, is_distractor=<factory>, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One artifact’s graded relevance to one or more themes.

Unlike [`DiscoveryCase`](#ir.eval.DiscoveryCase) (one query -> gold ids, flat-binary gold), a
case here is one **artifact** carrying its graded label per theme. The
per-theme *probe text* (the query each theme is scored with) is **not** on the
case — it lives in the JSONL `__meta__` header (see
[`save_package_cases()`](#ir.eval.save_package_cases)), alongside the corpus signature
([`ir.eval_gen.corpus_signature()`](ir.eval_gen.md#ir.eval_gen.corpus_signature)), so a frozen label set pins to the
corpus snapshot it was judged against.

#### artifact_id

the package id (matches the corpus’ `artifact_id`).

#### labels

`{theme: level}` with each level in [`RELEVANCE_LEVELS`](#ir.eval.RELEVANCE_LEVELS).

#### evidence

`{theme: short reason}` — the human-auditable *why*.

#### observed

`{theme: score}` the ranking gave this artifact in the frozen
run (optional; used to *derive* the named distractor set).

#### thin_description

True when the package had an empty/near-empty pyproject
description (used to derive the hard-positive set).

#### is_distractor

optional cached `{theme: bool}`; when absent it is
derived (see [`derive_named_sets()`](#ir.eval.derive_named_sets)).

#### metadata

free-form per-case metadata.

#### *classmethod* from_dict(d)

Inverse of [`to_dict()`](#ir.eval.PackageRelevanceCase.to_dict); rejects any level outside [`RELEVANCE_LEVELS`](#ir.eval.RELEVANCE_LEVELS).

* **Return type:**
  [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)

#### gain(theme, , gains={'core': 3.0, 'none': 0.0, 'strong': 2.0, 'tangential': 0.0, 'uses-tools': 1.0})

The graded gain for `theme` under `gains` (default [`LEVEL_GAINS`](#ir.eval.LEVEL_GAINS)).

A level missing from `gains` (e.g. a partial caller-supplied mapping)
defaults to `0.0` — the non-relevant convention — rather than raising.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### level(theme)

This artifact’s graded level for `theme` (`"none"` if unlabeled).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### to_dict()

JSON-serializable form (omitting empty optional fields).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.eval.RELEVANCE_LEVELS *= ('none', 'tangential', 'uses-tools', 'strong', 'core')*

Graded relevance levels for the package-relevance harness, weakest -> strongest.
The SSOT consumed by [`LEVEL_GAINS`](#ir.eval.LEVEL_GAINS), [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase), and
[`to_graded_qrels()`](#ir.eval.to_graded_qrels).

### ir.eval.SWEEP_METRICS *= ('selection_f1', 'selection_precision', 'selection_recall', 'conditional_commit_rate', 'abstention_accuracy', 'mean_selected_size')*

The scalar selection metrics a sweep can optimize / rank by.

### *class* ir.eval.SelectionGridPoint(max_k, rel, min_score, report)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One cell of a selector sweep: a parameter setting and the report it scored.

### max_k / rel / min_score

the swept selection parameters at this cell.

#### report

the [`SelectionReport`](#ir.eval.SelectionReport) for those parameters.

#### *property* params *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

The swept parameters as a plain dict (`max_k` / `rel` / `min_score`).

#### to_dict()

JSON-serializable form: the params plus the full report dict.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.eval.SelectionReport(selection_precision, selection_recall, selection_f1, conditional_commit_rate, n_gold_retrieved, abstention_accuracy, mean_selected_size, n_cases, n_gold, n_abstention, strategy='conservative', mode='hybrid', k=10)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_selection()`](#ir.eval.evaluate_selection) — selection quality, isolated.

Where [`evaluate_discovery()`](#ir.eval.evaluate_discovery) scores the *ranking*, this scores the
*commit*: given the same `k` candidates, what subset did the selector keep?
The headline is [`conditional_commit_rate`](#ir.eval.SelectionReport.conditional_commit_rate) — accuracy of the selection
decision **conditioned on retrieval having surfaced the gold** — which is the
one number that separates a selection failure from a retrieval failure.

Every selection-quality metric shares that conditioning: precision, recall
and F1 are computed **only over the** [`n_gold_retrieved`](#ir.eval.SelectionReport.n_gold_retrieved) **cases** whose
gold reached the `k` candidates, so a retrieval miss is never charged to the
selector (it is [`evaluate_discovery()`](#ir.eval.evaluate_discovery)’s to report).

#### selection_precision

mean over committed gold-retrieved cases of
`|selected ∩ gold| / |selected|` (`None` if nothing was ever
committed) — how clean the committed sets are.

#### selection_recall

mean over gold-retrieved cases of
`|selected ∩ gold| / |gold ∩ retrieved|` — of the gold the selector
was actually shown, how much it kept (denominator is the *retrievable*
gold, so retrieval misses do not depress it).

#### selection_f1

mean per-case F1 over gold-retrieved cases (an empty commit
scores 0).

#### conditional_commit_rate

among gold cases whose gold was in the `k`
candidates, the fraction where the selector committed to ≥1 gold —
the selection decision isolated from retrieval recall (`None` if
retrieval never surfaced any gold).

#### n_gold_retrieved

the shared denominator of every selection-quality
metric above (cases whose gold reached the `k` candidates).

#### abstention_accuracy

fraction of abstention cases (empty gold) the
selector correctly committed nothing to (`None` if none).

#### mean_selected_size

mean committed-set size over gold-retrieved cases.

### n_cases / n_gold / n_abstention

case counts.

### strategy / mode / k

the configuration scored (so a number is never
mistaken for a different setup’s).

#### to_dict()

JSON-serializable form (for the qh / HTTP surface).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.eval.SelectionSweep(points, objective, strategy, mode, k, n_cases, n_gold, n_abstention, n_gold_retrieved)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`sweep_selector()`](#ir.eval.sweep_selector) — a grid of selector settings scored.

Every point shares the same cases, retrieval `mode` / candidate depth `k`
and `strategy`; they differ only in the swept selection knobs (`max_k`,
`rel`, `min_score`). [`best()`](#ir.eval.SelectionSweep.best) picks a single setting by an objective;
[`frontier()`](#ir.eval.SelectionSweep.frontier) returns the precision/recall Pareto front for the honest
trade-off view; [`table()`](#ir.eval.SelectionSweep.table) renders the whole grid.

#### points

one [`SelectionGridPoint`](#ir.eval.SelectionGridPoint) per grid cell (grid order).

#### objective

the default metric [`best()`](#ir.eval.SelectionSweep.best) optimizes.

### strategy / mode / k

the shared configuration the grid was scored under.

### n_cases / n_gold / n_abstention / n_gold_retrieved

case counts (echoed
from the reports; selection metrics share the `n_gold_retrieved`
denominator, so it is surfaced here too).

#### best(metric=None, , minimize=None)

The grid point that optimizes `metric` (default [`objective`](#ir.eval.SelectionSweep.objective)).

Ties are broken toward the *cheaper, tighter* commit — smaller
`mean_selected_size`, then smaller `max_k`, then larger `rel` —
encoding the capability-discovery stance that fewer, higher-precision
commits beat more (ir_01 §3). `minimize` is inferred from the metric
(only `mean_selected_size` minimizes) but can be forced.

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the sweep is empty or `metric` is unknown.
* **Return type:**
  [`SelectionGridPoint`](#ir.eval.SelectionGridPoint)

#### frontier(, x='selection_recall', y='selection_precision')

The Pareto-optimal points trading `x` off against `y` (both maximized).

A point is on the frontier when no other point is at least as good on both
axes and strictly better on one. `mean_selected_size` on an axis is read
as “smaller is better” automatically. Returned sorted by `x` ascending.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SelectionGridPoint`](#ir.eval.SelectionGridPoint)]

#### table(, sort_by=None)

A human-readable grid table, one row per setting (best objective first).

`sort_by` defaults to [`objective`](#ir.eval.SelectionSweep.objective). `min_score` is shown only
when any cell sets one (an all-`None` column is elided as noise).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### to_dict()

JSON-serializable form: the grid, the best point, and the config.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.eval.as_doc_retriever(corpus, , mode='hybrid', surfaces=None, \*\*search_kw)

Adapt an `ir` corpus to `ef`’s retriever contract.

Returns a callable `retriever(query, *, limit=10) -> [artifact_id, …]` —
bare doc-id strings, best-ranked first, one per artifact. That is exactly
what `ef.evaluation.evaluate_retrieval()` consumes (it maps a bare
string straight to a document id), so the returned callable can be handed to
`ef` with no further adaptation.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – an [`Corpus`](ir.index.md#ir.index.Corpus) or a registered corpus *name*.
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode — `"dense"` / `"lexical"` / `"hybrid"`.
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict to these surface kinds (e.g. `{"description"}`).
  * **\*\*search_kw** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – any other [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search) keyword (`filter`,
    `rrf_k`, `rerank`, `bm25`, …).
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.eval.calibrate_min_score(corpus, cases, , mode='hybrid', k=10, surfaces=None, sensitivity_weight=0.5, floor_grid=None, persist=False, \*\*search_kw)

Calibrate the absolute abstention `min_score` floor for one `mode`.

Retrieves `k` candidates per case **once** (reusing [`evaluate_selection()`](#ir.eval.evaluate_selection)’s
retrieval pass), then treats floor-picking as a 1-D separation between the
*in-scope* top scores (gold-bearing cases whose gold reached the candidates)
and the *out-of-scope* top scores (empty-gold abstention cases). The floor
that best separates them — by `w·sensitivity + (1−w)·specificity` — is the
calibrated `min_score`.

`cases` must contain **both** gold-bearing and abstention cases; with only
one class the floor is undefined (`min_score=None`, `reason` records which
class was missing). Generate abstention cases with [`ir.eval_gen`](ir.eval_gen.md#module-ir.eval_gen) (its
`abstention` slice) — this module does not synthesize them.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – an [`Corpus`](ir.index.md#ir.index.Corpus) or a registered corpus *name*.
  * **cases** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`DiscoveryCase`](#ir.eval.DiscoveryCase)]) – mixed [`DiscoveryCase`](#ir.eval.DiscoveryCase)s (gold-bearing + abstention).
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode to calibrate (a floor is mode-specific).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – candidate depth retrieved before selection (match your `discover` k).
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict retrieval to these surface kinds.
  * **sensitivity_weight** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – `w` in the pick objective (see
    [`DFLT_CALIB_SENSITIVITY_WEIGHT`](#ir.eval.DFLT_CALIB_SENSITIVITY_WEIGHT)); `0.5` is balanced.
  * **floor_grid** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – candidate floors to evaluate; default derives them from the
    observed top scores (midpoints between adjacent values + the
    commit-all / abstain-all extremes), which is exact for a 1-D split.
  * **persist** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – when True, [`MinScoreCalibration.save()`](#ir.eval.MinScoreCalibration.save) the result on the
    corpus so `discover(min_score="auto")` will load it.
  * **\*\*search_kw** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – any other [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search) keyword.
* **Return type:**
  [`MinScoreCalibration`](#ir.eval.MinScoreCalibration)
* **Returns:**
  a [`MinScoreCalibration`](#ir.eval.MinScoreCalibration).

### ir.eval.compare_indexings(corpora, cases, , themes, probes, k=20, named_sets=None, mode='hybrid', rank_depth=1000, gains={'core': 3.0, 'none': 0.0, 'strong': 2.0, 'tangential': 0.0, 'uses-tools': 1.0}, surfaces=None, \*\*search_kw)

A/B (or N-way) regression gate over indexing / embedder configurations.

`corpora` maps a label -> a built corpus (an `ef` instruction-tuned-embedder
corpus or an `ir` deps-as-text corpus is just another entry — the harness is
embedder-agnostic). For each `(label, theme)` it computes graded nDCG@k (via
[`to_graded_qrels()`](#ir.eval.to_graded_qrels) + `ef.evaluation.ndcg_at_k`), the named-set FP-rate /
hard-positive recall@k (when `named_sets` is given), and the rank+score of
every named FP/FN id — so a change’s effect is quantified *per package*.
`probes` supplies the per-theme query text (normally loaded from the JSONL
`__meta__` via [`read_package_meta()`](#ir.eval.read_package_meta)). The first label is the baseline
that [`ComparisonReport.regressions()`](#ir.eval.ComparisonReport.regressions) compares against. Each ranking is
fetched once to depth `rank_depth` (deep enough that even a buried named id
gets a real rank) and sliced at `k` for the `@k` metrics.

* **Return type:**
  [`ComparisonReport`](#ir.eval.ComparisonReport)

### ir.eval.corpus_artifact_ids(corpus)

The set of `artifact_id`s present in a built corpus.

* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.eval.derive_named_sets(cases, theme, , observed_floor, gains={'core': 3.0, 'none': 0.0, 'strong': 2.0, 'tangential': 0.0, 'uses-tools': 1.0})

Derive the [`NamedSets`](#ir.eval.NamedSets) for `theme` deterministically from labels.

`distractors` = non-relevant (`none`/`tangential`) packages whose frozen
`observed` score for the theme is at or above `observed_floor` (they
ranked high enough to pollute the top-`k`) — unless a case carries an
explicit cached `is_distractor[theme]`, which then overrides the
observed-floor rule. `hard_positives` = relevant (`core`/`strong`)
packages flagged `thin_description` (the ones a description+README index
struggles to surface). Both lists are de-duplicated and sorted, so the result
is a stable, committable artifact.

* **Return type:**
  [`NamedSets`](#ir.eval.NamedSets)

### ir.eval.distractor_curve_from_cases(source, cases, \*\*kwargs)

Convenience: a distractor curve over a source’s scope from single-gold cases.

Probes are the single-gold cases (`(query, gold[0])`); the source’s
`scope` is the distractor pool and its `indexing_strategy` is reused
unless a `strategy` is passed in `kwargs`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### ir.eval.distractor_robustness_curve(scope, probes, , strategy=None, embedder='light', mode='dense', sizes=(1, 4, 8, 16, 32, 64, 128), trials=20, seed=0, k=1)

Retrieval accuracy as the catalog grows — the needle-in-a-haystack curve.

For each catalog size `N` and each `(query, gold_id)` probe, build many
`N`-artifact sub-corpora (the gold artifact plus `N−1` distractors
sampled from `scope`) and measure how often the gold artifact lands in the
top `k`. The accuracy at each `N` is averaged over the probes and over
`trials` random distractor draws. `N=1` is the trivial anchor (gold
alone) and is always 1.0; a steep decline as `N` grows is the signature of
retrieval that does not discriminate.

Sub-corpora are built in-memory with the `"light"` (numpy-only) embedder
by default, so the curve is fast and offline; pass `embedder="default"`
(and `mode="hybrid"`) to measure the production configuration.

Sizes larger than the corpus are **capped** at `len(scope)` (the largest
catalog the pool can fill) and de-duplicated, so the returned keys are the
catalog sizes actually built — the x-axis never claims an `N` larger than
the corpus could supply.

* **Parameters:**
  * **scope** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the full `{artifact_id: raw}` pool to draw the catalog from.
  * **probes** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `(query, gold_id)` pairs; `gold_id` must be a key of `scope`.
  * **strategy** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the [`IndexingStrategy`](ir.strategy.md#ir.strategy.IndexingStrategy) to index with
    (default [`WholeText`](ir.strategy.md#ir.strategy.WholeText)).
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – embedder spec for the sub-corpora.
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode for the probe searches.
  * **sizes** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – the catalog sizes to sweep (capped at `len(scope)`, deduped).
  * **trials** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – random distractor draws averaged per size (a size that consumes
    the whole pool has no sampling freedom, so it runs a single trial).
  * **seed** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – base RNG seed (reproducible; varied per size).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – the top-`k` the gold must reach to count as a hit.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]
* **Returns:**
  `{N: accuracy}` for each *achievable* `N` (the requested sizes capped
  at `len(scope)` and de-duplicated).

### ir.eval.evaluate_discovery(corpus, cases, , k_values=(1, 5, 10), primary_k=10, metrics=('ndcg', 'recall', 'precision', 'mrr', 'map'), mode='hybrid', surfaces=None, abstain_threshold=None, \*\*search_kw)

Score a corpus against `cases` — metrics, failure taxonomy, abstention.

Runs retrieval **once per case** and, from each ranking, computes the
`ef.evaluation` metric primitives (so the numbers match the pure-`ef`
path) *and* the capability-discovery extras in one pass.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – an [`Corpus`](ir.index.md#ir.index.Corpus) or a registered corpus *name*.
  * **k_values** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – rank cutoffs to report (`primary_k` is always included).
  * **primary_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – the cutoff the headline `primary` reports at.
  * **metrics** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – which `ef` metrics to compute (`ndcg` / `recall` /
    `precision` / `mrr` / `map`).
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode passed to [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search).
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict to these surface kinds.
  * **abstain_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – if given, an abstention case is scored *correct* when
    the top hit’s score is below this threshold (a retrieval-side proxy
    for “no artifact applies”); leave `None` to only count abstention
    cases without scoring them.
  * **\*\*search_kw** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – any other [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search) keyword.
* **Return type:**
  [`DiscoveryReport`](#ir.eval.DiscoveryReport)
* **Returns:**
  a [`DiscoveryReport`](#ir.eval.DiscoveryReport). If no case carries gold (an empty or
  all-abstention set) the retrieval metrics are omitted — `report.primary`
  is `None` — but the taxonomy and abstention counts are still returned
  (unlike [`retrieval_report()`](#ir.eval.retrieval_report), which raises `ValueError` on that
  input).

### ir.eval.evaluate_named_sets(corpus, named_sets, theme=None, , probe, mode='hybrid', k=10, surfaces=None, \*\*search_kw)

Score one theme’s named distractor / hard-positive sets with a single probe.

Runs the theme `probe` once, takes the per-artifact ranking, and reports the
distractor false-positive rate and hard-positive recall at `k` — plus the
distractor ids actually seen in the top-`k` and the hard-positives missed, so
a number is always auditable back to specific packages. `theme` defaults to
`named_sets.theme` (the only thing it labels); pass it only to override.

* **Return type:**
  [`NamedSetReport`](#ir.eval.NamedSetReport)

### ir.eval.evaluate_selection(corpus, cases, , strategy='conservative', mode='hybrid', k=10, surfaces=None, max_k=3, rel=0.9, gap_ratio=0.5, min_score=None, \*\*search_kw)

Score a selector against `cases` — selection quality, isolated.

Retrieves `k` candidates per case (the window the selector sees), commits
with `ir.select.select()`, and scores the *commit*. The key number,
[`SelectionReport.conditional_commit_rate`](#ir.eval.SelectionReport.conditional_commit_rate), conditions on retrieval
having surfaced the gold among those `k` — so a low value means the
*selector* dropped a gold it was shown, not that retrieval missed it.

`k` is the candidate window; hold it equal to the `k` used with
[`evaluate_discovery()`](#ir.eval.evaluate_discovery) to compare the two stages on the same footing.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – an [`Corpus`](ir.index.md#ir.index.Corpus) or a registered corpus *name*.
  * **cases** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`DiscoveryCase`](#ir.eval.DiscoveryCase)]) – the [`DiscoveryCase`](#ir.eval.DiscoveryCase)s (gold-bearing and/or abstention).
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – selection strategy (see `ir.select.select()`).
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode for retrieval.
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – candidate depth retrieved before selection.
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict retrieval to these surface kinds.
  * **max_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – selection parameters.
  * **rel** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – selection parameters.
  * **gap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – selection parameters.
  * **min_score** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – selection parameters.
  * **\*\*search_kw** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – any other [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search) keyword.
* **Return type:**
  [`SelectionReport`](#ir.eval.SelectionReport)
* **Returns:**
  a [`SelectionReport`](#ir.eval.SelectionReport).

### ir.eval.fp_rate_on_distractors(ranking, distractor_ids, , k)

Fraction of named distractors that appear in the top-`k` of `ranking`.

A distractor surfacing in the committed top-`k` is a false positive; this is
the rate over the named set (`0.0` when the set is empty). Duplicate ids are
counted once.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ir.eval.level_histogram(cases, theme)

Count of artifacts at each level for `theme` (every level represented).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### ir.eval.load_calibration(corpus, mode)

Load a persisted [`MinScoreCalibration`](#ir.eval.MinScoreCalibration) for `(corpus, mode)`.

Returns `None` when no calibration has been stored for that mode. Does not
validate the embedder stamp — that staleness check lives at the
`discover(min_score="auto")` boundary, which knows the live corpus’s
embedder id.

* **Return type:**
  [`MinScoreCalibration`](#ir.eval.MinScoreCalibration) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.eval.load_cases(path)

Read [`DiscoveryCase`](#ir.eval.DiscoveryCase)s from a JSONL file (skips a `__meta__` header).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`DiscoveryCase`](#ir.eval.DiscoveryCase)]

### ir.eval.load_package_cases(path)

Read [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)s from JSONL (skips a `__meta__` header).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)]

### ir.eval.read_package_meta(path)

Return the `__meta__` header dict from a package-cases JSONL (`{}` if absent).

The header carries the per-theme probe text (`probes`) and the corpus
signature the labels were frozen against — everything [`compare_indexings()`](#ir.eval.compare_indexings)
needs that is not on a per-artifact case.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.eval.recall_on_hard_positives(ranking, hard_positive_ids, , k)

Fraction of named hard-positives in the top-`k` (duplicate ids counted once).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ir.eval.retrieval_report(corpus, cases, , k_values=(1, 5, 10), metrics=('ndcg', 'recall', 'precision', 'mrr', 'map'), mode='hybrid', surfaces=None, limit=None, \*\*search_kw)

Score a corpus’s retrieval against the cases — the pure-`ef` path.

A thin wrapper that builds the retriever adapter and hands it, with the
cases’ `(queries, qrels)`, to `ef.evaluation.evaluate_retrieval()`.
Returns its `ef.evaluation.RetrievalEvalReport` (`.primary` is mean
[NDCG@10](mailto:NDCG@10)). Use this to A/B configurations — dense vs hybrid, with vs without
a reranker — on the standard BEIR/MTEB metrics.

For the richer report (taxonomy + abstention) use [`evaluate_discovery()`](#ir.eval.evaluate_discovery).

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if no case carries gold (`ef` requires at least one
      positively-judged query). [`evaluate_discovery()`](#ir.eval.evaluate_discovery) instead
      tolerates an all-abstention set — the two paths differ only on that
      degenerate input.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ir.eval.save_cases(cases, path, , meta=None)

Write `cases` to a JSONL file (one case per line).

An optional `meta` mapping is written as a leading `{"__meta__": …}`
line — the natural home for a corpus-version anchor that pins the cases to
the corpus snapshot they were generated against.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.eval.save_package_cases(cases, path, , meta=None)

Write [`PackageRelevanceCase`](#ir.eval.PackageRelevanceCase)s to JSONL (mirrors [`save_cases()`](#ir.eval.save_cases)).

`meta` is written as a leading `{"__meta__": …}` line — the home for the
per-theme probe text (`probes`) and the corpus signature
([`ir.eval_gen.corpus_signature()`](ir.eval_gen.md#ir.eval_gen.corpus_signature)) that pin the labels to a corpus snapshot.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.eval.sweep_selector(corpus, cases, , strategy='conservative', mode='hybrid', k=10, surfaces=None, max_k_grid=(1, 2, 3, 5, 8), rel_grid=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_score_grid=(None,), gap_ratio=0.5, objective='selection_f1', \*\*search_kw)

Tune the selector: score a grid of commit knobs against the cases.

The selection defaults (`DFLT_MAX_K`,
`DFLT_REL_THRESHOLD`) were tuned on real corpora (see
ir_06), not guessed. This sweeps `max_k × rel × min_score` and scores each
cell with [`evaluate_selection()`](#ir.eval.evaluate_selection)’s exact metric, so the right defaults
for a *given* corpus can be read off empirically (precision vs recall vs
commit size) rather than assumed.

Retrieval is run **once per case** (selector parameters do not change the
ranking) and the cached candidates are reused across the whole grid, so a
6×5 grid costs one retrieval pass, not thirty.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – an [`Corpus`](ir.index.md#ir.index.Corpus) or a registered corpus *name*.
  * **cases** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`DiscoveryCase`](#ir.eval.DiscoveryCase)]) – the [`DiscoveryCase`](#ir.eval.DiscoveryCase)s to score against.
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the selection strategy held fixed across the grid (the swept
    knobs are `conservative`’s; a non-`conservative` strategy that
    ignores some knobs simply yields duplicate rows).
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ranking mode for retrieval.
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – candidate depth retrieved before selection (the selector’s window).
  * **surfaces** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – restrict retrieval to these surface kinds.
  * **min_score_grid** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – the axes of the sweep.
  * **gap_ratio** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – held fixed (used only by the `score_gap` strategy).
  * **objective** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the default metric [`SelectionSweep.best()`](#ir.eval.SelectionSweep.best) optimizes.
  * **\*\*search_kw** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – any other [`ir.retrieve.search()`](ir.retrieve.md#ir.retrieve.search) keyword.
* **Return type:**
  [`SelectionSweep`](#ir.eval.SelectionSweep)
* **Returns:**
  a [`SelectionSweep`](#ir.eval.SelectionSweep) (`.best()` / `.frontier()` / `.table()` /
  `.to_dict()`).

### ir.eval.to_graded_qrels(cases, theme, , probe=None, gains={'core': 3.0, 'none': 0.0, 'strong': 2.0, 'tangential': 0.0, 'uses-tools': 1.0}, query_id=None)

Build `ef`’s **graded** `(queries, qrels)` for one `theme`.

Unlike [`to_qrels()`](#ir.eval.to_qrels) (which hardcodes grade `1`), every positive artifact
is judged at its graded [`LEVEL_GAINS`](#ir.eval.LEVEL_GAINS) value, so graded gains reach
`ef.evaluation`’s nDCG with zero `ef` change. There is exactly **one**
query per theme (the theme *probe*); `qrels` maps it to `{artifact_id:
gain}` for every artifact with a positive gain. Non-positive
(`none`/`tangential`) artifacts are omitted, matching `ef`’s
judged-positive qrels convention.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]

### ir.eval.to_qrels(cases, , grade=1.0)

Build `ef`’s `(queries, qrels)` from the gold-bearing cases.

Abstention cases (empty `gold`) are skipped — `ef`’s retrieval metrics
require at least one positively-judged document per query. Query ids are
derived from each case’s position so they stay stable across the gap left
by skipped abstention cases.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]

### ir.eval.validate_cases(corpus, cases)

Gold ids a case references that are absent from the corpus.

Returns `{case_index: [missing_artifact_id, …]}` for every case whose gold
has drifted out of the corpus (an empty dict means the cases are aligned).
The eval corpora are live and machine-specific, so a committed case file can
quietly fall out of sync; run this before trusting a report.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
