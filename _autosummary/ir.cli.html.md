# ir.cli

Command-line surface for `ir` (dispatched by cw).

Commands operate on **named** corpora from the registry (see
[`ir.registry`](ir.registry.html.md#module-ir.registry)):

```default
ir build skills                 # build/update the skills preset corpus
ir search skills "deploy app"   # rank candidates (retrieval only)
ir discover skills "deploy app" # retrieve -> commit to a high-precision subset
ir discover skills "deploy app" --disclose   # + load each selected item's body
ir ls                           # list corpora + record counts
ir coverage reports             # disk-vs-index coverage (silent-gap detector)
ir info packages                # config + stats for a corpus
ir maintain --all               # run due background work (idempotent)
ir schedule                     # install/inspect the OS job that runs it
ir register notes files --root ~/notes --pattern '.*\.md$'
ir rm notes                     # unregister (keeps built data)
ir eval-gen skills skills_eval.jsonl --k 5        # generate cases (needs aix/LLM)
ir eval skills skills_eval.jsonl --mode hybrid    # score retrieval on a case file
ir eval-select skills skills_eval.jsonl           # score the selection stage
ir sweep-select skills skills_eval.jsonl          # tune max_k/rel for the selector
ir calibrate-min-score skills skills_eval.jsonl --persist  # abstention floor
ir discover skills "deploy app" --min-score auto  # use the calibrated floor
```

### Functions

| [`build`](#ir.cli.build)(name, \*[, embedder, full])                  | Build or incrementally update a registered (or preset) corpus.                 |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`calibrate_min_score`](#ir.cli.calibrate_min_score)(name, cases, \*[, mode, ...])  | Calibrate the absolute abstention min_score floor for a corpus + mode.         |
| [`coverage`](#ir.cli.coverage)([name])                                   | Report disk-vs-index coverage for a reports corpus (silent-gap detector).      |
| [`discover`](#ir.cli.discover)(name, query, \*[, k, mode, ...])          | Search a corpus, commit to a distractor-robust subset, and show it.            |
| [`eval`](#ir.cli.eval)(name, cases, \*[, mode, k])                   | Score a built corpus's retrieval against a DiscoveryCase JSONL file.           |
| [`eval_gen`](#ir.cli.eval_gen)(name, out, \*[, k, abstention_frac, ...]) | Generate an eval-case file for a corpus by back-translation (needs aix/LLM).   |
| [`eval_select`](#ir.cli.eval_select)(name, cases, \*[, strategy, ...])      | Score a selector against a DiscoveryCase JSONL file (selection quality).       |
| [`info`](#ir.cli.info)(name)                                         | Show a corpus's stored config, stats, policy, and any abstention floors.       |
| [`ls`](#ir.cli.ls)()                                               | List registered corpora with their kind, embedder, and record count.           |
| [`maintain`](#ir.cli.maintain)([name, all, dry_run])                     | Run due background work: incremental reindex, synopsis in its downtime window. |
| [`register`](#ir.cli.register)(name, kind, \*[, root, pattern, ...])     | Register a named corpus.                                                       |
| [`rm`](#ir.cli.rm)(name)                                           | Unregister a corpus (does not delete its built data).                          |
| [`schedule`](#ir.cli.schedule)(\*[, every, status, restart, ...])        | Install, inspect, and operate the OS job that runs `ir maintain`.              |
| [`search`](#ir.cli.search)(name, query, \*[, k, mode, fusion])         | Search a built corpus and print the top-k hits.                                |
| [`sweep_select`](#ir.cli.sweep_select)(name, cases, \*[, strategy, ...])     | Sweep selector knobs (max_k × rel) against a case file; print the grid + best. |

### ir.cli.build(name, , embedder=None, full=True)

Build or incrementally update a registered (or preset) corpus.

### ir.cli.calibrate_min_score(name, cases, , mode='hybrid', k=10, sensitivity_weight=0.5, persist=False, fusion='rrf')

Calibrate the absolute abstention min_score floor for a corpus + mode.

Separates in-scope (gold-bearing) from out-of-scope (empty-gold) query top
scores and reports the floor that best splits them, with sensitivity /
specificity / Youden’s J. `cases` must include BOTH gold-bearing and
abstention cases — generate them with `ir eval-gen` (it adds an abstention
slice). `--persist` stores the floor so `ir discover ... --min-score auto`
will abstain by it. `--sensitivity-weight` (0..1, balanced default);
lower it to abstain more readily (precision-leaning). For hybrid, calibrate
with `--fusion blend` (rank-based RRF barely separates — see ir_08) and
query under the same fusion. mode: dense | lexical | hybrid. fusion (hybrid
only): rrf | blend.

### ir.cli.coverage(name='reports')

Report disk-vs-index coverage for a reports corpus (silent-gap detector).

Independently walks every `*/*/docs` and `*/*/misc/docs` tree on disk and
diffs the includable reports against what is actually indexed, listing any
file that is on disk but missing from the index (rebuild with `ir build` to
fix). Catches the ingestion gaps that back-translation evals structurally
cannot — a doc that was never indexed can never be an eval gold id.

### ir.cli.discover(name, query, , k=10, mode='hybrid', strategy='conservative', disclose=False, min_score=None, fusion='rrf')

Search a corpus, commit to a distractor-robust subset, and show it.

Retrieves `k` candidates, then *selects* the few high-precision results an
agent should act on (or abstains). `--disclose` additionally loads each
selected item’s body (SKILL.md / file text) via its stored pointer.
`--min-score auto` turns on absolute abstention using the floor calibrated
by `ir calibrate-min-score` (or pass a float). mode: dense | lexical |
hybrid. fusion (hybrid only): rrf (default) | blend (magnitude-preserving;
pair with –min-score for abstention — see ir_08). strategy: conservative |
top_k | rel_threshold | score_gap.

### ir.cli.eval(name, cases, , mode='hybrid', k=10)

Score a built corpus’s retrieval against a DiscoveryCase JSONL file.

cases: path to a JSONL file of cases (see [`ir.eval`](ir.eval.html.md#module-ir.eval)); each line is a
`{"query": ..., "gold": [artifact_id, ...]}` record (empty `gold` = an
abstention case). Prints recall@k / NDCG@k / MRR / MAP plus the failure-mode
taxonomy. mode: dense | lexical | hybrid.

### ir.cli.eval_gen(name, out, , k=5, abstention_frac=0.15, max_artifacts=None)

Generate an eval-case file for a corpus by back-translation (needs aix/LLM).

Writes a DiscoveryCase JSONL set (gold cases + an abstention slice) for the
registered corpus *name* to *out*, stamping a corpus-signature into the
header so the frozen file can be checked against the live corpus later. This
command calls an LLM via aix; scoring it afterwards (`ir eval`) is offline.

### ir.cli.eval_select(name, cases, , strategy='conservative', mode='hybrid', k=10, max_k=3, rel=0.9, min_score=None)

Score a selector against a DiscoveryCase JSONL file (selection quality).

Reports the conditional commit rate (the selection decision isolated from
retrieval) plus selection precision / recall / F1 and abstention accuracy.
strategy: conservative | top_k | rel_threshold | score_gap. mode: dense |
lexical | hybrid. max_k / rel / min_score tune the commit (see `ir.select`);
`ir sweep-select` sweeps them to find good values.

### ir.cli.info(name)

Show a corpus’s stored config, stats, policy, and any abstention floors.

### ir.cli.ls()

List registered corpora with their kind, embedder, and record count.

### ir.cli.maintain(name=None, , all=False, dry_run=False)

Run due background work: incremental reindex, synopsis in its downtime window.

With a name, maintains that corpus; with –all (or no name), every registered
corpus. Idempotent and safe to schedule (cron/launchd): it no-ops what is not
due. –dry-run reports what would run without doing it.

### ir.cli.register(name, kind, , root=None, pattern=None, embedder='default', reindex_on=None, every_hours=None, synopsis=False)

Register a named corpus. kind: skills | packages | reports | files.

Background-work policy (optional; smart per-kind defaults otherwise — see
`ir.policy`): reindex_on (source-change | interval | manual), every_hours (for
interval), synopsis (enable LLM synopses, run only in the policy’s downtime
window by `ir maintain`).

### ir.cli.rm(name)

Unregister a corpus (does not delete its built data).

### ir.cli.schedule(, every=None, status=False, restart=False, remove=False, dry_run=False, backend=None)

Install, inspect, and operate the OS job that runs `ir maintain`.

Bare `ir schedule` is idempotent: it creates the schedule when there is
none, and when there already is one it reports it (interval, definition,
whether the OS has it loaded, last run) plus how to operate it, rather than
silently reinstalling over a schedule you tuned.

An existing definition’s environment is carried forward by every operation,
so operating a working schedule from a shell that happens to lack $PP never
silently re-points it at a different corpus store.

every: interval as 15m / 2h / 1d, or a plain number of minutes.
status: report only; never mutates.
restart: reload the definition and re-pin it to this interpreter.
remove: stop the schedule and delete its definition.
dry_run: print the definition that would be written, and write nothing.
backend: launchd | cron (default: whichever this machine provides).

### ir.cli.search(name, query, , k=10, mode='dense', fusion='rrf')

Search a built corpus and print the top-k hits.

mode: dense (cosine) | lexical (BM25) | hybrid (dense + BM25 fused). fusion
(hybrid only): rrf (rank-based, default) | blend (magnitude-preserving;
better abstention separability — see ir_08).

### ir.cli.sweep_select(name, cases, , strategy='conservative', mode='hybrid', k=10, objective='selection_f1')

Sweep selector knobs (max_k × rel) against a case file; print the grid + best.

Retrieves once per case and reuses the candidates across the whole grid, so
the sweep is cheap. Prints a table (one row per setting, best objective first)
and the winning max_k / rel. Use it to pick `ir.select` defaults empirically.
objective: selection_f1 | selection_precision | selection_recall |
conditional_commit_rate | mean_selected_size. mode: dense | lexical | hybrid.
