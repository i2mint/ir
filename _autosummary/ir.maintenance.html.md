# ir.maintenance

Idempotent background-work runner — `ir maintain` (issue #58).

The *executing* half of ir’s maintenance story (the declarative half is
[`ir.policy`](ir.policy.html.md#module-ir.policy)). [`maintain()`](#ir.maintenance.maintain) reads each corpus’s resolved
[`MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy) and does the work that is **due**:

- an **incremental rebuild** when reindex is due (`source-change` is always
  due — the build is a near-no-op when nothing changed; `interval` only when
  stale; `manual` never);
- when `synopsis.enabled`, the rebuild’s strategy is wrapped in
  [`ir.with_synopsis()`](ir.html.md#ir.with_synopsis), so new / changed artifacts gain an LLM synopsis —
  > and because that build may call an LLM, it is run \*\*only inside the policy’s
  > downtime window\*\*.

It is safe to call as often as a scheduler likes: it no-ops when nothing is due,
and records `last_maintained` so interval policies converge. ir runs the work;
*scheduling* it is external — a cron / launchd entry calls `ir maintain --all`
every N minutes, and the downtime window lives in the policy (data), not in ir.

[`ir.schedule`](ir.schedule.html.md#module-ir.schedule) installs that entry for you (`ir schedule`) and reports on it;
it writes a definition and exits, so the executor is still cron / launchd:

```default
# what `ir schedule --every 15m` writes on a cron machine:
*/15 * * * *  <python> -m ir maintain --all >> ~/.cache/ir/maintain.log 2>&1
```

Because a scheduled run has nobody watching it, [`maintain()`](#ir.maintenance.maintain) is fault-isolated
per corpus: one corpus failing is recorded on its own result and the sweep carries
on, rather than aborting and leaving the corpora after it silently unmaintained.

### Module Attributes

| [`DFLT_STALE_LOCK_AFTER`](#ir.maintenance.DFLT_STALE_LOCK_AFTER)   | A lock older than this is assumed to belong to a run that was killed rather than one still working.   |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|

### Functions

| [`lock_path`](#ir.maintenance.lock_path)()                                     | Where the single-run lock lives (regenerable, so the cache dir).                                      |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [`maintain`](#ir.maintenance.maintain)([name, all, now, dry_run])             | Run due background work for one corpus (*name*) or every registered one (*all*).                      |
| [`maintain_corpus`](#ir.maintenance.maintain_corpus)(name, \*[, now, dry_run, full]) | Do the due background work for one corpus (idempotent).                                               |
| [`single_run`](#ir.maintenance.single_run)(\*[, path, stale_after])             | Hold the maintenance lock, or raise [`MaintenanceBusy`](#ir.maintenance.MaintenanceBusy). |

### Classes

| [`MaintenanceResult`](#ir.maintenance.MaintenanceResult)(name, ran, reason[, ...])   | What [`maintain_corpus()`](#ir.maintenance.maintain_corpus) did (or would do) for one corpus.   |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|

### Exceptions

| [`MaintenanceBusy`](#ir.maintenance.MaintenanceBusy)   | Another maintenance run holds the lock.   |
|--------------------------------------------------------------------|-------------------------------------------|

### ir.maintenance.DFLT_STALE_LOCK_AFTER *= datetime.timedelta(seconds=21600)*

A lock older than this is assumed to belong to a run that was killed rather
than one still working. Generous, because a first full build of a large corpus
legitimately takes a long time and reclaiming a *live* lock is the bad outcome.

### *exception* ir.maintenance.MaintenanceBusy

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

Another maintenance run holds the lock.

### *class* ir.maintenance.MaintenanceResult(name, ran, reason, reindex=False, synopsis=False, records=None, error=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What [`maintain_corpus()`](#ir.maintenance.maintain_corpus) did (or would do) for one corpus.

### ir.maintenance.lock_path()

Where the single-run lock lives (regenerable, so the cache dir).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.maintenance.maintain(name=None, , all=False, now=None, dry_run=False)

Run due background work for one corpus (*name*) or every registered one (*all*).

With neither, defaults to all registered corpora. Returns one
[`MaintenanceResult`](#ir.maintenance.MaintenanceResult) per corpus considered.

**A named corpus raises; a sweep records.** Asking for one corpus by name is a
direct request, and swallowing its error would turn a typo’d name into a
result that reads as success to any caller written before `error` existed.
A sweep is the unattended case, where the opposite is true: one corpus
failing must not leave every corpus after it stale with nothing to say so.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`MaintenanceResult`](#ir.maintenance.MaintenanceResult)]

### ir.maintenance.maintain_corpus(name, , now=None, dry_run=False, full=True)

Do the due background work for one corpus (idempotent).

Reads the corpus’s resolved policy and its `last_maintained` time, decides
whether a (synopsis-aware) reindex is due *and* permitted now, and — unless
`dry_run` — runs the incremental build and records the run.

* **Return type:**
  [`MaintenanceResult`](#ir.maintenance.MaintenanceResult)

### ir.maintenance.single_run(, path=None, stale_after=None)

Hold the maintenance lock, or raise [`MaintenanceBusy`](#ir.maintenance.MaintenanceBusy).

Two maintenance runs on one corpus store are not safe: the packed store
rewrites `matrix`/`ids`/`metas` as separate files, so interleaved
writers can leave a matrix whose rows no longer line up with its ids — an
index that answers confidently and wrongly, with nothing raised. launchd
already runs one instance per label, but that does not cover cron (which
happily stacks runs) or the manual `ir maintain --all` this tool’s own
menu suggests while the agent may be mid-run.

A lock whose owning process is gone, or which is older than *stale_after*, is
reclaimed — a killed run must not wedge maintenance forever.
