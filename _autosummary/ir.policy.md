# ir.policy

Per-corpus policy — how a corpus is segmented/stored and what background work it gets.

This is the *declarative* half of ir’s maintenance story (issue #58): a corpus’s
policy lives as **data** in its registry entry, and ir resolves an effective
policy by layering `entry` over per-`kind` defaults over a global default.
An idempotent [`ir.maintenance.maintain()`](ir.maintenance.md#ir.maintenance.maintain) reads the policy and does the due
work; *scheduling* that work (cron / launchd, or an orchestration layer’s budget
governor) stays **external** — ir never imports an orchestration layer, and there
is no global `Settings` singleton (policy is per-corpus data, injected, not a
process-wide mutable). The light path stays a one-liner: every field has a smart
default, so `ir build skills` needs no policy at all.

The three policy axes:

- **reindex** — *when* to rebuild: `"source-change"` (default; rebuild is a
  near-no-op when nothing changed, so it is always safe to run), `"interval"`
  (only when older than `every_hours`), or `"manual"` (never automatic).
- **synopsis** — *whether/when* to attach LLM synopses (the expensive,
  off-by-default work): only the `recent` slice, only during `downtime_hours`.
  Synopsis is realized as a strategy wrapper ([`ir.with_synopsis()`](ir.md#ir.with_synopsis)), so it is
  incremental — only new/changed artifacts are synthesized.
- **storage** — the persistence backend. Today only `"local"` (the file store);
  a `vd.Collection` backend is the documented future seam (issue #28).

### Module Attributes

| [`REINDEX_TRIGGERS`](#ir.policy.REINDEX_TRIGGERS)   | Accepted `reindex.on` triggers.                                   |
|---------------------------------------------------------------------|-------------------------------------------------------------------|
| [`STORAGE_BACKENDS`](#ir.policy.STORAGE_BACKENDS)   | Storage backends ir can resolve today.                            |
| [`GLOBAL_DEFAULT`](#ir.policy.GLOBAL_DEFAULT)     | The global fallback policy (any kind without a specific default). |
| [`DEFAULTS_BY_KIND`](#ir.policy.DEFAULTS_BY_KIND)   | Per-kind maintenance defaults.                                    |

### Functions

| [`default_policy_for_kind`](#ir.policy.default_policy_for_kind)(kind)                | The smart-default [`MaintenancePolicy`](#ir.policy.MaintenancePolicy) for a corpus `kind`.   |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [`in_downtime`](#ir.policy.in_downtime)(policy, now)                     | Whether `now` falls inside the synopsis `downtime_hours` window.                                            |
| [`is_reindex_due`](#ir.policy.is_reindex_due)(policy, last_maintained, now) | Whether a reindex is due under `policy` given the last-maintained time.                                     |
| [`resolve_policy`](#ir.policy.resolve_policy)(entry)                        | The effective policy for a registry `entry`: entry over kind over global.                                   |
| [`resolve_storage`](#ir.policy.resolve_storage)(entry)                       | The effective storage spec for an `entry` (default `{"backend": "local"}`).                                 |

### Classes

| [`MaintenancePolicy`](#ir.policy.MaintenancePolicy)([reindex, synopsis])   | The background-work policy for one corpus (reindex + synopsis).   |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`ReindexPolicy`](#ir.policy.ReindexPolicy)([on, every_hours])         | When to (incrementally) rebuild a corpus.                         |
| [`SynopsisPolicy`](#ir.policy.SynopsisPolicy)([enabled, scope, ...])    | Whether/when to attach (expensive, LLM-generated) synopses.       |

### ir.policy.DEFAULTS_BY_KIND *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)]* *= {'files': {'reindex': {'on': 'source-change'}}, 'packages': {'reindex': {'on': 'source-change'}}, 'records': {'reindex': {'every_hours': 24, 'on': 'interval'}}, 'reports': {'reindex': {'on': 'source-change'}}, 'sessions': {'reindex': {'every_hours': 24, 'on': 'interval'}, 'synopsis': {'downtime_hours': [2, 6], 'enabled': False, 'scope': 'recent', 'window_days': 30}}, 'skills': {'reindex': {'on': 'source-change'}}}*

Per-kind maintenance defaults. The rule of thumb encoded here:
*small, fully-enumerable* corpora (skills/packages/reports/files) rebuild on
source change (cheap, exact); *large, append-mostly, time-stamped* corpora
(sessions) rebuild on an interval and keep synopsis off by default, with a
downtime window ready for when it is turned on. New kinds register here.

### ir.policy.GLOBAL_DEFAULT *= MaintenancePolicy(reindex=ReindexPolicy(on='source-change', every_hours=None), synopsis=SynopsisPolicy(enabled=False, scope='recent', window_days=30, downtime_hours=None))*

The global fallback policy (any kind without a specific default).

### *class* ir.policy.MaintenancePolicy(reindex=<factory>, synopsis=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The background-work policy for one corpus (reindex + synopsis).

#### merged(override)

Layer an `override` dict on top of this policy (entry over defaults).

* **Return type:**
  [`MaintenancePolicy`](#ir.policy.MaintenancePolicy)

### ir.policy.REINDEX_TRIGGERS *= ('source-change', 'interval', 'manual')*

Accepted `reindex.on` triggers.

### *class* ir.policy.ReindexPolicy(on='source-change', every_hours=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

When to (incrementally) rebuild a corpus.

### ir.policy.STORAGE_BACKENDS *= ('local',)*

Storage backends ir can resolve today. `"vd"` (vd.Collection) is the
documented future backend, gated on issue #28.

### *class* ir.policy.SynopsisPolicy(enabled=False, scope='recent', window_days=30, downtime_hours=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Whether/when to attach (expensive, LLM-generated) synopses.

`downtime_hours` is a `[start, end)` pair of local-clock hours
(wrapping past midnight is allowed, e.g. `(22, 6)`); `None` means
“any time”. `scope="recent"` limits synthesis to artifacts whose
timestamp is within `window_days` (corpora that expose a time signal);
`scope="all"` synthesizes every artifact (bounded by incrementality).

### ir.policy.default_policy_for_kind(kind)

The smart-default [`MaintenancePolicy`](#ir.policy.MaintenancePolicy) for a corpus `kind`.

* **Return type:**
  [`MaintenancePolicy`](#ir.policy.MaintenancePolicy)

### ir.policy.in_downtime(policy, now)

Whether `now` falls inside the synopsis `downtime_hours` window.

`None` window means “any time”. A window whose start hour is greater than
its end hour wraps past midnight (e.g. `(22, 6)` is 22:00–06:00).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ir.policy.is_reindex_due(policy, last_maintained, now)

Whether a reindex is due under `policy` given the last-maintained time.

- `source-change` is always due (the build is incremental and a no-op when
  nothing changed, so running it cannot do harm).
- `interval` is due when never maintained or older than `every_hours`.
- `manual` is never automatically due.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ir.policy.resolve_policy(entry)

The effective policy for a registry `entry`: entry over kind over global.

A v1 entry (no `maintenance` key) resolves to its kind’s smart default, so
existing corpora gain a sensible policy without a migration.

* **Return type:**
  [`MaintenancePolicy`](#ir.policy.MaintenancePolicy)

### ir.policy.resolve_storage(entry)

The effective storage spec for an `entry` (default `{"backend": "local"}`).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
