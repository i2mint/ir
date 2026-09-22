# ir.schedule

Install and operate the OS job that runs `ir maintain` — `ir schedule` (issue #75).

`ir maintain` is idempotent and cron-shaped, but something has to *call* it.
This module manages that caller: it writes a **definition** (a launchd plist or a
crontab line), hands it to the OS, and exits.

**\`\`ir\`\` still does not run a scheduler.** The boundary from issue #58 — “ir holds
the declarative policy as data and exposes an idempotent `ir maintain`; it does
not run a scheduler” — is preserved exactly. There is no daemon here, no event
loop, no in-process timer, and no orchestration import: launchd/cron remain the
executor, and this module is the *installer* for the crontab snippet
[`ir.maintenance`](ir.maintenance.md#module-ir.maintenance) already prescribes in its docstring. Turning that snippet
into a command is packaging, not a change of architecture.

The command is idempotent and tells you how to operate what it finds:

```default
ir schedule                # ensure a schedule exists; report + menu if one does
ir schedule --status       # report only, never mutates
ir schedule --every 30m    # set/change the interval
ir schedule --restart      # reload the definition, re-pinning the interpreter
ir schedule --remove       # stop it and delete the definition
ir schedule --dry-run      # print what would be written
```

**An existing definition is data, not a template to re-derive.** Only a *fresh*
install snapshots the calling shell’s environment. Every operation on a schedule
that already exists (`--every`, `--restart`) carries the stored environment
forward untouched, because the alternative silently re-points a working job at a
different corpus store the first time you operate it from a shell that happens to
lack `$PP` — which is the exact failure this module exists to prevent. Only
`--restart` re-pins the interpreter, and only because repairing a dead
interpreter is what it is for.

Two seams, both one keyword argument:

- `backend` — which OS scheduler holds the definition. Feature-detected
  (`launchctl` then `crontab`), never hardcoded per-platform; `systemd
  --user` timers are the documented next backend. A backend owns its own
  definition format, its own preview, and whether “loaded” means anything for it,
  so adding one touches [`BACKENDS`](#ir.schedule.BACKENDS) and nothing else.
- [`ScheduleSpec.args`](#ir.schedule.ScheduleSpec.args) — the command the job runs, defaulting to
  `-m ir maintain --all` pinned to the installing interpreter. An orchestration
  layer’s own entry point (#58 / ADR #43) drops in here without touching backends.

Everything else — the plist keys, the crontab line syntax, interval parsing,
output formatting — is written directly, on purpose.

### Module Attributes

| [`DFLT_LABEL`](#ir.schedule.DFLT_LABEL)            | Reverse-DNS job identifier.                                                                 |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`DFLT_EVERY_MINUTES`](#ir.schedule.DFLT_EVERY_MINUTES)    | How often the job fires, in minutes.                                                        |
| [`INHERITED_ENV_VARS`](#ir.schedule.INHERITED_ENV_VARS)    | Variables a job must inherit to see what the shell that installed it sees.                  |
| [`LOAD_BEARING_ENV_VARS`](#ir.schedule.LOAD_BEARING_ENV_VARS) | Of those, the ones whose absence breaks a corpus outright rather than merely relocating it. |
| [`BACKENDS`](#ir.schedule.BACKENDS)              | Backends in preference order.                                                               |

### Functions

| [`cron_expression`](#ir.schedule.cron_expression)(minutes)                          | The five-field cron time spec for an every-*minutes* schedule.            |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`cron_line`](#ir.schedule.cron_line)(spec)                                   | ir's crontab entry for *spec* — deliberately **one self-contained line**. |
| [`cron_marker`](#ir.schedule.cron_marker)(label)                                | The trailing comment that identifies ir's crontab line.                   |
| [`crontab_with_line`](#ir.schedule.crontab_with_line)(text, spec)                     | *text* with ir's line replaced by (or appended as) *spec*'s line.         |
| [`crontab_without_line`](#ir.schedule.crontab_without_line)(text[, label])               | *text* with ir's line removed (unchanged if there is none).               |
| [`ensure`](#ir.schedule.ensure)(\*[, every, backend, label])               | Make sure a schedule exists — the idempotent front door.                  |
| [`format_every`](#ir.schedule.format_every)(minutes)                             | Human rendering of an interval (`90` -> `'1h30m'`).                       |
| [`install`](#ir.schedule.install)(\*[, every, backend, label, python, ...]) | Install (or overwrite) the schedule, and return the resulting status.     |
| [`parse_every`](#ir.schedule.parse_every)(value)                                | Minutes from `"30m"` / `"2h"` / `"1d"` / `"45"` (`None` passes through).  |
| [`plist_bytes`](#ir.schedule.plist_bytes)(spec)                                 | *spec* rendered as launchd plist XML.                                     |
| [`plist_dict`](#ir.schedule.plist_dict)(spec)                                  | The launchd job description for *spec*, as a plist-ready dict.            |
| [`remove`](#ir.schedule.remove)(\*[, backend, label])                      | Stop the schedule and delete its definition.                              |
| [`render`](#ir.schedule.render)(state)                                     | A human report of *state*, including how to operate what was found.       |
| [`resolve_backend`](#ir.schedule.resolve_backend)([backend, require_available])     | The backend named *backend*, or the first available one.                  |
| [`restart`](#ir.schedule.restart)(\*[, backend, label])                     | Reload the existing definition, re-pinning it to the current interpreter. |
| [`status`](#ir.schedule.status)(\*[, backend, label])                      | Report the current schedule without changing anything.                    |

### Classes

| [`ScheduleSpec`](#ir.schedule.ScheduleSpec)([every_minutes, python, args, ...])   | What to run and how often — backend-independent.               |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`ScheduleStatus`](#ir.schedule.ScheduleStatus)(installed, backend[, label, ...])   | What the scheduler holds for ir, and what just happened to it. |

### Exceptions

| [`ScheduleError`](#ir.schedule.ScheduleError)   | A schedule could not be read, written, or expressed on this platform.   |
|------------------------------------------------------------------|-------------------------------------------------------------------------|

### ir.schedule.BACKENDS *= (<class 'ir.schedule._LaunchdBackend'>, <class 'ir.schedule._CronBackend'>)*

Backends in preference order. launchd first on macOS because a LaunchAgent
survives reboots and logs where the OS expects; cron is the POSIX fallback and
also works on macOS if asked for by name.

This tuple is the whole seam. A backend owns its definition format, its
`preview`, and its `reports_loaded` answer, so adding `systemd --user`
means adding a class here and nothing else — no branch in the renderer, the
previewer, or the CLI.

### ir.schedule.DFLT_EVERY_MINUTES *= 60*

How often the job fires, in minutes. Hourly rather than the docstring
example’s 15 minutes: a `source-change` rebuild is a near-no-op only *after*
it has walked the sources, and a large corpus (tens of thousands of records)
makes that walk worth an hour’s spacing. `--every` moves it either way.

### ir.schedule.DFLT_LABEL *= 'com.i2mint.ir.maintain'*

Reverse-DNS job identifier. Also the launchd `Label` and the token that marks
ir’s line in the user crontab, so both backends find their own work and never
touch anybody else’s.

### ir.schedule.INHERITED_ENV_VARS *= ('IR_CONFIG_DIR', 'IR_DATA_DIR', 'IR_CACHE_DIR', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'PP', 'PTH_FILEPATH')*

Variables a job must inherit to see what the shell that installed it sees.
launchd starts jobs with a near-empty environment and cron with a minimal one,
so anything ir reads from the environment has to be carried into the
definition or the job maintains a *different* (or empty) set of corpora.

`IR_*`/`XDG_*` locate the stores ([`ir.config`](ir.config.md#module-ir.config)); `PP` and
`PTH_FILEPATH` are read by [`ir.sources`](ir.sources.md#module-ir.sources) itself to find the package
manifest and the projects root, and without them the `packages` and
`reports` corpora fail outright — which is precisely how a scheduled job
ends up “running fine” while two of three corpora go stale.

### ir.schedule.LOAD_BEARING_ENV_VARS *= ('PP', 'PTH_FILEPATH')*

Of those, the ones whose absence breaks a corpus outright rather than merely
relocating it. [`status()`](#ir.schedule.status) reports a job that lacks one the current shell
has, because that job is failing silently right now.

### *exception* ir.schedule.ScheduleError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

A schedule could not be read, written, or expressed on this platform.

### *class* ir.schedule.ScheduleSpec(every_minutes=60, python=<factory>, args=('-u', '-m', 'ir', 'maintain', '--all'), label='com.i2mint.ir.maintain', log_path=<factory>, env=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What to run and how often — backend-independent.

`python` is pinned to an absolute interpreter path (by default the one
installing the schedule) because neither launchd nor cron inherits the shell’s
`PATH`-resolved `ir`. Pinning is also what makes staleness *detectable*:
[`status()`](#ir.schedule.status) can check the recorded interpreter still exists.

#### args *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]* *= ('-u', '-m', 'ir', 'maintain', '--all')*

Python block-buffers stdout when it is not a tty, so
without it a scheduled run’s log stays empty until the process exits —
exactly when you most want to watch a slow rebuild.

* **Type:**
  `-u` matters

#### *property* command *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

The argv the scheduler executes.

#### *property* env_dict *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

The job environment as a plain dict.

### *class* ir.schedule.ScheduleStatus(installed, backend, label='com.i2mint.ir.maintain', action='reported', every_minutes=None, definition=None, command=None, python=None, env=(), log_path=None, loaded=None, last_run=None, last_log_line=None, problems=(), detail='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What the scheduler holds for ir, and what just happened to it.

One return type for every operation: `action` says what the call did, so a
caller never has to pair a status with a separate “did it change?” flag.

#### *property* changed *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether this call actually mutated the schedule.

#### to_dict()

JSON-ready view (the shape a non-CLI surface would consume).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.schedule.cron_expression(minutes)

The five-field cron time spec for an every-*minutes* schedule.

cron restarts its `*/n` count each hour (and each day), so only intervals
that divide evenly fire evenly. Rejecting the rest with the valid values named
beats installing a job that quietly skips.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> cron_expression(15), cron_expression(60), cron_expression(360)
('*/15 * * * *', '0 * * * *', '0 */6 * * *')
```

### ir.schedule.cron_line(spec)

ir’s crontab entry for *spec* — deliberately **one self-contained line**.

An earlier design used a begin/end marker block with the environment on its
own `NAME=value` crontab lines. Both halves of that were wrong:

- a block has an *interior*, and anything that lands in it (a job the user
  typed just above the end marker, or everything after a begin marker whose
  end marker got lost) is deleted the next time ir rewrites its block;
- crontab-scope `NAME=value` lines apply to \*\*every command after them in
  the file\*\*, so ir’s `HOME` and `PATH` would silently attach themselves
  to whatever job the user appends next — and detach again the next time ir
  moved its block. A cron job whose environment depends on when you last ran
  `ir schedule` is close to undiagnosable.

One line with an inline `env` prefix has no interior to corrupt and no
scope beyond itself. `%` is escaped because cron reads an unescaped one as
a newline that terminates the command.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.schedule.cron_marker(label)

The trailing comment that identifies ir’s crontab line.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.schedule.crontab_with_line(text, spec)

*text* with ir’s line replaced by (or appended as) *spec*’s line.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.schedule.crontab_without_line(text, label='com.i2mint.ir.maintain')

*text* with ir’s line removed (unchanged if there is none).

Only lines ir itself wrote are dropped, identified by the trailing marker.
Nothing else in the file is read, moved, or rewritten.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.schedule.ensure(, every=None, backend=None, label='com.i2mint.ir.maintain')

Make sure a schedule exists — the idempotent front door.

Installs when there is none. When one already exists it changes nothing
unless *every* differs, so running it twice is safe and running it out of
habit never silently reinstalls over a schedule you tuned.

* **Return type:**
  [`ScheduleStatus`](#ir.schedule.ScheduleStatus)

### ir.schedule.format_every(minutes)

Human rendering of an interval (`90` -> `'1h30m'`).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.schedule.install(, every=None, backend=None, label='com.i2mint.ir.maintain', python=None, inherit=True, dry_run=False)

Install (or overwrite) the schedule, and return the resulting status.

*inherit* carries an existing definition’s environment, interpreter and log
path forward; pass `False` to re-snapshot them from this process.

* **Return type:**
  [`ScheduleStatus`](#ir.schedule.ScheduleStatus)

### ir.schedule.parse_every(value)

Minutes from `"30m"` / `"2h"` / `"1d"` / `"45"` (`None` passes through).

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> parse_every("30m"), parse_every("2h"), parse_every("1d"), parse_every(45)
(30, 120, 1440, 45)
```

### ir.schedule.plist_bytes(spec)

*spec* rendered as launchd plist XML.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### ir.schedule.plist_dict(spec)

The launchd job description for *spec*, as a plist-ready dict.

`StartInterval` (seconds) and `EnvironmentVariables` are what make the
interval and the job environment readable back out of an installed job, so
[`status()`](#ir.schedule.status) needs no separate bookkeeping file.

`ProcessType` is deliberately `Standard` rather than `Background`:
launchd throttles a `Background` job’s CPU and I/O to keep it from
disrupting the user, and a corpus rebuild is exactly the file-walking,
embedding-heavy work that throttling would stretch past its own interval.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.schedule.remove(, backend=None, label='com.i2mint.ir.maintain')

Stop the schedule and delete its definition.

* **Return type:**
  [`ScheduleStatus`](#ir.schedule.ScheduleStatus)

### ir.schedule.render(state)

A human report of *state*, including how to operate what was found.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.schedule.resolve_backend(backend=None, , require_available=True)

The backend named *backend*, or the first available one.

Naming one this machine cannot run is an error rather than a half-install:
`--backend launchd` on Linux would otherwise write a plist into a directory
nothing reads. Pass `require_available=False` to obtain a backend purely to
render its definition format.

```pycon
>>> resolve_backend("cron", require_available=False).name
'cron'
```

### ir.schedule.restart(, backend=None, label='com.i2mint.ir.maintain')

Reload the existing definition, re-pinning it to the current interpreter.

This is the repair for a schedule left pointing at a rebuilt or deleted
interpreter. The stored environment is carried forward untouched — restarting
from a shell that lacks `$PP` must not quietly re-point a working job at a
different (or empty) set of corpora.

* **Return type:**
  [`ScheduleStatus`](#ir.schedule.ScheduleStatus)

### ir.schedule.status(, backend=None, label='com.i2mint.ir.maintain')

Report the current schedule without changing anything.

* **Return type:**
  [`ScheduleStatus`](#ir.schedule.ScheduleStatus)
