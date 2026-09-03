"""Install and operate the OS job that runs ``ir maintain`` — ``ir schedule`` (issue #75).

``ir maintain`` is idempotent and cron-shaped, but something has to *call* it.
This module manages that caller: it writes a **definition** (a launchd plist or a
crontab line), hands it to the OS, and exits.

**``ir`` still does not run a scheduler.** The boundary from issue #58 — "ir holds
the declarative policy as data and exposes an idempotent ``ir maintain``; it does
not run a scheduler" — is preserved exactly. There is no daemon here, no event
loop, no in-process timer, and no orchestration import: launchd/cron remain the
executor, and this module is the *installer* for the crontab snippet
:mod:`ir.maintenance` already prescribes in its docstring. Turning that snippet
into a command is packaging, not a change of architecture.

The command is idempotent and tells you how to operate what it finds::

    ir schedule                # ensure a schedule exists; report + menu if one does
    ir schedule --status       # report only, never mutates
    ir schedule --every 30m    # set/change the interval
    ir schedule --restart      # reload the definition, re-pinning the interpreter
    ir schedule --remove       # stop it and delete the definition
    ir schedule --dry-run      # print what would be written

**An existing definition is data, not a template to re-derive.** Only a *fresh*
install snapshots the calling shell's environment. Every operation on a schedule
that already exists (``--every``, ``--restart``) carries the stored environment
forward untouched, because the alternative silently re-points a working job at a
different corpus store the first time you operate it from a shell that happens to
lack ``$PP`` — which is the exact failure this module exists to prevent. Only
``--restart`` re-pins the interpreter, and only because repairing a dead
interpreter is what it is for.

Two seams, both one keyword argument:

- ``backend`` — which OS scheduler holds the definition. Feature-detected
  (``launchctl`` then ``crontab``), never hardcoded per-platform; ``systemd
  --user`` timers are the documented next backend. A backend owns its own
  definition format, its own preview, and whether "loaded" means anything for it,
  so adding one touches :data:`BACKENDS` and nothing else.
- :attr:`ScheduleSpec.args` — the command the job runs, defaulting to
  ``-m ir maintain --all`` pinned to the installing interpreter. An orchestration
  layer's own entry point (#58 / ADR #43) drops in here without touching backends.

Everything else — the plist keys, the crontab line syntax, interval parsing,
output formatting — is written directly, on purpose.
"""

from __future__ import annotations

import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import cache_dir

#: Reverse-DNS job identifier. Also the launchd ``Label`` and the token that marks
#: ir's line in the user crontab, so both backends find their own work and never
#: touch anybody else's.
DFLT_LABEL = "com.i2mint.ir.maintain"

#: How often the job fires, in minutes. Hourly rather than the docstring
#: example's 15 minutes: a ``source-change`` rebuild is a near-no-op only *after*
#: it has walked the sources, and a large corpus (tens of thousands of records)
#: makes that walk worth an hour's spacing. ``--every`` moves it either way.
DFLT_EVERY_MINUTES = 60

#: Variables a job must inherit to see what the shell that installed it sees.
#: launchd starts jobs with a near-empty environment and cron with a minimal one,
#: so anything ir reads from the environment has to be carried into the
#: definition or the job maintains a *different* (or empty) set of corpora.
#:
#: ``IR_*``/``XDG_*`` locate the stores (:mod:`ir.config`); ``PP`` and
#: ``PTH_FILEPATH`` are read by :mod:`ir.sources` itself to find the package
#: manifest and the projects root, and without them the ``packages`` and
#: ``reports`` corpora fail outright — which is precisely how a scheduled job
#: ends up "running fine" while two of three corpora go stale.
INHERITED_ENV_VARS = (
    "IR_CONFIG_DIR",
    "IR_DATA_DIR",
    "IR_CACHE_DIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "PP",
    "PTH_FILEPATH",
)

#: Of those, the ones whose absence breaks a corpus outright rather than merely
#: relocating it. :func:`status` reports a job that lacks one the current shell
#: has, because that job is failing silently right now.
LOAD_BEARING_ENV_VARS = ("PP", "PTH_FILEPATH")

#: Minute intervals that map to a clean cron expression (divisors of 60), and
#: hour intervals that do (divisors of 24). launchd accepts any interval; cron's
#: ``*/n`` restarts each hour, so a non-divisor would fire unevenly.
_CRON_MINUTE_STEPS = tuple(m for m in range(1, 60) if 60 % m == 0)
_CRON_HOUR_STEPS = tuple(h for h in range(1, 24) if 24 % h == 0)

_UNITS = {"m": 1, "h": 60, "d": 1440}

#: Exit status for "the tool itself is not here" (the shell convention). A missing
#: binary has to look like a failed command rather than an exception: naming a
#: backend is not the same as being on a machine that has it, and asking about one
#: that is absent must report rather than crash.
_NO_SUCH_TOOL = 127

#: How much of the job's log to read when reporting its last line. The log is
#: append-only and unbounded; reading it whole to get one line would grow into a
#: real cost on a machine that has been maintaining hourly for a year.
_LOG_TAIL_BYTES = 8192


class ScheduleError(Exception):
    """A schedule could not be read, written, or expressed on this platform."""


# --------------------------------------------------------------------------- #
# Data model (pure; no I/O beyond resolving default paths)
# --------------------------------------------------------------------------- #


def _job_env() -> tuple[tuple[str, str], ...]:
    """The environment a scheduled job needs, as sorted ``(name, value)`` pairs."""
    env: dict[str, str] = {"HOME": str(Path.home())}
    for name in ("PATH", *INHERITED_ENV_VARS):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return tuple(sorted(env.items()))


@dataclass(frozen=True)
class ScheduleSpec:
    """What to run and how often — backend-independent.

    ``python`` is pinned to an absolute interpreter path (by default the one
    installing the schedule) because neither launchd nor cron inherits the shell's
    ``PATH``-resolved ``ir``. Pinning is also what makes staleness *detectable*:
    :func:`status` can check the recorded interpreter still exists.
    """

    every_minutes: int = DFLT_EVERY_MINUTES
    python: str = field(default_factory=lambda: sys.executable)
    #: ``-u`` matters: Python block-buffers stdout when it is not a tty, so
    #: without it a scheduled run's log stays empty until the process exits —
    #: exactly when you most want to watch a slow rebuild.
    args: tuple[str, ...] = ("-u", "-m", "ir", "maintain", "--all")
    label: str = DFLT_LABEL
    log_path: str = field(default_factory=lambda: str(cache_dir() / "maintain.log"))
    env: tuple[tuple[str, str], ...] = field(default_factory=_job_env)

    def __post_init__(self):
        if self.every_minutes < 1:
            raise ScheduleError("every_minutes must be at least 1")
        for name, value in self.env:
            # A newline in an environment value would inject arbitrary crontab
            # lines; cron additionally turns an unescaped % into a newline.
            if "\n" in value or "\r" in value or "\n" in name:
                raise ScheduleError(
                    f"environment variable {name!r} contains a newline and cannot "
                    "be written into a schedule definition"
                )

    @property
    def command(self) -> list[str]:
        """The argv the scheduler executes."""
        return [self.python, *self.args]

    @property
    def env_dict(self) -> dict[str, str]:
        """The job environment as a plain dict."""
        return dict(self.env)


@dataclass(frozen=True)
class ScheduleStatus:
    """What the scheduler holds for ir, and what just happened to it.

    One return type for every operation: ``action`` says what the call did, so a
    caller never has to pair a status with a separate "did it change?" flag.
    """

    installed: bool
    backend: str
    label: str = DFLT_LABEL
    action: str = "reported"
    every_minutes: int | None = None
    definition: str | None = None
    command: str | None = None
    python: str | None = None
    env: tuple[tuple[str, str], ...] = ()
    log_path: str | None = None
    loaded: bool | None = None
    last_run: str | None = None
    last_log_line: str | None = None
    problems: tuple[str, ...] = ()
    detail: str = ""

    @property
    def changed(self) -> bool:
        """Whether this call actually mutated the schedule."""
        return self.action in ("installed", "updated", "removed", "restarted")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view (the shape a non-CLI surface would consume)."""
        return {
            "installed": self.installed,
            "backend": self.backend,
            "label": self.label,
            "action": self.action,
            "every_minutes": self.every_minutes,
            "definition": self.definition,
            "command": self.command,
            "python": self.python,
            "env": dict(self.env),
            "log_path": self.log_path,
            "loaded": self.loaded,
            "last_run": self.last_run,
            "last_log_line": self.last_log_line,
            "problems": list(self.problems),
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# Interval parsing and rendering (pure)
# --------------------------------------------------------------------------- #


def parse_every(value: str | int | None) -> int | None:
    """Minutes from ``"30m"`` / ``"2h"`` / ``"1d"`` / ``"45"`` (``None`` passes through).

    >>> parse_every("30m"), parse_every("2h"), parse_every("1d"), parse_every(45)
    (30, 120, 1440, 45)
    """
    if value is None:
        return None
    if isinstance(value, int):
        minutes = value
    else:
        text = str(value).strip().lower()
        if not text:
            raise ScheduleError("empty interval; try 15m, 1h, or 1d")
        unit = _UNITS.get(text[-1])
        number, factor = (text[:-1], unit) if unit else (text, 1)
        try:
            minutes = int(number) * factor
        except ValueError:
            raise ScheduleError(
                f"cannot read interval {value!r}; expected e.g. 15m, 2h, 1d, or a "
                "plain number of minutes"
            ) from None
    if minutes < 1:
        raise ScheduleError(f"interval must be at least 1 minute, got {minutes}")
    return minutes


def cron_expression(minutes: int) -> str:
    """The five-field cron time spec for an every-*minutes* schedule.

    cron restarts its ``*/n`` count each hour (and each day), so only intervals
    that divide evenly fire evenly. Rejecting the rest with the valid values named
    beats installing a job that quietly skips.

    >>> cron_expression(15), cron_expression(60), cron_expression(360)
    ('*/15 * * * *', '0 * * * *', '0 */6 * * *')
    """
    if minutes < 1:
        raise ScheduleError("interval must be at least 1 minute")
    if minutes < 60:
        if 60 % minutes:
            raise ScheduleError(
                f"cron cannot fire evenly every {minutes} minutes; use one of "
                f"{', '.join(f'{m}m' for m in _CRON_MINUTE_STEPS)} "
                "(or the launchd backend, which accepts any interval)"
            )
        return f"*/{minutes} * * * *"
    if minutes == 60:
        return "0 * * * *"
    if minutes == 1440:
        return "0 0 * * *"
    if minutes % 60 == 0 and (hours := minutes // 60) < 24:
        if 24 % hours:
            raise ScheduleError(
                f"cron cannot fire evenly every {hours} hours; use one of "
                f"{', '.join(f'{h}h' for h in _CRON_HOUR_STEPS)}"
            )
        return f"0 */{hours} * * *"
    raise ScheduleError(
        f"cron cannot express an every-{minutes}-minute schedule; use a divisor of "
        "an hour (e.g. 15m), a divisor of a day (e.g. 6h), or 1d"
    )


def format_every(minutes: int | None) -> str:
    """Human rendering of an interval (``90`` -> ``'1h30m'``)."""
    if minutes is None or minutes < 1:
        return "unknown"
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}h{mins}m"
    return f"{hours}h" if hours else f"{mins}m"


# --------------------------------------------------------------------------- #
# launchd definition rendering (pure)
# --------------------------------------------------------------------------- #


def plist_dict(spec: ScheduleSpec) -> dict[str, Any]:
    """The launchd job description for *spec*, as a plist-ready dict.

    ``StartInterval`` (seconds) and ``EnvironmentVariables`` are what make the
    interval and the job environment readable back out of an installed job, so
    :func:`status` needs no separate bookkeeping file.

    ``ProcessType`` is deliberately ``Standard`` rather than ``Background``:
    launchd throttles a ``Background`` job's CPU and I/O to keep it from
    disrupting the user, and a corpus rebuild is exactly the file-walking,
    embedding-heavy work that throttling would stretch past its own interval.
    """
    return {
        "Label": spec.label,
        "ProgramArguments": list(spec.command),
        "StartInterval": spec.every_minutes * 60,
        "RunAtLoad": False,
        "EnvironmentVariables": spec.env_dict,
        "StandardOutPath": spec.log_path,
        "StandardErrorPath": spec.log_path,
        "ProcessType": "Standard",
        # launchd starts jobs in "/"; a relative path in a source resolver would
        # then behave differently under the agent than in the shell.
        "WorkingDirectory": spec.env_dict.get("HOME", str(Path.home())),
    }


def plist_bytes(spec: ScheduleSpec) -> bytes:
    """*spec* rendered as launchd plist XML."""
    return plistlib.dumps(plist_dict(spec))


# --------------------------------------------------------------------------- #
# cron definition rendering and parsing (pure)
# --------------------------------------------------------------------------- #


def cron_marker(label: str) -> str:
    """The trailing comment that identifies ir's crontab line."""
    return f"# {label}"


def cron_line(spec: ScheduleSpec) -> str:
    """ir's crontab entry for *spec* — deliberately **one self-contained line**.

    An earlier design used a begin/end marker block with the environment on its
    own ``NAME=value`` crontab lines. Both halves of that were wrong:

    - a block has an *interior*, and anything that lands in it (a job the user
      typed just above the end marker, or everything after a begin marker whose
      end marker got lost) is deleted the next time ir rewrites its block;
    - crontab-scope ``NAME=value`` lines apply to **every command after them in
      the file**, so ir's ``HOME`` and ``PATH`` would silently attach themselves
      to whatever job the user appends next — and detach again the next time ir
      moved its block. A cron job whose environment depends on when you last ran
      ``ir schedule`` is close to undiagnosable.

    One line with an inline ``env`` prefix has no interior to corrupt and no
    scope beyond itself. ``%`` is escaped because cron reads an unescaped one as
    a newline that terminates the command.
    """
    parts = [_ENV_TOOL, *(f"{name}={value}" for name, value in spec.env), *spec.command]
    command = " ".join(shlex.quote(part) for part in parts)
    redirect = f">> {shlex.quote(spec.log_path)} 2>&1"
    line = (
        f"{cron_expression(spec.every_minutes)} {command} {redirect} "
        f"{cron_marker(spec.label)} every={spec.every_minutes}"
    )
    return line.replace("%", r"\%")


def _is_our_line(line: str, label: str) -> bool:
    """Whether *line* is the crontab entry ir manages."""
    return cron_marker(label) in line and not line.lstrip().startswith("#")


def crontab_without_line(text: str, label: str = DFLT_LABEL) -> str:
    """*text* with ir's line removed (unchanged if there is none).

    Only lines ir itself wrote are dropped, identified by the trailing marker.
    Nothing else in the file is read, moved, or rewritten.
    """
    kept = [line for line in _split_lines(text) if not _is_our_line(line, label)]
    body = "\n".join(kept).strip("\n")
    return f"{body}\n" if body else ""


def crontab_with_line(text: str, spec: ScheduleSpec) -> str:
    """*text* with ir's line replaced by (or appended as) *spec*'s line."""
    body = crontab_without_line(text, spec.label).rstrip("\n")
    parts = [p for p in (body, cron_line(spec)) if p]
    return "\n".join(parts) + "\n"


def _split_lines(text: str) -> list[str]:
    """Split crontab *text* on real line terminators only.

    ``str.splitlines`` also breaks on ``\\x0b``, ``\\x0c``, ``\\x1c`` and
    ``U+2028``, none of which end a line in a crontab — a command containing one
    would be silently rewritten as two entries.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _find_our_line(text: str, label: str) -> str | None:
    """ir's crontab line, or ``None``."""
    for line in _split_lines(text):
        if _is_our_line(line, label):
            return line
    return None


def _parse_cron_line(line: str, label: str) -> dict[str, Any]:
    """Read interval, command, interpreter, env and log path back out of *line*."""
    line = line.replace(r"\%", "%")
    head, _, tail = line.rpartition(cron_marker(label))
    every = None
    match = re.search(r"every=(\d+)", tail)
    if match:
        every = int(match.group(1)) or None
    fields = head.strip().split(None, 5)
    command = fields[5].strip() if len(fields) == 6 else None
    env: dict[str, str] = {}
    python = log_path = None
    if command:
        try:
            tokens = shlex.split(command)
        except ValueError:
            # An unbalanced quote (a hand-added trailing comment with an
            # apostrophe, say) must degrade to "cannot read the details", never
            # take the status report down with it.
            tokens = []
        seen_env_tool = False
        for token in tokens:
            if not seen_env_tool:
                # Tolerant of a hand-edited path to env (and of a .exe suffix, so
                # a definition written elsewhere still reads back here).
                seen_env_tool = Path(token).stem.lower() == "env"
                continue
            if "=" in token and not token.startswith("/"):
                name, _, value = token.partition("=")
                env[name] = value
                continue
            python = token
            break
        if ">>" in command:
            redirect = command.partition(">>")[2].replace("2>&1", "").strip()
            try:
                log_path = shlex.split(redirect)[0] if redirect else None
            except ValueError:
                log_path = None
    return {
        "every_minutes": every,
        "command": command,
        "python": python,
        "env": tuple(sorted(env.items())),
        "log_path": log_path,
    }


# --------------------------------------------------------------------------- #
# Log inspection and staleness checks (shared by both backends)
# --------------------------------------------------------------------------- #


def _log_facts(log_path: str | None) -> tuple[str | None, str | None]:
    """``(last_run, last_log_line)`` from the tail of the job's log."""
    if not log_path:
        return None, None
    path = Path(log_path)
    try:
        stat = path.stat()
    except OSError:
        return None, None
    when = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    last_line = None
    try:
        with path.open("rb") as stream:
            if stat.st_size > _LOG_TAIL_BYTES:
                stream.seek(-_LOG_TAIL_BYTES, os.SEEK_END)
            tail = stream.read().decode("utf-8", errors="replace")
        lines = [line for line in tail.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1][:200]
    except OSError:
        pass
    return when, last_line


def _interpreter_problems(python: str | None) -> tuple[str, ...]:
    """Staleness checks on the interpreter a job is pinned to.

    A pinned interpreter that has been rebuilt or removed (a pyenv version bump,
    a recreated venv) leaves a job that keeps firing and doing nothing, so this is
    the check that turns a silent failure into a visible one.

    The comparison is deliberately on the *unresolved* paths. Two virtualenvs
    have different ``site-packages`` — and so possibly different installed
    versions of ``ir`` — while ``Path.resolve()`` collapses both onto the same
    base interpreter and reports no difference at all.
    """
    if not python:
        return ()
    if not Path(python).exists():
        return (
            f"the interpreter this job runs ({python}) no longer exists — the job "
            "fires and fails silently; fix with: ir schedule --restart",
        )
    if Path(python) != Path(sys.executable):
        return (
            f"this job runs {python}, but you are running ir from "
            f"{sys.executable} — the job may be using a different installed "
            "version of ir; re-pin with: ir schedule --restart",
        )
    return ()


def _env_problems(env: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Flag a job whose environment is missing something this shell has.

    A job installed from a shell without ``$PP`` fails on the ``packages`` and
    ``reports`` corpora every time it fires, while exiting 0 and looking healthy.
    """
    stored = dict(env)
    missing = [
        name
        for name in LOAD_BEARING_ENV_VARS
        if os.environ.get(name) and not stored.get(name)
    ]
    if not missing:
        return ()
    return (
        f"the job's environment has no {', '.join(missing)}, which ir reads to "
        "find your corpora — it will fail on the packages and reports corpora "
        "while still exiting 0. Re-create it from this shell with: "
        "ir schedule --remove && ir schedule",
    )


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _run(
    argv: Sequence[str], *, stdin: str | None = None
) -> subprocess.CompletedProcess:
    """Run *argv* capturing text output; never raises, not even if it is missing."""
    try:
        return subprocess.run(
            list(argv),
            input=stdin,
            capture_output=True,
            check=False,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        return subprocess.CompletedProcess(
            list(argv), _NO_SUCH_TOOL, stdout=b"", stderr=str(exc).encode()
        )


def _text(raw: bytes | str | None) -> str:
    """Decode captured process output, never raising on odd bytes.

    A crontab with one latin-1 byte in a comment must not take ``ir schedule
    --status`` down with a ``UnicodeDecodeError``.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


#: The ``env`` binary used to carry a job's environment inline on a cron line.
#: Hardcoded rather than looked up: cron is POSIX-only, ``/usr/bin/env`` is where
#: it lives on every POSIX system, and cron runs jobs with a minimal ``PATH`` of
#: its own anyway. A ``shutil.which`` here would make the definition depend on the
#: machine that wrote it — and on Windows it finds Git-Bash's ``env.EXE``, which
#: is not a thing any crontab should ever contain.
_ENV_TOOL = "/usr/bin/env"


class _LaunchdBackend:
    """macOS launchd — a per-user LaunchAgent plist under ``~/Library/LaunchAgents``."""

    name = "launchd"
    #: launchd genuinely loads/unloads a job, so "loaded" is a fact worth showing.
    reports_loaded = True

    @staticmethod
    def available() -> bool:
        return bool(shutil.which("launchctl")) and hasattr(os, "getuid")

    @property
    def _domain(self) -> str:
        return f"gui/{os.getuid()}"

    def plist_path(self, label: str) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"

    def preview(self, spec: ScheduleSpec) -> str:
        return (
            f"{self.plist_path(spec.label)}:\n"
            f"{plist_bytes(spec).decode('utf-8').strip()}"
        )

    def status(self, label: str) -> ScheduleStatus:
        path = self.plist_path(label)
        if not path.exists():
            return ScheduleStatus(
                installed=False,
                backend=self.name,
                label=label,
                definition=str(path),
                detail="no LaunchAgent plist for ir",
            )
        try:
            with path.open("rb") as stream:
                data = plistlib.load(stream)
        except Exception as exc:  # a hand-edited or truncated plist
            return ScheduleStatus(
                installed=True,
                backend=self.name,
                label=label,
                definition=str(path),
                problems=(f"cannot read {path}: {exc}",),
                detail="the plist exists but could not be parsed",
            )
        argv = [str(part) for part in (data.get("ProgramArguments") or [])]
        interval = data.get("StartInterval")
        log_path = data.get("StandardOutPath")
        env = tuple(sorted((data.get("EnvironmentVariables") or {}).items()))
        last_run, last_line = _log_facts(log_path)
        loaded, last_exit = self._launchctl_facts(label)
        problems = _interpreter_problems(argv[0] if argv else None) + _env_problems(env)
        if last_exit:
            problems += (
                f"the last run exited {last_exit} — the job is firing but failing; "
                f"read {log_path or 'the log'} for the reason",
            )
        if not loaded:
            problems += (
                "the plist exists but launchd has not loaded it — nothing is "
                "running; fix with: ir schedule --restart",
            )
        return ScheduleStatus(
            installed=True,
            backend=self.name,
            label=label,
            every_minutes=int(interval // 60) if interval else None,
            definition=str(path),
            command=" ".join(shlex.quote(part) for part in argv),
            python=argv[0] if argv else None,
            env=env,
            log_path=log_path,
            loaded=loaded,
            last_run=last_run,
            last_log_line=last_line,
            problems=problems,
        )

    @staticmethod
    def _launchctl_facts(label: str) -> tuple[bool, int | None]:
        """``(loaded, last_exit_status)`` from ``launchctl list``.

        The exit status is the cheapest health signal there is: a job that fires
        on time and fails every time looks identical to a healthy one until you
        read it.
        """
        done = _run(["launchctl", "list", label])
        if done.returncode != 0:
            return False, None
        match = re.search(r'"LastExitStatus"\s*=\s*(-?\d+)', _text(done.stdout))
        return True, int(match.group(1)) if match else None

    def install(self, spec: ScheduleSpec) -> None:
        path = self.plist_path(spec.label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plist_bytes(spec))
        try:
            self._reload(spec.label, path)
        except ScheduleError:
            # Never leave a plist on disk that launchd refused: --status would
            # then report a schedule that is installed and does nothing.
            path.unlink(missing_ok=True)
            raise

    def remove(self, label: str) -> None:
        path = self.plist_path(label)
        _run(["launchctl", "bootout", f"{self._domain}/{label}"])
        path.unlink(missing_ok=True)

    def _reload(self, label: str, path: Path) -> None:
        """Boot the job out (if loaded) and back in, so edits take effect."""
        _run(["launchctl", "bootout", f"{self._domain}/{label}"])
        done = _run(["launchctl", "bootstrap", self._domain, str(path)])
        if done.returncode != 0:
            # Pre-bootstrap launchctl (and some managed Macs) still take load -w.
            legacy = _run(["launchctl", "load", "-w", str(path)])
            if legacy.returncode != 0:
                message = (_text(done.stderr) or _text(done.stdout)).strip()
                raise ScheduleError(
                    f"launchctl could not load {path}: {message or 'unknown error'}"
                )


class _CronBackend:
    """POSIX cron — one marked, self-contained line in the invoking user's crontab."""

    name = "cron"
    #: cron re-reads the crontab itself; there is no load step to report on.
    reports_loaded = False

    @staticmethod
    def available() -> bool:
        return bool(shutil.which("crontab"))

    def preview(self, spec: ScheduleSpec) -> str:
        return f"appended to the user crontab:\n{cron_line(spec)}"

    def read(self) -> str:
        done = _run(["crontab", "-l"])
        if done.returncode == _NO_SUCH_TOOL:
            raise ScheduleError(
                "the cron backend needs a `crontab` command and this machine has "
                "none; run `ir schedule` without --backend to use whatever this "
                "platform does provide"
            )
        if done.returncode != 0:
            # "no crontab for <user>" is a normal empty state, not a failure.
            stderr = _text(done.stderr)
            if "no crontab" in stderr.lower():
                return ""
            raise ScheduleError(f"could not read the crontab: {stderr.strip()}")
        return _text(done.stdout)

    def write(self, text: str) -> None:
        done = _run(["crontab", "-"], stdin=text.encode("utf-8"))
        if done.returncode != 0:
            raise ScheduleError(
                f"could not write the crontab: {_text(done.stderr).strip()}"
            )

    def status(self, label: str) -> ScheduleStatus:
        line = _find_our_line(self.read(), label)
        if line is None:
            return ScheduleStatus(
                installed=False,
                backend=self.name,
                label=label,
                definition="the user crontab",
                detail="no ir line in the crontab",
            )
        parsed = _parse_cron_line(line, label)
        last_run, last_line = _log_facts(parsed["log_path"])
        return ScheduleStatus(
            installed=True,
            backend=self.name,
            label=label,
            every_minutes=parsed["every_minutes"],
            definition="the user crontab",
            command=parsed["command"],
            python=parsed["python"],
            env=parsed["env"],
            log_path=parsed["log_path"],
            last_run=last_run,
            last_log_line=last_line,
            problems=(
                _interpreter_problems(parsed["python"]) + _env_problems(parsed["env"])
            ),
            detail="cron re-reads the crontab itself; there is nothing to load",
        )

    def install(self, spec: ScheduleSpec) -> None:
        self.write(crontab_with_line(self.read(), spec))

    def remove(self, label: str) -> None:
        self.write(crontab_without_line(self.read(), label))


class _UnsupportedBackend:
    """Neither launchd nor cron is present (Windows, or a stripped container)."""

    name = "unsupported"
    reports_loaded = False

    @staticmethod
    def available() -> bool:
        return True

    _HOW = (
        "ir schedule needs a supported OS scheduler and this machine provides "
        "none. On Windows, schedule it with Task Scheduler, e.g.\n"
        '    schtasks /create /tn "ir maintain" /sc hourly '
        '/tr "\\"<python>\\" -u -m ir maintain --all"'
    )

    def preview(self, spec: ScheduleSpec) -> str:
        return self._HOW

    def status(self, label: str) -> ScheduleStatus:
        return ScheduleStatus(
            installed=False, backend=self.name, label=label, detail=self._HOW
        )

    def install(self, spec: ScheduleSpec) -> None:
        raise ScheduleError(self._HOW)

    def remove(self, label: str) -> None:
        raise ScheduleError(self._HOW)


#: Backends in preference order. launchd first on macOS because a LaunchAgent
#: survives reboots and logs where the OS expects; cron is the POSIX fallback and
#: also works on macOS if asked for by name.
#:
#: This tuple is the whole seam. A backend owns its definition format, its
#: ``preview``, and its ``reports_loaded`` answer, so adding ``systemd --user``
#: means adding a class here and nothing else — no branch in the renderer, the
#: previewer, or the CLI.
BACKENDS = (_LaunchdBackend, _CronBackend)


def resolve_backend(backend: str | None = None, *, require_available: bool = True):
    """The backend named *backend*, or the first available one.

    Naming one this machine cannot run is an error rather than a half-install:
    ``--backend launchd`` on Linux would otherwise write a plist into a directory
    nothing reads. Pass ``require_available=False`` to obtain a backend purely to
    render its definition format.

    >>> resolve_backend("cron", require_available=False).name
    'cron'
    """
    if backend is not None:
        for candidate in BACKENDS:
            if candidate.name == backend:
                if require_available and not candidate.available():
                    raise ScheduleError(
                        f"the {backend} backend is not available on this machine; "
                        "run `ir schedule` without --backend to use whatever this "
                        "platform does provide"
                    )
                return candidate()
        known = ", ".join(candidate.name for candidate in BACKENDS)
        raise ScheduleError(f"unknown backend {backend!r}; expected one of {known}")
    for candidate in BACKENDS:
        if candidate.available():
            return candidate()
    return _UnsupportedBackend()


# --------------------------------------------------------------------------- #
# Operations (the public API; one return type for all of them)
# --------------------------------------------------------------------------- #


def status(*, backend: str | None = None, label: str = DFLT_LABEL) -> ScheduleStatus:
    """Report the current schedule without changing anything."""
    return resolve_backend(backend).status(label)


def _spec_for(
    current: ScheduleStatus,
    *,
    minutes: int | None,
    python: str | None,
    label: str,
    inherit: bool,
) -> ScheduleSpec:
    """Build the spec to write.

    When *inherit* is set and a schedule already exists, its stored environment
    and interpreter are carried forward rather than re-derived from this process.
    Re-deriving is how a job installed from a full shell silently loses ``$PP``
    the first time it is operated from one without it.
    """
    fields: dict[str, Any] = {"label": label}
    fields["every_minutes"] = minutes or current.every_minutes or DFLT_EVERY_MINUTES
    if python:
        fields["python"] = python
    elif inherit and current.installed and current.python:
        fields["python"] = current.python
    if inherit and current.installed and current.env:
        fields["env"] = current.env
    if inherit and current.installed and current.log_path:
        fields["log_path"] = current.log_path
    return ScheduleSpec(**fields)


def install(
    *,
    every: str | int | None = None,
    backend: str | None = None,
    label: str = DFLT_LABEL,
    python: str | None = None,
    inherit: bool = True,
    dry_run: bool = False,
) -> ScheduleStatus:
    """Install (or overwrite) the schedule, and return the resulting status.

    *inherit* carries an existing definition's environment, interpreter and log
    path forward; pass ``False`` to re-snapshot them from this process.
    """
    impl = resolve_backend(backend, require_available=not dry_run)
    current = impl.status(label) if not dry_run else _safe_status(impl, label)
    spec = _spec_for(
        current,
        minutes=parse_every(every),
        python=python,
        label=label,
        inherit=inherit,
    )
    if dry_run:
        return ScheduleStatus(
            installed=current.installed,
            backend=impl.name,
            label=label,
            action="would-install",
            every_minutes=spec.every_minutes,
            definition=current.definition,
            command=" ".join(shlex.quote(part) for part in spec.command),
            python=spec.python,
            env=spec.env,
            log_path=spec.log_path,
            detail=impl.preview(spec),
        )
    impl.install(spec)
    action = "updated" if current.installed else "installed"
    return _replace_action(impl.status(label), action)


def _safe_status(impl, label: str) -> ScheduleStatus:
    """*impl*'s status, or an empty one — used where a report must not fail."""
    try:
        return impl.status(label)
    except ScheduleError:
        return ScheduleStatus(installed=False, backend=impl.name, label=label)


def remove(*, backend: str | None = None, label: str = DFLT_LABEL) -> ScheduleStatus:
    """Stop the schedule and delete its definition."""
    impl = resolve_backend(backend)
    current = impl.status(label)
    if not current.installed:
        return _replace_action(current, "unchanged")
    impl.remove(label)
    return _replace_action(impl.status(label), "removed")


def restart(*, backend: str | None = None, label: str = DFLT_LABEL) -> ScheduleStatus:
    """Reload the existing definition, re-pinning it to the current interpreter.

    This is the repair for a schedule left pointing at a rebuilt or deleted
    interpreter. The stored environment is carried forward untouched — restarting
    from a shell that lacks ``$PP`` must not quietly re-point a working job at a
    different (or empty) set of corpora.
    """
    impl = resolve_backend(backend)
    current = impl.status(label)
    if not current.installed:
        raise ScheduleError(
            "nothing to restart — no schedule is installed. Create one with: ir schedule"
        )
    return _replace_action(
        install(
            every=current.every_minutes,
            backend=impl.name,
            label=label,
            python=sys.executable,
        ),
        "restarted",
    )


def ensure(
    *,
    every: str | int | None = None,
    backend: str | None = None,
    label: str = DFLT_LABEL,
) -> ScheduleStatus:
    """Make sure a schedule exists — the idempotent front door.

    Installs when there is none. When one already exists it changes nothing
    unless *every* differs, so running it twice is safe and running it out of
    habit never silently reinstalls over a schedule you tuned.
    """
    impl = resolve_backend(backend)
    current = impl.status(label)
    minutes = parse_every(every)
    if not current.installed:
        return install(every=minutes, backend=impl.name, label=label)
    if minutes is not None and minutes != current.every_minutes:
        return install(every=minutes, backend=impl.name, label=label)
    return _replace_action(current, "unchanged")


def _replace_action(state: ScheduleStatus, action: str) -> ScheduleStatus:
    """*state* with its ``action`` set."""
    return dc_replace(state, action=action)


# --------------------------------------------------------------------------- #
# Human rendering (the CLI's formatter; kept here so the report has one home)
# --------------------------------------------------------------------------- #

_ACTION_HEADLINES = {
    "installed": "Installed the ir maintenance schedule.",
    "updated": "Updated the ir maintenance schedule.",
    "restarted": "Reloaded the ir maintenance schedule.",
    "removed": "Removed the ir maintenance schedule.",
    "unchanged": "ir maintain is already scheduled.",
    "reported": "ir maintenance schedule:",
    "would-install": "Dry run — nothing was written.",
}

_MENU = """\
To operate it:
  ir schedule --status       show this without changing anything
  ir schedule --every 30m    change the interval (rewrites and reloads)
  ir schedule --restart      reload it (after upgrading ir or changing interpreter)
  ir schedule --remove       stop it and delete the definition
  ir maintain --all          run the work right now, in the foreground"""


def _rows(state: ScheduleStatus) -> Iterable[tuple[str, Any]]:
    yield "backend", state.backend
    yield "every", format_every(state.every_minutes)
    yield "definition", state.definition
    yield "runs", state.command
    yield "log", state.log_path
    if state.loaded is not None:
        yield "loaded", "yes" if state.loaded else "NO"
    yield "last run", state.last_run or "never (no log yet)"
    yield "last log line", state.last_log_line


def render(state: ScheduleStatus) -> str:
    """A human report of *state*, including how to operate what was found."""
    lines = [_ACTION_HEADLINES.get(state.action, state.action)]
    if not state.installed and state.action in ("reported", "unchanged"):
        lines = ["Nothing is scheduled to run `ir maintain`."]
        if state.detail:
            lines += ["", state.detail]
        if state.backend != "unsupported":
            lines += ["", "Create one with:  ir schedule"]
        return "\n".join(lines)
    lines.append("")
    width = max(len(name) for name, _ in _rows(state))
    lines += [
        f"  {name.ljust(width)}  {value}" for name, value in _rows(state) if value
    ]
    if state.problems:
        lines += ["", "Problems:"]
        lines += [f"  ! {problem}" for problem in state.problems]
    if state.detail and state.action == "would-install":
        lines += ["", state.detail]
    elif state.detail:
        lines += ["", f"note: {state.detail}"]
    if state.action != "removed":
        lines += ["", _MENU]
    return "\n".join(lines)
