"""Install and operate the OS job that runs ``ir maintain`` — ``ir schedule`` (issue #75).

``ir maintain`` is idempotent and cron-shaped, but something has to *call* it.
This module manages that caller: it writes a **definition** (a launchd plist or a
crontab block), hands it to the OS, and exits.

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
    ir schedule --restart      # reload the definition (after upgrading ir)
    ir schedule --remove       # stop it and delete the definition
    ir schedule --dry-run      # print what would be written

Two seams, both one keyword argument:

- ``backend`` — which OS scheduler holds the definition. Feature-detected
  (``launchctl`` then ``crontab``), never hardcoded per-platform; ``systemd
  --user`` timers are the documented next backend.
- :attr:`ScheduleSpec.args` — the command the job runs, defaulting to
  ``-m ir maintain --all`` pinned to the installing interpreter. An orchestration
  layer's own entry point (#58 / ADR #43) drops in here without touching backends.

Everything else — the plist keys, the crontab marker syntax, interval parsing,
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import cache_dir

#: Reverse-DNS job identifier. Also the launchd ``Label`` and the token that
#: delimits ir's block in the user crontab, so both backends find their own work
#: and never touch anybody else's.
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

#: Minute intervals that map to a clean cron expression (divisors of 60), and
#: hour intervals that do (divisors of 24). launchd accepts any interval; cron's
#: ``*/n`` restarts each hour, so a non-divisor would fire unevenly.
_CRON_MINUTE_STEPS = tuple(m for m in range(1, 60) if 60 % m == 0)
_CRON_HOUR_STEPS = tuple(h for h in range(1, 24) if 24 % h == 0)

_UNITS = {"m": 1, "h": 60, "d": 1440}


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

    @property
    def command(self) -> list[str]:
        """The argv the scheduler executes."""
        return [self.python, *self.args]

    @property
    def env_dict(self) -> dict[str, str]:
        """The job environment as a plain dict."""
        return dict(self.env)

    def shell_line(self) -> str:
        """The command as one shell line, appending to the log (for crontab)."""
        cmd = " ".join(shlex.quote(part) for part in self.command)
        return f"{cmd} >> {shlex.quote(self.log_path)} 2>&1"


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
    if minutes is None:
        return "unknown"
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}h{mins}m"
    return f"{hours}h" if hours else f"{mins}m"


# --------------------------------------------------------------------------- #
# Definition rendering (pure — the two backends' file formats)
# --------------------------------------------------------------------------- #


def plist_dict(spec: ScheduleSpec) -> dict[str, Any]:
    """The launchd job description for *spec*, as a plist-ready dict.

    ``StartInterval`` (seconds) is what makes the interval readable back out of
    an installed job, so :func:`status` needs no separate bookkeeping file.
    """
    return {
        "Label": spec.label,
        "ProgramArguments": list(spec.command),
        "StartInterval": spec.every_minutes * 60,
        "RunAtLoad": False,
        "EnvironmentVariables": spec.env_dict,
        "StandardOutPath": spec.log_path,
        "StandardErrorPath": spec.log_path,
        "ProcessType": "Background",
    }


def plist_bytes(spec: ScheduleSpec) -> bytes:
    """*spec* rendered as launchd plist XML."""
    return plistlib.dumps(plist_dict(spec))


def cron_markers(label: str) -> tuple[str, str]:
    """The ``(begin, end)`` comment lines delimiting ir's crontab block."""
    return (f"# >>> {label} >>>", f"# <<< {label} <<<")


def cron_block(spec: ScheduleSpec) -> str:
    """ir's crontab block for *spec*, markers included.

    The interval is written into the begin marker as ``every=<minutes>`` so it
    reads back exactly, rather than being re-derived from the cron expression
    (``0 * * * *`` and ``*/60 * * * *`` mean the same thing but do not spell it).
    """
    begin, end = cron_markers(spec.label)
    lines = [
        f"{begin} every={spec.every_minutes} (managed by `ir schedule`; edit with it, not by hand)"
    ]
    lines += [f"{name}={value}" for name, value in spec.env]
    lines.append(f"{cron_expression(spec.every_minutes)} {spec.shell_line()}")
    lines.append(end)
    return "\n".join(lines)


def _split_crontab(text: str, label: str) -> tuple[list[str], list[str]]:
    """Split crontab *text* into ``(other_lines, ir_block_lines)``."""
    begin, end = cron_markers(label)
    other: list[str] = []
    block: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith(begin):
            inside = True
            block.append(line)
        elif inside:
            block.append(line)
            if line.strip() == end:
                inside = False
        else:
            other.append(line)
    return other, block


def crontab_without_block(text: str, label: str = DFLT_LABEL) -> str:
    """*text* with ir's block removed (unchanged if there is none)."""
    other, _ = _split_crontab(text, label)
    body = "\n".join(other).strip("\n")
    return f"{body}\n" if body else ""


def crontab_with_block(text: str, spec: ScheduleSpec) -> str:
    """*text* with ir's block replaced by (or appended as) *spec*'s block."""
    body = crontab_without_block(text, spec.label).rstrip("\n")
    parts = [p for p in (body, cron_block(spec)) if p]
    return "\n".join(parts) + "\n"


def _every_from_cron_block(block: Sequence[str], label: str) -> int | None:
    """Read ``every=<minutes>`` back out of a crontab block's begin marker."""
    begin, _ = cron_markers(label)
    for line in block:
        if line.startswith(begin):
            for token in line.split():
                if token.startswith("every="):
                    try:
                        return int(token.partition("=")[2])
                    except ValueError:
                        return None
    return None


def _command_from_cron_block(block: Sequence[str]) -> str | None:
    """The command half of the block's schedule line (fields 6 onward)."""
    for line in block:
        if line.startswith("#") or "=" in line.split(" ")[0] or not line.strip():
            continue
        fields = line.split(None, 5)
        if len(fields) == 6:
            return fields[5]
    return None


# --------------------------------------------------------------------------- #
# Log inspection (shared by both backends)
# --------------------------------------------------------------------------- #


def _log_facts(log_path: str | None) -> tuple[str | None, str | None]:
    """``(last_run, last_log_line)`` from the job's log, or ``(None, None)``."""
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
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
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
    """
    if not python:
        return ()
    if not Path(python).exists():
        return (
            f"the interpreter this job runs ({python}) no longer exists — the job "
            "fires and fails silently; fix with: ir schedule --restart",
        )
    if Path(python).resolve() != Path(sys.executable).resolve():
        return (
            f"this job runs {python}, but you are running ir from "
            f"{sys.executable} — they may see different corpora; "
            "re-pin with: ir schedule --restart",
        )
    return ()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


#: Exit status for "the tool itself is not here" (the shell convention). A missing
#: binary has to look like a failed command rather than an exception: naming a
#: backend is not the same as being on a machine that has it, and asking about one
#: that is absent must report rather than crash.
_NO_SUCH_TOOL = 127


def _run(
    argv: Sequence[str], *, stdin: str | None = None
) -> subprocess.CompletedProcess:
    """Run *argv* capturing text output; never raises, not even if it is missing."""
    try:
        return subprocess.run(
            list(argv),
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        return subprocess.CompletedProcess(
            list(argv), _NO_SUCH_TOOL, stdout="", stderr=f"{argv[0]}: {exc.strerror}"
        )


class _LaunchdBackend:
    """macOS launchd — a per-user LaunchAgent plist under ``~/Library/LaunchAgents``."""

    name = "launchd"

    @staticmethod
    def available() -> bool:
        return bool(shutil.which("launchctl")) and hasattr(os, "getuid")

    @property
    def _domain(self) -> str:
        return f"gui/{os.getuid()}"

    def plist_path(self, label: str) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"

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
        argv = list(data.get("ProgramArguments") or [])
        interval = data.get("StartInterval")
        log_path = data.get("StandardOutPath")
        last_run, last_line = _log_facts(log_path)
        loaded, last_exit = self._launchctl_facts(label)
        problems = _interpreter_problems(argv[0] if argv else None)
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
        match = re.search(r'"LastExitStatus"\s*=\s*(-?\d+)', done.stdout or "")
        return True, int(match.group(1)) if match else None

    def install(self, spec: ScheduleSpec) -> None:
        path = self.plist_path(spec.label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plist_bytes(spec))
        self._reload(spec.label, path)

    def remove(self, label: str) -> None:
        path = self.plist_path(label)
        _run(["launchctl", "bootout", f"{self._domain}/{label}"])
        if path.exists():
            path.unlink()

    def restart(self, label: str) -> None:
        path = self.plist_path(label)
        if not path.exists():
            raise ScheduleError(f"no schedule to restart ({path} does not exist)")
        self._reload(label, path)

    def _reload(self, label: str, path: Path) -> None:
        """Boot the job out (if loaded) and back in, so edits take effect."""
        _run(["launchctl", "bootout", f"{self._domain}/{label}"])
        done = _run(["launchctl", "bootstrap", self._domain, str(path)])
        if done.returncode != 0:
            # Pre-bootstrap launchctl (and some managed Macs) still take load -w.
            legacy = _run(["launchctl", "load", "-w", str(path)])
            if legacy.returncode != 0:
                message = (done.stderr or done.stdout or "").strip()
                raise ScheduleError(
                    f"launchctl could not load {path}: {message or 'unknown error'}"
                )


class _CronBackend:
    """POSIX cron — a marker-delimited block in the invoking user's crontab."""

    name = "cron"

    @staticmethod
    def available() -> bool:
        return bool(shutil.which("crontab"))

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
            if "no crontab" in (done.stderr or "").lower():
                return ""
            raise ScheduleError(
                f"could not read the crontab: {(done.stderr or '').strip()}"
            )
        return done.stdout

    def write(self, text: str) -> None:
        done = _run(["crontab", "-"], stdin=text)
        if done.returncode != 0:
            raise ScheduleError(
                f"could not write the crontab: {(done.stderr or '').strip()}"
            )

    def status(self, label: str) -> ScheduleStatus:
        _, block = _split_crontab(self.read(), label)
        if not block:
            return ScheduleStatus(
                installed=False,
                backend=self.name,
                label=label,
                definition="the user crontab",
                detail="no ir block in the crontab",
            )
        command = _command_from_cron_block(block)
        log_path = None
        argv0 = None
        if command:
            parts = shlex.split(command.split(">>")[0])
            argv0 = parts[0] if parts else None
            _, _, redirect = command.partition(">>")
            redirect = redirect.replace("2>&1", "").strip()
            log_path = shlex.split(redirect)[0] if redirect else None
        last_run, last_line = _log_facts(log_path)
        return ScheduleStatus(
            installed=True,
            backend=self.name,
            label=label,
            every_minutes=_every_from_cron_block(block, label),
            definition="the user crontab",
            command=command,
            log_path=log_path,
            loaded=True,
            last_run=last_run,
            last_log_line=last_line,
            problems=_interpreter_problems(argv0),
            detail="cron re-reads the crontab itself; there is nothing to load",
        )

    def install(self, spec: ScheduleSpec) -> None:
        self.write(crontab_with_block(self.read(), spec))

    def remove(self, label: str) -> None:
        self.write(crontab_without_block(self.read(), label))

    def restart(self, label: str) -> None:
        text = self.read()
        _, block = _split_crontab(text, label)
        if not block:
            raise ScheduleError("no schedule to restart (no ir block in the crontab)")
        self.write(text)  # rewriting is what makes cron re-read it immediately


class _UnsupportedBackend:
    """Neither launchd nor cron is present (Windows, or a stripped container)."""

    name = "unsupported"

    @staticmethod
    def available() -> bool:
        return True

    _HOW = (
        "ir schedule needs launchd (macOS) or cron (POSIX); neither is available "
        "here. On Windows, schedule it with Task Scheduler, e.g.\n"
        '    schtasks /create /tn "ir maintain" /sc hourly '
        '/tr "\\"<python>\\" -m ir maintain --all"'
    )

    def status(self, label: str) -> ScheduleStatus:
        return ScheduleStatus(
            installed=False,
            backend=self.name,
            label=label,
            detail=self._HOW,
        )

    def install(self, spec: ScheduleSpec) -> None:
        raise ScheduleError(self._HOW)

    def remove(self, label: str) -> None:
        raise ScheduleError(self._HOW)

    def restart(self, label: str) -> None:
        raise ScheduleError(self._HOW)


#: Backends in preference order. launchd first on macOS because a LaunchAgent
#: survives reboots and logs where the OS expects; cron is the POSIX fallback and
#: also works on macOS if asked for by name.
BACKENDS = (_LaunchdBackend, _CronBackend)


def resolve_backend(backend: str | None = None):
    """The backend named *backend*, or the first available one.

    >>> resolve_backend("cron").name
    'cron'
    """
    if backend is not None:
        for candidate in BACKENDS:
            if candidate.name == backend:
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


def install(
    *,
    every: str | int | None = None,
    backend: str | None = None,
    label: str = DFLT_LABEL,
    python: str | None = None,
    dry_run: bool = False,
) -> ScheduleStatus:
    """Install (or overwrite) the schedule, and return the resulting status."""
    impl = resolve_backend(backend)
    current = impl.status(label)
    minutes = parse_every(every)
    if minutes is None:
        minutes = current.every_minutes or DFLT_EVERY_MINUTES
    spec = ScheduleSpec(
        every_minutes=minutes,
        label=label,
        **({"python": python} if python else {}),
    )
    if dry_run:
        return ScheduleStatus(
            installed=current.installed,
            backend=impl.name,
            label=label,
            action="would-install",
            every_minutes=minutes,
            definition=current.definition,
            command=" ".join(shlex.quote(part) for part in spec.command),
            log_path=spec.log_path,
            detail=preview(spec, backend=impl.name),
        )
    impl.install(spec)
    after = impl.status(label)
    action = "updated" if current.installed else "installed"
    return _replace_action(after, action)


def remove(*, backend: str | None = None, label: str = DFLT_LABEL) -> ScheduleStatus:
    """Stop the schedule and delete its definition."""
    impl = resolve_backend(backend)
    current = impl.status(label)
    if not current.installed:
        return _replace_action(current, "unchanged")
    impl.remove(label)
    return _replace_action(impl.status(label), "removed")


def restart(*, backend: str | None = None, label: str = DFLT_LABEL) -> ScheduleStatus:
    """Reload the existing definition (after upgrading ir or moving interpreter).

    Re-pins the job to the interpreter running this call, which is what repairs a
    schedule left pointing at a rebuilt or deleted interpreter.
    """
    impl = resolve_backend(backend)
    current = impl.status(label)
    if not current.installed:
        raise ScheduleError(
            "nothing to restart — no schedule is installed. Create one with: ir schedule"
        )
    return _replace_action(
        install(every=current.every_minutes, backend=impl.name, label=label),
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


def preview(spec: ScheduleSpec, *, backend: str | None = None) -> str:
    """The definition text *spec* would write, for ``--dry-run``."""
    impl = resolve_backend(backend)
    if impl.name == "launchd":
        return (
            f"{impl.plist_path(spec.label)}:\n"
            f"{plist_bytes(spec).decode('utf-8').strip()}"
        )
    if impl.name == "cron":
        return f"appended to the user crontab:\n{cron_block(spec)}"
    return _UnsupportedBackend._HOW


def _replace_action(state: ScheduleStatus, action: str) -> ScheduleStatus:
    """*state* with its ``action`` set (dataclasses.replace, kept local and typed)."""
    from dataclasses import replace as dc_replace

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
    if state.backend == "launchd" and state.loaded is not None:
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
