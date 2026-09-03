"""Tests for ``ir schedule`` — the OS-job installer for ``ir maintain`` (#75).

Everything here runs on any OS: the definition formats, interval maths and status
reporting are pure functions, and the install/update/remove round trip drives the
cron backend against an in-memory crontab rather than the real one. Nothing in
this file shells out, writes to ``~/Library/LaunchAgents``, or touches a crontab.
"""

import sys

import pytest

from ir import schedule
from ir.schedule import (
    DFLT_LABEL,
    ScheduleError,
    ScheduleSpec,
    cron_block,
    cron_expression,
    crontab_with_block,
    crontab_without_block,
    format_every,
    parse_every,
    plist_dict,
    resolve_backend,
)


@pytest.fixture
def fake_crontab(monkeypatch):
    """Drive the cron backend against an in-memory crontab."""
    box = {"text": ""}
    monkeypatch.setattr(schedule._CronBackend, "read", lambda self: box["text"])
    monkeypatch.setattr(
        schedule._CronBackend, "write", lambda self, text: box.__setitem__("text", text)
    )
    return box


# ----- intervals ----------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [("15m", 15), ("2h", 120), ("1d", 1440), ("45", 45), (30, 30), (None, None)],
)
def test_parse_every(value, expected):
    assert parse_every(value) == expected


@pytest.mark.parametrize("bad", ["", "soon", "0", "-5m", "m"])
def test_parse_every_rejects_nonsense(bad):
    with pytest.raises(ScheduleError):
        parse_every(bad)


@pytest.mark.parametrize(
    "minutes,expr",
    [
        (1, "*/1 * * * *"),
        (15, "*/15 * * * *"),
        (30, "*/30 * * * *"),
        (60, "0 * * * *"),
        (360, "0 */6 * * *"),
        (1440, "0 0 * * *"),
    ],
)
def test_cron_expression(minutes, expr):
    assert cron_expression(minutes) == expr


@pytest.mark.parametrize("minutes", [7, 45, 50, 300, 2880])
def test_cron_expression_refuses_uneven_intervals(minutes):
    # cron restarts its */n count each hour/day, so a non-divisor fires unevenly.
    # Refusing beats installing a job that quietly skips -- and the message has to
    # name the intervals that do work.
    with pytest.raises(ScheduleError) as excinfo:
        cron_expression(minutes)
    assert "15m" in str(excinfo.value) or "6h" in str(excinfo.value)


def test_format_every():
    assert format_every(30) == "30m"
    assert format_every(60) == "1h"
    assert format_every(90) == "1h30m"
    assert format_every(None) == "unknown"


# ----- definition rendering ------------------------------------------------ #


def test_plist_dict_pins_interpreter_and_interval():
    spec = ScheduleSpec(every_minutes=30, python="/opt/py/bin/python")
    data = plist_dict(spec)
    assert data["Label"] == DFLT_LABEL
    assert data["ProgramArguments"] == [
        "/opt/py/bin/python",
        "-u",  # unbuffered: without it the job's log stays empty until it exits
        "-m",
        "ir",
        "maintain",
        "--all",
    ]
    # StartInterval is seconds, and is what status() reads the interval back from.
    assert data["StartInterval"] == 1800
    assert data["RunAtLoad"] is False
    assert data["StandardOutPath"] == spec.log_path


def test_plist_bytes_round_trips_through_plistlib():
    import plistlib

    spec = ScheduleSpec(every_minutes=45)
    assert plistlib.loads(schedule.plist_bytes(spec)) == plist_dict(spec)


def test_launchd_accepts_intervals_cron_cannot():
    # The backends genuinely differ here; the plist path must not inherit cron's
    # divisor restriction.
    assert plist_dict(ScheduleSpec(every_minutes=45))["StartInterval"] == 2700


def test_cron_block_is_marker_delimited_and_self_describing():
    block = cron_block(ScheduleSpec(every_minutes=15, python="/opt/py/bin/python"))
    lines = block.splitlines()
    assert lines[0].startswith(f"# >>> {DFLT_LABEL} >>>")
    assert "every=15" in lines[0]
    assert lines[-1] == f"# <<< {DFLT_LABEL} <<<"
    assert any(line.startswith("*/15 * * * * ") for line in lines)
    assert any("maintain" in line and "2>&1" in line for line in lines)


def test_cron_block_carries_the_ir_environment(monkeypatch):
    # launchd/cron start with a near-empty environment: an IR_DATA_DIR set in the
    # installing shell and not in the job would maintain *different* corpora.
    monkeypatch.setenv("IR_DATA_DIR", "/data/ir")
    block = cron_block(ScheduleSpec(every_minutes=60))
    assert "IR_DATA_DIR=/data/ir" in block
    assert plist_dict(ScheduleSpec())["EnvironmentVariables"]["IR_DATA_DIR"] == "/data/ir"


# ----- crontab editing ----------------------------------------------------- #

OTHER_JOBS = "0 3 * * * /usr/local/bin/backup.sh\n@reboot /usr/bin/tunnel\n"


def test_crontab_with_block_preserves_other_entries():
    result = crontab_with_block(OTHER_JOBS, ScheduleSpec(every_minutes=60))
    assert "/usr/local/bin/backup.sh" in result
    assert "@reboot /usr/bin/tunnel" in result
    assert DFLT_LABEL in result


def test_crontab_with_block_replaces_rather_than_appends():
    once = crontab_with_block(OTHER_JOBS, ScheduleSpec(every_minutes=60))
    twice = crontab_with_block(once, ScheduleSpec(every_minutes=15))
    assert twice.count(f"# >>> {DFLT_LABEL} >>>") == 1
    assert "every=15" in twice
    assert "every=60" not in twice


def test_crontab_without_block_leaves_foreign_entries_alone():
    populated = crontab_with_block(OTHER_JOBS, ScheduleSpec())
    assert crontab_without_block(populated) == OTHER_JOBS
    # Removing from a crontab that never had an ir block is a no-op, not an error.
    assert crontab_without_block(OTHER_JOBS) == OTHER_JOBS


def test_crontab_without_block_on_empty_crontab():
    assert crontab_without_block("") == ""


# ----- backend resolution -------------------------------------------------- #


def test_resolve_backend_by_name():
    assert resolve_backend("cron").name == "cron"
    assert resolve_backend("launchd").name == "launchd"


def test_resolve_backend_rejects_unknown_names():
    with pytest.raises(ScheduleError) as excinfo:
        resolve_backend("systemd")
    assert "launchd" in str(excinfo.value) and "cron" in str(excinfo.value)


def test_resolve_backend_falls_back_to_unsupported(monkeypatch):
    monkeypatch.setattr(schedule._LaunchdBackend, "available", staticmethod(lambda: False))
    monkeypatch.setattr(schedule._CronBackend, "available", staticmethod(lambda: False))
    impl = resolve_backend()
    assert impl.name == "unsupported"
    # It must report, not raise -- ir imports and runs fine on a platform with
    # neither scheduler.
    state = impl.status(DFLT_LABEL)
    assert state.installed is False
    assert "Task Scheduler" in state.detail
    with pytest.raises(ScheduleError):
        impl.install(ScheduleSpec())


def test_status_never_raises_on_this_platform():
    # Whatever machine CI runs on, asking is always safe.
    assert schedule.status().backend in {"launchd", "cron", "unsupported"}


# ----- the operations round trip (cron backend, in-memory crontab) --------- #


def test_install_inspect_update_remove(fake_crontab):
    absent = schedule.status(backend="cron")
    assert absent.installed is False

    created = schedule.ensure(backend="cron", every="15m")
    assert (created.installed, created.action, created.every_minutes) == (
        True,
        "installed",
        15,
    )
    assert created.changed is True
    assert created.command and "maintain" in created.command

    # Re-running the bare command must NOT silently reinstall over a tuned schedule.
    again = schedule.ensure(backend="cron")
    assert (again.action, again.changed, again.every_minutes) == ("unchanged", False, 15)

    changed = schedule.ensure(backend="cron", every="1h")
    assert (changed.action, changed.every_minutes) == ("updated", 60)
    assert fake_crontab["text"].count(f"# >>> {DFLT_LABEL} >>>") == 1

    gone = schedule.remove(backend="cron")
    assert (gone.installed, gone.action) == (False, "removed")
    assert DFLT_LABEL not in fake_crontab["text"]

    # Removing twice is idempotent rather than an error.
    assert schedule.remove(backend="cron").action == "unchanged"


def test_dry_run_writes_nothing(fake_crontab):
    state = schedule.install(backend="cron", every="15m", dry_run=True)
    assert state.action == "would-install"
    assert fake_crontab["text"] == ""
    assert "*/15 * * * *" in state.detail


def test_restart_requires_an_installed_schedule(fake_crontab):
    with pytest.raises(ScheduleError) as excinfo:
        schedule.restart(backend="cron")
    assert "ir schedule" in str(excinfo.value)


def test_restart_repins_the_interpreter(fake_crontab):
    schedule.install(backend="cron", every="30m", python="/gone/bin/python")
    stale = schedule.status(backend="cron")
    assert stale.problems and "no longer exists" in stale.problems[0]

    healed = schedule.restart(backend="cron")
    assert healed.action == "restarted"
    assert healed.every_minutes == 30  # the interval survives a restart
    assert healed.problems == ()
    assert sys.executable in (healed.command or "")


def test_install_keeps_the_existing_interval_when_none_is_given(fake_crontab):
    schedule.install(backend="cron", every="15m")
    assert schedule.install(backend="cron").every_minutes == 15


# ----- reporting ----------------------------------------------------------- #


def test_render_absent_schedule_says_how_to_create_one():
    text = schedule.render(schedule.status(backend="cron"))
    assert "Nothing is scheduled" in text
    assert "ir schedule" in text


def test_render_existing_schedule_shows_the_operations_menu(fake_crontab):
    text = schedule.render(schedule.ensure(backend="cron", every="15m"))
    for expected in ("--status", "--every", "--restart", "--remove", "ir maintain"):
        assert expected in text


def test_render_surfaces_problems(fake_crontab):
    schedule.install(backend="cron", every="1h", python="/gone/bin/python")
    text = schedule.render(schedule.status(backend="cron"))
    assert "Problems:" in text
    assert "--restart" in text


def test_status_to_dict_is_json_ready(fake_crontab):
    import json

    schedule.ensure(backend="cron", every="1h")
    payload = schedule.status(backend="cron").to_dict()
    assert json.loads(json.dumps(payload))["every_minutes"] == 60


# ----- the CLI command ----------------------------------------------------- #


def test_cli_schedule_round_trip(fake_crontab):
    from ir.cli import schedule as cli_schedule

    assert "Nothing is scheduled" in cli_schedule(status=True, backend="cron")
    assert "Installed" in cli_schedule(every="15m", backend="cron")
    assert "already scheduled" in cli_schedule(backend="cron")
    assert "Removed" in cli_schedule(remove=True, backend="cron")


def test_cli_schedule_reports_errors_without_a_traceback(fake_crontab):
    # A bad interval is a user mistake: it should read as a message, not crash.
    out = schedule.render(schedule.status(backend="cron"))
    assert "Traceback" not in out
    from ir.cli import schedule as cli_schedule

    message = cli_schedule(every="45m", backend="cron")
    assert message.startswith("ir schedule: ")
    assert "cron cannot fire evenly" in message


def test_schedule_command_is_registered():
    from ir.cli import COMMANDS, schedule as cli_schedule

    assert cli_schedule in COMMANDS


def test_job_inherits_the_env_irs_own_sources_read(monkeypatch):
    # ir.sources reads $PP and $PTH_FILEPATH to locate the projects root and the
    # package manifest. A job that does not carry them fails outright on the
    # `packages` and `reports` corpora -- while looking like it ran.
    monkeypatch.setenv("PP", "/home/me/proj")
    monkeypatch.setenv("PTH_FILEPATH", "/home/me/proj/my_packages.pth")
    env = plist_dict(ScheduleSpec())["EnvironmentVariables"]
    assert env["PP"] == "/home/me/proj"
    assert env["PTH_FILEPATH"] == "/home/me/proj/my_packages.pth"
    assert "PP=/home/me/proj" in cron_block(ScheduleSpec(every_minutes=60))


def test_launchd_status_flags_a_failing_job(monkeypatch, tmp_path):
    # A job that fires on time and fails every time looks healthy until someone
    # reads the exit status, so status() has to read it for you.
    plist = tmp_path / "job.plist"
    plist.write_bytes(schedule.plist_bytes(ScheduleSpec(every_minutes=60)))
    monkeypatch.setattr(
        schedule._LaunchdBackend, "plist_path", lambda self, label: plist
    )
    monkeypatch.setattr(
        schedule._LaunchdBackend,
        "_launchctl_facts",
        staticmethod(lambda label: (True, 1)),
    )
    state = schedule._LaunchdBackend().status(DFLT_LABEL)
    assert state.installed and state.loaded
    assert any("exited 1" in problem for problem in state.problems)
    assert "Problems:" in schedule.render(state)


def test_maintain_cli_exits_nonzero_when_a_corpus_fails(monkeypatch):
    from ir import cli, maintenance

    monkeypatch.setattr(
        maintenance,
        "maintain",
        lambda **kwargs: [
            maintenance.MaintenanceResult("good", True, "reindexed"),
            maintenance.MaintenanceResult("bad", False, "error: boom", error="E: boom"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.maintain(all=True)
    message = str(excinfo.value)
    assert "1 of 2 corpora failed" in message
    assert "FAILED" in message  # the per-corpus detail travels with the failure
