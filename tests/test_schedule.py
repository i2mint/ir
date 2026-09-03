"""Tests for ``ir schedule`` — the OS-job installer for ``ir maintain`` (#75).

Everything here runs on any OS: the definition formats, interval maths and status
reporting are pure functions, and the install/update/remove round trip drives the
cron backend against an in-memory crontab rather than the real one. Nothing in
this file shells out, writes to ``~/Library/LaunchAgents``, or touches a crontab.

A large share of these tests exist because an adversarial review of the first
draft found the failure each one now locks down.
"""

import sys

import pytest

from ir import schedule
from ir.schedule import (
    DFLT_LABEL,
    ScheduleError,
    ScheduleSpec,
    cron_expression,
    cron_line,
    cron_marker,
    crontab_with_line,
    crontab_without_line,
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
    # The Windows CI leg has no `crontab` binary; the backend is exercised here
    # entirely through the two methods patched above.
    monkeypatch.setattr(schedule._CronBackend, "available", staticmethod(lambda: True))
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
    # A hand-edited `every=-5` must not render as "-1h55m".
    assert format_every(-5) == "unknown"
    assert format_every(0) == "unknown"


# ----- launchd definition -------------------------------------------------- #


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


def test_plist_process_type_is_not_throttled():
    # launchd throttles a `Background` job's CPU and I/O; a corpus rebuild is
    # exactly the work that would stretch past its own interval under throttling.
    assert plist_dict(ScheduleSpec())["ProcessType"] == "Standard"


def test_plist_sets_a_working_directory():
    # launchd starts jobs in "/", so a relative path in a source resolver would
    # behave differently under the agent than in the shell.
    assert plist_dict(ScheduleSpec())["WorkingDirectory"]


def test_plist_bytes_round_trips_through_plistlib():
    import plistlib

    spec = ScheduleSpec(every_minutes=45)
    assert plistlib.loads(schedule.plist_bytes(spec)) == plist_dict(spec)


def test_launchd_accepts_intervals_cron_cannot():
    # The backends genuinely differ here; the plist path must not inherit cron's
    # divisor restriction.
    assert plist_dict(ScheduleSpec(every_minutes=45))["StartInterval"] == 2700


# ----- cron definition ----------------------------------------------------- #


def test_cron_entry_is_exactly_one_line():
    # The block format this replaced had an interior: a job the user typed inside
    # it, or everything after a begin marker whose end marker was lost, got
    # deleted on the next rewrite. One line has no interior.
    assert len(cron_line(ScheduleSpec(every_minutes=15)).splitlines()) == 1


def test_cron_line_carries_env_inline_not_as_crontab_assignments(monkeypatch):
    # Bare NAME=value crontab lines apply to EVERY command after them in the
    # file, so ir's HOME/PATH would silently attach to whatever job the user
    # appends next. Inline `env` has no scope beyond its own line.
    monkeypatch.setenv("PP", "/home/me/proj")
    line = cron_line(ScheduleSpec(every_minutes=60))
    assert "PP=/home/me/proj" in line
    # The five schedule fields must not be environment assignments.
    assert not any("=" in field for field in line.split()[:5])
    assert "env" in line.split()[5]


def test_cron_line_escapes_percent():
    # cron reads an unescaped % as a newline that terminates the command.
    spec = ScheduleSpec(every_minutes=60, log_path="/tmp/100%/maintain.log")
    line = cron_line(spec)
    assert "%" in line
    assert line.count("%") == line.count("\\%")


def test_cron_line_is_self_describing():
    line = cron_line(ScheduleSpec(every_minutes=15, python="/opt/py/bin/python"))
    assert line.startswith("*/15 * * * * ")
    assert cron_marker(DFLT_LABEL) in line
    assert "every=15" in line
    assert "2>&1" in line


def test_spec_rejects_env_values_that_would_inject_crontab_lines():
    with pytest.raises(ScheduleError):
        ScheduleSpec(env=(("PP", "/proj\n*/1 * * * * evil.sh"),))


# ----- crontab editing (the destructive-edit surface) ---------------------- #

OTHER_JOBS = (
    "MAILTO=ops@example.com\n"
    "0 3 * * * /usr/local/bin/backup.sh\n"
    "@reboot /usr/bin/tunnel\n"
    "*/5 * * * * /usr/local/bin/heartbeat.sh\n"
)


def test_crontab_with_line_preserves_every_other_entry():
    result = crontab_with_line(OTHER_JOBS, ScheduleSpec(every_minutes=60))
    for kept in ("MAILTO=ops@example.com", "backup.sh", "@reboot", "heartbeat.sh"):
        assert kept in result


def test_crontab_with_line_replaces_rather_than_appends():
    once = crontab_with_line(OTHER_JOBS, ScheduleSpec(every_minutes=60))
    twice = crontab_with_line(once, ScheduleSpec(every_minutes=15))
    assert twice.count(cron_marker(DFLT_LABEL)) == 1
    assert "every=15" in twice and "every=60" not in twice


def test_crontab_without_line_removes_only_ours():
    populated = crontab_with_line(OTHER_JOBS, ScheduleSpec())
    assert crontab_without_line(populated) == OTHER_JOBS
    # Removing from a crontab that never had an ir line is a no-op, not an error.
    assert crontab_without_line(OTHER_JOBS) == OTHER_JOBS
    assert crontab_without_line("") == ""


def test_a_user_job_adjacent_to_ours_survives_every_rewrite():
    # The block format deleted a job typed just above its end marker. Prove the
    # line format cannot: put user jobs on both sides and rewrite twice.
    text = crontab_with_line(OTHER_JOBS, ScheduleSpec(every_minutes=60))
    text += "*/5 * * * * /usr/local/bin/typed-after-ir.sh\n"
    text = crontab_with_line(text, ScheduleSpec(every_minutes=15))
    text = crontab_without_line(text)
    assert "typed-after-ir.sh" in text
    assert "heartbeat.sh" in text
    assert cron_marker(DFLT_LABEL) not in text


def test_a_user_comment_mentioning_the_label_is_not_eaten():
    text = f"# reminder about {DFLT_LABEL}\n0 3 * * * /usr/local/bin/backup.sh\n"
    assert crontab_without_line(text) == text


def test_crontab_editing_does_not_split_on_exotic_characters():
    # str.splitlines also breaks on \x0b \x0c \x1c and U+2028, none of which end a
    # crontab line -- a command containing one would become two entries.
    text = "0 3 * * * /bin/echo 'a\x0bb\x1cc'\n"
    assert crontab_without_line(text) == text
    assert len(schedule._split_lines(text.rstrip("\n"))) == 1


def test_crontab_round_trip_reads_back_what_it_wrote(monkeypatch):
    monkeypatch.setenv("PP", "/home/me/proj")
    spec = ScheduleSpec(every_minutes=15, python="/opt/py/bin/python")
    text = crontab_with_line(OTHER_JOBS, spec)
    parsed = schedule._parse_cron_line(
        schedule._find_our_line(text, DFLT_LABEL), DFLT_LABEL
    )
    assert parsed["every_minutes"] == 15
    assert parsed["python"] == "/opt/py/bin/python"
    assert dict(parsed["env"])["PP"] == "/home/me/proj"
    assert parsed["log_path"] == spec.log_path


def test_parsing_survives_an_unbalanced_quote():
    # A hand-added trailing comment with an apostrophe must degrade to "cannot
    # read the details", not take the status report down with a ValueError.
    line = (
        "*/15 * * * * /usr/bin/env /py -u -m ir maintain --all >> /log 2>&1 "
        f"{cron_marker(DFLT_LABEL)} every=15 # don't touch"
    )
    parsed = schedule._parse_cron_line(line, DFLT_LABEL)
    assert parsed["every_minutes"] == 15  # the interval still reads back


# ----- backend resolution -------------------------------------------------- #


def test_resolve_backend_by_name():
    assert resolve_backend("cron", require_available=False).name == "cron"
    assert resolve_backend("launchd", require_available=False).name == "launchd"


def test_resolve_backend_rejects_unknown_names():
    with pytest.raises(ScheduleError) as excinfo:
        resolve_backend("systemd")
    assert "launchd" in str(excinfo.value) and "cron" in str(excinfo.value)


def test_naming_a_backend_this_machine_lacks_is_an_error_not_a_half_install(
    monkeypatch,
):
    # `--backend launchd` on Linux would otherwise write a plist into a directory
    # nothing reads, and then report installed=True.
    monkeypatch.setattr(
        schedule._LaunchdBackend, "available", staticmethod(lambda: False)
    )
    with pytest.raises(ScheduleError) as excinfo:
        schedule.status(backend="launchd")
    assert "not available" in str(excinfo.value)


def test_resolve_backend_falls_back_to_unsupported(monkeypatch):
    monkeypatch.setattr(
        schedule._LaunchdBackend, "available", staticmethod(lambda: False)
    )
    monkeypatch.setattr(schedule._CronBackend, "available", staticmethod(lambda: False))
    impl = resolve_backend()
    assert impl.name == "unsupported"
    state = impl.status(DFLT_LABEL)
    assert state.installed is False
    assert "Task Scheduler" in state.detail
    with pytest.raises(ScheduleError):
        impl.install(ScheduleSpec())


def test_status_never_raises_on_this_platform():
    # Whatever machine CI runs on, asking is always safe.
    assert schedule.status().backend in {"launchd", "cron", "unsupported"}


def test_a_new_backend_needs_only_the_BACKENDS_tuple(monkeypatch):
    # The seam's whole claim: adding `systemd --user` must not require editing the
    # previewer or the renderer. A backend owns its own preview and whether
    # "loaded" means anything for it.
    class _Fake:
        name = "systemd"
        reports_loaded = True

        @staticmethod
        def available():
            return True

        def preview(self, spec):
            return "systemd --user timer definition"

        def status(self, label):
            return schedule.ScheduleStatus(
                installed=True,
                backend=self.name,
                label=label,
                every_minutes=60,
                loaded=True,
                command="/py -u -m ir maintain --all",
            )

        def install(self, spec):
            raise AssertionError("not reached")

        def remove(self, label):
            raise AssertionError("not reached")

    monkeypatch.setattr(schedule, "BACKENDS", (_Fake,))
    state = schedule.status(backend="systemd")
    assert state.backend == "systemd"
    rendered = schedule.render(state)
    assert "loaded" in rendered  # not gated on the backend's name
    assert resolve_backend("systemd").preview(ScheduleSpec()) == (
        "systemd --user timer definition"
    )


def test_backends_expose_the_same_protocol():
    for backend in (*schedule.BACKENDS, schedule._UnsupportedBackend):
        for method in ("available", "preview", "status", "install", "remove"):
            assert hasattr(backend, method), f"{backend.__name__} lacks {method}"
        assert isinstance(backend.reports_loaded, bool)


# ----- process failures must report, not crash ----------------------------- #


def test_a_missing_scheduler_binary_reports_instead_of_crashing(monkeypatch):
    def no_such_tool(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(schedule.subprocess, "run", no_such_tool)
    monkeypatch.setattr(schedule._CronBackend, "available", staticmethod(lambda: True))
    with pytest.raises(ScheduleError) as excinfo:
        schedule.status(backend="cron")
    assert "crontab" in str(excinfo.value)


def test_a_non_utf8_crontab_does_not_crash_status(monkeypatch):
    # One latin-1 byte in a comment must not surface a UnicodeDecodeError.
    import subprocess as sp

    def latin1_crontab(*args, **kwargs):
        return sp.CompletedProcess(args[0], 0, stdout=b"# caf\xe9\n", stderr=b"")

    monkeypatch.setattr(schedule.subprocess, "run", latin1_crontab)
    monkeypatch.setattr(schedule._CronBackend, "available", staticmethod(lambda: True))
    assert schedule.status(backend="cron").installed is False


# ----- staleness checks ---------------------------------------------------- #


def test_interpreter_check_sees_two_different_venvs(tmp_path):
    # Path.resolve() collapses two venv interpreters onto the same base python and
    # reports no difference -- while their site-packages hold different irs. The
    # comparison has to be on the unresolved paths.
    other = tmp_path / "venv" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    problems = schedule._interpreter_problems(str(other))
    assert any("different installed" in problem for problem in problems)


def test_interpreter_check_is_quiet_for_the_running_interpreter():
    assert schedule._interpreter_problems(sys.executable) == ()


def test_missing_interpreter_is_reported_with_the_repair():
    problems = schedule._interpreter_problems("/gone/bin/python")
    assert any("--restart" in problem for problem in problems)


def test_a_job_missing_load_bearing_env_is_reported(monkeypatch):
    # A job installed from a shell without $PP fails on packages and reports every
    # time it fires, while exiting 0 and looking healthy.
    monkeypatch.setenv("PP", "/home/me/proj")
    monkeypatch.delenv("PTH_FILEPATH", raising=False)
    problems = schedule._env_problems((("HOME", "/home/me"),))
    assert any("PP" in problem for problem in problems)
    # A job that carries what this shell has is not flagged.
    assert schedule._env_problems((("PP", "/home/me/proj"),)) == ()


# ----- the operations round trip ------------------------------------------- #


def test_install_inspect_update_remove(fake_crontab):
    assert schedule.status(backend="cron").installed is False

    created = schedule.ensure(backend="cron", every="15m")
    assert (created.installed, created.action, created.every_minutes) == (
        True,
        "installed",
        15,
    )
    assert created.changed is True and "maintain" in (created.command or "")

    # Re-running the bare command must NOT silently reinstall over a tuned schedule.
    again = schedule.ensure(backend="cron")
    assert (again.action, again.changed, again.every_minutes) == ("unchanged", False, 15)

    changed = schedule.ensure(backend="cron", every="1h")
    assert (changed.action, changed.every_minutes) == ("updated", 60)
    assert fake_crontab["text"].count(cron_marker(DFLT_LABEL)) == 1

    gone = schedule.remove(backend="cron")
    assert (gone.installed, gone.action) == (False, "removed")
    assert cron_marker(DFLT_LABEL) not in fake_crontab["text"]
    assert schedule.remove(backend="cron").action == "unchanged"


def test_changing_the_interval_preserves_the_stored_environment(
    fake_crontab, monkeypatch
):
    # THE critical regression. An existing definition is data, not a template to
    # re-derive: operating a working job from a shell that lacks $PP must not
    # silently re-point it at a different (or empty) corpus store.
    monkeypatch.setenv("PP", "/home/me/proj")
    monkeypatch.setenv("PTH_FILEPATH", "/home/me/proj/my_packages.pth")
    schedule.ensure(backend="cron", every="15m")

    monkeypatch.delenv("PP")
    monkeypatch.delenv("PTH_FILEPATH")
    after = schedule.ensure(backend="cron", every="1h")
    assert dict(after.env)["PP"] == "/home/me/proj"
    assert dict(after.env)["PTH_FILEPATH"] == "/home/me/proj/my_packages.pth"


def test_restart_preserves_the_environment_while_repinning_the_interpreter(
    fake_crontab, monkeypatch
):
    monkeypatch.setenv("PP", "/home/me/proj")
    schedule.install(backend="cron", every="30m", python="/gone/bin/python")
    stale = schedule.status(backend="cron")
    assert any("no longer exists" in problem for problem in stale.problems)

    monkeypatch.delenv("PP")
    healed = schedule.restart(backend="cron")
    assert healed.action == "restarted"
    assert healed.every_minutes == 30  # the interval survives
    assert dict(healed.env)["PP"] == "/home/me/proj"  # so does the environment
    assert healed.python == sys.executable  # but the interpreter is re-pinned
    assert healed.problems == ()


def test_a_fresh_install_snapshots_the_current_environment(fake_crontab, monkeypatch):
    monkeypatch.setenv("PP", "/first/proj")
    schedule.ensure(backend="cron", every="1h")
    schedule.remove(backend="cron")
    monkeypatch.setenv("PP", "/second/proj")
    assert dict(schedule.ensure(backend="cron").env)["PP"] == "/second/proj"


def test_dry_run_writes_nothing(fake_crontab):
    state = schedule.install(backend="cron", every="15m", dry_run=True)
    assert state.action == "would-install"
    assert fake_crontab["text"] == ""
    assert "*/15 * * * *" in state.detail


def test_restart_requires_an_installed_schedule(fake_crontab):
    with pytest.raises(ScheduleError) as excinfo:
        schedule.restart(backend="cron")
    assert "ir schedule" in str(excinfo.value)


def test_install_keeps_the_existing_interval_when_none_is_given(fake_crontab):
    schedule.install(backend="cron", every="15m")
    assert schedule.install(backend="cron").every_minutes == 15


# ----- reporting ----------------------------------------------------------- #


def test_render_absent_schedule_says_how_to_create_one(fake_crontab):
    text = schedule.render(schedule.status(backend="cron"))
    assert "Nothing is scheduled" in text and "ir schedule" in text


def test_render_existing_schedule_shows_the_operations_menu(fake_crontab):
    text = schedule.render(schedule.ensure(backend="cron", every="15m"))
    for expected in ("--status", "--every", "--restart", "--remove", "ir maintain"):
        assert expected in text


def test_render_surfaces_problems(fake_crontab):
    schedule.install(backend="cron", every="1h", python="/gone/bin/python")
    text = schedule.render(schedule.status(backend="cron"))
    assert "Problems:" in text and "--restart" in text


def test_cron_status_does_not_claim_a_loaded_state(fake_crontab):
    # cron has no load step; claiming "loaded: yes" would be an invented fact.
    state = schedule.ensure(backend="cron", every="1h")
    assert state.loaded is None
    assert "loaded" not in schedule.render(state).split("To operate it:")[0]


def test_status_to_dict_is_json_ready(fake_crontab):
    import json

    schedule.ensure(backend="cron", every="1h")
    payload = schedule.status(backend="cron").to_dict()
    assert json.loads(json.dumps(payload))["every_minutes"] == 60


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


# ----- the CLI command ----------------------------------------------------- #


def test_cli_schedule_round_trip(fake_crontab):
    from ir.cli import schedule as cli_schedule

    assert "Nothing is scheduled" in cli_schedule(status=True, backend="cron")
    assert "Installed" in cli_schedule(every="15m", backend="cron")
    assert "already scheduled" in cli_schedule(backend="cron")
    assert "Removed" in cli_schedule(remove=True, backend="cron")


def test_cli_schedule_reports_errors_without_a_traceback(fake_crontab):
    from ir.cli import schedule as cli_schedule

    message = cli_schedule(every="45m", backend="cron")
    assert message.startswith("ir schedule: ")
    assert "cron cannot fire evenly" in message


def test_cli_schedule_refuses_contradictory_flags(fake_crontab):
    from ir.cli import schedule as cli_schedule

    # Silently letting one flag win made `--restart --every 15m` report success
    # while ignoring the interval.
    assert "cannot be combined" in cli_schedule(status=True, remove=True, backend="cron")
    assert "cannot be combined" in cli_schedule(
        restart=True, every="15m", backend="cron"
    )


def test_schedule_command_is_registered():
    from ir.cli import COMMANDS, schedule as cli_schedule

    assert cli_schedule in COMMANDS
