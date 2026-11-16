"""Pin ``ir``'s command-line grammar against what argh produced before the cw migration.

The golden in ``tests/cli_goldens/ir.json`` was recorded by running the real
``ir`` entry point while it still dispatched through ``argh`` (0.31.3, with
``NameMappingPolicy.BY_NAME_IF_KWONLY``). Every vector's exit code and
normalised ``usage:`` line is asserted here, so a grammar change cannot land
unnoticed.

What is deliberately *not* asserted: full ``--help`` bodies and argparse's error
text. CPython rewrites both between versions -- 3.12 changed the "invalid
choice" quoting and the option column -- and this repo's CI spans 3.10, 3.12 and
Windows, so pinning them would fail for reasons that have nothing to do with
``ir``. The stronger check (byte-identical stdout/stderr across all 31 vectors,
run through both ``python -m ir`` and the ``ir`` console script) was done at
migration time and is recorded in the pull request.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import cw

from ir.__main__ import CONVENTION, main, mk_parser

GOLDEN = json.loads(
    (Path(__file__).parent / "cli_goldens" / "ir.json").read_text(encoding="utf-8")
)
CASES = GOLDEN["cases"]


def _usage(text: str) -> str:
    """The first ``usage:`` block, whitespace-collapsed, prog neutralised."""
    match = re.search(r"^usage: (.*?)(?=\n\S|\n\n|\Z)", text, re.S | re.M)
    if not match:
        return ""
    return re.sub(r"^PROGNAME", "PROG", " ".join(match.group(1).split()))


def _run(argv):
    """Run the CLI in a subprocess with a pinned prog, as a shell would see it."""
    code = (
        "import sys; sys.argv[0] = 'PROGNAME';"
        "from ir.__main__ import main; sys.exit(main())"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "80"},
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: " ".join(c["argv"]) or "(no args)")
def test_grammar_matches_the_argh_recording(case):
    """Every recorded vector still exits the same way with the same usage line."""
    proc = _run(case["argv"])
    assert proc.returncode == case["rc"], (
        f"{case['argv']} exited {proc.returncode}, expected {case['rc']}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert _usage(proc.stdout or proc.stderr) == case["usage"]


def test_no_argument_invocation_prints_usage_to_stdout_and_exits_zero():
    """argh's no-argument behaviour, which plain argparse does NOT reproduce.

    A bare argparse parser with a required subparser exits 2 to stderr. argh
    printed usage to stdout and exited 0, and ``ir`` has always done that; CI
    steps and shell wrappers can depend on it.
    """
    proc = _run([])
    assert proc.returncode == 0
    assert proc.stdout.startswith("usage:")
    assert proc.stderr == ""


def test_argument_errors_exit_non_zero():
    """``main()`` must RETURN the exit code and the entry point must raise it.

    ``cw.run`` returns the code where ``argh`` exited by itself. Forgetting the
    ``sys.exit``/``SystemExit`` wiring makes every argument error exit 0, which
    no other test in this suite would notice.
    """
    assert _run(["no-such-command"]).returncode == 2
    assert _run(["build"]).returncode == 2


def test_maintain_takes_an_optional_positional_not_an_option():
    """``ir maintain [name]`` -- the load-bearing consequence of BY_NAME_IF_KWONLY."""
    usage = mk_parser().format_usage()
    sub = _subparsers()
    assert "[name]" in sub["maintain"].format_usage()
    assert "--name" not in sub["maintain"].format_usage()
    assert "[name]" in sub["coverage"].format_usage()
    assert usage  # top level builds at all


def test_the_naming_convention_is_load_bearing_and_this_test_can_fail():
    """Prove the parity check would catch losing ``naming=BY_NAME_IF_KWONLY``.

    Under cw's ARGH default (``BY_NAME_IF_HAS_DEFAULT``) the two commands with a
    defaulted *positional* parameter change grammar: ``[name]`` becomes
    ``--name``. If this assertion ever stops holding, the convention no longer
    matters and ``ir/__main__.py``'s comment is stale -- not the other way round.
    """
    from ir.cli import COMMANDS

    default_naming = cw.mk_parser(
        COMMANDS, convention=dataclasses.replace(CONVENTION, naming=cw.ARGH.naming)
    )
    changed = _subparsers(default_naming)
    for name in ("maintain", "coverage"):
        usage = changed[name].format_usage()
        assert "[name]" not in usage and "NAME" in usage, (
            f"{name} was expected to lose its optional positional under the "
            f"default naming policy, but its usage is {usage!r}"
        )


def _subparsers(parser=None):
    parser = parser if parser is not None else mk_parser()
    for action in parser._actions:
        if getattr(action, "choices", None):
            return dict(action.choices)
    raise AssertionError("no subparsers found")


def test_every_command_in_COMMANDS_is_reachable_on_the_command_line():
    """The parser is built from ``ir.cli.COMMANDS``, so it cannot drift from it."""
    from ir.cli import COMMANDS

    expected = {f.__name__.replace("_", "-") for f in COMMANDS}
    assert expected == set(_subparsers())


def test_main_is_importable_without_argh():
    """``ir`` no longer depends on argh; nothing in the CLI path may import it."""
    assert main is not None
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ir.__main__; "
            "sys.exit(1 if 'argh' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "importing ir.__main__ pulled in argh"
