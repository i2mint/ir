"""``python -m ir`` / ``ir`` CLI entry point (cw dispatch)."""

from __future__ import annotations

import dataclasses

import cw

from .cli import COMMANDS

#: ir's grammar takes ordinary parameters positionally (``ir build NAME``,
#: ``ir search NAME QUERY``) and exposes keyword-only parameters as options.
#: That is ``BY_NAME_IF_KWONLY``, and it is load-bearing rather than decorative:
#: under cw's ARGH default (``BY_NAME_IF_HAS_DEFAULT``) the two commands with a
#: defaulted *positional* parameter change grammar -- ``ir maintain [name]``
#: becomes ``ir maintain --name NAME`` and ``ir coverage [name]`` likewise.
#: ``tests/test_cli_surface.py`` pins that difference so this line cannot be
#: dropped silently.
CONVENTION = dataclasses.replace(cw.ARGH, naming=cw.BY_NAME_IF_KWONLY)


def mk_parser():
    """Build ir's argument parser (a plain :class:`argparse.ArgumentParser`)."""
    return cw.mk_parser(COMMANDS, convention=CONVENTION)


def main():
    """Dispatch the ``ir`` command-line interface; return the exit code."""
    return cw.run(mk_parser())


if __name__ == "__main__":
    raise SystemExit(main())
