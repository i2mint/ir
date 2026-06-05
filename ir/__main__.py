"""``python -m ir`` / ``ir`` CLI entry point (argh dispatch)."""

from __future__ import annotations

from .cli import COMMANDS


def main():
    """Dispatch the ``ir`` command-line interface."""
    import argh

    parser = argh.ArghParser()
    argh.add_commands(parser, COMMANDS)
    parser.dispatch()


if __name__ == "__main__":
    main()
