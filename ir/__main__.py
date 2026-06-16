"""``python -m ir`` / ``ir`` CLI entry point (argh dispatch)."""

from __future__ import annotations

from .cli import COMMANDS


def main():
    """Dispatch the ``ir`` command-line interface."""
    import argh

    parser = argh.ArghParser()
    # argh >= 0.30 requires an explicit name-mapping policy as soon as a command
    # has an *optional positional* (e.g. ``maintain(name=None, ...)``).
    # ``BY_NAME_IF_KWONLY`` keeps positional params positional and maps
    # keyword-only params to options — exactly ir's existing command convention,
    # so it changes nothing for the other commands. Fall back gracefully on
    # older argh that lacks the policy.
    try:
        policy = argh.NameMappingPolicy.BY_NAME_IF_KWONLY
    except AttributeError:
        try:
            from argh.assembling import NameMappingPolicy

            policy = NameMappingPolicy.BY_NAME_IF_KWONLY
        except ImportError:
            policy = None
    try:
        if policy is not None:
            argh.add_commands(parser, COMMANDS, name_mapping_policy=policy)
        else:
            argh.add_commands(parser, COMMANDS)
    except TypeError:  # very old argh without the kwarg
        argh.add_commands(parser, COMMANDS)
    parser.dispatch()


if __name__ == "__main__":
    main()
