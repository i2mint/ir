"""Dependency check for ``ir``: name what is missing and how to install it.

``ir`` imports its dependencies deep inside ``ir/base.py`` and friends; on an
interpreter that lacks them (a fresh server's system ``python3``) that surfaces
as a bare ``ModuleNotFoundError: No module named 'numpy'`` that says nothing
about ``ir`` or the fix. :func:`check_requirements` runs first, from
``ir/__init__.py``, using only the standard library, and raises
:class:`MissingRequirements` (a :class:`ModuleNotFoundError`) listing every
missing import, the interpreter it was looked up in, and the install commands.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable, Mapping

#: import name -> distribution name, for everything ``import ir`` needs.
REQUIRED: Mapping[str, str] = {
    "numpy": "numpy",
    "dol": "dol",
    "ef": "ef",
    "vd": "vd",
    "cw": "cw",
}


class MissingRequirements(ModuleNotFoundError):
    """``ir``'s dependencies are missing from the running interpreter."""


def missing_requirements(required: Mapping[str, str] = REQUIRED) -> list[str]:
    """Distribution names in *required* whose import name cannot be found."""
    return [
        dist
        for module, dist in required.items()
        if importlib.util.find_spec(module) is None
    ]


def requirements_message(missing: Iterable[str]) -> str:
    """The actionable message for *missing* distributions."""
    missing = list(missing)
    return (
        f"ir cannot run in this interpreter ({sys.executable}): missing "
        f"{', '.join(missing)}. Install ir as a standalone command, in its own "
        f"environment (works even where the system Python is externally "
        f"managed):\n"
        f"  uv tool install ir\n"
        f"or into an environment you own:\n"
        f"  uv pip install --python <that-env>/bin/python ir\n"
        f"From a source checkout: uv pip install --python <that-env>/bin/python "
        f"-e <path-to-ir>"
    )


def check_requirements(required: Mapping[str, str] = REQUIRED) -> None:
    """Raise :class:`MissingRequirements` unless every dependency is importable."""
    missing = missing_requirements(required)
    if missing:
        raise MissingRequirements(requirements_message(missing), name=missing[0])
