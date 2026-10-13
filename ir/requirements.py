"""Dependency check for ``ir``: name what is missing and how to install it.

``ir`` imports its dependencies deep inside ``ir/base.py`` and friends; on an
interpreter that lacks them (a fresh server's system ``python3``) that surfaces
as a bare ``ModuleNotFoundError: No module named 'numpy'`` that says nothing
about ``ir`` or the fix. :func:`check_requirements` runs first, from
``ir/__init__.py``, using only the standard library, and raises
:class:`MissingRequirements` (a :class:`ModuleNotFoundError`) listing every
missing import, the interpreter it was looked up in, and the install commands.
It also checks the version bounds ``ir`` declares for them (``cw>=0.1.1,<0.2``),
read from ``ir``'s installed metadata, and raises :class:`UnsatisfiedRequirements`
for an installed dependency outside them, instead of an opaque error later.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
from collections.abc import Iterable, Mapping

#: the distribution whose declared requirements hold the version bounds
DIST = "ir"

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


class UnsatisfiedRequirements(ImportError):
    """A dependency is installed in a version outside the bounds ``ir`` declares."""


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


_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;]*)")
_CLAUSE = re.compile(r"^\s*(===|==|!=|~=|>=|<=|>|<)\s*([A-Za-z0-9.*+!_-]+)\s*$")
_RELEASE = re.compile(r"^v?(?:\d+!)?(\d+(?:\.\d+)*)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _release(version: str) -> tuple[int, ...] | None:
    """The numeric release segments of *version* (``'0.2.0rc1'`` -> ``(0, 2, 0)``)."""
    found = _RELEASE.match(version.strip())
    return tuple(int(n) for n in found.group(1).split(".")) if found else None


def _padded(a: tuple[int, ...], b: tuple[int, ...]):
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def _clause_holds(op: str, bound: str, installed: str) -> bool:
    """One ``op bound`` clause against *installed*, comparing numeric releases only
    (the stdlib fallback for ``packaging``); a clause it cannot read holds."""
    have, want = _release(installed), _release(bound.rstrip(".*"))
    if have is None or want is None:
        return True
    if op in ("==", "!=") and bound.endswith(".*"):
        same = have[: len(want)] == want
        return same if op == "==" else not same
    if op == "~=":  # compatible release: >= bound, and == bound with its last part dropped
        keep = max(len(want) - 1, 1)
        have, want = _padded(have, want)
        return have >= want and have[:keep] == want[:keep]
    have, want = _padded(have, want)
    return {
        "==": have == want,
        "===": have == want,
        "!=": have != want,
        ">=": have >= want,
        "<=": have <= want,
        ">": have > want,
        "<": have < want,
    }[op]


def version_satisfies(installed: str, specifier: str) -> bool:
    """Whether *installed* meets *specifier* (``'>=0.1.1,<0.2'``).

    Uses ``packaging`` when it is installed, else compares numeric releases with the
    standard library alone; a specifier it cannot read is taken to hold.

    >>> version_satisfies("0.1.6", ">=0.1.1,<0.2"), version_satisfies("0.2.0", "<0.2")
    (True, False)
    >>> version_satisfies("1.4", "~=1.2"), version_satisfies("2.0", "~=1.2")
    (True, False)
    """
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion
    except ImportError:
        pass
    else:
        try:
            return SpecifierSet(specifier).contains(installed, prereleases=True)
        except (InvalidSpecifier, InvalidVersion):
            return True
    for clause in filter(None, (c.strip() for c in specifier.strip("() ").split(","))):
        parsed = _CLAUSE.match(clause)
        if parsed and not _clause_holds(parsed.group(1), parsed.group(2), installed):
            return False
    return True


def declared_bounds(dist: str = DIST) -> dict[str, str]:
    """``{normalized dependency name: specifier}`` that *dist*'s installed metadata
    declares (extras excluded); empty when *dist* is not installed or has none."""
    try:
        lines = importlib.metadata.requires(dist) or []
    except importlib.metadata.PackageNotFoundError:
        return {}
    bounds = {}
    for line in lines:
        if "extra ==" in line or "extra==" in line:
            continue
        parsed = _REQUIREMENT.match(line)
        if parsed and parsed.group(2).strip():
            bounds[_normalize(parsed.group(1))] = parsed.group(2).strip()
    return bounds


def unsatisfied_requirements(
    required: Mapping[str, str] = REQUIRED, *, bounds: Mapping[str, str] | None = None
) -> list[tuple[str, str, str]]:
    """``(distribution, installed version, specifier)`` of each dependency in
    *required* that is installed outside its declared bounds."""
    bounds = declared_bounds() if bounds is None else bounds
    found = []
    for dist in required.values():
        specifier = bounds.get(_normalize(dist))
        if not specifier:
            continue
        try:
            installed = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        if not version_satisfies(installed, specifier):
            found.append((dist, installed, specifier))
    return found


def unsatisfied_message(unsatisfied: Iterable[tuple[str, str, str]]) -> str:
    """The actionable message for *unsatisfied* ``(dist, installed, specifier)``."""
    unsatisfied = list(unsatisfied)
    have = ", ".join(f"{d} {v} (needs {s})" for d, v, s in unsatisfied)
    pins = " ".join(f'"{d}{s}"' for d, _, s in unsatisfied)
    return (
        f"ir cannot run in this interpreter ({sys.executable}): installed "
        f"dependencies outside the bounds ir declares: {have}. Install versions "
        f"within them:\n"
        f"  uv pip install --python {sys.executable} {pins}"
    )


def check_requirements(
    required: Mapping[str, str] = REQUIRED, *, bounds: Mapping[str, str] | None = None
) -> None:
    """Raise :class:`MissingRequirements` unless every dependency is importable, then
    :class:`UnsatisfiedRequirements` unless each is within the bounds ``ir`` declares
    (*bounds*: ``{name: specifier}``, default read from ``ir``'s metadata)."""
    missing = missing_requirements(required)
    if missing:
        raise MissingRequirements(requirements_message(missing), name=missing[0])
    unsatisfied = unsatisfied_requirements(required, bounds=bounds)
    if unsatisfied:
        raise UnsatisfiedRequirements(
            unsatisfied_message(unsatisfied), name=unsatisfied[0][0]
        )
