"""``import ir`` names what is missing instead of a bare ModuleNotFoundError (#93)."""

import sys

import pytest

from ir.requirements import (
    MissingRequirements,
    check_requirements,
    missing_requirements,
    requirements_message,
)


def test_all_present_is_silent():
    assert missing_requirements() == []
    check_requirements()


def test_missing_dependency_raises_actionable_error():
    required = {"os": "os", "no_such_module_xyz": "no-such-dist"}
    assert missing_requirements(required) == ["no-such-dist"]
    with pytest.raises(MissingRequirements, match="no-such-dist") as info:
        check_requirements(required)
    assert isinstance(info.value, ModuleNotFoundError)
    assert "uv tool install ir" in str(info.value)


def test_message_names_the_interpreter_and_leads_with_uv_tool():
    import sys

    message = requirements_message(["numpy"])
    assert sys.executable in message
    assert message.index("uv tool install") < message.index("uv pip install")


def test_required_matches_the_declared_dependencies():
    """REQUIRED is a hand copy of [project].dependencies; it must not drift from it."""
    import re
    from pathlib import Path

    tomllib = pytest.importorskip("tomllib")  # Python 3.11+; 3.12 CI covers it

    from ir.requirements import REQUIRED

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    names = {
        re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].strip().lower() for d in declared
    }
    assert set(REQUIRED.values()) == names


# --- version bounds (#96) --------------------------------------------------------

from ir import requirements as _req  # noqa: E402
from ir.requirements import UnsatisfiedRequirements  # noqa: E402


@pytest.mark.parametrize("use_packaging", [True, False])
@pytest.mark.parametrize(
    "installed, specifier, ok",
    [
        ("0.1.6", ">=0.1.1,<0.2", True),
        ("0.2", ">=0.1.1,<0.2", False),
        ("0.2.0", ">=0.1.1,<0.2", False),
        ("0.1.0", ">=0.1.1,<0.2", False),
        ("0.2.0rc1", "<0.2", False),  # PEP 440: a pre-release of 0.2 is not < 0.2
        ("1.4", "~=1.2", True),
        ("2.0", "~=1.2", False),
        ("1.2.5", "==1.2.*", True),
        ("1.3.0", "==1.2.*", False),
        ("3.0", "!=3.0", False),
        ("1.0", "(>=1.0)", True),
        ("1.0", "@weird", True),  # unreadable: assumed to hold
    ],
)
def test_version_satisfies(monkeypatch, use_packaging, installed, specifier, ok):
    if use_packaging:
        pytest.importorskip("packaging")
    else:  # force the stdlib fallback
        monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
    if use_packaging and specifier in ("@weird",):
        pytest.skip("packaging rejects it; covered by the fallback")
    assert _req.version_satisfies(installed, specifier) is ok


def test_installed_dependency_outside_the_bounds_is_named(monkeypatch):
    monkeypatch.setattr(_req.importlib.metadata, "version", lambda dist: "0.2.0")
    bounds = {"cw": ">=0.1.1,<0.2", "numpy": ""}
    found = _req.unsatisfied_requirements(bounds=bounds)
    assert found == [("cw", "0.2.0", ">=0.1.1,<0.2")]
    with pytest.raises(UnsatisfiedRequirements, match=r"cw 0\.2\.0 \(needs >=0\.1\.1,<0\.2\)"):
        check_requirements(bounds=bounds)


def test_declared_bounds_come_from_the_installed_metadata():
    bounds = _req.declared_bounds()
    assert "cw" in bounds and "<0.2" in bounds["cw"]
    assert "pytest" not in bounds and "aix" not in bounds  # extras are not required
    assert _req.declared_bounds("no-such-dist-xyz") == {}


def test_the_installed_environment_meets_its_own_bounds():
    assert _req.unsatisfied_requirements() == []


def test_missing_beats_unsatisfied():
    required = {"no_such_module_xyz": "no-such-dist"}
    with pytest.raises(MissingRequirements):
        check_requirements(required, bounds={"no-such-dist": ">=1"})
