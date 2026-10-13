"""``import ir`` names what is missing instead of a bare ModuleNotFoundError (#93)."""

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
