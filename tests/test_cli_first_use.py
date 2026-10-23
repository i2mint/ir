"""A never-built preset corpus is built on first ``search``/``discover`` use (#110)."""

from ir import cli, registry


def test_empty_preset_is_built_on_first_use(monkeypatch, capsys):
    built = []
    state = {"n": 0}

    class _C:
        def __len__(self):
            return state["n"]

    monkeypatch.setattr(cli, "open_corpus", lambda name: _C())
    monkeypatch.setattr(
        cli, "build", lambda name: built.append(name) or state.update(n=3)
    )
    cli._open_ready("skills")
    assert built == ["skills"]
    assert "never built" in capsys.readouterr().err


def test_empty_non_preset_is_left_alone(monkeypatch):
    built = []

    class _C:
        def __len__(self):
            return 0

    monkeypatch.setattr(cli, "open_corpus", lambda name: _C())
    monkeypatch.setattr(cli, "build", lambda name: built.append(name))
    assert "notes" not in registry.PRESETS
    cli._open_ready("notes")
    assert built == []
