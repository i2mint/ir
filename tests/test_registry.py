"""Tests for the named-corpus registry, facade, and CLI (hermetic)."""

import ir
from ir import cli, registry


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))


def test_register_get_unregister(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    registry.register("foo", "files", root="/some/dir", pattern=r".*\.md$")
    assert registry.get("foo")["kind"] == "files"
    assert "foo" in registry.registered()
    registry.unregister("foo")
    assert registry.get("foo") is None


def test_register_rejects_unknown_kind(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    try:
        registry.register("bad", "nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kind")


def test_files_corpus_build_search_roundtrip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text("How to deploy the app to the server with systemd.")
    (docs / "baking.md").write_text(
        "Recipe to bake a cake in the oven with flour sugar."
    )

    ir.register("notes", "files", root=str(docs), pattern=r".*\.md$")
    corpus = ir.build_corpus("notes", embedder="light")
    assert len(corpus) >= 2

    hits = ir.search("notes", "bake a cake in the oven", k=1)
    assert hits[0].artifact_id == "baking.md"


def test_source_for_auto_registers_preset(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    # A preset name resolves even if not explicitly registered (auto-registered).
    # Use the registry layer; building 'skills'/'packages' needs priv/$PP, so we
    # only assert the registration happens, not the build.
    try:
        registry.source_for("reports")
    except Exception:
        pass  # building the source may need $PP; registration is what we check
    assert registry.get("reports") is not None
    assert registry.get("reports")["kind"] == "reports"


def test_cli_commands_return_strings(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha widgets and gadgets for testing the cli output.")
    cli.register("notes", "files", root=str(docs), pattern=r".*\.md$")
    cli.build("notes", embedder="light")
    assert "notes" in cli.ls()
    assert "alpha" in cli.search("notes", "widgets", k=3) or "a.md" in cli.search(
        "notes", "widgets", k=3
    )
    assert "records:" in cli.info("notes")
