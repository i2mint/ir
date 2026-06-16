"""Per-corpus policy + registry v2 strategy persistence + idempotent maintain (#58)."""

from datetime import datetime, timedelta

import pytest

import ir
from ir import policy as P
from ir.policy import MaintenancePolicy, in_downtime, is_reindex_due, resolve_policy
from ir.strategy import Chunked, strategy_from_spec, strategy_to_spec

# --------------------------------------------------------------------------- #
# Strategy spec round-trip (the v1 gap: registry could not persist a strategy)
# --------------------------------------------------------------------------- #


def test_strategy_spec_roundtrip():
    spec = strategy_to_spec(Chunked(chunk_size=800, overlap=100))
    assert spec["name"] == "Chunked"
    assert spec["params"]["chunk_size"] == 800
    s2 = strategy_from_spec(spec)
    assert isinstance(s2, Chunked) and s2.chunk_size == 800 and s2.overlap == 100


def test_strategy_from_spec_none_is_none():
    assert strategy_from_spec(None) is None


def test_strategy_from_spec_unknown_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        strategy_from_spec({"name": "Nope", "params": {}})


# --------------------------------------------------------------------------- #
# Policy resolution + smart per-kind defaults
# --------------------------------------------------------------------------- #


def test_sessions_kind_default_is_interval_with_downtime():
    pol = P.default_policy_for_kind("sessions")
    assert pol.reindex.on == "interval" and pol.reindex.every_hours == 24
    assert pol.synopsis.enabled is False
    assert pol.synopsis.downtime_hours == (2, 6)


def test_small_corpus_kinds_default_to_source_change():
    for kind in ("skills", "packages", "reports", "files"):
        assert P.default_policy_for_kind(kind).reindex.on == "source-change"


def test_entry_maintenance_overrides_kind_default():
    pol = resolve_policy(
        {"kind": "sessions", "maintenance": {"synopsis": {"enabled": True}}}
    )
    assert pol.synopsis.enabled is True  # overridden
    assert pol.reindex.on == "interval"  # kept from the kind default


def test_v1_entry_resolves_to_kind_default():
    assert resolve_policy({"kind": "skills"}).reindex.on == "source-change"


# --------------------------------------------------------------------------- #
# Timing predicates
# --------------------------------------------------------------------------- #


def test_source_change_is_always_due():
    assert is_reindex_due(MaintenancePolicy(), None, datetime(2026, 6, 16, 12)) is True


def test_interval_due_only_when_stale():
    pol = resolve_policy({"kind": "sessions"})  # interval 24h
    now = datetime(2026, 6, 16, 12)
    assert is_reindex_due(pol, None, now) is True
    assert is_reindex_due(pol, now - timedelta(hours=1), now) is False
    assert is_reindex_due(pol, now - timedelta(hours=25), now) is True


def test_manual_is_never_due():
    pol = resolve_policy({"kind": "x", "maintenance": {"reindex": {"on": "manual"}}})
    assert is_reindex_due(pol, None, datetime(2026, 6, 16, 12)) is False


def test_downtime_window_and_midnight_wrap():
    day = resolve_policy(
        {"kind": "s", "maintenance": {"synopsis": {"downtime_hours": [2, 6]}}}
    )
    assert in_downtime(day, datetime(2026, 6, 16, 3)) is True
    assert in_downtime(day, datetime(2026, 6, 16, 7)) is False
    wrap = resolve_policy(
        {"kind": "s", "maintenance": {"synopsis": {"downtime_hours": [22, 6]}}}
    )
    assert in_downtime(wrap, datetime(2026, 6, 16, 23)) is True
    assert in_downtime(wrap, datetime(2026, 6, 16, 5)) is True
    assert in_downtime(wrap, datetime(2026, 6, 16, 12)) is False


def test_no_downtime_window_is_always():
    assert in_downtime(MaintenancePolicy(), datetime(2026, 6, 16, 12)) is True


# --------------------------------------------------------------------------- #
# Registry v2 persistence (isolated config dir)
# --------------------------------------------------------------------------- #


def test_register_persists_and_reconstructs_strategy(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    ir.register(
        "notes",
        "files",
        root=str(tmp_path),
        strategy=Chunked(chunk_size=900),
        maintenance={"reindex": {"on": "interval", "every_hours": 12}},
    )
    entry = ir.corpora()["notes"]
    assert entry["strategy"]["params"]["chunk_size"] == 900
    assert entry["maintenance"]["reindex"]["every_hours"] == 12

    from ir.registry import source_from_entry

    src = source_from_entry("notes", entry)
    assert isinstance(src.indexing_strategy, Chunked)
    assert src.indexing_strategy.chunk_size == 900


def test_register_rejects_bad_reindex_trigger(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    with pytest.raises(ValueError, match="unknown reindex trigger"):
        ir.register("x", "skills", maintenance={"reindex": {"on": "whenever"}})


# --------------------------------------------------------------------------- #
# maintain: builds when due, no-ops within the interval (idempotent)
# --------------------------------------------------------------------------- #


def test_maintain_builds_then_noops_within_interval(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha apple avocado", encoding="utf-8")
    ir.register(
        "n",
        "files",
        root=str(docs),
        embedder="light",
        maintenance={"reindex": {"on": "interval", "every_hours": 24}},
    )

    now = datetime(2026, 6, 16, 12)
    (r1,) = ir.maintain("n", now=now)
    assert r1.ran and r1.reindex and r1.records >= 1

    (r2,) = ir.maintain("n", now=now + timedelta(hours=1))  # within interval
    assert not r2.ran and "not due" in r2.reason

    (r3,) = ir.maintain("n", now=now + timedelta(hours=25))  # past interval
    assert r3.ran


def test_maintain_defers_synopsis_build_outside_downtime(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    ir.register(
        "n",
        "files",
        root=str(docs),
        embedder="light",
        maintenance={
            "reindex": {"on": "source-change"},
            "synopsis": {"enabled": True, "downtime_hours": [2, 6]},
        },
    )
    # Noon is outside [2, 6): the (LLM-touching) synopsis build is deferred, so
    # nothing is built — and crucially no aix/LLM is reached.
    (r,) = ir.maintain("n", now=datetime(2026, 6, 16, 12))
    assert not r.ran and "downtime" in r.reason
    assert len(ir.open_corpus("n")) == 0


def test_maintain_dry_run_does_not_build(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    ir.register("n", "files", root=str(docs), embedder="light")
    (r,) = ir.maintain("n", now=datetime(2026, 6, 16, 12), dry_run=True)
    assert not r.ran and "dry-run" in r.reason
    # nothing built: corpus is still empty
    assert len(ir.open_corpus("n")) == 0
