"""Claude Code sessions corpus — ClaudeTurn strategy + from_claude_sessions (#57).

The transcript parsing lives in ``priv``; here we inject a ``fetcher`` of
turn-pair records so the ir-side corpus (surfaces, filter fields, presets) is
tested without ``priv`` or real transcripts.
"""

import ir
from ir.store import CorpusStore
from ir.strategy import ClaudeTurn, strategy_from_spec, strategy_to_spec

FAKE = [
    {
        "id": "s1:u1",
        "user_prompt": "how do I deploy the app to the server",
        "assistant_summary": "I deployed it via the deploy script and verified it live",
        "assistant_full": "Let me check the config... I deployed it via the deploy "
        "script and verified it live",
        "session_id": "s1",
        "cwd": "/x/myproj",
        "project": "myproj",
        "git_branch": "main",
        "timestamp": "2026-06-10T10:00:00Z",
        "model": "claude-opus-4-8",
        "has_tool_use": True,
    },
    {
        "id": "s1:u2",
        "user_prompt": "fix the failing numpy import error",
        "assistant_summary": "The numpy 2.x ABI mismatch caused it; pinned numpy "
        "below 2 and the tests pass now",
        "assistant_full": "Investigating the traceback... The numpy 2.x ABI mismatch "
        "caused it; pinned numpy below 2 and the tests pass now",
        "session_id": "s1",
        "cwd": "/x/myproj",
        "project": "myproj",
        "git_branch": "main",
        "timestamp": "2026-06-11T09:00:00Z",
        "model": "claude-opus-4-8",
        "has_tool_use": False,
    },
    {
        "id": "s2:u1",
        "user_prompt": "add a dark mode toggle",
        "assistant_summary": "Added a dark mode toggle wired to a theme context",
        "assistant_full": "Added a dark mode toggle wired to a theme context",
        "session_id": "s2",
        "cwd": "/x/other",
        "project": "other",
        "git_branch": "dev",
        "timestamp": "2026-06-12T08:00:00Z",
        "model": "claude-sonnet-4-6",
        "has_tool_use": True,
    },
]


def _corpus(**kw):
    src = ir.CorpusSource.from_claude_sessions(fetcher=lambda: FAKE, **kw)
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def test_claudeturn_default_surfaces_and_filter_fields():
    plan = ClaudeTurn().decompose("s1:u1", FAKE[0])
    assert {s.kind for s in plan.surfaces} == {"user_prompt", "assistant_summary"}
    assert plan.filter_fields["project"] == "myproj"
    assert plan.filter_fields["has_tool_use"] is True
    assert plan.filter_fields["model"] == "claude-opus-4-8"


def test_claudeturn_include_full_adds_surface():
    plan = ClaudeTurn(include_full=True).decompose("s1:u1", FAKE[0])
    assert "assistant_full" in {s.kind for s in plan.surfaces}


def test_claudeturn_session_title_record_indexes_the_title():
    rec = {
        "id": "s9:__title__",
        "record_type": "session_title",
        "session_title": "Fixing the numpy ABI crash",
        "session_id": "s9",
        "project": "ir",
        "user_prompt": "",
        "assistant_summary": "",
    }
    plan = ClaudeTurn().decompose("s9:__title__", rec)
    assert {s.kind for s in plan.surfaces} == {"session_title"}
    assert plan.surfaces[0].text == "Fixing the numpy ABI crash"
    assert plan.filter_fields["session_title"] == "Fixing the numpy ABI crash"


def test_claudeturn_turn_record_carries_session_title_as_metadata():
    rec = dict(FAKE[0], session_title="the deploy session")
    plan = ClaudeTurn().decompose(rec["id"], rec)
    # a turn record still indexes user/assistant — not a title surface
    assert {s.kind for s in plan.surfaces} == {"user_prompt", "assistant_summary"}
    assert plan.filter_fields["session_title"] == "the deploy session"


def test_build_indexes_two_surfaces_per_turn():
    corpus = _corpus()
    assert len(corpus) == 3 * 2  # 3 turns × {user_prompt, assistant_summary}


def test_search_targets_assistant_side():
    corpus = _corpus()
    hits = corpus.search(
        "numpy abi mismatch", surfaces={"assistant_summary"}, mode="lexical", k=3
    )
    assert hits and hits[0].artifact_id == "s1:u2"


def test_search_targets_user_side():
    corpus = _corpus()
    hits = corpus.search(
        "deploy the app", surfaces={"user_prompt"}, mode="lexical", k=3
    )
    assert hits and hits[0].artifact_id == "s1:u1"


def test_filter_by_project_discriminates():
    corpus = _corpus()
    hits = corpus.search("toggle", filter={"project": "other"}, mode="lexical", k=5)
    assert hits and all(h.metadata.get("project") == "other" for h in hits)


def test_claudeturn_spec_roundtrips_for_registry_persistence():
    spec = strategy_to_spec(ClaudeTurn(include_full=True))
    assert spec["name"] == "ClaudeTurn" and spec["params"]["include_full"] is True
    s2 = strategy_from_spec(spec)
    assert isinstance(s2, ClaudeTurn) and s2.include_full is True


def test_sessions_preset_resolves_and_has_interval_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    # The sessions kind's smart default is interval reindex with a downtime window.
    pol = ir.default_policy_for_kind("sessions")
    assert pol.reindex.on == "interval"
    assert pol.synopsis.downtime_hours == (2, 6)
