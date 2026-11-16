"""The ``records`` source: a corpus whose records another package owns.

``from_mapping`` takes an in-process object, so a registry entry can never name
it; ``from_records`` takes a ``"module:attr"`` fetcher reference and a
JSON-friendly ``metadata_keys`` list instead, which is what lets a corpus owned
by another package (astern's session synopses / turns) round-trip through
``corpora.json`` and rebuild in a fresh process.
"""

import json

import pytest

import ir
from ir import registry
from ir.sources import resolve_fetcher
from ir.store import CorpusStore

RECORDS = [
    {
        "id": "s1",
        "text": "the ledger keeps a resumed session from being judged twice",
        "project": "astern",
        "timestamp": "2026-09-01T10:00:00Z",
    },
    {
        "id": "s2",
        "text": "a dark mode toggle wired to a theme context",
        "project": "other",
        "timestamp": "2026-09-02T10:00:00Z",
    },
]


def fetch():
    """Module-level so it is reachable as a ``"module:attr"`` reference."""
    return RECORDS


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))


# --- resolve_fetcher ---------------------------------------------------------


def test_resolve_fetcher_accepts_callable_colon_ref_and_dotted_ref():
    assert resolve_fetcher(fetch) is fetch
    assert resolve_fetcher("test_records_source:fetch")() == RECORDS
    assert resolve_fetcher("test_records_source.fetch")() == RECORDS


@pytest.mark.parametrize(
    "bad, exc",
    [
        (None, TypeError),
        ("nodots", ValueError),
        ("no.such.module:fetch", ImportError),
        ("test_records_source:nope", AttributeError),
    ],
)
def test_resolve_fetcher_says_what_is_wrong(bad, exc):
    with pytest.raises(exc):
        resolve_fetcher(bad)


# --- from_records ------------------------------------------------------------


def test_from_records_ids_text_and_lifted_metadata():
    src = ir.CorpusSource.from_records(
        name="notes", fetcher=fetch, metadata_keys=["project", "timestamp"]
    )
    assert sorted(src.scope) == ["s1", "s2"]
    assert src.metadata_of("s1", src.scope["s1"]) == {
        "project": "astern",
        "timestamp": "2026-09-01T10:00:00Z",
    }


def test_from_records_without_ids_keeps_every_record():
    src = ir.CorpusSource.from_records(
        name="n", fetcher=lambda: [{"text": "a"}, {"text": "b"}]
    )
    assert sorted(src.scope) == ["n_0", "n_1"]


def test_from_records_metadata_is_a_hard_filter_after_build():
    src = ir.CorpusSource.from_records(
        name="notes", fetcher=fetch, metadata_keys=["project"]
    )
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    hits = corpus.search("toggle", filter={"project": "other"}, mode="lexical", k=5)
    assert hits and all(h.metadata.get("project") == "other" for h in hits)


def test_from_records_takes_any_shipped_strategy():
    turn = {
        "id": "s1:u1",
        "user_prompt": "why is CI red",
        "assistant_summary": "a stale lockfile",
        "session_id": "s1",
        "turn_index": 3,
    }
    src = ir.CorpusSource.from_records(
        name="turns",
        fetcher=lambda: [turn],
        strategy=ir.ClaudeTurn(),
        metadata_keys=["turn_index"],
    )
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    hits = corpus.search("stale lockfile", mode="lexical", k=3)
    assert hits[0].artifact_id == "s1:u1"
    assert hits[0].metadata["turn_index"] == 3


# --- the registry round-trip (the point of the string fetcher) ----------------


def test_records_corpus_round_trips_through_the_registry(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ir.register(
        "notes",
        "records",
        embedder="light",
        strategy={"name": "Chunked", "params": {"text_key": "text"}},
        fetcher="test_records_source:fetch",
        metadata_keys=["project"],
    )
    # Rebuilt from the persisted entry alone — no in-process object involved.
    corpus = ir.build_corpus("notes")
    assert len(corpus) == 2
    hits = ir.search("notes", "judged twice", mode="lexical", k=1)
    assert hits[0].artifact_id == "s1"


def test_records_is_not_auto_registered_from_a_bare_name(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    with pytest.raises(KeyError):  # needs a fetcher; never guessed
        registry.source_for("records")


def test_records_kind_defaults_to_interval_reindex():
    assert ir.default_policy_for_kind("records").reindex.on == "interval"


def test_metadata_only_change_is_reindexed(tmp_path, monkeypatch):
    """#137: a record whose filter metadata changed (same text) is rebuilt."""
    import sys
    import types

    _isolate(tmp_path, monkeypatch)
    mod = types.ModuleType("tmpfetch137")
    mod.state = [{"id": "x", "text": "same text here", "pages": "p.1"}]
    mod.fetch = lambda: [dict(r) for r in mod.state]
    monkeypatch.setitem(sys.modules, "tmpfetch137", mod)
    ir.register(
        "metatest", "records", fetcher="tmpfetch137:fetch", metadata_keys=["pages"]
    )

    def pages():
        hits = ir.search("metatest", "same text", k=1, mode="lexical")
        return [h.metadata["pages"] for h in hits]

    ir.build_corpus("metatest", embedder="light")
    assert pages() == ["p.1"]
    mod.state = [{"id": "x", "text": "same text here", "pages": "p.2"}]
    ir.build_corpus("metatest", embedder="light")
    assert pages() == ["p.2"]


def test_records_signal_is_unchanged_without_metadata_keys():
    import ef
    from ir.sources import records_change_signal

    r = {"id": "x", "text": "hello", "pages": "p.1"}
    assert records_change_signal()("x", r) == ef.content_hash(r)


def test_records_signal_sees_every_field_for_a_custom_text_key():
    from ir.sources import records_change_signal

    sig = records_change_signal(text_key="body")
    assert sig("x", {"body": "t", "title": "a"}) != sig(
        "x", {"body": "t", "title": "b"}
    )


def test_records_signal_is_stable_for_set_valued_metadata():
    """#144: set order varies per process; the signal must not."""
    import subprocess
    import sys

    from ir.sources import records_change_signal

    code = (
        "from ir.sources import records_change_signal as r;"
        "s=r(metadata_keys=['tags'],text_key='text');"
        "print(s('x',{'text':'t','tags':{'alpha','beta','gamma','delta','eps'}}))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": str(seed)},
            check=True,
        ).stdout
        for seed in (1, 2, 3)
    }
    assert len(outs) == 1
    sig = records_change_signal(metadata_keys=["tags"])
    assert sig("x", {"text": "t", "tags": {"a", "b"}}) == sig(
        "x", {"text": "t", "tags": {"b", "a"}}
    )
    assert sig("x", {"text": "t", "tags": {"a"}}) != sig(
        "x", {"text": "t", "tags": {"b"}}
    )


def test_signal_ignores_what_it_does_not_hash_and_matches_the_old_hash_without_sets():
    import ef

    from ir.sources import records_change_signal

    cyclic = {"text": "t"}
    cyclic["parent"] = cyclic  # an unhashed cycle used to be fine and must stay fine
    assert records_change_signal()("x", cyclic) == ef.content_hash(cyclic)
    plain = {"text": "t", "pages": ["p.1", "p.2"], "n": (1, 2)}
    sig = records_change_signal(metadata_keys=["pages"])
    assert sig("x", plain) == ef.content_hash(plain, content_keys=["pages"])
    whole = records_change_signal(text_key="body")
    rec = {"body": "t", "tags": ["a", "b"]}
    assert whole("x", rec) == ef.content_hash(
        json.dumps(rec, sort_keys=True, default=str)
    )
    assert whole("x", {"body": "t", "tags": {"a", "b"}}) == whole(
        "x", {"body": "t", "tags": {"b", "a"}}
    )
