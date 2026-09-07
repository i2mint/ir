"""The ``records`` source: a corpus whose records another package owns.

``from_mapping`` takes an in-process object, so a registry entry can never name
it; ``from_records`` takes a ``"module:attr"`` fetcher reference and a
JSON-friendly ``metadata_keys`` list instead, which is what lets a corpus owned
by another package (astern's session synopses / turns) round-trip through
``corpora.json`` and rebuild in a fresh process.
"""

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
