"""Regression tests for incremental-maintenance edge cases."""

import ir
from ir.base import storage_key
from ir.store import CorpusStore


def test_strategy_change_triggers_reindex():
    # Same content, different strategy -> must re-decompose (not skipped).
    text = "\n\n".join(f"paragraph {i} with several words here" for i in range(30))
    docs = {"d": {"text": text}}
    store = CorpusStore.memory()

    ir.build(
        ir.CorpusSource.from_mapping(docs, name="m", strategy=ir.WholeText()),
        store=store,
        embedder="light",
    )
    assert len(store) == 1  # WholeText -> one surface

    ir.build(
        ir.CorpusSource.from_mapping(
            docs, name="m", strategy=ir.Chunked(chunk_size=120, overlap=20)
        ),
        store=store,
        embedder="light",
    )
    assert len(store) > 1  # Chunked -> many surfaces, despite unchanged content
    assert "Chunked" in store.get_ledger_entry(storage_key("d"))["strategy_id"]


def test_from_skills_fetcher_injection():
    fake = [
        {"name": "alpha", "description": "manage alpha widgets", "parent": "pkgA"},
        {"name": "beta", "description": "manage beta gadgets", "parent": "pkgB"},
    ]
    src = ir.CorpusSource.from_skills(fetcher=lambda: fake)
    assert set(src.scope) == {"alpha", "beta"}
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    assert len(corpus) == 2
    assert ir.search(corpus, "alpha widgets", k=1)[0].metadata["name"] == "alpha"


def test_sweep_is_fault_isolated_per_corpus(monkeypatch):
    # A scheduled run has nobody watching it: one corpus blowing up must not
    # leave the corpora after it silently unmaintained (the failure that left a
    # real corpus stale for months while the sweep "succeeded").
    from ir import maintenance

    monkeypatch.setattr(
        maintenance.registry, "registered", lambda: {"good": {}, "bad": {}, "also": {}}
    )

    def fake_maintain_corpus(name, **kwargs):
        if name == "bad":
            raise RuntimeError("source vanished")
        return maintenance.MaintenanceResult(name, True, "reindexed", reindex=True)

    monkeypatch.setattr(maintenance, "maintain_corpus", fake_maintain_corpus)

    results = maintenance.maintain(all=True)
    assert [r.name for r in results] == ["good", "bad", "also"]
    assert [r.ran for r in results] == [True, False, True]

    failed = results[1]
    assert failed.error == "RuntimeError: source vanished"
    assert "FAILED" in str(failed) and "source vanished" in str(failed)
