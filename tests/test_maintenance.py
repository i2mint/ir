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
