"""End-to-end pipeline tests using the light (numpy-only) hashing embedder.

These are hermetic: no model download, no API keys, no network. They exercise
build/search, incremental maintenance, deletion, metadata filtering, surface
selection, and local persistence.
"""

import ir
from ir.base import storage_key
from ir.store import CorpusStore

DOCS = {
    "git": "git version control branching merge commit repository clone",
    "cooking": "recipe oven bake flour sugar cake dessert kitchen knife",
    "astronomy": "telescope galaxy nebula star planet orbit cosmos comet",
}


def _src(docs=DOCS, name="t", **kw):
    return ir.CorpusSource.from_mapping(docs, name=name, strategy=ir.WholeText(), **kw)


def test_build_and_rank():
    corpus = ir.build(_src(), store=CorpusStore.memory(), embedder="light")
    assert len(corpus) == 3
    assert (
        ir.search(corpus, "merge a git branch in the repository", k=1)[0].artifact_id
        == "git"
    )
    assert ir.search(corpus, "bake a cake in the oven", k=1)[0].artifact_id == "cooking"


def test_idempotent_rebuild_is_stable():
    store = CorpusStore.memory()
    ir.build(_src(), store=store, embedder="light")
    v1 = store.get_ledger_entry(storage_key("git"))["version"]
    ir.build(_src(), store=store, embedder="light")
    assert len(store) == 3
    assert store.get_ledger_entry(storage_key("git"))["version"] == v1


def test_incremental_edit_updates_version():
    store = CorpusStore.memory()
    ir.build(_src(), store=store, embedder="light")
    v1 = store.get_ledger_entry(storage_key("git"))["version"]
    edited = dict(DOCS, git=DOCS["git"] + " rebase stash tag")
    ir.build(_src(edited), store=store, embedder="light")
    assert len(store) == 3
    assert store.get_ledger_entry(storage_key("git"))["version"] != v1


def test_full_refresh_prunes_deletions():
    store = CorpusStore.memory()
    ir.build(_src(), store=store, embedder="light")
    smaller = {k: v for k, v in DOCS.items() if k != "astronomy"}
    ir.build(_src(smaller), store=store, embedder="light")
    assert len(store) == 2
    assert store.get_ledger_entry(storage_key("astronomy")) is None


def test_metadata_filter():
    src = _src(
        {k: {"text": v} for k, v in DOCS.items()},
        name="tf",
        metadata_of=lambda aid, raw: {"kind": "tech" if aid == "git" else "other"},
    )
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    hits = ir.search(corpus, "anything at all", k=5, filter={"kind": "tech"})
    assert [h.artifact_id for h in hits] == ["git"]


def test_surface_dedupe_per_artifact():
    # One long doc -> many chunk surfaces; per_artifact collapses to one hit.
    text = "\n\n".join(
        f"deployment paragraph {i} server systemd caddy" for i in range(20)
    )
    src = ir.CorpusSource.from_mapping(
        {"big": {"text": text}},
        name="tc",
        strategy=ir.Chunked(chunk_size=120, overlap=20),
    )
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    assert len(corpus) > 1  # multiple chunk records
    hits = ir.search(corpus, "deploy to the server", k=5)
    assert len(hits) == 1 and hits[0].artifact_id == "big"


def test_local_persistence_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    ir.build(_src(name="persist"), embedder="light")  # default = local store
    reopened = ir.open_corpus("persist")
    assert len(reopened) == 3
    assert ir.search(reopened, "merge a git branch", k=1)[0].artifact_id == "git"
