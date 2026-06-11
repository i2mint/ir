"""Tests for the agent-ready retrieval seams (ir_09 / epic #38).

Covers `ir.as_retriever` (the Retriever leaf adapter an orchestration layer
registers as one source), `SearchHit.to_dict` / `.pointer` (the ir_09 `Result`
mapping + the serialization-clean edge for a subagent boundary), and
`ir.disclose(store=...)` (the injectable resource-store / pointer-passing
contract of ir_09 §5). Hermetic: the light embedder, in-memory store.
"""

import json

import numpy as np
import pytest

import ir
from ir.base import SearchHit
from ir.select import Selection, disclose
from ir.store import CorpusStore

DOCS = {
    "alpha": "alpha apple avocado almond",
    "beta": "beta banana blueberry blackberry",
    "gamma": "gamma grape guava greengage",
}


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="seams", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _selection(metadata):
    hit = SearchHit("alpha", "description", 1.0, "snippet", metadata)
    return Selection(selected=[hit], candidates=[hit], abstained=False, reason="x")


# ----- as_retriever (the Retriever leaf) ----------------------------------- #


def test_as_retriever_matches_search():
    corpus = _corpus()
    retr = ir.as_retriever(corpus)
    assert [h.artifact_id for h in retr("alpha apple")] == [
        h.artifact_id for h in ir.search(corpus, "alpha apple")
    ]


def test_as_retriever_binds_defaults_and_per_call_override_wins():
    corpus = _corpus()
    retr = ir.as_retriever(corpus, k=2)
    assert len(retr("alpha apple")) <= 2
    assert len(retr("alpha apple", k=1)) <= 1  # per-call override beats the default


def test_as_retriever_exposes_bound_corpus():
    corpus = _corpus()
    retr = ir.as_retriever(corpus)
    assert retr.corpus is corpus


def test_retriever_is_exported():
    assert hasattr(ir, "Retriever")
    assert "Retriever" in ir.__all__ and "as_retriever" in ir.__all__


# ----- SearchHit.to_dict / .pointer (the Result mapping) ------------------- #


def test_searchhit_to_dict_is_json_clean():
    corpus = _corpus()
    hit = ir.search(corpus, "alpha apple")[0]
    d = hit.to_dict()
    assert isinstance(d["score"], float) and not isinstance(d["score"], np.floating)
    json.dumps(d)  # must not raise — no numpy scalar leaks across the boundary


def test_searchhit_pointer_reads_pointer_keys():
    assert SearchHit("a", "k", 1.0, "t", {"skill_path": "/x"}).pointer == "/x"
    assert SearchHit("a", "k", 1.0, "t", {"path": "/y"}).pointer == "/y"
    assert SearchHit("a", "k", 1.0, "t", {}).pointer is None


# ----- disclose(store=...) — pointer-passing over a Mapping (§5) ------------ #


def test_disclose_dereferences_injected_store():
    out = disclose(_selection({"path": "P1", "name": "A"}), store={"P1": "full body"})
    assert out[0].body == "full body"


def test_disclose_store_is_stale_tolerant():
    out = disclose(_selection({"path": "MISSING", "name": "A"}), store={})
    assert out[0].body is None
    assert out[0].metadata.get("disclosure") == "pointer_unreadable"


def test_disclose_rejects_loader_and_store_together():
    with pytest.raises(ValueError, match="either loader= or store="):
        disclose(_selection({}), loader=lambda m: "x", store={})


# ----- POINTER_KEYS SSOT --------------------------------------------------- #


def test_pointer_keys_single_source_of_truth():
    import importlib

    base = importlib.import_module("ir.base")
    # ir.select the attribute is the re-exported *function*, not the submodule.
    select_mod = importlib.import_module("ir.select")
    assert select_mod.POINTER_KEYS is base.POINTER_KEYS


# ----- Selection.sufficient signal (#35) ----------------------------------- #


def test_selection_sufficient_signal():
    committed = _selection({})  # selected=[hit], abstained=False
    abstained = Selection(selected=[], candidates=[], abstained=True, reason="abstain")
    assert committed.sufficient is True
    assert abstained.sufficient is False
    assert committed.to_dict()["sufficient"] is True
    assert abstained.to_dict()["sufficient"] is False


# ----- registry.retrievers() view (#34) ------------------------------------ #


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))


def _build_notes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text("deploy the app to the server with systemd")
    (docs / "baking.md").write_text("bake a cake in the oven with flour and sugar")
    ir.register("notes", "files", root=str(docs), pattern=r".*\.md$")
    ir.build_corpus("notes", embedder="light")


def test_retrievers_view_projects_registry_ssot(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    from ir import registry

    _build_notes(tmp_path)
    view = registry.retrievers()
    assert set(view.keys()) == set(registry.registered().keys())  # a view, not a copy
    assert "notes" in view
    retr = view["notes"]  # lazily opens the corpus, returns a Retriever
    assert [h.artifact_id for h in retr("bake a cake")] == [
        h.artifact_id for h in ir.search("notes", "bake a cake")
    ]


def test_retrievers_view_keyerror_on_unregistered(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    from ir import registry

    with pytest.raises(KeyError):
        registry.retrievers()["does_not_exist"]


def test_retriever_for_binds_defaults(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    from ir import registry

    _build_notes(tmp_path)
    retr = registry.retriever_for("notes", k=1)
    hits = retr("bake a cake in the oven")
    assert hits and hits[0].artifact_id == "baking.md" and len(hits) == 1
