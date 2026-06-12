"""Tests for ``SearchHit.surface_index`` + ledger-backed sibling addressing (#44).

Three contracts of the expansion prerequisite: a hit names its surface position
(the plan-global ``surface_index``, distinct from the per-kind ``chunk_index``
on multi-kind strategies), siblings resolve through the ledger (never by
re-deriving record ids), and the dataclass / JSON contracts stay additive.
Hermetic: in-memory stores + the light hashing embedder.
"""

import json

import pytest

import ir
from ir.base import Record, SearchHit
from ir.store import CorpusStore

LONG = "\n\n".join(f"deployment paragraph {i} server systemd caddy" for i in range(12))

PKG_DOCS = {
    "dol": {
        "name": "dol",
        "description": "dict-like facades over storage backends",
        "readme": LONG,
        "owner": "i2mint",
    },
    "i2": {
        "name": "i2",
        "description": "signature tooling and meta-programming",
        "readme": "",
        "owner": "i2mint",
    },
}


def _chunked_corpus():
    src = ir.CorpusSource.from_mapping(
        {"big": {"text": LONG}},
        name="chunked",
        strategy=ir.Chunked(chunk_size=120, overlap=20),
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _package_corpus(docs=PKG_DOCS, name="pkgs"):
    src = ir.CorpusSource.from_mapping(
        docs, name=name, strategy=ir.Package(chunk_size=120, overlap=20)
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


# --------------------------------------------------------------------------- #
# SearchHit contract: trailing defaulted field, additive to_dict
# --------------------------------------------------------------------------- #


def test_searchhit_surface_index_is_trailing_and_defaults_none():
    # 6-positional construction (the pre-#44 full-arity form, used downstream)
    # must keep meaning the same thing: the 6th positional is still `source`.
    hit = SearchHit("a", "chunk", 0.5, "text", {"k": "v"}, "corpus")
    assert hit.source == "corpus"
    assert hit.surface_index is None


def test_searchhit_to_dict_gains_surface_index_and_stays_json_clean():
    d = SearchHit("a", "chunk", 0.5, "t", surface_index=3).to_dict()
    assert d["surface_index"] == 3
    assert SearchHit("a", "chunk", 0.5, "t").to_dict()["surface_index"] is None
    json.dumps(d)  # additive key is JSON-serializable


# --------------------------------------------------------------------------- #
# Hits carry surface_index — single- and multi-kind strategies
# --------------------------------------------------------------------------- #


def test_hits_carry_surface_index_single_kind():
    corpus = _chunked_corpus()
    hits = ir.search(corpus, "deploy to the server", k=50, per_artifact=False)
    assert len(hits) > 1
    for h in hits:
        assert isinstance(h.surface_index, int)
        # Single-kind plan: the plan-global index IS the per-kind chunk index.
        assert h.surface_index == h.metadata["chunk_index"]


def test_hits_carry_surface_index_multi_kind_offset():
    corpus = _package_corpus()
    chunk_hits = ir.search(
        corpus, "deploy server", k=50, per_artifact=False, surfaces=["readme_chunk"]
    )
    assert len(chunk_hits) > 1
    for h in chunk_hits:
        # `description` occupies plan position 0, shifting readme chunks by 1.
        assert h.surface_index == h.metadata["chunk_index"] + 1
    desc_hits = ir.search(
        corpus, "storage facades", k=5, per_artifact=False, surfaces=["description"]
    )
    assert desc_hits and all(h.surface_index == 0 for h in desc_hits)


# --------------------------------------------------------------------------- #
# records_for_artifact — ledger-backed, ordered sibling addressing
# --------------------------------------------------------------------------- #


def test_records_for_artifact_ordered_single_kind():
    corpus = _chunked_corpus()
    records = ir.records_for_artifact(corpus, "big")
    assert len(records) == len(corpus) > 1
    assert [r.surface_index for r in records] == list(range(len(records)))
    assert all(r.artifact_id == "big" for r in records)


def test_records_for_artifact_multi_kind_ordered_and_filtered():
    corpus = _package_corpus()
    records = ir.records_for_artifact(corpus, "dol")
    assert records[0].surface_kind == "description"
    assert records[0].surface_index == 0
    chunks = ir.records_for_artifact(corpus, "dol", surface_kind="readme_chunk")
    assert len(chunks) == len(records) - 1
    for j, r in enumerate(chunks):
        assert r.surface_index == j + 1
        assert r.metadata["chunk_index"] == j


def test_id_derivation_from_chunk_index_is_wrong_on_multi_kind():
    # The regression #44 guards against: deriving a sibling id from the
    # per-kind chunk_index fetches a missing or *wrong* record on multi-kind
    # plans. The ledger path is the only sound route.
    corpus = _package_corpus()
    chunks = ir.records_for_artifact(corpus, "dol", surface_kind="readme_chunk")
    stored_ids = {r.id for r in chunks}
    derived_for_chunk0 = Record.make_id("dol", "readme_chunk", 0)
    assert derived_for_chunk0 not in stored_ids  # missing sibling
    derived_for_chunk1 = Record.make_id("dol", "readme_chunk", 1)
    assert derived_for_chunk1 == chunks[0].id  # the WRONG sibling (off by one)


def test_records_for_artifact_accepts_store_or_corpus():
    corpus = _chunked_corpus()
    via_corpus = ir.records_for_artifact(corpus, "big")
    via_store = ir.records_for_artifact(corpus.store, "big")
    assert [r.id for r in via_corpus] == [r.id for r in via_store]


def test_records_for_artifact_unknown_artifact_raises_keyerror():
    corpus = _chunked_corpus()
    with pytest.raises(KeyError, match="no ledger entry"):
        ir.records_for_artifact(corpus, "nope")


def test_records_for_artifact_immune_to_plan_dependent_offset():
    # An empty name + description drops the description surface, so readme
    # chunks start at plan position 0 — the offset is plan-dependent. The
    # ledger path returns the right siblings either way.
    corpus = _package_corpus(
        {"anon": {"name": "", "description": "", "readme": LONG}}, name="anonpkg"
    )
    records = ir.records_for_artifact(corpus, "anon")
    assert all(r.surface_kind == "readme_chunk" for r in records)
    for j, r in enumerate(records):
        assert r.surface_index == j == r.metadata["chunk_index"]


# --------------------------------------------------------------------------- #
# Package stamps n_chunks (id-neutral metadata; optional at read time)
# --------------------------------------------------------------------------- #


def test_package_stamps_n_chunks_on_readme_chunks():
    plan = ir.Package(chunk_size=120, overlap=20).decompose("dol", PKG_DOCS["dol"])
    chunks = [s for s in plan.surfaces if s.kind == "readme_chunk"]
    assert len(chunks) > 1
    for j, s in enumerate(chunks):
        assert s.metadata["chunk_index"] == j
        assert s.metadata["n_chunks"] == len(chunks)
        assert type(s.metadata["n_chunks"]) is int  # JSON-clean, no numpy
    (desc,) = [s for s in plan.surfaces if s.kind == "description"]
    assert "n_chunks" not in desc.metadata


def test_n_chunks_flows_to_hits():
    corpus = _package_corpus()
    hits = ir.search(
        corpus, "deploy server", k=50, per_artifact=False, surfaces=["readme_chunk"]
    )
    assert hits and all(isinstance(h.metadata.get("n_chunks"), int) for h in hits)
