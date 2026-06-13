"""Tests for the semantic link graph (#46) — links view, GraphStore, ingest.

Pins the #46 acceptance: the ``links`` view round-trips (memory + file), edge
ingest turns Package ``deps`` / Skill ``parent`` into REF / PARENT neighbors,
the corpus-backed adapter satisfies the ``GraphStore`` protocol, edges are
derived state outside build identity, and an absent view degrades to "no
edges". Hermetic: light embedder + in-memory / tmp stores.
"""

import json

import pytest

import ir
from ir.base import ledger_key
from ir.graph import (
    CHILD,
    NEXT,
    PARENT,
    PREV,
    REF,
    CorpusGraph,
    GraphStore,
    _dep_name,
    canonical_node_id,
    default_edge_extractor,
)
from ir.retrieve import NoLedgerEntry
from ir.store import CorpusStore

PKG = {
    "aa": {
        "name": "aa",
        "description": "package a",
        "readme": "",
        "owner": "ours",
        "deps": ["dol>=1.0", "numpy[fast]>=1.2 ; python_version>='3.9'", "aa"],
    },
    "bb": {"name": "bb", "description": "package b", "readme": "", "deps": ["aa"]},
    "cc": {"name": "cc", "description": "package c", "readme": "", "deps": []},
}

SKILLS = {
    "deploy": {"name": "deploy", "description": "push apps", "parent": "tw"},
    "build": {"name": "build", "description": "compile", "parent": "tw"},
}


def _packages(name="pkgs", store=None, edges=True):
    # NB: explicit `is None` — an empty CorpusStore is falsy (len 0), so
    # `store or CorpusStore.memory()` would discard a freshly-passed store.
    src = ir.CorpusSource.from_mapping(PKG, name=name, strategy=ir.Package())
    return ir.build(
        src,
        store=CorpusStore.memory() if store is None else store,
        embedder="light",
        edge_extractor=default_edge_extractor if edges else None,
    )


def _skills(store=None):
    src = ir.CorpusSource.from_mapping(SKILLS, name="sk", strategy=ir.Skill())
    return ir.build(
        src,
        store=CorpusStore.memory() if store is None else store,
        embedder="light",
        edge_extractor=default_edge_extractor,
    )


# --------------------------------------------------------------------------- #
# links view — round-trip, hygiene
# --------------------------------------------------------------------------- #


def test_links_view_round_trips_in_memory():
    s = CorpusStore.memory()
    s.set_links("aa", {"REF": ["dol", "i2"]})
    assert s.get_links("aa") == {"REF": ["dol", "i2"]}
    assert list(s.link_items()) == [("aa", {"REF": ["dol", "i2"]})]


def test_links_view_round_trips_file_backed(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    s = CorpusStore.local("g")
    s.set_links("aa", {"REF": ["dol"], "PARENT": [["packages", "tw"]]})
    reopened = CorpusStore.local("g")
    got = reopened.get_links("aa")
    assert got == {"REF": ["dol"], "PARENT": [["packages", "tw"]]}
    json.dumps(got)  # JSON-clean


def test_set_links_drops_empty_edge_lists_and_rows():
    s = CorpusStore.memory()
    s.set_links("aa", {"REF": ["dol"], "PARENT": []})
    assert s.get_links("aa") == {"REF": ["dol"]}
    s.set_links("aa", {"REF": []})  # all-empty → row removed
    assert s.get_links("aa") == {}
    assert list(s.link_items()) == []


def test_get_links_returns_a_copy():
    s = CorpusStore.memory()
    s.set_links("aa", {"REF": ["dol"]})
    got = s.get_links("aa")
    got["REF"].append("MUTATED")
    assert s.get_links("aa") == {"REF": ["dol"]}  # store untouched


def test_absent_links_view_degrades_to_no_edges():
    s = CorpusStore.memory()  # 4-arg construction, links defaults to {}
    assert s.get_links("anything") == {}
    assert CorpusGraph(s).neighbors("anything") == []


# --------------------------------------------------------------------------- #
# default_edge_extractor + _dep_name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("dol", "dol"),
        ("dol>=1.0", "dol"),
        ("numpy[fast]>=1.2", "numpy"),
        ("Requests >= 2 ; python_version>='3.9'", "requests"),
        ("pkg (>=1.0)", "pkg"),
        ("dol@git+https://github.com/i2mint/dol.git", "dol"),  # PEP 508 URL dep
        ("dol @ git+https://github.com/i2mint/dol.git@main", "dol"),  # spaced
    ],
)
def test_dep_name_strips_specifiers(spec, expected):
    assert _dep_name(spec) == expected


def test_default_extractor_self_edge_drop_is_case_insensitive():
    # _dep_name lower-cases, so an "AA" package depending on "aa" (or "AA")
    # is recognized as a self-reference and dropped.
    assert default_edge_extractor("AA", {"deps": ["AA", "dol"]}) == {"REF": ["dol"]}


def test_default_extractor_dedups_duplicate_deps():
    edges = default_edge_extractor("x", {"deps": ["dol", "dol>=1.0", "i2"]})
    assert edges == {"REF": ["dol", "i2"]}


def test_default_extractor_handles_url_dep():
    edges = default_edge_extractor("x", {"deps": ["dol@git+https://h/i2mint/dol"]})
    assert edges == {"REF": ["dol"]}


@pytest.mark.parametrize(
    "target,source,expected",
    [
        ("dol", "packages", ("packages", "dol")),
        (["skills", "deploy"], "packages", ("skills", "deploy")),
        ("x", None, (None, "x")),
    ],
)
def test_canonical_node_id(target, source, expected):
    assert canonical_node_id(target, source=source) == expected


def test_default_extractor_deps_to_ref_and_parent():
    edges = default_edge_extractor("aa", PKG["aa"])
    # version specifiers / extras stripped; self-edge "aa" dropped; de-duped
    assert edges == {"REF": ["dol", "numpy"]}
    assert default_edge_extractor("deploy", SKILLS["deploy"]) == {"PARENT": ["tw"]}
    assert default_edge_extractor("cc", PKG["cc"]) == {}  # empty deps → no edges


# --------------------------------------------------------------------------- #
# CorpusGraph — neighbors, node payload, protocol
# --------------------------------------------------------------------------- #


def test_neighbors_returns_package_deps():
    g = CorpusGraph(_packages())
    assert g.neighbors("aa", edge_type=REF) == ["dol", "numpy"]
    assert g.neighbors("bb", edge_type=REF) == ["aa"]
    assert g.neighbors("cc", edge_type=REF) == []  # empty deps


def test_neighbors_all_edge_types_when_unfiltered():
    s = CorpusStore.memory()
    s.set_links("aa", {"REF": ["dol", "i2"], "PARENT": ["grp"]})
    assert CorpusGraph(s).neighbors("aa") == ["dol", "i2", "grp"]


def test_neighbors_dedups_preserving_order():
    s = CorpusStore.memory()
    s.set_links("aa", {"REF": ["dol", "i2"], "PARENT": ["dol"]})
    assert CorpusGraph(s).neighbors("aa") == ["dol", "i2"]


def test_neighbors_preserves_cross_corpus_pair_targets():
    s = CorpusStore.memory()
    s.set_links("deploy", {"PARENT": [["packages", "tw"]]})
    assert CorpusGraph(s).neighbors("deploy", edge_type=PARENT) == [["packages", "tw"]]


def test_neighbors_dedups_cross_corpus_pairs_unfiltered():
    s = CorpusStore.memory()
    s.set_links("a", {"REF": [["c", "x"], "y"], "PARENT": [["c", "x"]]})
    # The [c, x] pair appears under two edge types — deduped once (lists are
    # unhashable, so dedup keys on the tuple form).
    assert CorpusGraph(s).neighbors("a") == [["c", "x"], "y"]


def test_neighbors_round_trips_through_a_reopened_file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    src = ir.CorpusSource.from_mapping(PKG, name="fpkgs", strategy=ir.Package())
    ir.build(src, embedder="light", edge_extractor=default_edge_extractor)
    reopened = ir.open_corpus("fpkgs")
    assert CorpusGraph(reopened).neighbors("aa", edge_type=REF) == ["dol", "numpy"]


def test_skill_collision_ids_keep_correct_parent_edges():
    # Same-named skills from different parents get collision-safe ids
    # ("name@parent"); the PARENT edge must still point at each one's parent,
    # and the id ("deploy@ops") must not be mistaken for a self-edge.
    skills = {
        "deploy": {"name": "deploy", "description": "d", "parent": "tw"},
        "deploy@ops": {"name": "deploy", "description": "d2", "parent": "ops"},
    }
    src = ir.CorpusSource.from_mapping(skills, name="skc", strategy=ir.Skill())
    g = CorpusGraph(
        ir.build(
            src,
            store=CorpusStore.memory(),
            embedder="light",
            edge_extractor=default_edge_extractor,
        )
    )
    assert g.neighbors("deploy", edge_type=PARENT) == ["tw"]
    assert g.neighbors("deploy@ops", edge_type=PARENT) == ["ops"]


def test_skill_parent_becomes_parent_edge():
    g = CorpusGraph(_skills())
    assert g.neighbors("deploy", edge_type=PARENT) == ["tw"]


def test_getitem_returns_node_records():
    corpus = _packages()
    g = CorpusGraph(corpus)
    recs = g["aa"]
    assert recs and all(r.artifact_id == "aa" for r in recs)
    assert any(r.surface_kind == "description" for r in recs)


def test_getitem_unknown_artifact_raises():
    g = CorpusGraph(_packages())
    with pytest.raises(NoLedgerEntry):
        g["nope"]


def test_corpus_graph_satisfies_graphstore_protocol():
    g = CorpusGraph(_packages())
    assert isinstance(g, GraphStore)
    # A bare object missing neighbors is NOT a GraphStore.
    assert not isinstance(object(), GraphStore)


def test_corpus_graph_accepts_store_or_corpus():
    corpus = _packages()
    via_corpus = CorpusGraph(corpus)
    via_store = CorpusGraph(corpus.store)
    assert via_corpus.neighbors("aa", edge_type=REF) == via_store.neighbors(
        "aa", edge_type=REF
    )
    assert via_corpus.source == "pkgs" and via_store.source is None


def test_edge_vocabulary_constants():
    assert (NEXT, PREV, PARENT, CHILD, REF) == (
        "NEXT",
        "PREV",
        "PARENT",
        "CHILD",
        "REF",
    )


# --------------------------------------------------------------------------- #
# Edges are derived state, outside build identity
# --------------------------------------------------------------------------- #


def test_edges_do_not_change_build_identity():
    with_edges = _packages(name="x", edges=True)
    without = _packages(name="x", edges=False)
    # Same records, same ledger entries (edge ingest touches only the links view).
    assert set(with_edges.store.record_ids()) == set(without.store.record_ids())
    for aid in PKG:
        e1 = with_edges.store.get_ledger_entry(ledger_key(aid))
        e0 = without.store.get_ledger_entry(ledger_key(aid))
        assert e1 == e0  # ledger carries no edge data
    # ... yet only the edge-ingested build has edges.
    assert CorpusGraph(with_edges).neighbors("aa", edge_type=REF) == ["dol", "numpy"]
    assert CorpusGraph(without).neighbors("aa", edge_type=REF) == []


def test_edge_ingest_is_eager_on_rebuild():
    # Build WITHOUT edges, then rebuild the same store WITH an extractor:
    # every artifact gains edges (not just content-changed ones), because
    # ingest decomposes the whole scope, not only the changed set.
    store = CorpusStore.memory()
    _packages(name="x", store=store, edges=False)
    assert CorpusGraph(store).neighbors("aa", edge_type=REF) == []
    _packages(name="x", store=store, edges=True)  # nothing changed but edges
    assert CorpusGraph(store).neighbors("aa", edge_type=REF) == ["dol", "numpy"]
    assert CorpusGraph(store).neighbors("bb", edge_type=REF) == ["aa"]


def test_prune_deletes_links():
    store = CorpusStore.memory()
    src_full = ir.CorpusSource.from_mapping(PKG, name="x", strategy=ir.Package())
    ir.build(
        src_full, store=store, embedder="light", edge_extractor=default_edge_extractor
    )
    assert CorpusGraph(store).neighbors("bb", edge_type=REF) == ["aa"]
    smaller = {k: v for k, v in PKG.items() if k != "bb"}
    src_small = ir.CorpusSource.from_mapping(smaller, name="x", strategy=ir.Package())
    ir.build(
        src_small, store=store, embedder="light", edge_extractor=default_edge_extractor
    )
    assert store.get_links("bb") == {}  # pruned artifact's edges removed
    assert CorpusGraph(store).neighbors("bb") == []


def test_custom_edge_extractor_is_injectable():
    def only_first_dep(aid, fields):
        deps = fields.get("deps") or []
        return {"REF": [str(deps[0]).split(">")[0]]} if deps else {}

    g = CorpusGraph(_packages_custom(only_first_dep))
    assert g.neighbors("aa", edge_type=REF) == ["dol"]


def _packages_custom(extractor):
    src = ir.CorpusSource.from_mapping(PKG, name="pc", strategy=ir.Package())
    return ir.build(
        src, store=CorpusStore.memory(), embedder="light", edge_extractor=extractor
    )
