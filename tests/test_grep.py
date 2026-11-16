"""The grep Retriever leaf (#136): index-free, scored, filtered before reading."""

import os
import shutil
import time

import pytest

import ir
from ir.grep import grep_hits, make_grep_retriever

HAS_RG = shutil.which("rg") is not None
BACKENDS = [
    False,
    pytest.param(None, marks=pytest.mark.skipif(not HAS_RG, reason="no rg")),
]


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "proj1").mkdir()
    (tmp_path / "proj2").mkdir()
    (tmp_path / "proj1" / "a.md").write_text("Acme invoice\nnothing\nacme again\n")
    (tmp_path / "proj1" / "b.md").write_text("only one ACME line\n")
    (tmp_path / "proj2" / "c.txt").write_text("acme\nacme\nacme\n")
    (tmp_path / "proj2" / "d.md").write_text("unrelated\n")
    return tmp_path


@pytest.mark.parametrize("use_rg", BACKENDS)
def test_scores_are_matching_line_counts_best_first(tree, use_rg):
    hits = grep_hits("acme", roots=tree, use_rg=use_rg)
    assert [(h.artifact_id, h.score) for h in hits] == [
        ("proj2/c.txt", 3.0),
        ("proj1/a.md", 2.0),
        ("proj1/b.md", 1.0),
    ]
    assert hits[1].text == "Acme invoice" and hits[1].surface_kind == "grep"
    assert hits[0].metadata["n_matches"] == 3


@pytest.mark.parametrize("use_rg", BACKENDS)
def test_case_fixed_glob_and_k(tree, use_rg):
    exact = grep_hits("Acme", roots=tree, ignore_case=False, use_rg=use_rg)
    assert [h.artifact_id for h in exact] == ["proj1/a.md"]
    assert len(grep_hits("acme", roots=tree, k=1, use_rg=use_rg)) == 1
    md = grep_hits("acme", roots=tree, glob="**/*.md", use_rg=use_rg)
    assert {h.artifact_id for h in md} == {"proj1/a.md", "proj1/b.md"}
    assert grep_hits("a.m", roots=tree, fixed=True, use_rg=use_rg) == []
    assert grep_hits("ac.e", roots=tree, use_rg=use_rg)


@pytest.mark.parametrize("use_rg", BACKENDS)
def test_single_file_root_and_missing_matches(tree, use_rg):
    only = grep_hits("acme", roots=tree / "proj1" / "a.md", use_rg=use_rg)
    assert [h.artifact_id for h in only] == ["a.md"]
    assert grep_hits("zzzz", roots=tree, use_rg=use_rg) == []


def test_filter_is_pushed_into_the_scan_before_files_are_read(tree, monkeypatch):
    from ir import grep as g

    seen = []
    real = g._python_counts
    monkeypatch.setattr(
        g, "_python_counts", lambda files, rx: seen.extend(files) or real(files, rx)
    )
    hits = grep_hits(
        "acme",
        roots=tree,
        use_rg=False,
        metadata_of=lambda p: {"project": p.parent.name},
        filter={"project": "proj2"},
    )
    assert [h.artifact_id for h in hits] == ["proj2/c.txt"]
    assert all(f.parent.name == "proj2" for f in seen)  # proj1 was never opened
    assert hits[0].metadata["project"] == "proj2"


def test_filter_without_metadata_of_sees_path_and_filename(tree):
    hits = grep_hits("acme", roots=tree, use_rg=False, filter={"filename": "b.md"})
    assert [h.artifact_id for h in hits] == ["proj1/b.md"]


def test_weight_and_recency_reorder_hits(tree):
    heavy = grep_hits(
        "acme",
        roots=tree,
        use_rg=False,
        weight_of=lambda p: 10.0 if p.name == "b.md" else 1.0,
    )
    assert heavy[0].artifact_id == "proj1/b.md" and heavy[0].score == 10.0
    now = time.time()
    old = tree / "proj2" / "c.txt"
    os.utime(old, (now - 30 * 86400, now - 30 * 86400))
    decayed = grep_hits(
        "acme", roots=tree, use_rg=False, recency_half_life_days=30, now=now
    )
    scores = {h.artifact_id: h.score for h in decayed}
    assert scores["proj2/c.txt"] == pytest.approx(
        1.5, rel=1e-3
    )  # 3 lines, one half-life
    assert decayed[0].artifact_id == "proj1/a.md"


def test_bad_regex_and_missing_rg_are_clear_errors(tree, monkeypatch):
    with pytest.raises(ValueError, match="invalid regular expression"):
        grep_hits("(", roots=tree)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="rg"):
        grep_hits("acme", roots=tree, use_rg=True)
    assert grep_hits("acme", roots=tree)  # default degrades to Python


@pytest.mark.skipif(not HAS_RG, reason="no rg")
def test_rg_incompatible_regex_falls_back_to_python(tree):
    # a lookahead: rg's default engine cannot compile it, Python re can
    hits = grep_hits(r"acme(?= invoice)", roots=tree, use_rg=None)
    assert [h.artifact_id for h in hits] == ["proj1/a.md"]


def test_retriever_contract_binds_roots_and_allows_overrides(tree):
    retr = make_grep_retriever(tree, use_rg=False, source="notes")
    hits = retr("acme", k=1)
    assert len(hits) == 1 and hits[0].source == "notes"
    assert retr.roots == tree
    assert ir.grep_hits is grep_hits and ir.make_grep_retriever is make_grep_retriever


def test_grep_hits_fuse_with_an_index_ranking(tree):
    """Grep hits are ordinary SearchHits: they merge with any other Retriever's."""
    from ir.base import best_per_artifact

    src = ir.CorpusSource.from_mapping(
        {"proj1/a.md": "acme invoice", "zzz": "bake a cake"},
        name="fuse",
        strategy=ir.WholeText(),
    )
    corpus = ir.build(src, store=ir.CorpusStore.memory(), embedder="light")
    index_hits = ir.search(corpus, "acme", mode="lexical")
    merged = best_per_artifact(index_hits + grep_hits("acme", roots=tree, use_rg=False))
    assert {"proj1/a.md", "proj2/c.txt"} <= {h.artifact_id for h in merged}
