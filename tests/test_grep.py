"""The grep Retriever leaf (#136): index-free, scored, filtered before reading."""

import os
import shutil
import subprocess
import time

import pytest

import ir
from ir.grep import grep_hits, make_grep_retriever

HAS_RG = shutil.which("rg") is not None
BACKENDS = [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not HAS_RG, reason="no rg")),
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
    assert hits[1].metadata["path"] == "proj1/a.md"  # relative: no local path leaks
    assert hits[0].metadata["n_matches"] == 3


@pytest.mark.parametrize("use_rg", BACKENDS)
def test_case_fixed_glob_and_k(tree, use_rg):
    exact = grep_hits("Acme", roots=tree, ignore_case=False, use_rg=use_rg)
    assert [h.artifact_id for h in exact] == ["proj1/a.md"]
    assert len(grep_hits("acme", roots=tree, k=1, use_rg=use_rg)) == 1
    md = grep_hits("acme", roots=tree, glob="*.md", use_rg=use_rg)
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
    real = g._count_and_first
    monkeypatch.setattr(
        g, "_count_and_first", lambda f, rx: seen.append(f) or real(f, rx)
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
    hits = grep_hits(r"acme(?= invoice)", roots=tree, use_rg=True)
    assert [h.artifact_id for h in hits] == ["proj1/a.md"]


def test_retriever_contract_binds_roots_and_allows_overrides(tree):
    retr = make_grep_retriever(tree, use_rg=False, source="notes")
    hits = retr("acme", k=1)
    assert len(hits) == 1 and hits[0].source == "notes"
    assert retr.roots == [tree]
    other = retr("acme", roots=tree / "proj1")
    assert {h.artifact_id for h in other} == {"a.md", "b.md"}
    with pytest.raises(TypeError):
        retr("acme", mode="hybrid")  # unknown overrides are not silently dropped
    assert ir.grep_hits is grep_hits and ir.make_grep_retriever is make_grep_retriever


def test_a_generator_of_roots_survives_repeated_calls(tree):
    retr = make_grep_retriever((p for p in [tree]), use_rg=False)
    assert retr("acme") and retr("acme")


def test_grep_hits_are_ordinary_hits_that_merge_with_an_index_ranking(tree):
    from ir.base import best_per_artifact

    src = ir.CorpusSource.from_mapping(
        {"proj1/a.md": "acme invoice", "zzz": "bake a cake"},
        name="fuse",
        strategy=ir.WholeText(),
    )
    corpus = ir.build(src, store=ir.CorpusStore.memory(), embedder="light")
    merged = best_per_artifact(
        ir.search(corpus, "acme", mode="lexical")
        + grep_hits("acme", roots=tree, use_rg=False)
    )
    assert {"proj1/a.md", "proj2/c.txt"} <= {h.artifact_id for h in merged}


# ----- review findings: backend equivalence, ids, privacy, guards -------------- #


@pytest.mark.skipif(not HAS_RG, reason="no rg")
def test_rg_really_runs_and_a_newline_in_a_filename_is_fine(tree, monkeypatch):
    weird = tree / "line\nbreak.txt"
    try:
        weird.write_text("acme\n")
    except OSError:
        pytest.skip("filesystem forbids newlines in names")
    calls = []
    real = subprocess.run
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: calls.append(a[0]) or real(*a, **k)
    )
    hits = grep_hits("acme", roots=tree, use_rg=True)
    assert calls and "line\nbreak.txt" in {h.artifact_id for h in hits}


@pytest.mark.parametrize("use_rg", BACKENDS)
def test_crlf_anchors_and_cr_only_lines_agree_across_backends(tmp_path, use_rg):
    (tmp_path / "crlf.txt").write_bytes(b"acme\r\nx acme\r\nacme\r\n")
    (tmp_path / "cr.txt").write_bytes(b"acme\racme\racme\r")
    anchored = grep_hits(r"acme$", roots=tmp_path, glob="crlf.txt", use_rg=use_rg)
    assert [h.score for h in anchored] == [3.0]  # `$` sees through the CR of CRLF
    cr = grep_hits("acme", roots=tmp_path, glob="cr.txt", use_rg=use_rg)
    assert [h.score for h in cr] == [1.0]  # a lone CR does not end a line


def test_same_relative_path_in_two_roots_is_an_error_unless_id_of_is_given(tmp_path):
    for r in ("A", "B"):
        (tmp_path / r).mkdir()
        (tmp_path / r / "n.md").write_text("acme\n")
    with pytest.raises(ValueError, match="share the id"):
        grep_hits("acme", roots=[tmp_path / "A", tmp_path / "B"], use_rg=False)
    hits = grep_hits(
        "acme",
        roots=[tmp_path / "A", tmp_path / "B"],
        use_rg=False,
        id_of=lambda root, p: f"{root.name}/{p.name}",
    )
    assert {h.artifact_id for h in hits} == {"A/n.md", "B/n.md"}


def test_absolute_paths_are_opt_in_and_user_metadata_cannot_shadow_path(tree):
    h = grep_hits(
        "acme", roots=tree, use_rg=False, metadata_of=lambda p: {"path": "x"}
    )[0]
    assert not os.path.isabs(h.metadata["path"])
    h = grep_hits("acme", roots=tree, use_rg=False, absolute_paths=True)[0]
    assert os.path.isabs(h.metadata["path"])


def test_symlinks_out_of_the_root_and_excluded_dirs_are_not_read(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("acme\n")
    root = tmp_path / "root"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "cfg").write_text("acme\n")
    (root / "ok.md").write_text("acme\n")
    try:
        (root / "link.md").symlink_to(outside)
    except OSError:
        pytest.skip("no symlinks")
    assert [h.artifact_id for h in grep_hits("acme", roots=root, use_rg=False)] == [
        "ok.md"
    ]
    allowed = grep_hits("acme", roots=root, use_rg=False, exclude_dirs=())
    assert {h.artifact_id for h in allowed} == {"ok.md", ".git/cfg"}


@pytest.mark.parametrize(
    "kw",
    [
        {"k": 0},
        {"k": -1},
        {"k": None},
        {"recency_half_life_days": 0},
        {"recency_half_life_days": -3},
        {"weight_of": lambda p: -1.0},
        {"weight_of": lambda p: float("nan")},
        {"weight_of": lambda p: float("inf")},
    ],
)
def test_scoring_inputs_are_validated(tree, kw):
    with pytest.raises(ValueError):
        grep_hits("acme", roots=tree, use_rg=False, **kw)
