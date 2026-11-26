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
    assert grep_hits("ac.e", roots=tree, use_rg=False)


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
        g, "_count_and_first", lambda f, rx, *a: seen.append(f) or real(f, rx, *a)
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


def test_regex_patterns_are_read_in_python_and_use_rg_true_refuses_them(tree):
    # a lookahead: only Python can read it; the default quietly uses Python
    hits = grep_hits(r"acme(?= invoice)", roots=tree)
    assert [h.artifact_id for h in hits] == ["proj1/a.md"]
    with pytest.raises(ValueError, match="ASCII literal"):
        grep_hits(r"acme(?= invoice)", roots=tree, use_rg=True)
    with pytest.raises(ValueError, match="ASCII literal"):
        grep_hits("caf\u00e9", roots=tree, use_rg=True)


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
def test_crlf_and_cr_only_lines_agree_across_backends(tmp_path, use_rg):
    (tmp_path / "crlf.txt").write_bytes(b"acme\r\nx acme\r\nacme\r\n")
    (tmp_path / "cr.txt").write_bytes(b"acme\racme\racme\r")
    crlf = grep_hits("acme", roots=tmp_path, glob="crlf.txt", use_rg=use_rg)
    assert [h.score for h in crlf] == [3.0]
    cr = grep_hits("acme", roots=tmp_path, glob="cr.txt", use_rg=use_rg)
    assert [h.score for h in cr] == [1.0]  # a lone CR does not end a line


def test_anchors_see_through_the_cr_of_crlf(tmp_path):
    (tmp_path / "crlf.txt").write_bytes(b"acme\r\nx acme\r\nacme\r\n")
    assert [h.score for h in grep_hits(r"acme$", roots=tmp_path)] == [3.0]


@pytest.mark.parametrize("ignore_case", [True, False])
@pytest.mark.parametrize(
    "pattern", ["acme", "istanbul", "Kelvin", "stra\u00dfe", "a b"]
)
def test_rg_prefilter_never_changes_the_result(tmp_path, pattern, ignore_case):
    """Differential: with or without rg, the hits are identical (incl. fold traps)."""
    texts = [
        "acme here",
        "\u0130stanbul and istanbul",
        "\u212aelvin sign, kelvin, Kelvin",
        "STRASSE stra\u00dfe",
        "a b",
        "caf\u00e9",
        "\ufeffacme at bom",
    ]
    for i, t in enumerate(texts):
        (tmp_path / f"f{i}.txt").write_text(t + "\n", encoding="utf-8")
    kw = dict(roots=tmp_path, ignore_case=ignore_case, fixed=True)
    with_rg = grep_hits(pattern, use_rg=None, **kw)
    without = grep_hits(pattern, use_rg=False, **kw)
    assert [(h.artifact_id, h.score) for h in with_rg] == [
        (h.artifact_id, h.score) for h in without
    ]


@pytest.mark.skipif(not HAS_RG, reason="no rg")
def test_rg_prefilter_result_is_what_is_read(tree, monkeypatch):
    from ir import grep as g

    read = []
    real = g._count_and_first
    monkeypatch.setattr(
        g, "_count_and_first", lambda f, rx, *a: read.append(f.name) or real(f, rx, *a)
    )
    grep_hits("acme", roots=tree, use_rg=True)
    assert sorted(read) == [
        "a.md",
        "b.md",
        "c.txt",
    ]  # d.md was dropped by rg, never read


def test_bom_nul_and_bad_arguments(tmp_path):
    (tmp_path / "b.md").write_bytes(b"\xef\xbb\xbf# title acme\n")
    assert grep_hits("^# title", roots=tmp_path, use_rg=False)  # BOM stripped
    assert grep_hits("\x00", roots=tmp_path, fixed=True) == []  # no rg crash on NUL
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="rg_batch"):
            grep_hits("acme", roots=tmp_path, rg_batch=bad)
    with pytest.raises(ValueError, match="k must"):
        grep_hits("acme", roots=tmp_path, k=True)
    with pytest.raises(ValueError, match="does not exist"):
        grep_hits("acme", roots=tmp_path / "missing")


def test_symlink_loop_does_not_crash_the_search(tmp_path):
    (tmp_path / "ok.md").write_text("acme\n")
    try:
        (tmp_path / "x").symlink_to(tmp_path / "x")
    except OSError:
        pytest.skip("no symlinks")
    assert [h.artifact_id for h in grep_hits("acme", roots=tmp_path, use_rg=False)] == [
        "ok.md"
    ]


def test_overlapping_roots_return_a_file_once(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "n.md").write_text("acme\n")
    hits = grep_hits("acme", roots=[tmp_path, tmp_path / "sub"], use_rg=False)
    assert len(hits) == 1


def test_metadata_of_runs_only_for_filtered_or_returned_files(tree):
    calls = []

    def meta(p):
        calls.append(p.name)
        return {"project": p.parent.name}

    hits = grep_hits("acme", roots=tree, use_rg=False, metadata_of=meta, k=1)
    assert (
        len(calls) == 1 and hits[0].metadata["project"] == "proj2"
    )  # one returned hit
    calls.clear()
    grep_hits(
        "acme", roots=tree, use_rg=False, metadata_of=meta, filter={"project": "proj1"}
    )
    assert set(calls) >= {
        "a.md",
        "b.md",
        "c.txt",
        "d.md",
    }  # the filter needs every file


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


def test_recency_weight_factory_composes_with_weight_of(tree):
    import ir

    now = 10 * 86400.0
    for p in tree.rglob("*.md"):
        os.utime(p, (0, 0))
    rw = ir.recency_weight(10, now=now)
    assert rw(next(tree.rglob("*.md"))) == 0.5
    a = ir.grep_hits("acme", roots=tree, use_rg=False, weight_of=rw)
    b = ir.grep_hits(
        "acme", roots=tree, use_rg=False, recency_half_life_days=10, now=now
    )
    assert [(h.artifact_id, h.score) for h in a] == [
        (h.artifact_id, h.score) for h in b
    ]
    with pytest.raises(ValueError):
        ir.recency_weight(0)


def test_rg_prefilter_is_public_with_defaults(tmp_path):
    import shutil

    from ir.grep import DFLT_RG_BATCH, rg_can_prefilter, rg_prefilter

    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("hello acme\n")
    b.write_text("nothing\n")
    assert rg_can_prefilter("acme") and not rg_can_prefilter("ac.me")
    assert DFLT_RG_BATCH > 0
    if shutil.which("rg"):
        assert rg_prefilter([a, b], "ACME", ignore_case=True) == [a]
