"""Ingestion-coverage regression tests for the ``reports`` source.

These pin the corpus *coverage* contract — that every report on disk that
should be indexed actually reaches ``source.scope`` — at the tightest layer,
independent of any embedder, model, or the live ecosystem (``priv`` / ``$PP``).

They exist because a non-recursive ``glob("*.md")`` once silently dropped every
report living in a subdirectory of ``docs/`` / ``misc/docs/`` (``research/``,
``decisions/``, ``adr/`` …) — a whole class of documents unsearchable with no
error. A back-translation eval can never catch that (its gold is drawn from what
is *already* indexed), so coverage needs its own filesystem-vs-scope assertion.
"""

from pathlib import Path

import ir
from ir.sources import DFLT_EXCLUDE_DIRS


def _make_projects_tree(root: Path) -> None:
    """A minimal ``<group>/<pkg>/(misc/)docs`` tree with nested + noise files."""
    misc_docs = root / "t" / "pkg" / "misc" / "docs"
    (misc_docs / "research").mkdir(parents=True)
    (misc_docs / "top.md").write_text("top-level report about deployment")
    (misc_docs / "research" / "multiplatform.md").write_text(
        "nested research report on platform independence"
    )
    (misc_docs / "research" / "deep" / "nested").mkdir(parents=True)
    (misc_docs / "research" / "deep" / "nested" / "buried.md").write_text(
        "a report several levels deep"
    )
    # ALL-CAPS filenames are reports metadata, not reports — always excluded.
    (misc_docs / "README.md").write_text("readme, excluded")
    # A vendored subtree must be pruned even though it holds a .md.
    (misc_docs / "node_modules").mkdir()
    (misc_docs / "node_modules" / "junk.md").write_text("vendored noise")
    # A hidden subtree must be pruned too.
    (misc_docs / ".cache").mkdir()
    (misc_docs / ".cache" / "stale.md").write_text("hidden noise")
    # A plain ``docs/`` tree (not ``misc/docs``) with a nested decision record.
    decisions = root / "tt" / "app" / "docs" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / "adr1.md").write_text("a decision record")


def test_reports_ingest_includes_nested_docs(tmp_path):
    _make_projects_tree(tmp_path)
    ids = set(ir.CorpusSource.from_md_reports(projects_root=tmp_path).scope)

    # Nested reports at any depth are indexed (the bug this guards against).
    assert "t/pkg/misc/docs/research/multiplatform.md" in ids
    assert "t/pkg/misc/docs/research/deep/nested/buried.md" in ids
    assert "tt/app/docs/decisions/adr1.md" in ids
    # Top-level reports still indexed (unchanged behavior).
    assert "t/pkg/misc/docs/top.md" in ids


def test_reports_ingest_excludes_allcaps_and_vendored(tmp_path):
    _make_projects_tree(tmp_path)
    ids = set(ir.CorpusSource.from_md_reports(projects_root=tmp_path).scope)

    assert not any(Path(i).name == "README.md" for i in ids)
    assert not any("node_modules" in i for i in ids)  # DFLT_EXCLUDE_DIRS
    assert not any("/.cache/" in f"/{i}" for i in ids)  # dotted dir pruned
    assert "node_modules" in DFLT_EXCLUDE_DIRS


def test_reports_ingest_shallow_escape_hatch(tmp_path):
    _make_projects_tree(tmp_path)
    ids = set(
        ir.CorpusSource.from_md_reports(projects_root=tmp_path, recursive=False).scope
    )

    # recursive=False restores the old top-level-only behavior.
    assert "t/pkg/misc/docs/top.md" in ids
    assert "t/pkg/misc/docs/research/multiplatform.md" not in ids


def test_reports_custom_exclude_dirs(tmp_path):
    _make_projects_tree(tmp_path)
    # A caller can widen the exclusion set (e.g. drop 'research/' too).
    ids = set(
        ir.CorpusSource.from_md_reports(
            projects_root=tmp_path, exclude_dirs={"research"}
        ).scope
    )
    assert not any("/research/" in f"/{i}" for i in ids)
    assert "tt/app/docs/decisions/adr1.md" in ids  # unrelated tree untouched


def test_exclude_reason_is_shared_ssot(tmp_path):
    """The inclusion predicate ingestion uses is the one coverage reports on."""
    from ir.sources import report_exclude_reason

    docs = tmp_path / "t" / "pkg" / "misc" / "docs"
    (docs / "research").mkdir(parents=True)
    (docs / "node_modules").mkdir()

    assert report_exclude_reason(docs / "top.md", docs) is None
    assert report_exclude_reason(docs / "research" / "x.md", docs) is None
    assert report_exclude_reason(docs / "README.md", docs) == "allcaps"
    assert report_exclude_reason(docs / "node_modules" / "j.md", docs) == "excluded_dir"


def test_coverage_full_index_reports_100pct(tmp_path):
    """A fully-built index reports coverage 1.0 with nothing missing."""
    from ir.coverage import reports_coverage
    from ir.store import CorpusStore

    _make_projects_tree(tmp_path)
    src = ir.CorpusSource.from_md_reports(projects_root=tmp_path)
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    indexed_ids = [
        e["artifact_id"]
        for _k, e in corpus.store.ledger_items()
        if e.get("artifact_id")
    ]

    rep = reports_coverage(projects_root=tmp_path, indexed_ids=indexed_ids)
    assert rep.coverage_ratio == 1.0
    assert rep.missing == []
    assert rep.indexed == rep.should_index > 0
    assert rep.on_disk > rep.should_index  # README + node_modules on disk, excluded
    assert rep.excluded.get("allcaps", 0) >= 1
    assert rep.excluded.get("excluded_dir", 0) >= 1


def test_coverage_flags_on_disk_but_unindexed(tmp_path):
    """The nested-doc-dropping bug's signature: on disk, includable, not indexed."""
    from ir.coverage import reports_coverage

    _make_projects_tree(tmp_path)
    # Simulate the old shallow index: only the top-level doc got in.
    shallow_index = ["t/pkg/misc/docs/top.md"]
    rep = reports_coverage(projects_root=tmp_path, indexed_ids=shallow_index)

    assert rep.coverage_ratio < 1.0
    assert "t/pkg/misc/docs/research/multiplatform.md" in rep.missing
    assert "t/pkg/misc/docs/research/deep/nested/buried.md" in rep.missing
    # Excluded files never count as "missing".
    assert not any("README" in m or "node_modules" in m for m in rep.missing)
