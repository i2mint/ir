"""Source-adapter tests against the real local ecosystem.

These need ``priv`` and the projects folder (``$PP`` / ``$PTH_FILEPATH``), so
they are skipped where unavailable (e.g. CI). They use the light embedder to
stay fast — they validate the *source plumbing* (scope, metadata, ids), not
semantic quality.
"""

import os

import pytest

import ir
from ir.store import CorpusStore

_has_priv = False
try:
    import priv.skills_index  # noqa: F401

    _has_priv = True
except Exception:
    pass

_has_pp = bool(os.environ.get("PP") or os.environ.get("PTH_FILEPATH"))

skills_only = pytest.mark.skipif(not _has_priv, reason="priv not importable")
ecosystem = pytest.mark.skipif(
    not (_has_priv and _has_pp), reason="priv/$PP not available"
)


@skills_only
def test_skills_source_scope_and_build():
    src = ir.CorpusSource.from_skills()
    assert len(src.scope) > 20  # the ecosystem has many skills
    sample = next(iter(src.scope.values()))
    assert "name" in sample and "description" in sample
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    assert len(corpus) == len(src.scope)  # Skill -> one surface per skill
    hits = ir.search(corpus, "anything", k=3)
    assert hits and hits[0].metadata.get("name")


@ecosystem
def test_md_reports_source_excludes_allcaps():
    src = ir.CorpusSource.from_md_reports()
    # No ALL-CAPS report filenames (README/CLAUDE/MEMORY/SKILL).
    for meta in (src.metadata_of(k, v) for k, v in list(src.scope.items())[:200]):
        fn = meta.get("filename", "")
        assert not fn.isupper() or not fn.endswith(".md")


@ecosystem
def test_packages_source_scope_shape():
    src = ir.CorpusSource.from_packages()
    assert len(src.scope) > 50
    rec = src.scope.get("dol") or next(iter(src.scope.values()))
    assert rec["owner"] == "ours"
    assert "name" in rec and "path" in rec
