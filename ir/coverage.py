"""Corpus-coverage diagnostic — what's on disk vs. what's actually indexed.

Retrieval can only return what was indexed, and an ingestion gap is *silent*:
a whole folder of reports can be missing from the index with no error (a
non-recursive glob once dropped every doc in a ``research/`` / ``decisions/``
subtree — 29% of the corpus — invisibly). Back-translation evals cannot catch
this: their gold is drawn from what is *already* indexed, so an un-indexed doc
can never lower a metric. Coverage therefore needs its own check.

:func:`reports_coverage` **independently** walks the reports doc-trees on disk
(the raw ``*/*/docs`` and ``*/*/misc/docs`` files, before any filtering),
classifies each file with the same inclusion SSOT the ingestion walk uses
(:func:`ir.sources.report_exclude_reason`), and diffs the *should-be-indexed*
set against the built corpus's ledger — surfacing exactly the on-disk-but-absent
reports that a search would silently miss.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .index import open_corpus
from .sources import (
    DFLT_EXCLUDE_DIRS,
    _projects_root,
    iter_report_doc_folders,
    report_exclude_reason,
)


@dataclass(frozen=True)
class CoverageReport:
    """Disk-vs-index coverage for a filesystem-backed corpus."""

    name: str
    on_disk: int
    indexed: int
    excluded: dict[str, int]
    missing: list[str] = field(default_factory=list)

    @property
    def should_index(self) -> int:
        """On-disk files that pass the inclusion rule (indexed + missing)."""
        return self.indexed + len(self.missing)

    @property
    def coverage_ratio(self) -> float:
        """Fraction of should-index files that are actually indexed (1.0 = full)."""
        denom = self.should_index
        return 1.0 if denom == 0 else self.indexed / denom

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "on_disk": self.on_disk,
            "indexed": self.indexed,
            "should_index": self.should_index,
            "excluded": dict(self.excluded),
            "coverage_ratio": self.coverage_ratio,
            "missing": list(self.missing),
        }

    def __str__(self) -> str:
        exc = ", ".join(f"{k}={v}" for k, v in sorted(self.excluded.items())) or "none"
        lines = [
            f"coverage[{self.name}]: {self.indexed}/{self.should_index} indexed "
            f"({self.coverage_ratio:.1%}); {self.on_disk} on disk, excluded: {exc}",
        ]
        if self.missing:
            lines.append(f"  {len(self.missing)} on disk but NOT indexed (rebuild?):")
            shown = self.missing[:20]
            lines += [f"    {m}" for m in shown]
            if len(self.missing) > len(shown):
                lines.append(f"    … and {len(self.missing) - len(shown)} more")
        else:
            lines.append("  ✓ every includable report on disk is indexed")
        return "\n".join(lines)


def reports_coverage(
    name: str = "reports",
    *,
    projects_root: str | Path | None = None,
    indexed_ids: Iterable[str] | None = None,
    exclude_dirs: Iterable[str] = DFLT_EXCLUDE_DIRS,
) -> CoverageReport:
    """Diff the reports docs tree on disk against the built ``name`` corpus.

    Walks every ``*/*/docs`` and ``*/*/misc/docs`` folder recursively (raw, no
    filtering), classifies each ``*.md`` with the shared inclusion SSOT, and
    reports how many *includable* reports are indexed vs. silently missing. A
    ``coverage_ratio`` below 1.0 means on-disk reports are absent from the index —
    rebuild with ``ir build <name>``.

    ``indexed_ids`` (the built corpus's artifact ids) is read from the registered
    ``name`` corpus by default; pass it explicitly (with ``projects_root``) to
    diff an arbitrary tree against an arbitrary index without opening a corpus.
    """
    if indexed_ids is None:
        corpus = open_corpus(name)
        indexed_ids = {
            e["artifact_id"]
            for _key, e in corpus.store.ledger_items()
            if e.get("artifact_id")
        }
    indexed_ids = set(indexed_ids)
    root = Path(projects_root or _projects_root())
    exclude = frozenset(exclude_dirs)

    on_disk = 0
    indexed = 0
    excluded: dict[str, int] = {}
    missing: list[str] = []
    seen: set[str] = set()
    for folder in iter_report_doc_folders(root):
        for path in folder.rglob("*.md"):
            if not path.is_file():
                continue
            # Match the POSIX-normalized ids from_md_reports stores (forward
            # slashes on every platform), so the disk↔index diff is exact.
            rel = path.relative_to(root).as_posix()
            if rel in seen:  # a file can't be double-counted across globs
                continue
            seen.add(rel)
            on_disk += 1
            reason = report_exclude_reason(path, folder, exclude_dirs=exclude)
            if reason is not None:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            if rel in indexed_ids:
                indexed += 1
            else:
                missing.append(rel)
    missing.sort()
    return CoverageReport(
        name=name,
        on_disk=on_disk,
        indexed=indexed,
        excluded=excluded,
        missing=missing,
    )
