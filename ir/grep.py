"""The grep Retriever leaf: an index-free first stage over a corpus's raw files.

Embedding indexes find paraphrases but miss the exact things a keyword search is for
(a name, an identifier, an error string) and need a build step that may be stale. This
leaf reads the files where they lie: it runs a regex (``rg`` when available, else
Python ``re``) over the files under some roots and returns
:class:`~ir.base.SearchHit`\\ s, so it can be fused with a dense or BM25 ranking like
any other Retriever (:data:`ir.retrieve.Retriever`).

Scoring is explicit and tunable: a file's score is its **matching-line count**, times a
per-file weight (``weight_of``: e.g. a title or a recent folder counts more), times an
optional recency decay (``recency_half_life_days``: a file's weight halves for each
half-life of age). Metadata filters are pushed into the scan: ``metadata_of(path)``
derives a file's metadata and a Mongo-style ``filter`` (the same one
:func:`ir.search` takes) drops files **before** they are read.

It returns hits, never a loop and never a plan; whether to grep or to use the index is
the caller's (or the agent layer's) choice.

>>> import tempfile, pathlib
>>> with tempfile.TemporaryDirectory() as d:
...     _ = pathlib.Path(d, "a.md").write_text("Acme invoice\\nnothing\\nacme again")
...     _ = pathlib.Path(d, "b.md").write_text("unrelated")
...     hits = grep_hits("acme", roots=d, use_rg=False)
>>> [(h.artifact_id, h.score, h.metadata["n_matches"]) for h in hits]
[('a.md', 2.0, 2)]
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .base import SearchHit

#: How many files one ``rg`` call is handed (keeps argv under the OS limit).
RG_BATCH = 500

#: Longest snippet kept per hit (the first matching line, trimmed).
DFLT_SNIPPET_CHARS = 200

#: The ``surface_kind`` of a grep hit.
GREP_KIND = "grep"

_SECONDS_PER_DAY = 86400.0


def _as_roots(roots: str | os.PathLike | Iterable[str | os.PathLike]) -> list[Path]:
    if isinstance(roots, (str, os.PathLike)):
        return [Path(roots)]
    return [Path(r) for r in roots]


def _candidate_files(roots: list[Path], glob: str) -> Iterator[tuple[Path, Path]]:
    """``(root, file)`` for each regular file under the roots matching ``glob``."""
    for root in roots:
        if root.is_file():
            yield root.parent, root
            continue
        for f in sorted(root.glob(glob)):
            if f.is_file():
                yield root, f


def _compile(pattern: str, *, ignore_case: bool, fixed: bool) -> re.Pattern:
    try:
        return re.compile(
            re.escape(pattern) if fixed else pattern, re.I if ignore_case else 0
        )
    except re.error as e:
        raise ValueError(f"grep: invalid regular expression {pattern!r}: {e}") from None


def _python_counts(files: list[Path], rx: re.Pattern) -> dict[Path, int]:
    counts = {}
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                n = sum(1 for line in fh if rx.search(line))
        except OSError:
            continue
        if n:
            counts[f] = n
    return counts


def _rg_counts(
    files: list[Path], pattern: str, *, ignore_case: bool, fixed: bool
) -> dict[Path, int] | None:
    """Matching-line counts per file via ``rg``, or ``None`` if it is missing or
    cannot compile the pattern (the caller then falls back to Python ``re``)."""
    rg = shutil.which("rg")
    if not rg:
        return None
    flags = ["--no-config", "--null", "--no-messages", "--no-ignore", "-j1", "-c"]
    flags += ["--with-filename"]  # even for a one-file batch
    flags += (["-i"] if ignore_case else []) + (["-F"] if fixed else [])
    counts: dict[Path, int] = {}
    for i in range(0, len(files), RG_BATCH):
        batch = [str(f) for f in files[i : i + RG_BATCH]]
        proc = subprocess.run(
            [rg, *flags, "-e", pattern, "--", *batch],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 2 and not proc.stdout:  # rg cannot compile the regex
            return None
        for rec in proc.stdout.split(b"\n"):
            path, _, n = rec.partition(b"\0")
            if n.strip():
                counts[Path(path.decode())] = int(n)
    return counts


def _first_match(path: Path, rx: re.Pattern, *, chars: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if rx.search(line):
                    return line.strip()[:chars]
    except OSError:
        pass
    return ""


def _recency(path: Path, half_life_days: float | None, now: float) -> float:
    if not half_life_days:
        return 1.0
    try:
        age_days = max(0.0, (now - path.stat().st_mtime) / _SECONDS_PER_DAY)
    except OSError:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def grep_hits(
    pattern: str,
    *,
    roots: str | os.PathLike | Iterable[str | os.PathLike],
    glob: str = "**/*",
    k: int = 10,
    ignore_case: bool = True,
    fixed: bool = False,
    filter: Mapping[str, Any] | None = None,
    metadata_of: Callable[[Path], Mapping[str, Any]] | None = None,
    weight_of: Callable[[Path], float] | None = None,
    recency_half_life_days: float | None = None,
    id_of: Callable[[Path, Path], str] | None = None,
    use_rg: bool | None = None,
    snippet_chars: int = DFLT_SNIPPET_CHARS,
    source: str | None = None,
    now: float | None = None,
) -> list[SearchHit]:
    """The top-``k`` files under ``roots`` whose lines match ``pattern``, as hits.

    ``pattern`` is a regex (``fixed=True``: a literal). Files are ``glob``-ed under
    each root (a root that is a file stands for itself). ``filter`` is applied to
    ``metadata_of(path)`` *before* any file is read (a filter without ``metadata_of``
    sees only ``{"path": ..., "filename": ...}``). ``id_of(root, path)`` names the
    artifact (default: the path relative to its root). ``use_rg``: ``None`` = ``rg``
    if installed, ``True`` = require it (``RuntimeError`` otherwise), ``False`` =
    Python ``re`` only; ``rg`` that cannot compile the regex also falls back to
    Python. The two count the same thing (matching lines).
    """
    if use_rg is True and not shutil.which("rg"):
        raise RuntimeError("grep_hits(use_rg=True) but `rg` is not installed")
    rx = _compile(pattern, ignore_case=ignore_case, fixed=fixed)
    from vd.filters import matches_filter

    rel = id_of or (lambda root, path: str(path.relative_to(root)))
    kept: dict[Path, tuple[Path, dict]] = {}
    for root, f in _candidate_files(_as_roots(roots), glob):
        meta = {
            "path": str(f),
            "filename": f.name,
            **(dict(metadata_of(f)) if metadata_of else {}),
        }
        if filter and not matches_filter(meta, dict(filter)):
            continue
        kept[f] = (root, meta)

    files = list(kept)
    counts = None
    if use_rg is not False:
        counts = _rg_counts(files, pattern, ignore_case=ignore_case, fixed=fixed)
    if counts is None:
        counts = _python_counts(files, rx)

    t = time.time() if now is None else now
    scored = []
    for f, n in counts.items():
        weight = weight_of(f) if weight_of else 1.0
        scored.append((n * weight * _recency(f, recency_half_life_days, t), n, f))
    scored.sort(key=lambda s: (-s[0], str(s[2])))

    hits = []
    for score, n, f in scored[:k]:
        root, meta = kept[f]
        hits.append(
            SearchHit(
                artifact_id=rel(root, f),
                surface_kind=GREP_KIND,
                score=float(score) if math.isfinite(score) else 0.0,
                text=_first_match(f, rx, chars=snippet_chars),
                metadata={**meta, "n_matches": n},
                source=source,
            )
        )
    return hits


def make_grep_retriever(
    roots: str | os.PathLike | Iterable[str | os.PathLike], **grep_defaults: Any
) -> Callable[..., list[SearchHit]]:
    """Bind roots (and defaults) to the uniform :data:`ir.retrieve.Retriever` contract.

    The result is ``retrieve(pattern, **overrides) -> list[SearchHit]``, the same
    shape as :func:`ir.as_retriever`, so a grep leaf can sit in a
    ``Mapping[name, Retriever]`` registry beside index-backed corpora.
    """

    def retrieve(pattern: str, **overrides: Any) -> list[SearchHit]:
        return grep_hits(pattern, roots=roots, **{**grep_defaults, **overrides})

    retrieve.roots = roots
    return retrieve
