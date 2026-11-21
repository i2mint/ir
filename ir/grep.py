"""The grep Retriever leaf: an index-free first stage over a corpus's raw files.

Embedding indexes find paraphrases but miss the exact things a keyword search is for
(a name, an identifier, an error string) and need a build step that may be stale. This
leaf reads the files where they lie: it runs a regex (``rg`` when available, else
Python ``re``) over the files under some roots and returns
:class:`~ir.base.SearchHit`\\ s, so it can be fused with a dense or BM25 ranking like
any other Retriever (:data:`ir.retrieve.Retriever`).

Python's ``re`` is the authority: ``pattern`` is a Python regex, lines are split on
``\\n`` only (a trailing ``\\r`` is ignored), and counts and snippets come from Python,
so scores never depend on whether ``rg`` is installed. ``rg`` is only a prefilter that
skips files with no possible match, and it is used **only when that is provably safe**:
an ASCII *literal* pattern (no regex metacharacters, see :func:`_rg_can_prefilter`),
since ``rg`` and Python read ``\\w``, ``\\s``, ``.``, character classes and Unicode case
folding differently and a prefilter that disagrees would silently drop files. Any
other pattern is read in Python (``use_rg=True`` then raises).

Scoring is explicit and tunable: a file's score is its **matching-line count**, times a
per-file weight (``weight_of``: e.g. a title or a recent folder counts more), times an
optional recency decay (``recency_half_life_days``: a file's weight halves for each
half-life of age). Metadata filters are pushed into the scan: ``metadata_of(path)``
derives a file's metadata and a Mongo-style ``filter`` (the same one
:func:`ir.search` takes) drops files **before** they are read.

Hit metadata carries only the id-relative ``path`` and ``filename`` (never an absolute
local path, which would leak into anything published) unless ``absolute_paths=True``.

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

import fnmatch
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
from .retrieve import Retriever

#: How many files one ``rg`` call is handed (keeps argv under the OS limit).
DFLT_RG_BATCH = 500

#: Longest snippet kept per hit (the first matching line, trimmed).
DFLT_SNIPPET_CHARS = 200

#: Directories never descended into (VCS and tooling noise; override with ``exclude_dirs``).
DFLT_EXCLUDE_DIRS = (".git", "node_modules", "__pycache__", ".venv")

#: The ``surface_kind`` of a grep hit.
GREP_KIND = "grep"

_SECONDS_PER_DAY = 86400.0


def _as_roots(roots: str | os.PathLike | Iterable[str | os.PathLike]) -> list[Path]:
    if isinstance(roots, (str, os.PathLike)):
        return [Path(roots)]
    return [Path(r) for r in roots]


def _candidate_files(
    roots: list[Path], glob: str, exclude_dirs: Iterable[str]
) -> Iterator[tuple[Path, Path]]:
    """``(root, file)`` for each regular file under the roots whose root-relative path
    matches the fnmatch ``glob``; symlinks that leave the root, and excluded
    directories, are skipped (and never walked)."""
    skip = set(exclude_dirs)
    for root in roots:
        if root.is_file():
            yield root.parent, root
            continue
        if not root.is_dir():
            raise ValueError(f"grep: root {str(root)!r} does not exist")
        real_root = root.resolve()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
            for name in sorted(filenames):
                f = Path(dirpath, name)
                if not fnmatch.fnmatchcase(f.relative_to(root).as_posix(), glob):
                    continue
                if f.is_symlink() and not _inside(f, real_root):
                    continue
                if f.is_file():
                    yield root, f


def _inside(path: Path, real_root: Path) -> bool:
    """Does the symlink ``path`` resolve to somewhere under ``real_root``?"""
    try:
        return path.resolve().is_relative_to(real_root)
    except (OSError, RuntimeError):  # a symlink loop
        return False


def _compile(pattern: str, *, ignore_case: bool, fixed: bool) -> re.Pattern:
    try:
        return re.compile(
            re.escape(pattern) if fixed else pattern, re.I if ignore_case else 0
        )
    except re.error as e:
        raise ValueError(f"grep: invalid regular expression {pattern!r}: {e}") from None


def _lines(path: Path) -> Iterator[str]:
    """The file's lines, split on ``\\n`` only, without a trailing ``\\r`` or ``\\n``."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="\n") as fh:
        for line in fh:
            yield line.rstrip("\r\n")


def _count_and_first(path: Path, rx: re.Pattern) -> tuple[int, str]:
    n, first = 0, ""
    try:
        for line in _lines(path):
            if rx.search(line):
                n += 1
                first = first or line
    except OSError:
        return 0, ""
    return n, first


#: Regex metacharacters: a pattern containing none is a plain literal.
_META = re.compile(r"[\\.^$*+?{}\[\]|()]")

#: Letters with non-ASCII case-insensitive equivalents (İ ı K ſ) that Python's
#: ``re.I`` matches and ``rg -i`` does not.
_FOLD_TRAPS = re.compile(r"[iIkKsS]")


def _rg_can_prefilter(pattern: str, *, ignore_case: bool, fixed: bool) -> bool:
    """Whether ``rg`` is guaranteed to select a superset of the files Python matches."""
    if not pattern.isascii() or "\x00" in pattern or not pattern:
        return False
    if not (fixed or not _META.search(pattern)):
        return False
    return not (ignore_case and _FOLD_TRAPS.search(pattern))


def _rg_prefilter(
    files: list[Path],
    pattern: str,
    *,
    ignore_case: bool,
    fixed: bool,
    batch: int,
) -> list[Path] | None:
    """The files ``rg`` says may match, or ``None`` if ``rg`` is missing or cannot be
    used for this pattern/batch (the caller then reads every file in Python)."""
    rg = shutil.which("rg")
    if not rg:
        return None
    flags = ["--no-config", "--null", "--no-messages", "--no-ignore", "--crlf", "-l"]
    flags += (["-i"] if ignore_case else []) + (["-F"] if fixed else [])
    keep: list[Path] = []
    for i in range(0, len(files), batch):
        chunk = [str(f) for f in files[i : i + batch]]
        try:
            proc = subprocess.run(
                [rg, *flags, "-e", pattern, "--", *chunk],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except OSError:  # e.g. argument list too long
            return None
        if proc.returncode == 2 and not proc.stdout:  # rg cannot compile the regex
            return None
        keep += [Path(os.fsdecode(p)) for p in proc.stdout.split(b"\0") if p]
    return keep


def _recency(path: Path, half_life_days: float | None, now: float) -> float:
    if not half_life_days:
        return 1.0
    try:
        age_days = max(0.0, (now - path.stat().st_mtime) / _SECONDS_PER_DAY)
    except OSError:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def recency_weight(
    half_life_days: float, *, now: float | None = None
) -> Callable[[Path], float]:
    """A composable ``weight_of`` seam: a file's weight halves per ``half_life_days`` of age.

    ``now`` pins the clock (default: the time of each call). Multiply it into your own
    ``weight_of`` to combine it with other weights; it is what
    ``recency_half_life_days`` uses.

    >>> import os, tempfile, time
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as d:
    ...     f = Path(d, "a.md"); _ = f.write_text("x")
    ...     os.utime(f, (0, 0))
    ...     recency_weight(10, now=10 * _SECONDS_PER_DAY)(f)
    0.5
    """
    _check_args(1, half_life_days, 1)
    return lambda path: _recency(path, half_life_days, time.time() if now is None else now)


def _check_args(k: int, half_life: float | None, rg_batch: int) -> None:
    for name, value in (("k", k), ("rg_batch", rg_batch)):
        if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if half_life is not None and not (math.isfinite(half_life) and half_life > 0):
        raise ValueError(f"recency_half_life_days must be positive, got {half_life!r}")


def _weight(weight_of: Callable[[Path], float] | None, path: Path) -> float:
    if weight_of is None:
        return 1.0
    w = weight_of(path)
    if not (isinstance(w, (int, float)) and math.isfinite(w) and w >= 0):
        raise ValueError(f"weight_of({str(path)!r}) must be finite and >= 0, got {w!r}")
    return float(w)


def grep_hits(
    pattern: str,
    *,
    roots: str | os.PathLike | Iterable[str | os.PathLike],
    glob: str = "*",
    k: int = 10,
    ignore_case: bool = True,
    fixed: bool = False,
    filter: Mapping[str, Any] | None = None,
    metadata_of: Callable[[Path], Mapping[str, Any]] | None = None,
    weight_of: Callable[[Path], float] | None = None,
    recency_half_life_days: float | None = None,
    id_of: Callable[[Path, Path], str] | None = None,
    use_rg: bool | None = None,
    rg_batch: int = DFLT_RG_BATCH,
    exclude_dirs: Iterable[str] = DFLT_EXCLUDE_DIRS,
    absolute_paths: bool = False,
    snippet_chars: int = DFLT_SNIPPET_CHARS,
    source: str | None = None,
    now: float | None = None,
) -> list[SearchHit]:
    """The top-``k`` files under ``roots`` whose lines match ``pattern``, as hits.

    ``pattern`` is a Python regex (``fixed=True``: a literal). Files are those whose
    root-relative posix path matches the fnmatch ``glob`` (``*`` crosses ``/``, so
    ``"*.md"`` means every markdown file below the root); a root that is a file
    stands for itself. ``filter`` is applied to ``metadata_of(path)`` *before* any file
    is read (a filter without ``metadata_of`` sees ``{"path", "filename"}``, the path
    relative to its root). ``id_of(root, path)`` names the artifact (default: the
    relative path); two files with the same id raise ``ValueError`` (several roots with
    a common relative path: pass ``id_of``). ``use_rg``: ``None`` = prefilter with
    ``rg`` if installed, ``True`` = require it (``RuntimeError`` otherwise), ``False`` =
    Python only. The score is described in the module docstring.
    """
    _check_args(k, recency_half_life_days, rg_batch)
    if use_rg is True and not _rg_can_prefilter(
        pattern, ignore_case=ignore_case, fixed=fixed
    ):
        raise ValueError(
            "grep_hits(use_rg=True) needs an ASCII literal pattern that rg and "
            "Python read identically; use use_rg=None to let ir decide"
        )
    if use_rg is True and not shutil.which("rg"):
        raise RuntimeError("grep_hits(use_rg=True) but `rg` is not installed")
    rx = _compile(pattern, ignore_case=ignore_case, fixed=fixed)
    from vd.filters import matches_filter

    name_of = id_of or (lambda root, path: path.relative_to(root).as_posix())
    user_meta = (lambda f: dict(metadata_of(f))) if metadata_of else (lambda f: {})

    def meta_for(f: Path, aid: str) -> dict:
        shown = str(f) if absolute_paths else aid
        return {**user_meta(f), "path": shown, "filename": f.name}

    kept: dict[Path, str] = {}
    ids: dict[str, Path] = {}
    seen_real: set[Path] = set()
    for root, f in _candidate_files(_as_roots(roots), glob, exclude_dirs):
        aid = name_of(root, f)
        if filter and not matches_filter(meta_for(f, aid), dict(filter)):
            continue  # metadata_of runs here only when there is a filter
        real = f.resolve()
        if real in seen_real:  # overlapping roots: one file, once
            continue
        if aid in ids and ids[aid] != f:
            raise ValueError(
                f"grep: {str(ids[aid])!r} and {str(f)!r} share the id {aid!r}; "
                "pass id_of to tell them apart"
            )
        seen_real.add(real)
        ids[aid] = f
        kept[f] = aid

    files = list(kept)
    if use_rg is not False and _rg_can_prefilter(
        pattern, ignore_case=ignore_case, fixed=fixed
    ):
        prefiltered = _rg_prefilter(
            files, pattern, ignore_case=ignore_case, fixed=fixed, batch=rg_batch
        )
        if prefiltered is not None:
            wanted = set(prefiltered)
            files = [f for f in files if f in wanted]

    t = time.time() if now is None else now
    scored = []
    for f in files:
        n, first = _count_and_first(f, rx)
        if n:
            score = n * _weight(weight_of, f) * _recency(f, recency_half_life_days, t)
            scored.append((score, n, f, first))
    scored.sort(key=lambda s: (-s[0], kept[s[2]]))

    return [
        SearchHit(
            artifact_id=kept[f],
            surface_kind=GREP_KIND,
            score=float(score),
            text=first.strip()[:snippet_chars],
            metadata={**meta_for(f, kept[f]), "n_matches": n},
            source=source,
        )
        for score, n, f, first in scored[:k]
    ]


def make_grep_retriever(
    roots: str | os.PathLike | Iterable[str | os.PathLike], **grep_defaults: Any
) -> Retriever:
    """Bind roots (and defaults) to the uniform :data:`ir.retrieve.Retriever` contract.

    The result is ``retrieve(pattern, **overrides) -> list[SearchHit]`` (``roots`` may
    be overridden per call too), the same shape as :func:`ir.as_retriever`, so a grep
    leaf can sit in a ``Mapping[name, Retriever]`` registry beside index-backed
    corpora. Overrides are the keywords of :func:`grep_hits`; an unknown one
    (``mode=``) is a ``TypeError``, not silently dropped.
    """
    bound = _as_roots(roots)  # materialised: a generator would be spent after one call

    def retrieve(pattern: str, **overrides: Any) -> list[SearchHit]:
        return grep_hits(pattern, **{"roots": bound, **grep_defaults, **overrides})

    retrieve.roots = bound
    return retrieve
