"""The precision stage: a cross-encoder rerank over the first-stage candidates.

First-stage retrieval (dense / lexical / hybrid) is tuned for *recall*; many of its worst
false positives are artefacts a joint ``(query, passage)`` scorer catches -- a package
whose single "DAG" bullet inflates its pooled-vector cosine, or a pair of siblings that
share jargon rather than topic. :func:`ir.search` therefore takes ``rerank=``: it
over-fetches ``fetch_k`` candidates, hands them to the reranker, and keeps the top ``k``.
This is single-shot and agent-free: a better *one* search, not the evaluator back-edge
(that belongs to ``raglab``).

``ir`` composes ``ef`` here and adds no reranker class. ``ef.cross_encoder_reranker`` is the
implementation; this module adds the three things ``ef`` cannot know about an ``ir`` corpus:

- **What the reranker reads.** ``rerank_text="surface"`` (default) scores the matched
  surface, exactly as before. ``"artifact"`` scores the artifact's *whole* stored text --
  its sibling surfaces (name + description + README chunks) joined in plan order -- so a
  thin description is judged against the full README. A callable
  ``(corpus, meta) -> str`` replaces both.
- **Opt-in, with a graceful fallback.** ``rerank="cross-encoder"`` (or
  ``"cross-encoder:<model>"``) names the ef cross-encoder without importing it, so
  ``import ir`` needs no torch. When ``sentence-transformers`` (the ``ir[rerank]`` extra) is
  missing, the search warns once and returns the first-stage ranking rather than failing.
- **Calibration that stays honest.** A reranked hit's score is the reranker's, on a scale
  unrelated to cosine / RRF / BM25, so an abstention floor calibrated for ``hybrid`` would
  be mis-scaled if applied to it. :func:`calibration_key` gives a reranked configuration its
  own key (``hybrid+rerank:<model>+artifact``), so ``calibrate_min_score(...,
  rerank=...)`` and ``discover(..., min_score="auto", rerank=...)`` read and write a floor
  for *that* configuration, and fall back to "no floor" (with the usual warning) rather
  than a wrong one. The fallback to first-stage ranking maps back to the plain key.

Rerankers used with relative selection strategies should return scores in ``[0, 1]`` (the
sentence-transformers cross-encoder applies a sigmoid for single-label models).

>>> resolve_reranker(None) is None
True
>>> calibration_key("hybrid", None)
'hybrid'
>>> calibration_key("hybrid", rerank_tag(lambda q, s: [], "artifact"))
'hybrid+rerank:<lambda>+artifact'
"""

from __future__ import annotations

import re
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "RERANK_TEXTS",
    "DFLT_RERANK_MODEL",
    "DFLT_RERANK_MAX_CHARS",
    "CROSS_ENCODER",
    "resolve_reranker",
    "rerank_tag",
    "calibration_key",
    "passage_texts",
]

#: The built-in ``rerank_text`` choices (a callable is also accepted).
RERANK_TEXTS = ("surface", "artifact")

#: The cross-encoder ``rerank="cross-encoder"`` resolves to (ef's default).
DFLT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Cap on the characters of one ``rerank_text="artifact"`` passage. Cross-encoders truncate
#: at ~512 tokens anyway; the cap keeps the pair list cheap to build and the cut at a
#: surface boundary-agnostic, front-loaded point (name + description come first).
DFLT_RERANK_MAX_CHARS = 2000

#: The ``rerank=`` string that names ef's cross-encoder.
CROSS_ENCODER = "cross-encoder"

_model_cache: dict[str, Callable] = {}
_cache_lock = threading.Lock()
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, stacklevel=4)


def resolve_reranker(spec: Any) -> Callable | None:
    """The reranker ``spec`` names, or ``None`` when there is none to apply.

    ``spec`` is ``None`` (no rerank), any ``ef.Reranker`` callable (returned as is), or the
    string ``"cross-encoder"`` / ``"cross-encoder:<model>"`` -- ef's sentence-transformers
    cross-encoder, built lazily and cached per model name (loading one is the expensive
    part). If its optional dependency is absent this warns once and returns ``None``, so
    the caller degrades to first-stage ranking; resolve *before* deriving a calibration key
    so the degraded run uses the plain key.

    >>> resolve_reranker(len) is len
    True
    >>> resolve_reranker("nope")
    Traceback (most recent call last):
        ...
    ValueError: unknown rerank spec 'nope'; expected a callable, 'cross-encoder' or 'cross-encoder:<model>'.
    """
    if spec is None or callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"rerank must be a callable or str, got {type(spec).__name__}")
    name, _, model = spec.partition(":")
    if name != CROSS_ENCODER:
        raise ValueError(
            f"unknown rerank spec {spec!r}; expected a callable, "
            f"'{CROSS_ENCODER}' or '{CROSS_ENCODER}:<model>'."
        )
    model = model or DFLT_RERANK_MODEL
    with _cache_lock:
        if model in _model_cache:
            return _model_cache[model]
        try:
            from ef import cross_encoder_reranker

            reranker = cross_encoder_reranker(model)
        except ImportError as exc:
            _warn_once(
                model,
                f"rerank={spec!r} needs sentence-transformers ({exc}); install the "
                f"extra with `pip install 'ir[rerank]'`. Falling back to the "
                f"first-stage ranking (no rerank).",
            )
            return None
        _model_cache[model] = reranker
        return reranker


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.<>-]+", "-", text).strip("-")


def rerank_tag(reranker: Callable | None, rerank_text: Any = "surface") -> str | None:
    """A short, key-safe identity for one reranked configuration (``None`` if no rerank).

    Built from the reranker's ``model_name`` (ef's cross-encoder sets it) or ``__name__``,
    plus the ``rerank_text`` choice, because both change the score scale.
    """
    if reranker is None:
        return None
    ident = getattr(reranker, "model_name", None) or getattr(
        reranker, "__name__", type(reranker).__name__
    )
    text = (
        rerank_text
        if isinstance(rerank_text, str)
        else getattr(rerank_text, "__name__", "custom")
    )
    return f"rerank:{_slug(str(ident))}+{_slug(text)}"


def calibration_key(mode: str, tag: str | None = None) -> str:
    """The key an abstention floor is stored and loaded under.

    ``mode`` alone for a first-stage ranking (today's behaviour, so existing floors keep
    working); ``"<mode>+<tag>"`` for a reranked one.

    >>> calibration_key("hybrid")
    'hybrid'
    >>> calibration_key("hybrid", "rerank:m+artifact")
    'hybrid+rerank:m+artifact'
    """
    return mode if not tag else f"{mode}+{tag}"


def passage_texts(
    corpus: Any,
    candidates: Sequence[Mapping[str, Any]],
    rerank_text: Any = "surface",
    *,
    max_chars: int = DFLT_RERANK_MAX_CHARS,
) -> list[str]:
    """The text the reranker scores for each candidate's store ``meta`` row.

    ``"surface"`` is the matched surface's own text. ``"artifact"`` stitches every stored
    surface of the candidate's artifact in plan order (chunk overlap deduped, as in
    :mod:`ir.expand`) (siblings looked up once per
    artifact, through the ledger) and cuts at ``max_chars``; an artifact the ledger does not
    know falls back to its matched surface. A callable is called as ``rerank_text(corpus,
    meta)``.
    """
    if callable(rerank_text):
        return [str(rerank_text(corpus, meta)) for meta in candidates]
    if rerank_text == "surface":
        return [meta["text"] for meta in candidates]
    if rerank_text != "artifact":
        raise ValueError(
            f"unknown rerank_text {rerank_text!r}; expected one of {RERANK_TEXTS} "
            f"or a callable (corpus, meta) -> str."
        )
    from .expand import _stitch
    from .retrieve import NoLedgerEntry, records_for_artifact

    whole: dict[str, str] = {}

    def artifact_text(meta: Mapping[str, Any]) -> str:
        artifact_id = meta["artifact_id"]
        if artifact_id not in whole:
            try:
                records = records_for_artifact(corpus, artifact_id)
            except NoLedgerEntry:
                records = []
            records = [r for r in records if r.text]
            stitched = _stitch(records) if records else ""
            whole[artifact_id] = stitched[:max_chars] or meta["text"]
        return whole[artifact_id]

    return [artifact_text(meta) for meta in candidates]
