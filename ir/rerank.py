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
  own key (``hybrid+rerank-<model>+artifact``), so ``calibrate_min_score(...,
  rerank=...)`` and ``discover(..., min_score="auto", rerank=...)`` read and write a floor
  for *that* configuration, and fall back to "no floor" (with the usual warning) rather
  than a wrong one. The fallback to first-stage ranking maps back to the plain key.

Relative selection strategies (``rel_threshold``, ``score_gap``...) compare score *ratios*,
so a reranker used with them must return non-negative scores. The ``"cross-encoder"`` spec
guarantees that: most cross-encoders (including ef's default ms-marco model) return raw
logits that run negative, so ir passes them through a sigmoid into ``(0, 1)``. (A model that already applies a sigmoid would be squashed a second time, which keeps the
order but compresses the scale until relative selectors can no longer tell hits apart: use
such a model as a callable, ``ef.cross_encoder_reranker(model)`` with a ``rerank_id``.) A callable you pass yourself is used as is: keep it in
``[0, 1]``, or give it a stable ``rerank_id`` attribute so its calibrated floor is kept apart.

>>> resolve_reranker(None) is None
True
>>> calibration_key("hybrid", None)
'hybrid'
>>> calibration_key("hybrid", rerank_tag(lambda q, s: [], "artifact"))
'hybrid+rerank-~anonymous+artifact'
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import re
import sys
import threading
import types
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
    "is_anonymous",
    "spec_tag",
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


def _external_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside ``ir``."""
    package = __file__.rsplit("ir", 1)[0] + "ir"
    level, frame = 1, sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame, level = frame.f_back, level + 1
    return level


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, stacklevel=_external_stacklevel())


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
        if model in _warned:  # the extra is missing: do not retry the import per search
            return None
        try:
            from ef import cross_encoder_reranker

            raw = cross_encoder_reranker(model)
        except ImportError as exc:
            _warn_once(
                model,
                f"rerank={spec!r} needs sentence-transformers ({exc}); install the "
                f"extra with `pip install 'ir[rerank]'`. Falling back to the "
                f"first-stage ranking (no rerank).",
            )
            return None
        reranker = _sigmoid_scores(raw, model)
        _model_cache[model] = reranker
        return reranker


def _sigmoid_scores(raw: Callable, model: str) -> Callable:
    """``raw`` with its logits squashed into ``(0, 1)`` (see the module docstring)."""

    def reranker(query, segments):
        return [_sigmoid(score) for score in raw(query, segments)]

    reranker.model_name = model  # type: ignore[attr-defined]
    reranker.rerank_id = f"{model}+sigmoid"  # type: ignore[attr-defined]
    return reranker


def _sigmoid(x: float) -> float:
    x = float(x)
    if (
        x != x
    ):  # NaN: an unscorable pair must not outrank everything (min/max would say 60)
        return 0.0
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


#: Marks a tag resting on an unnamed callable. ``~`` is never produced by :func:`_slug`
#: (it escapes it), so no real model name or ``rerank_id`` can spell it.
ANONYMOUS = "~anonymous"

#: Longest tag kept verbatim: a calibration key is a file name on the file store.
MAX_TAG_CHARS = 120


def _slug(text: str) -> str:
    """``text`` as key-safe characters, losslessly: ``org/m`` and ``org-m`` stay apart."""
    return re.sub(r"[^A-Za-z0-9.-]", lambda m: f"_{ord(m.group()):x}_", text) or "_"


def _identity(obj: Any) -> str:
    """A stable name for a reranker / text callable, or ``ANONYMOUS``.

    ``rerank_id`` (set it on your callable) beats ``model_name`` beats ``__name__``.
    Lambdas, ``functools.partial``, bare instances, closures made by a factory and bound
    methods have no stable name -- two of them would share one calibration key while
    scoring on different scales -- unless they carry a ``rerank_id`` / ``model_name``.
    """
    for attr in ("rerank_id", "model_name"):
        value = getattr(obj, attr, None)
        if value:
            return _slug(str(value))
    name = getattr(obj, "__name__", None)
    qualname = getattr(obj, "__qualname__", "")
    unstable = (
        not name
        or name in ("<lambda>", "partial")
        or "<locals>" in qualname
        or inspect.ismethod(obj)
    )
    if unstable:
        return ANONYMOUS
    module = getattr(obj, "__module__", None)
    return _slug(f"{module}.{qualname or name}" if module else str(name))


def rerank_tag(
    reranker: Callable | None,
    rerank_text: Any = "surface",
    *,
    max_chars: int = DFLT_RERANK_MAX_CHARS,
) -> str | None:
    """A short, key-safe identity for one reranked configuration (``None`` if no rerank).

    Built from the reranker's identity (see :func:`_identity`), the ``rerank_text`` choice
    and, for ``"artifact"``, a non-default ``max_chars`` -- everything that changes the
    score scale. An unnamed reranker or text callable puts ``~anonymous`` in its tag, which
    ir refuses to persist a floor under (:func:`is_anonymous`). A tag longer than
    ``MAX_TAG_CHARS`` is shortened with a hash of the whole, so it stays a usable file name.
    """
    if reranker is None:
        return None
    if isinstance(rerank_text, str):
        text = _slug(rerank_text)
        if rerank_text == "artifact" and max_chars != DFLT_RERANK_MAX_CHARS:
            text += f"@{max_chars}"
    else:
        text = f"fn-{_identity(rerank_text)}"
    tag = f"rerank-{_identity(reranker)}+{text}"
    if len(tag) > MAX_TAG_CHARS:
        digest = hashlib.sha1(tag.encode()).hexdigest()[:12]
        # Keep the marker: anonymity must survive the cut.
        head = f"rerank-{ANONYMOUS}" if ANONYMOUS in tag else tag[: MAX_TAG_CHARS - 13]
        tag = f"{head}.{digest}"
    return tag


def is_anonymous(tag: str | None) -> bool:
    """Does ``tag`` rest on an unnamed callable (so its calibration key may collide)?"""
    return bool(tag) and ANONYMOUS in tag


def _has_sentence_transformers() -> bool:
    """Is the optional extra importable? (Without importing it: that loads torch.)"""
    if "sentence_transformers" in sys.modules:
        return sys.modules["sentence_transformers"] is not None
    return importlib.util.find_spec("sentence_transformers") is not None


def spec_tag(
    spec: Any, rerank_text: Any = "surface", *, max_chars: int = DFLT_RERANK_MAX_CHARS
):
    """The tag a ``rerank`` spec *would* get, without loading a model; ``None`` if the
    spec cannot run here (a ``"cross-encoder"`` spec whose extra is not installed).

    For lookups such as :func:`ir.eval.load_calibration`, which need the key, not the model.
    """
    if not isinstance(spec, str):
        return rerank_tag(spec, rerank_text, max_chars=max_chars)
    name, _, model = spec.partition(":")
    if name != CROSS_ENCODER:
        resolve_reranker(spec)  # raises the usual error
    if not _has_sentence_transformers():
        return None
    ghost = types.SimpleNamespace(rerank_id=f"{model or DFLT_RERANK_MODEL}+sigmoid")
    return rerank_tag(ghost, rerank_text, max_chars=max_chars)


def calibration_key(mode: str, tag: str | None = None) -> str:
    """The key an abstention floor is stored and loaded under.

    ``mode`` alone for a first-stage ranking (today's behaviour, so existing floors keep
    working); ``"<mode>+<tag>"`` for a reranked one.

    >>> calibration_key("hybrid")
    'hybrid'
    >>> calibration_key("hybrid", "rerank-m+artifact")
    'hybrid+rerank-m+artifact'
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
    if max_chars < 1:
        raise ValueError(f"rerank_max_chars must be positive, got {max_chars}")
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
    from .retrieve import records_for_artifact

    whole: dict[str, str] = {}

    def artifact_text(meta: Mapping[str, Any]) -> str:
        artifact_id = meta["artifact_id"]
        if artifact_id not in whole:
            try:
                records = records_for_artifact(corpus, artifact_id)
            except (
                KeyError
            ):  # no ledger entry, or a stale one: fall back to the surface
                records = []
            records = [r for r in records if r.text]
            stitched = _stitch(records) if records else ""
            whole[artifact_id] = stitched[:max_chars] or meta["text"]
        return whole[artifact_id]

    return [artifact_text(meta) for meta in candidates]
