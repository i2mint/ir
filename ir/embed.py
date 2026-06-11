"""Embedder resolution for ``ir`` — a decent local default, a light fallback.

``ir`` favors a *decent local* embedding so retrieval works offline and tests
need no API keys, while keeping a *light* (numpy-only) option for when semantic
power is not what's under test.

- ``"default"`` / ``"local"`` / ``"minilm"`` → ``all-MiniLM-L6-v2`` (384-dim)
  via :func:`ef.embedder_adapters.sentence_transformers_embedder`
  (``normalize=True``), wrapped in :class:`ef.CachedEmbedder` over a ``dol``
  cache under ``~/.cache/ir/embeddings/<model>``. If ``sentence-transformers``
  is unavailable, it degrades to the hashing embedder with a warning.
- ``"light"`` / ``"hashing"`` → :class:`ef.HashingEmbedder` (numpy only).
- any other string → treated as a sentence-transformers model name.
- a callable / existing ``Embedder`` → passed through ``ef.as_embedder``.

:func:`make_embedder` returns ``(embedder, embedder_id)``; the id pins the
model in the corpus ledger so a model change triggers a re-embed (the SSOT
discipline that keeps the index from silently drifting).

Importing this module sets ``USE_TF=0`` so ``sentence-transformers`` (via
``transformers``) does not import TensorFlow, which crashes on this stack's
numpy ABI. Import ``ir`` before anything that imports ``transformers``.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

# Force-disable TensorFlow in transformers (it crashes on this stack's numpy
# ABI). Assignment (not setdefault) so a stray ``USE_TF=1`` in the shell can't
# re-enable it. Must precede any transformers import; ``ir`` never uses TF.
os.environ["USE_TF"] = "0"

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_LIGHT = {"light", "hashing", "hash"}
_LOCAL = {"default", "local", "minilm", "st", "sentence-transformers"}


def _hashing():
    import ef

    return ef.HashingEmbedder(), "hashing__dim512"


def _sentence_transformers(model_name: str):
    from ef import embedder_adapters

    emb = embedder_adapters.sentence_transformers_embedder(model_name, normalize=True)
    return emb, f"st__{model_name}"


def _with_cache(emb, model_id: str):
    """Wrap *emb* in a disk-cached embedder keyed under this model's cache dir."""
    import ef

    from .config import embeddings_cache_dir
    from .store import _ndarray_store

    store = _ndarray_store(embeddings_cache_dir(model_id))
    return ef.CachedEmbedder(emb, store)


def make_embedder(spec: Any = "default", *, cache: bool = True) -> tuple[Callable, str]:
    """Resolve *spec* to ``(embedder, embedder_id)``.

    ``embedder`` is a batch callable ``Iterable[str] -> ndarray(n, dim)`` that
    also accepts ``input_type=`` (``"query"`` / ``"document"``).
    """
    import ef

    if callable(spec) and not isinstance(spec, str):
        return ef.as_embedder(spec), getattr(spec, "_ir_id", "custom")

    key = (spec or "default").strip().lower()

    if key in _LIGHT:
        return _hashing()

    if key in _LOCAL:
        model_name = DEFAULT_MODEL
    else:
        model_name = spec  # any other string is a model name

    try:
        emb, model_id = _sentence_transformers(model_name)
    except Exception as e:  # ImportError or model-load failure
        warnings.warn(
            f"Local embedder {model_name!r} unavailable ({e}); "
            f"falling back to the light hashing embedder. "
            f"Install with `pip install sentence-transformers` for semantic search.",
            stacklevel=2,
        )
        return _hashing()

    if cache:
        emb = _with_cache(emb, model_id)
    return emb, model_id
