"""``ir`` — an information-retrieval substrate for agentic systems.

One uniform "find the relevant things in this corpus" contract that scales from
an ad-hoc search over an ephemeral list to a maintained search engine. Retrieval
is the core; generation/selection/reranking are layered on top.

Quick start::

    import ir

    # Define a corpus source (abstract strategy + parameters, smart defaults):
    source = ir.CorpusSource.from_md_reports()          # project docs/ reports
    corpus = ir.build(source)                            # index (incremental)
    hits = ir.search(corpus, "how do I deploy the app")  # ranked SearchHits

    # Light, dependency-free embedding for fast tests:
    corpus = ir.build(source, embedder="light")

A corpus source is defined by a ``scope`` (what is in the corpus), a
``change_signal`` (what counts as stale), an ``indexing_strategy`` (how a raw
item becomes filter fields + embeddable surfaces), and an ``embedder``. The
default embedder is a decent *local* model (``all-MiniLM-L6-v2``); ``"light"``
selects a numpy-only hashing embedder. Data persists under XDG dirs through a
``dol`` repository layer.
"""

from __future__ import annotations

from . import embed as _embed  # noqa: F401  (sets USE_TF=0 before transformers)
from . import registry
from .base import Artifact, IndexPlan, Record, SearchHit, Surface, tag_source
from .formulate import Formulator, make_llm_formulator
from .index import Corpus, build, open_corpus
from .registry import retriever_for, retrievers
from .retrieve import Retriever, as_retriever, fuse_hits
from .retrieve import search as _search
from .select import (
    Disclosure,
    DiscoveryResult,
    Selection,
    disclose,
    discover,
    select,
)
from .sources import CorpusSource
from .store import CorpusStore
from .strategy import Chunked, IndexingStrategy, Package, Skill, WholeText

__all__ = [
    "Artifact",
    "Surface",
    "Record",
    "SearchHit",
    "IndexPlan",
    "IndexingStrategy",
    "WholeText",
    "Chunked",
    "Skill",
    "Package",
    "CorpusSource",
    "CorpusStore",
    "Corpus",
    "build",
    "open_corpus",
    "search",
    "as_retriever",
    "Retriever",
    "retrievers",
    "retriever_for",
    "fuse_hits",
    "tag_source",
    "make_llm_formulator",
    "Formulator",
    "select",
    "disclose",
    "discover",
    "Selection",
    "Disclosure",
    "DiscoveryResult",
    "register",
    "corpora",
    "build_corpus",
]

register = registry.register
corpora = registry.registered


def build_corpus(name, **kwargs):
    """Build (or update) a registered/preset corpus by *name*; returns a :class:`~ir.index.Corpus`.

    ``**kwargs`` are forwarded to :func:`ir.build` — notably ``store``,
    ``embedder`` (e.g. ``"light"`` for the numpy-only hashing embedder),
    ``full`` (prune artifacts no longer in the source), and ``batch_size``.
    """
    return build(registry.source_for(name), **kwargs)


def search(corpus, query, **kwargs):
    """Search a :class:`~ir.index.Corpus` (or a corpus *name*, reopened lazily).

    Thin facade over :func:`ir.retrieve.search`; ``**kwargs`` are forwarded to
    it — the useful ones are ``k`` (how many hits), ``mode`` (``"dense"`` /
    ``"lexical"`` / ``"hybrid"``), ``filter`` (a ``vd`` Mongo-style metadata
    filter), ``surfaces`` (restrict to surface kinds), and ``per_artifact``
    (collapse to the best surface per artifact). See :func:`ir.retrieve.search`
    for the full signature and defaults.
    """
    if isinstance(corpus, str):
        corpus = open_corpus(corpus)
    return _search(corpus, query, **kwargs)


# The evaluation harness is reachable as ``ir.eval`` (its ``ef`` imports are
# lazy, so this does not weigh down ``import ir``). Kept out of ``__all__`` so a
# star-import does not shadow the ``eval`` builtin. ``ir.eval_gen`` is the
# build-time case generator (its ``oa`` import is lazy too).
from . import eval  # noqa: E402,F401  (submodule attribute: ir.eval)
from . import eval_gen  # noqa: E402,F401  (submodule attribute: ir.eval_gen)
