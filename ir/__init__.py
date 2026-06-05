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
from .base import Artifact, IndexPlan, Record, SearchHit, Surface
from .index import Corpus, build, open_corpus
from .retrieve import search as _search
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
]


def search(corpus, query, **kwargs):
    """Search a :class:`~ir.index.Corpus`, or a corpus *name* (reopened lazily)."""
    if isinstance(corpus, str):
        corpus = open_corpus(corpus)
    return _search(corpus, query, **kwargs)
