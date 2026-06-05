"""Command-line surface for ``ir`` (argh-dispatched).

Commands operate on **named** corpora from the registry (see
:mod:`ir.registry`)::

    ir build skills                 # build/update the skills preset corpus
    ir search skills "deploy app"   # query it
    ir ls                           # list corpora + record counts
    ir info packages                # config + stats for a corpus
    ir register notes files --root ~/notes --pattern '.*\\.md$'
    ir rm notes                     # unregister (keeps built data)
"""

from __future__ import annotations

from . import registry
from .index import build as _build
from .index import open_corpus


def ls():
    """List registered corpora with their kind, embedder, and record count."""
    entries = registry.registered()
    if not entries:
        return "No corpora registered. Try: ir build skills"
    lines = []
    for name, e in entries.items():
        try:
            count = len(open_corpus(name))
        except Exception:
            count = 0
        lines.append(
            f"{name:18} {e['kind']:10} {e.get('embedder', 'default'):10} records={count}"
        )
    return "\n".join(lines)


def register(name, kind, *, root=None, pattern=None, embedder="default"):
    """Register a named corpus. kind: skills | packages | reports | files."""
    params = {}
    if root:
        params["root"] = root
    if pattern:
        params["pattern"] = pattern
    registry.register(name, kind, embedder=embedder, **params)
    return f"registered {name!r} (kind={kind}, embedder={embedder})"


def build(name, *, embedder=None, full=True):
    """Build or incrementally update a registered (or preset) corpus."""
    source = registry.source_for(name)
    corpus = _build(source, embedder=embedder, full=full)
    return f"built {name!r}: {len(corpus)} records (embedder {corpus.embedder_id})"


def search(name, query, *, k=10, mode="dense"):
    """Search a built corpus and print the top-k hits.

    mode: dense (cosine) | lexical (BM25) | hybrid (dense + BM25 fused via RRF).
    """
    corpus = open_corpus(name)
    if len(corpus) == 0:
        return f"corpus {name!r} is empty; build it first: ir build {name}"
    lines = []
    for h in corpus.search(query, k=k, mode=mode):
        # artifact_id is the unique label across corpora (skill name[@parent],
        # package name, or a report's relative path).
        lines.append(f"{h.score:+.3f}  {h.artifact_id}  [{h.surface_kind}]")
    return "\n".join(lines) or "(no matches)"


def info(name):
    """Show a corpus's stored config and stats."""
    corpus = open_corpus(name)
    cfg = corpus.store.get_config()
    reg = registry.get(name)
    return f"name: {name}\nregistered: {reg}\nrecords: {len(corpus)}\nconfig: {cfg}"


def rm(name):
    """Unregister a corpus (does not delete its built data)."""
    registry.unregister(name)
    return f"unregistered {name!r}"


COMMANDS = [ls, register, build, search, info, rm]
