"""The semantic link graph — typed edges between artifacts (report 12).

ir indexes artifacts that *refer to each other*: a package depends on other
packages, a skill belongs to a parent, a chunk follows the previous one, a
report cites another. This module models those references as **typed, directed
edges** between nodes identified by ``(source, artifact_id)`` (ir's canonical
node identity since 0.1.16), queryable at retrieval time — the substrate the
traversal operator (``traverse``, follow-up) walks.

Two pieces:

- the :data:`GraphStore` **protocol** — a minimal ``__getitem__`` +
  ``neighbors`` structural contract, so a traversal operator binds to *any*
  conforming store, not only ir's;
- :class:`CorpusGraph` — the corpus-backed implementation. Node payloads are
  the artifact's stored records (via :func:`ir.retrieve.records_for_artifact`);
  neighbors come from the ``links`` view on :class:`~ir.store.CorpusStore`.

Edges are **regenerable derived state** (persisted in the ``links`` view, like
``calibration``) — never part of build identity. They are ingested at build
time by an injectable :data:`EdgeExtractor`; the shipped
:func:`default_edge_extractor` turns the edge data already latent in the index
(``Package`` ``deps`` → ``REF``, ``Skill`` ``parent`` → ``PARENT``) into edges.

Vocabulary (documented, **not** enforced): :data:`NEXT` / :data:`PREV` /
:data:`PARENT` / :data:`CHILD` / :data:`REF`. ``NEXT`` / ``PREV`` are *not*
materialized — they are derivable from the ledger's surface order (#44); the
view stores only what isn't derivable.

**Naming guard** (ADR #43): this is the **semantic link graph** — directed,
**possibly cyclic** (citations, cross-references), traversed at query time. It
is *not* ``ef.artifact_graph``, which is an acyclic build-time
derivation/lineage DAG. The id conventions are kept compatible so a later
unification stays open, but the two are distinct substrates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Protocol, runtime_checkable

from .base import Record
from .retrieve import records_for_artifact

# ---- edge vocabulary (documented, not enforced) --------------------------- #

#: Next sibling surface in sequence (derivable from the ledger — not stored).
NEXT = "NEXT"
#: Previous sibling surface in sequence (derivable from the ledger — not stored).
PREV = "PREV"
#: Containment up: a child points at its parent (e.g. a skill → its package).
PARENT = "PARENT"
#: Containment down: a parent points at a child.
CHILD = "CHILD"
#: A generic reference / dependency / citation (e.g. a package → its deps).
REF = "REF"

#: A node's identity: ``artifact_id`` within one corpus, or ``(source,
#: artifact_id)`` across corpora. The visited-set of a traversal keys on this.
NodeId = "str | tuple[str, str]"


@runtime_checkable
class GraphStore(Protocol):
    """Structural contract a traversal operator binds to — node + neighbors.

    Deliberately minimal (two methods) so it is satisfied by ir's
    :class:`CorpusGraph` *and* by any external graph: ``__getitem__`` resolves
    a node id to its scorable payload, ``neighbors`` lists adjacent node ids
    (optionally of one edge type). Granularity-agnostic on purpose — an
    artifact graph and a surface-level tree are both ``GraphStore``s.
    """

    def __getitem__(self, node_id: Any) -> Any: ...

    def neighbors(self, node_id: Any, *, edge_type: str | None = None) -> Iterable: ...


class CorpusGraph:
    """A :data:`GraphStore` over one corpus — artifact nodes, ``links`` edges.

    ``node_id`` is an ``artifact_id``. ``graph[aid]`` is the artifact's stored
    records (its scorable surfaces, in plan order); ``graph.neighbors(aid,
    edge_type=...)`` reads the corpus store's ``links`` view. Single-corpus, so
    it resolves **intra-corpus** targets; cross-corpus ``[source, artifact_id]``
    targets are returned by :meth:`neighbors` verbatim but ``__getitem__`` only
    dereferences ids in *this* corpus (federated traversal is a follow-up).
    """

    def __init__(self, store_or_corpus: Any):
        self.store = getattr(store_or_corpus, "store", store_or_corpus)
        #: The corpus name, when known — the ``source`` half of this graph's
        #: node identities (``None`` for a bare store).
        self.source = getattr(store_or_corpus, "name", None)

    def __getitem__(self, node_id: str) -> list[Record]:
        """The artifact's records, plan-ordered (``NoLedgerEntry`` if unknown)."""
        return records_for_artifact(self.store, node_id)

    def neighbors(self, node_id: str, *, edge_type: str | None = None) -> list:
        """Outgoing neighbor ids of *node_id*, optionally of one *edge_type*.

        Returns target ids in stored form (a bare ``artifact_id``, or a
        ``[source, artifact_id]`` list for a cross-corpus edge), de-duplicated
        with first-seen order preserved. An artifact with no edges (or a store
        without a links view) yields ``[]`` — never raises.
        """
        edges = self.store.get_links(node_id)
        if edge_type is not None:
            targets = list(edges.get(edge_type, []))
        else:
            targets = [t for ts in edges.values() for t in ts]
        seen: set = set()
        out: list = []
        for t in targets:
            key = tuple(t) if isinstance(t, list) else t
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def edge_types(self, node_id: str) -> list[str]:
        """The edge types present on *node_id* (``[]`` if none)."""
        return list(self.store.get_links(node_id))


# ---- edge ingest ---------------------------------------------------------- #

#: An edge extractor: ``(artifact_id, filter_fields) -> {edge_type: [target]}``.
#: An injectable build-time seam (like :data:`~ir.formulate.Formulator`) that
#: turns an artifact's filter fields into outgoing edges. ``build(...,
#: edge_extractor=...)`` runs it; :func:`default_edge_extractor` is shipped.
EdgeExtractor = Callable[[str, Mapping[str, Any]], "Mapping[str, list]"]

#: Split a PEP 508 dependency spec at the first version/extra/marker delimiter,
#: leaving the bare distribution name (``"numpy[fast]>=1.2; python>='3.9'"`` →
#: ``"numpy"``).
_DEP_DELIM = re.compile(r"[<>=!~;\[\s(]")


def _dep_name(dep: str) -> str:
    """The bare distribution name of a dependency spec (lower-cased, stripped)."""
    return _DEP_DELIM.split(dep.strip(), 1)[0].strip().lower()


def default_edge_extractor(
    artifact_id: str, filter_fields: Mapping[str, Any]
) -> dict[str, list]:
    """Edges latent in the standard filter fields: ``deps`` → REF, ``parent`` → PARENT.

    - ``deps`` (Package) → :data:`REF` edges to each dependency's bare name
      (version specifiers / extras / markers stripped). Self-edges and blanks
      are dropped.
    - ``parent`` (Skill) → a single :data:`PARENT` edge.

    A package whose ``deps`` name other packages in the same corpus gets
    intra-corpus REF edges; third-party deps become REF edges to ids not in the
    corpus (harmless — :meth:`CorpusGraph.neighbors` lists them, and a
    traversal simply finds no node to expand).
    """
    edges: dict[str, list] = {}
    deps = filter_fields.get("deps")
    if isinstance(deps, (list, tuple)):
        refs = []
        for dep in deps:
            name = _dep_name(str(dep))
            if name and name != artifact_id:
                refs.append(name)
        if refs:
            edges[REF] = list(dict.fromkeys(refs))  # de-dup, keep order
    parent = filter_fields.get("parent")
    if isinstance(parent, str) and parent and parent != artifact_id:
        edges[PARENT] = [parent]
    return edges
