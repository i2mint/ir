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

    ``runtime_checkable`` makes ``isinstance(x, GraphStore)`` a *structural*
    check on attribute **names** only (not signatures) — enough to tell a
    conforming adapter from an arbitrary object, but it cannot validate that
    ``neighbors`` takes the right arguments; treat it as a smoke check.
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
        self._reverse: dict[str, dict[Any, list[str]]] = {}
        self._artifacts_by_key: dict[str, dict[Any, str]] = {}

    def __getitem__(self, node_id: str) -> list[Record]:
        """The artifact's records, plan-ordered (``NoLedgerEntry`` if unknown)."""
        return records_for_artifact(self.store, node_id)

    def neighbors(self, node_id: str, *, edge_type: str | None = None) -> list:
        """Outgoing neighbor ids of *node_id*, optionally of one *edge_type*.

        Returns target ids in stored form (a bare ``artifact_id``, whose source
        is *this* graph's :attr:`source`; or a ``[source, artifact_id]`` list
        for a cross-corpus edge), de-duplicated with first-seen order
        preserved. An artifact with no edges — or a store without a links view —
        yields ``[]``. Pass a canonical ``(source, artifact_id)`` to
        :func:`canonical_node_id` for a traversal's visited-set.

        *node_id* is an intra-corpus ``artifact_id`` (a ``str``); a cross-corpus
        target fetched from another graph is out of contract here (it has no
        edges *in this corpus*).
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

    def _reverse_index(self, edge_type: str) -> dict[Any, list[str]]:
        """``{target: [source artifact ids]}`` for one edge type, built once.

        Derived from the ``links`` view on first use and kept for the life of
        this graph instance (a snapshot, like the forward edges themselves):
        build a fresh :class:`CorpusGraph` to see edges written since.
        """
        index = self._reverse.get(edge_type)
        if index is None:
            index = {}
            link_items = getattr(self.store, "link_items", None)
            for artifact_id, edges in link_items() if callable(link_items) else ():
                for target in edges.get(edge_type, []):
                    key = _reverse_key(target, edge_type)
                    sources = index.setdefault(key, [])
                    if artifact_id not in sources:
                        sources.append(artifact_id)
            self._reverse[edge_type] = index
        return index

    def _artifacts_by_key_for(self, edge_type: str) -> dict[Any, str]:
        """``{reverse-index key: artifact_id}`` over this corpus's artifacts.

        Read from the ledger (whose keys are hashes, so the artifact id comes from
        each entry), built once and kept for the life of this graph instance, like
        :meth:`_reverse_index`.
        """
        index = self._artifacts_by_key.get(edge_type)
        if index is None:
            ledger_items = getattr(self.store, "ledger_items", None)
            entries = ledger_items() if callable(ledger_items) else ()
            index = {
                _reverse_key(aid, edge_type): aid
                for aid in (entry.get("artifact_id") for _, entry in entries)
                if aid is not None
            }
            self._artifacts_by_key[edge_type] = index
        return index

    def resolve(self, node_id: Any, *, edge_type: str = REF) -> Any:
        """The node id *node_id* names here, as :meth:`reverse_neighbors` keys it.

        A ``(source, artifact_id)`` pair whose *source* is this graph's
        :attr:`source` is read as its bare ``artifact_id``. An id naming an artifact
        of this corpus (through the *edge_type* reverse-index key, so for ``REF``
        case and ``-``/``_``/``.`` are ignored) resolves to that artifact's id;
        anything else (a third-party library, another corpus's pair) resolves to its
        own reverse-index key.

        >>> CorpusGraph({}).resolve("Scikit_Learn")
        'scikit-learn'
        >>> CorpusGraph({}).resolve(["other", "Meshed"])
        ('other', 'Meshed')
        """
        if (
            isinstance(node_id, (list, tuple))
            and self.source is not None
            and len(node_id) == 2
            and node_id[0] == self.source
        ):
            node_id = node_id[1]
        key = _reverse_key(node_id, edge_type)
        return self._artifacts_by_key_for(edge_type).get(key, key)

    def reverse_neighbors(self, node_id: Any, *, edge_type: str = REF) -> list[str]:
        """The artifact ids whose *edge_type* edges point **at** *node_id*.

        The reverse of :meth:`neighbors`: with the default :data:`REF` edges of
        :func:`default_edge_extractor` (``package -> its deps``), this lists a
        library's dependents. *node_id* may name an artifact of this corpus or a
        target outside it (a third-party dependency). For ``REF`` edges, string
        ids match through :func:`normalize_dist_name` (case, and ``-``/``_``/``.``,
        are ignored), so ``"scikit_learn"`` finds the dependents of
        ``"scikit-learn"``; other edge types match ids exactly. A cross-corpus
        target is a ``(source, artifact_id)`` pair, matched exactly. Order is
        first-seen over the links view, de-duplicated.
        """
        index = self._reverse_index(edge_type)
        return list(index.get(_reverse_key(node_id, edge_type), []))

    def fan_in(self, node_id: Any, *, edge_type: str = REF) -> int:
        """How many artifacts point at *node_id* by *edge_type* (its in-degree).

        With ``REF`` dependency edges this is the number of dependents, a
        PageRank-style centrality prior: a widely depended-on library is central
        to the corpus.
        """
        return len(self.reverse_neighbors(node_id, edge_type=edge_type))

    def max_fan_in(self, *, edge_type: str = REF) -> int:
        """The largest :meth:`fan_in` over all targets (``0`` without edges)."""
        return max(map(len, self._reverse_index(edge_type).values()), default=0)


def _reverse_key(target: Any, edge_type: str) -> Any:
    """The reverse-index key of an edge target.

    A cross-corpus pair matches exactly. A string target of a :data:`REF`
    (dependency) edge matches through :func:`normalize_dist_name`; any other
    edge type's ids (paths, ``name@parent``) match exactly.
    """
    if isinstance(target, (list, tuple)):
        return tuple(target)
    return normalize_dist_name(target) if edge_type == REF else target


def canonical_node_id(target: Any, *, source: str | None) -> tuple[str | None, str]:
    """Canonicalize a neighbor *target* to a ``(source, artifact_id)`` node id.

    The repo's node identity is ``(source, artifact_id)`` — the key a traversal
    visited-set must use so the same id in two corpora stays two nodes.
    :meth:`CorpusGraph.neighbors` returns targets in stored form: a bare
    ``artifact_id`` (implicitly in *source*, the graph it came from) or a
    ``[source, artifact_id]`` cross-corpus pair. This resolves either to the
    canonical tuple.

    >>> canonical_node_id("dol", source="packages")
    ('packages', 'dol')
    >>> canonical_node_id(["skills", "deploy"], source="packages")
    ('skills', 'deploy')
    """
    if isinstance(target, (list, tuple)) and len(target) == 2:
        return (target[0], target[1])
    return (source, target)


# ---- edge ingest ---------------------------------------------------------- #

#: An edge extractor: ``(artifact_id, filter_fields) -> {edge_type: [target]}``.
#: An injectable build-time seam (like :data:`~ir.formulate.Formulator`) that
#: turns an artifact's filter fields into outgoing edges. ``build(...,
#: edge_extractor=...)`` runs it; :func:`default_edge_extractor` is shipped.
EdgeExtractor = Callable[[str, Mapping[str, Any]], "Mapping[str, list]"]

#: Split a PEP 508 dependency spec at the first version/extra/marker/url
#: delimiter, leaving the bare distribution name (``"numpy[fast]>=1.2;
#: python>='3.9'"`` → ``"numpy"``; the direct-reference ``"dol@git+https://…"``
#: → ``"dol"``). ``@`` is included so PEP 508 URL deps yield the bare name.
_DEP_DELIM = re.compile(r"[<>=!~;@\[\s(]")


def _dep_name(dep: str) -> str:
    """The bare distribution name of a dependency spec (lower-cased, stripped)."""
    return _DEP_DELIM.split(dep.strip(), 1)[0].strip().lower()


def normalize_dist_name(name: str) -> str:
    """A dependency spec's bare name, PEP 503-normalized (``-``/``_``/``.`` -> ``-``).

    The one comparison key for library names: the reverse dependency index and
    :func:`ir.dependency_evidence_boost` both match through it, so
    ``"scikit_learn"`` finds a package that depends on ``"Scikit-Learn>=1"``.

    >>> normalize_dist_name("Scikit_Learn[extra]>=1.2")
    'scikit-learn'
    """
    return re.sub(r"[-_.]+", "-", _dep_name(str(name)))


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

    Self-edges are dropped case-insensitively (``_dep_name`` lower-cases, so a
    package ``"AA"`` depending on ``"aa"`` is recognized as a self-reference).
    """
    edges: dict[str, list] = {}
    self_id = artifact_id.lower()
    deps = filter_fields.get("deps")
    if isinstance(deps, (list, tuple)):
        refs = []
        for dep in deps:
            name = _dep_name(str(dep))
            if name and name != self_id:
                refs.append(name)
        if refs:
            edges[REF] = list(dict.fromkeys(refs))  # de-dup, keep order
    parent = filter_fields.get("parent")
    if isinstance(parent, str) and parent and parent.lower() != self_id:
        edges[PARENT] = [parent]
    return edges
