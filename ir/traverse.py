"""Query-time graph traversal — the ``traverse`` operator (report 12).

Recursive retrieval, summary-routing, RAPTOR collapsed-tree, PPR /
spreading-activation, and beam walks are **one operator** — *score frontier →
select → expand → test stop* — parameterized by a pluggable
:class:`WalkPolicy`. The engineering substance is termination, split in two:

- **safety** ("*will* it stop?") — structural, and **enforced by the operator**:
  a visited-set (keyed on the policy's node id), a depth cap, and a node
  budget live in :class:`WalkState` and are checked by :func:`traverse` itself,
  so a buggy or adversarial policy *cannot* loop forever — the highest-leverage
  robustness decision when the graph may contain directed cycles;
- **sufficiency** ("*should* it stop?") — injected and fallible: the policy's
  ``stop`` (default: never — run to budget).

The policy owns graph semantics; the operator owns the loop and the safety
primitives. ``store`` is passed to the policy verbatim and never interpreted by
the operator — collapsed-tree takes a :class:`~ir.index.Corpus` (search to
seed, ledger to expand to chunks), an artifact-link policy a
:class:`~ir.graph.CorpusGraph` (``neighbors``).

**Flat top-k + rerank stays the default** (report 12: a strong flat retriever
beats most graph methods on simple lookup, and global graph methods cost
orders of magnitude more); :func:`traverse` is opt-in, and a policy earns
promotion only by beating flat on the eval set. The shipped
:func:`collapsed_tree_policy` is pure-vector (no LLM in the query loop): it
routes a query that matches an artifact's *summary* surface down to that
artifact's best *chunk*.

Returned :class:`~ir.base.SearchHit`\\s carry additive, JSON-clean walk
provenance — ``metadata["walk_depth"]`` and ``metadata["seed"]`` (the routing
node) — and compose with :func:`ir.select` / :func:`ir.disclose` unchanged.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .base import Record, SearchHit
from .retrieve import records_for_artifact

#: Default summary/router surface kinds a collapsed-tree walk seeds on.
DFLT_SUMMARY_KINDS = ("description", "synopsis", "capability", "document")
#: Default leaf surface kinds a collapsed-tree walk descends to (the results).
DFLT_LEAF_KINDS = ("chunk", "readme_chunk")
#: Default traversal bounds (operator-enforced safety).
DFLT_MAX_DEPTH = 2
DFLT_NODE_BUDGET = 64
#: Default number of summary surfaces a collapsed-tree walk routes from.
DFLT_SEED_K = 10


@dataclass
class WalkState:
    """The operator-owned state of one :func:`traverse` call — the safety home.

    ``visited`` (node ids already committed), ``budget``, and ``max_depth`` are
    the structural safety primitives the operator enforces; ``results`` are the
    emitted hits; ``cache`` is scratch space a policy may use (e.g. to embed the
    query once). A policy reads this but the *operator* enforces the bounds —
    a policy cannot opt out of termination.
    """

    query: str
    max_depth: int
    budget: int
    visited: set = field(default_factory=set)
    results: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)


@runtime_checkable
class WalkPolicy(Protocol):
    """The pluggable strategy of a walk — graph semantics, not safety.

    ``seed`` produces the initial frontier; ``score`` ranks a node against the
    query; ``select`` chooses which scored frontier nodes to commit/expand this
    step (beam/greedy — default: all, best-first); ``expand`` yields a node's
    neighbors; ``node_id`` is the hashable visited-set key; ``stop`` is the
    injected sufficiency check; ``to_hit`` materializes a committed node as a
    :class:`~ir.base.SearchHit` — or ``None`` for a *router-only* node (a
    summary that routes but is not itself a result).
    """

    def seed(self, state: WalkState, store: Any) -> Iterable: ...
    def score(self, state: WalkState, node: Any, store: Any) -> float: ...
    def select(self, state: WalkState, scored: Sequence) -> Sequence: ...
    def expand(self, state: WalkState, node: Any, store: Any) -> Iterable: ...
    def node_id(self, node: Any) -> Hashable: ...
    def stop(self, state: WalkState) -> bool: ...
    def to_hit(
        self, state: WalkState, node: Any, score: float, depth: int
    ) -> "SearchHit | None": ...


def _finalize(results: list[SearchHit], k: int) -> list[SearchHit]:
    """Best-first, top-*k*. Visited-set dedup already guarantees one per node."""
    return sorted(results, key=lambda h: h.score, reverse=True)[:k]


def traverse(
    query: str,
    store: Any,
    *,
    policy: WalkPolicy,
    max_depth: int = DFLT_MAX_DEPTH,
    node_budget: int = DFLT_NODE_BUDGET,
    k: int = 10,
) -> list[SearchHit]:
    """Walk *store* from *query* under *policy*, returning the top-*k* hits.

    The loop — *score the frontier → select → commit → expand* — is the
    operator's; the **safety primitives are non-negotiable and enforced here**:
    a node id is committed at most once (the visited-set), expansion stops at
    ``max_depth``, and no more than ``node_budget`` nodes are ever committed.
    A policy whose ``expand`` cycles forever and whose ``stop`` never fires
    still terminates.

    Args:
        query: the user intent.
        store: passed to *policy* verbatim — a :class:`~ir.index.Corpus` for
            :func:`collapsed_tree_policy`, a :class:`~ir.graph.CorpusGraph` for
            an artifact-link policy. The operator never inspects it.
        policy: the :class:`WalkPolicy` (e.g. :func:`collapsed_tree_policy`).
        max_depth: maximum expansion depth from a seed (safety).
        node_budget: maximum nodes committed (safety).
        k: number of hits to return.

    Returns:
        the committed hits, best-first, top-*k* — each a
        :class:`~ir.base.SearchHit` with ``metadata["walk_depth"]`` / ``["seed"]``.
    """
    state = WalkState(query=query, max_depth=max_depth, budget=node_budget)
    frontier = [(node, 0) for node in policy.seed(state, store)]
    while frontier and len(state.visited) < node_budget:
        scored = [
            (node, depth, policy.score(state, node, store))
            for node, depth in frontier
            if policy.node_id(node) not in state.visited
        ]
        if not scored:
            break
        next_frontier: list[tuple[Any, int]] = []
        for node, depth, score in policy.select(state, scored):
            nid = policy.node_id(node)
            if nid in state.visited:
                continue
            if len(state.visited) >= node_budget:
                break
            state.visited.add(nid)
            hit = policy.to_hit(state, node, score, depth)
            if hit is not None:
                state.results.append(hit)
                if policy.stop(state):
                    return _finalize(state.results, k)
            if depth < max_depth:
                for nb in policy.expand(state, node, store):
                    if policy.node_id(nb) not in state.visited:
                        next_frontier.append((nb, depth + 1))
        frontier = next_frontier
    return _finalize(state.results, k)


# =========================================================================== #
# Collapsed-tree / summary-routing — the first shipped policy (pure-vector)
# =========================================================================== #


@dataclass(frozen=True)
class _WalkNode:
    """A traversal node: one surface record + the artifact that routed to it."""

    record: Record
    seed: str | None = None


def _cosine(query_vec: np.ndarray, vec: np.ndarray) -> float:
    """Cosine of a unit *query_vec* against a raw record *vec*."""
    norm = float(np.linalg.norm(vec))
    return float(query_vec @ (vec / norm)) if norm else 0.0


class _CollapsedTreePolicy:
    """Summary-routing: seed on summary surfaces, descend to leaf chunks.

    The document-summary-index / RAPTOR collapsed-tree pattern, surface-grained
    over ir's heterogeneous surfaces: a query that matches an artifact's
    ``description`` / ``synopsis`` surface (the *router*, never itself a result)
    surfaces that artifact's best leaf chunk (the *result*). Node id is the
    record id, so within-artifact descent (summary and chunks share an
    ``artifact_id``) is not blocked by the visited-set.
    """

    def __init__(self, *, summary_kinds, leaf_kinds, seed_k):
        self.summary_kinds = tuple(summary_kinds)
        self.leaf_kinds = tuple(leaf_kinds)
        self.seed_k = seed_k

    def _query_vec(self, state: WalkState, store: Any) -> np.ndarray:
        qv = state.cache.get("query_vec")
        if qv is None:
            from .retrieve import _embed_query

            embedder = getattr(store, "embedder", None)
            qv = _embed_query(embedder, state.query)
            state.cache["query_vec"] = qv
            state.cache["source"] = getattr(store, "name", None)
        return qv

    def seed(self, state: WalkState, store: Any) -> list[_WalkNode]:
        from .retrieve import search

        self._query_vec(state, store)  # warm the cache (query_vec + source)
        hits = search(
            store,
            state.query,
            surfaces=self.summary_kinds,
            k=self.seed_k,
            per_artifact=True,
        )
        nodes: list[_WalkNode] = []
        for h in hits:
            for r in records_for_artifact(store, h.artifact_id):
                if r.surface_kind in self.summary_kinds:
                    nodes.append(_WalkNode(record=r, seed=None))
                    break  # one router surface per artifact
        return nodes

    def score(self, state: WalkState, node: _WalkNode, store: Any) -> float:
        return _cosine(self._query_vec(state, store), node.record.vector)

    def select(self, state: WalkState, scored: Sequence) -> list:
        return sorted(scored, key=lambda t: t[2], reverse=True)

    def expand(self, state: WalkState, node: _WalkNode, store: Any) -> list[_WalkNode]:
        # Only summary routers expand (to their artifact's leaves); a leaf is
        # terminal, so a query that matches a leaf directly doesn't fan out.
        if node.record.surface_kind not in self.summary_kinds:
            return []
        aid = node.record.artifact_id
        return [
            _WalkNode(record=r, seed=aid)
            for r in records_for_artifact(store, aid)
            if r.surface_kind in self.leaf_kinds
        ]

    def node_id(self, node: _WalkNode) -> Hashable:
        return node.record.id

    def stop(self, state: WalkState) -> bool:
        return False

    def to_hit(
        self, state: WalkState, node: _WalkNode, score: float, depth: int
    ) -> "SearchHit | None":
        # Router (summary) nodes route but are not results.
        if node.record.surface_kind in self.summary_kinds:
            return None
        meta = dict(node.record.metadata)
        meta["walk_depth"] = depth
        if node.seed is not None:
            meta["seed"] = node.seed
        return SearchHit(
            artifact_id=node.record.artifact_id,
            surface_kind=node.record.surface_kind,
            score=float(score),
            text=node.record.text,
            metadata=meta,
            source=state.cache.get("source"),
            surface_index=node.record.surface_index,
        )


def collapsed_tree_policy(
    *,
    summary_kinds: Iterable[str] = DFLT_SUMMARY_KINDS,
    leaf_kinds: Iterable[str] = DFLT_LEAF_KINDS,
    seed_k: int = DFLT_SEED_K,
) -> WalkPolicy:
    """The pure-vector summary-routing / collapsed-tree :class:`WalkPolicy`.

    Seeds on the top *seed_k* matches among ``summary_kinds`` surfaces (routers,
    not emitted) and descends to each routed artifact's ``leaf_kinds`` surfaces
    (the emitted results), scored by cosine to the query. No LLM in the loop.

    >>> hits = traverse(q, corpus, policy=collapsed_tree_policy())   # doctest: +SKIP
    """
    return _CollapsedTreePolicy(
        summary_kinds=summary_kinds, leaf_kinds=leaf_kinds, seed_k=seed_k
    )
