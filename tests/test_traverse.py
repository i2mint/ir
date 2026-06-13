"""Tests for the traversal operator (#47) — safety, routing, composition.

Pins the #47 acceptance: operator-enforced termination on cyclic / never-stop
policies (the safety primitives are the operator's, not the policy's), the
collapsed-tree policy routing a summary match down to chunks and beating flat
top-k on a constructed routing case, PPR expressible as a degenerate policy
(shape only), and downstream select/disclose composition. Hermetic: light
embedder + memory store.
"""

import json

import ir
from ir.base import SearchHit
from ir.store import CorpusStore
from ir.traverse import WalkPolicy, WalkState, traverse

# --------------------------------------------------------------------------- #
# A constructed routing case: A is gold (summary matches, answer in its chunk);
# D is the flat trap (summary does NOT match, but a chunk matches routing terms
# strongly). Routing excludes D before its chunk can compete; flat does not.
# --------------------------------------------------------------------------- #

ROUTING_PKG = {
    "A": {
        "name": "A",
        "description": "rtok1 rtok2 rtok3 rtok4 alpha",
        "readme": "ANSTOK answer payload here.\n\nfiller one two three.",
    },
    "E": {
        "name": "E",
        "description": "rtok1 rtok2 beta gamma",
        "readme": "neutral content mu nu.\n\nmore neutral xi.",
    },
    "D": {
        "name": "D",
        "description": "delta epsilon zeta unrelated",
        "readme": "rtok1 rtok2 rtok3 rtok4 trap distractor strong.",
    },
    "F": {"name": "F", "description": "omega sigma", "readme": "generic tau upsilon."},
}
ROUTING_QUERY = "rtok1 rtok2 rtok3 rtok4 ANSTOK"


def _routing_corpus():
    src = ir.CorpusSource.from_mapping(
        ROUTING_PKG, name="rt", strategy=ir.Package(chunk_size=120, overlap=10)
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def _first_rank(hits, needle="ANSTOK"):
    return next((i for i, h in enumerate(hits) if needle in h.text), -1)


# --------------------------------------------------------------------------- #
# Collapsed-tree routing — beats flat on the constructed case
# --------------------------------------------------------------------------- #


def test_collapsed_tree_beats_flat_on_routing_case():
    corpus = _routing_corpus()
    flat = ir.search(corpus, ROUTING_QUERY, k=10, per_artifact=False)
    trav = ir.traverse(
        ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(seed_k=2), k=10
    )
    flat_rank = _first_rank(flat)
    trav_rank = _first_rank(trav)
    # Flat buries the answer chunk under A's summary + D's trap chunk; routing
    # surfaces it at the top.
    assert flat_rank > 0, f"expected the answer buried in flat, got rank {flat_rank}"
    assert trav_rank == 0, f"expected the answer first under routing, got {trav_rank}"
    assert trav[0].artifact_id == "A"


def test_collapsed_tree_excludes_unrouted_distractor():
    # D's summary does not match, so D is never seeded — its trap chunk (which
    # flat ranks highly) cannot appear in the traversal results at all.
    corpus = _routing_corpus()
    trav = ir.traverse(
        ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(seed_k=2), k=10
    )
    assert all(h.artifact_id != "D" for h in trav)


def test_collapsed_tree_emits_only_leaves_not_routers():
    corpus = _routing_corpus()
    trav = ir.traverse(ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(), k=10)
    assert trav  # something came back
    assert all(h.surface_kind == "readme_chunk" for h in trav)  # no description


def test_traverse_hits_carry_walk_provenance():
    corpus = _routing_corpus()
    trav = ir.traverse(
        ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(seed_k=2), k=10
    )
    top = trav[0]
    assert top.metadata["walk_depth"] == 1
    assert top.metadata["seed"] == "A"
    assert top.source == "rt"
    json.dumps(top.to_dict())  # additive provenance stays JSON-clean


def test_traverse_composes_with_select_and_disclose():
    corpus = _routing_corpus()
    trav = ir.traverse(
        ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(seed_k=2), k=10
    )
    selection = ir.select(trav)
    results = ir.disclose(selection, level="metadata")
    assert results and all(d.summary for d in results)
    assert selection.selected[0].artifact_id == "A"


def test_traverse_respects_k():
    corpus = _routing_corpus()
    assert (
        len(ir.traverse(ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(), k=1))
        <= 1
    )


def test_collapsed_tree_policy_is_a_walkpolicy():
    assert isinstance(ir.collapsed_tree_policy(), WalkPolicy)


# --------------------------------------------------------------------------- #
# Single-surface corpora (WholeText / Skill): every surface is a summary kind,
# none is a leaf kind. The footgun would route everything and emit nothing;
# the structural router check must instead emit the leaf-less summaries.
# --------------------------------------------------------------------------- #


def test_wholetext_corpus_does_not_silently_return_empty():
    # WholeText emits one "document" surface per artifact and no chunks. With
    # "document" in DFLT_SUMMARY_KINDS, the naive policy would mark every
    # surface a router (-> to_hit None) and find no leaves to descend to,
    # returning []. The leaf-less-emit fix must surface the documents instead.
    docs = {
        "d1": "rtok1 rtok2 rtok3 rtok4 alpha ANSTOK answer payload here filler",
        "d2": "unrelated beta gamma delta neutral content mu nu xi omicron",
        "d3": "rtok1 omega sigma tau upsilon generic content here",
    }
    src = ir.CorpusSource.from_mapping(docs, name="wt", strategy=ir.WholeText())
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    trav = ir.traverse(
        "rtok1 rtok2 rtok3 rtok4 ANSTOK", corpus, policy=ir.collapsed_tree_policy(), k=5
    )
    assert trav, "WholeText corpus must not silently return zero results"
    assert all(h.surface_kind == "document" for h in trav)
    assert trav[0].artifact_id == "d1"
    # A leaf-less summary is emitted at its seed position (depth 0), with no
    # routing parent — it IS the seed, not a descent target.
    assert trav[0].metadata["walk_depth"] == 0
    assert "seed" not in trav[0].metadata


def test_skill_corpus_does_not_silently_return_empty():
    # Skill emits one "capability" surface per artifact and no chunks — same
    # all-summary-kinds shape as WholeText, via a different summary kind.
    skills = {
        "s1": {"name": "deploy-app", "description": "rtok1 rtok2 rtok3 ANSTOK ship it"},
        "s2": {
            "name": "other-thing",
            "description": "unrelated beta gamma delta epsilon",
        },
    }
    src = ir.CorpusSource.from_mapping(skills, name="sk", strategy=ir.Skill())
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    trav = ir.traverse(
        "rtok1 rtok2 rtok3 ANSTOK deploy-app",
        corpus,
        policy=ir.collapsed_tree_policy(),
        k=5,
    )
    assert trav, "Skill corpus must not silently return zero results"
    assert all(h.surface_kind == "capability" for h in trav)
    assert trav[0].artifact_id == "s1"
    assert trav[0].metadata["walk_depth"] == 0


def test_package_corpus_still_routes_summaries_not_leaf_less_emit():
    # The fix must NOT regress the genuine-tree case: a Package artifact HAS
    # leaf surfaces, so its "description" summary stays a router (suppressed),
    # and only readme_chunk leaves are emitted.
    corpus = _routing_corpus()
    trav = ir.traverse(ROUTING_QUERY, corpus, policy=ir.collapsed_tree_policy(), k=10)
    assert trav
    assert all(h.surface_kind == "readme_chunk" for h in trav)  # no "description"


# --------------------------------------------------------------------------- #
# Operator-enforced safety — termination regardless of policy behavior
# --------------------------------------------------------------------------- #


def _hit(node_id, score, depth):
    return SearchHit(
        artifact_id=str(node_id),
        surface_kind="node",
        score=float(score),
        text=str(node_id),
        metadata={"walk_depth": depth},
    )


class _CyclicPolicy:
    """A graph with a directed cycle and a policy that never stops."""

    def __init__(self, adjacency):
        self.adj = adjacency

    def seed(self, state, store):
        return [0]

    def score(self, state, node, store):
        return 0.0

    def select(self, state, scored):
        return scored

    def expand(self, state, node, store):
        return self.adj.get(node, [])

    def node_id(self, node):
        return node

    def stop(self, state):
        return False

    def to_hit(self, state, node, score, depth):
        return _hit(node, score, depth)


class _ExplodingPolicy:
    """A never-stopping policy generating unbounded *fresh* node ids (binary)."""

    def seed(self, state, store):
        return [0]

    def score(self, state, node, store):
        return 0.0

    def select(self, state, scored):
        return scored

    def expand(self, state, node, store):
        return [2 * node + 1, 2 * node + 2]  # always-new ids; cannot be visited-deduped

    def node_id(self, node):
        return node

    def stop(self, state):
        return False

    def to_hit(self, state, node, score, depth):
        return _hit(node, score, depth)


def test_cyclic_graph_terminates_via_visited_set():
    # 0 -> 1 -> 2 -> 0 : the visited-set alone must break the cycle even with
    # a generous budget/depth and a policy that never stops.
    policy = _CyclicPolicy({0: [1], 1: [2], 2: [0]})
    hits = traverse("q", None, policy=policy, max_depth=100, node_budget=100, k=100)
    assert {h.artifact_id for h in hits} == {"0", "1", "2"}  # each node once


def test_unbounded_fresh_nodes_terminate_via_node_budget():
    # Every node id is new (visited-set can't help), so the operator's node
    # budget is the backstop that guarantees termination.
    hits = traverse(
        "q", None, policy=_ExplodingPolicy(), max_depth=100, node_budget=20, k=1000
    )
    assert len(hits) <= 20


def test_depth_cap_bounds_expansion():
    # A linear fresh-node chain: expansion must stop at max_depth.
    class _Linear(_ExplodingPolicy):
        def expand(self, state, node, store):
            return [node + 1]

    hits = traverse("q", None, policy=_Linear(), max_depth=3, node_budget=1000, k=1000)
    assert max(h.metadata["walk_depth"] for h in hits) <= 3


def test_injected_stop_halts_early():
    class _StopAfterTwo(_ExplodingPolicy):
        def stop(self, state):
            return len(state.results) >= 2

    hits = traverse(
        "q", None, policy=_StopAfterTwo(), max_depth=100, node_budget=100, k=100
    )
    assert len(hits) == 2


# --------------------------------------------------------------------------- #
# PPR expressible as a degenerate policy (shape only — not implemented)
# --------------------------------------------------------------------------- #


def test_ppr_is_expressible_as_a_degenerate_policy():
    # PPR / spreading-activation as a one-shot closed-form: score is the
    # personalization weight, stop() is immediately true (single relaxation),
    # expand walks the artifact links. We only assert the protocol *admits*
    # this shape over a CorpusGraph — we do not ship or promote it.
    pkg = {
        "aa": {"name": "aa", "description": "a", "readme": "", "deps": ["bb", "cc"]},
        "bb": {"name": "bb", "description": "b", "readme": "", "deps": ["cc"]},
        "cc": {"name": "cc", "description": "c", "readme": "", "deps": []},
    }
    src = ir.CorpusSource.from_mapping(pkg, name="ppr", strategy=ir.Package())
    corpus = ir.build(
        src,
        store=CorpusStore.memory(),
        embedder="light",
        edge_extractor=ir.default_edge_extractor,
    )
    graph = ir.CorpusGraph(corpus)

    class _PPRShape:
        damping = 0.85

        def seed(self, state, store):
            return ["aa"]  # the personalization vector's support

        def score(self, state, node, store):
            return self.damping  # closed-form weight (degenerate)

        def select(self, state, scored):
            return scored

        def expand(self, state, node, store):
            return [
                ir.canonical_node_id(t, source=store.source)[1]
                for t in store.neighbors(node)
            ]

        def node_id(self, node):
            return node

        def stop(self, state):
            return True  # one relaxation step, then halt

        def to_hit(self, state, node, score, depth):
            return _hit(node, score, depth)

    assert isinstance(_PPRShape(), WalkPolicy)
    hits = traverse("q", graph, policy=_PPRShape(), max_depth=2, node_budget=10)
    # stop() fires after the first committed node — the degenerate one-shot.
    assert [h.artifact_id for h in hits] == ["aa"]


# --------------------------------------------------------------------------- #
# WalkState bounds are the operator's
# --------------------------------------------------------------------------- #


def test_walkstate_carries_the_safety_bounds():
    s = WalkState(query="q", max_depth=3, budget=10)
    assert s.max_depth == 3 and s.budget == 10
    assert s.visited == set() and s.results == []
