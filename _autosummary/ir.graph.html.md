# ir.graph

The semantic link graph — typed edges between artifacts (report 12).

ir indexes artifacts that *refer to each other*: a package depends on other
packages, a skill belongs to a parent, a chunk follows the previous one, a
report cites another. This module models those references as \*\*typed, directed
edges\*\* between nodes identified by `(source, artifact_id)` (ir’s canonical
node identity since 0.1.16), queryable at retrieval time — the substrate the
traversal operator (`traverse`, follow-up) walks.

Two pieces:

- the [`GraphStore`](#ir.graph.GraphStore) **protocol** — a minimal `__getitem__` +
  `neighbors` structural contract, so a traversal operator binds to *any*
  conforming store, not only ir’s;
- [`CorpusGraph`](#ir.graph.CorpusGraph) — the corpus-backed implementation. Node payloads are
  the artifact’s stored records (via [`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact));
  neighbors come from the `links` view on [`CorpusStore`](ir.store.html.md#ir.store.CorpusStore).

Edges are **regenerable derived state** (persisted in the `links` view, like
`calibration`) — never part of build identity. They are ingested at build
time by an injectable [`EdgeExtractor`](#ir.graph.EdgeExtractor); the shipped
[`default_edge_extractor()`](#ir.graph.default_edge_extractor) turns the edge data already latent in the index
(`Package` `deps` → `REF`, `Skill` `parent` → `PARENT`) into edges.

Vocabulary (documented, **not** enforced): [`NEXT`](#ir.graph.NEXT) / [`PREV`](#ir.graph.PREV) /
[`PARENT`](#ir.graph.PARENT) / [`CHILD`](#ir.graph.CHILD) / [`REF`](#ir.graph.REF). `NEXT` / `PREV` are *not*
materialized — they are derivable from the ledger’s surface order (#44); the
view stores only what isn’t derivable.

**Naming guard** (ADR #43): this is the **semantic link graph** — directed,
**possibly cyclic** (citations, cross-references), traversed at query time. It
is *not* `ef.artifact_graph`, which is an acyclic build-time
derivation/lineage DAG. The id conventions are kept compatible so a later
unification stays open, but the two are distinct substrates.

### Module Attributes

| [`NEXT`](#ir.graph.NEXT)          | Next sibling surface in sequence (derivable from the ledger — not stored).     |
|----------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`PREV`](#ir.graph.PREV)          | Previous sibling surface in sequence (derivable from the ledger — not stored). |
| [`PARENT`](#ir.graph.PARENT)        | a child points at its parent (e.g. a skill → its package).                     |
| [`CHILD`](#ir.graph.CHILD)         | a parent points at a child.                                                    |
| [`REF`](#ir.graph.REF)           | A generic reference / dependency / citation (e.g. a package → its deps).       |
| [`NodeId`](#ir.graph.NodeId)        | `artifact_id` within one corpus, or `(source, artifact_id)` across corpora.    |
| [`EdgeExtractor`](#ir.graph.EdgeExtractor) | `(artifact_id, filter_fields) -> {edge_type: [target]}`.                       |

### Functions

| [`canonical_node_id`](#ir.graph.canonical_node_id)(target, \*, source)    | Canonicalize a neighbor *target* to a `(source, artifact_id)` node id.       |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`default_edge_extractor`](#ir.graph.default_edge_extractor)(artifact_id, ...) | Edges latent in the standard filter fields: `deps` → REF, `parent` → PARENT. |
| [`normalize_dist_name`](#ir.graph.normalize_dist_name)(name)                | A dependency spec's bare name, PEP 503-normalized (`-`/`_`/`.` -> `-`).      |

### Classes

| [`CorpusGraph`](#ir.graph.CorpusGraph)(store_or_corpus)   | A [`GraphStore`](#ir.graph.GraphStore) over one corpus — artifact nodes, `links` edges.   |
|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`GraphStore`](#ir.graph.GraphStore)(\*args, \*\*kwargs) | Structural contract a traversal operator binds to — node + neighbors.                                            |

### ir.graph.CHILD *= 'CHILD'*

a parent points at a child.

* **Type:**
  Containment down

### *class* ir.graph.CorpusGraph(store_or_corpus)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A [`GraphStore`](#ir.graph.GraphStore) over one corpus — artifact nodes, `links` edges.

`node_id` is an `artifact_id`. `graph[aid]` is the artifact’s stored
records (its scorable surfaces, in plan order); `graph.neighbors(aid,
edge_type=...)` reads the corpus store’s `links` view. Single-corpus, so
it resolves **intra-corpus** targets; cross-corpus `[source, artifact_id]`
targets are returned by [`neighbors()`](#ir.graph.CorpusGraph.neighbors) verbatim but `__getitem__` only
dereferences ids in *this* corpus (federated traversal is a follow-up).

#### edge_types(node_id)

The edge types present on *node_id* (`[]` if none).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### fan_in(node_id, , edge_type='REF')

How many artifacts point at *node_id* by *edge_type* (its in-degree).

With `REF` dependency edges this is the number of dependents, a
PageRank-style centrality prior: a widely depended-on library is central
to the corpus.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### max_fan_in(, edge_type='REF')

The largest [`fan_in()`](#ir.graph.CorpusGraph.fan_in) over all targets (`0` without edges).

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### neighbors(node_id, , edge_type=None)

Outgoing neighbor ids of *node_id*, optionally of one *edge_type*.

Returns target ids in stored form (a bare `artifact_id`, whose source
is *this* graph’s [`source`](#ir.graph.CorpusGraph.source); or a `[source, artifact_id]` list
for a cross-corpus edge), de-duplicated with first-seen order
preserved. An artifact with no edges — or a store without a links view —
yields `[]`. Pass a canonical `(source, artifact_id)` to
[`canonical_node_id()`](#ir.graph.canonical_node_id) for a traversal’s visited-set.

*node_id* is an intra-corpus `artifact_id` (a `str`); a cross-corpus
target fetched from another graph is out of contract here (it has no
edges *in this corpus*).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

#### reverse_neighbors(node_id, , edge_type='REF')

The artifact ids whose *edge_type* edges point **at** *node_id*.

The reverse of [`neighbors()`](#ir.graph.CorpusGraph.neighbors): with the default [`REF`](#ir.graph.REF) edges of
[`default_edge_extractor()`](#ir.graph.default_edge_extractor) (`package -> its deps`), this lists a
library’s dependents. *node_id* may name an artifact of this corpus or a
target outside it (a third-party dependency). For `REF` edges, string
ids match through [`normalize_dist_name()`](#ir.graph.normalize_dist_name) (case, and `-`/`_`/`.`,
are ignored), so `"scikit_learn"` finds the dependents of
`"scikit-learn"`; other edge types match ids exactly. A cross-corpus
target is a `(source, artifact_id)` pair, matched exactly. Order is first-seen over the links view, de-duplicated.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### source

The corpus name, when known — the `source` half of this graph’s
node identities (`None` for a bare store).

### ir.graph.EdgeExtractor

`(artifact_id, filter_fields) -> {edge_type: [target]}`.
An injectable build-time seam (like [`Formulator`](ir.formulate.html.md#ir.formulate.Formulator)) that
turns an artifact’s filter fields into outgoing edges. `build(...,
edge_extractor=...)` runs it; [`default_edge_extractor()`](#ir.graph.default_edge_extractor) is shipped.

* **Type:**
  An edge extractor

alias of `Callable`[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], `Mapping[str, list]`]

### *class* ir.graph.GraphStore(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Structural contract a traversal operator binds to — node + neighbors.

Deliberately minimal (two methods) so it is satisfied by ir’s
[`CorpusGraph`](#ir.graph.CorpusGraph) *and* by any external graph: `__getitem__` resolves
a node id to its scorable payload, `neighbors` lists adjacent node ids
(optionally of one edge type). Granularity-agnostic on purpose — an
artifact graph and a surface-level tree are both 

```
``
```

GraphStore\`\`s.

`runtime_checkable` makes `isinstance(x, GraphStore)` a *structural*
check on attribute **names** only (not signatures) — enough to tell a
conforming adapter from an arbitrary object, but it cannot validate that
`neighbors` takes the right arguments; treat it as a smoke check.

### ir.graph.NEXT *= 'NEXT'*

Next sibling surface in sequence (derivable from the ledger — not stored).

### ir.graph.NodeId *= 'str | tuple[str, str]'*

`artifact_id` within one corpus, or `(source,
artifact_id)` across corpora. The visited-set of a traversal keys on this.

* **Type:**
  A node’s identity

### ir.graph.PARENT *= 'PARENT'*

a child points at its parent (e.g. a skill → its package).

* **Type:**
  Containment up

### ir.graph.PREV *= 'PREV'*

Previous sibling surface in sequence (derivable from the ledger — not stored).

### ir.graph.REF *= 'REF'*

A generic reference / dependency / citation (e.g. a package → its deps).

### ir.graph.canonical_node_id(target, , source)

Canonicalize a neighbor *target* to a `(source, artifact_id)` node id.

The repo’s node identity is `(source, artifact_id)` — the key a traversal
visited-set must use so the same id in two corpora stays two nodes.
[`CorpusGraph.neighbors()`](#ir.graph.CorpusGraph.neighbors) returns targets in stored form: a bare
`artifact_id` (implicitly in *source*, the graph it came from) or a
`[source, artifact_id]` cross-corpus pair. This resolves either to the
canonical tuple.

```pycon
>>> canonical_node_id("dol", source="packages")
('packages', 'dol')
>>> canonical_node_id(["skills", "deploy"], source="packages")
('skills', 'deploy')
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.graph.default_edge_extractor(artifact_id, filter_fields)

Edges latent in the standard filter fields: `deps` → REF, `parent` → PARENT.

- `deps` (Package) → [`REF`](#ir.graph.REF) edges to each dependency’s bare name
  (version specifiers / extras / markers stripped). Self-edges and blanks
  are dropped.
- `parent` (Skill) → a single [`PARENT`](#ir.graph.PARENT) edge.

A package whose `deps` name other packages in the same corpus gets
intra-corpus REF edges; third-party deps become REF edges to ids not in the
corpus (harmless — [`CorpusGraph.neighbors()`](#ir.graph.CorpusGraph.neighbors) lists them, and a
traversal simply finds no node to expand).

Self-edges are dropped case-insensitively (`_dep_name` lower-cases, so a
package `"AA"` depending on `"aa"` is recognized as a self-reference).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]

### ir.graph.normalize_dist_name(name)

A dependency spec’s bare name, PEP 503-normalized (`-`/`_`/`.` -> `-`).

The one comparison key for library names: the reverse dependency index and
[`ir.dependency_evidence_boost()`](ir.html.md#ir.dependency_evidence_boost) both match through it, so
`"scikit_learn"` finds a package that depends on `"Scikit-Learn>=1"`.

```pycon
>>> normalize_dist_name("Scikit_Learn[extra]>=1.2")
'scikit-learn'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
