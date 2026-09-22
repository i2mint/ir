# ir.base

Core data model for `ir`.

Retrieval in `ir` flows through four small, explicit types:

- [`Artifact`](#ir.base.Artifact) — a logical item in a corpus (a file, a skill, a package).
  Opaque `raw` payload plus `metadata`.
- [`Surface`](#ir.base.Surface) — one *embeddable unit* derived from an artifact. A single
  artifact may yield several heterogeneous surfaces (a short description, an
  AI-authored synopsis, a list of problem classes, body chunks). The
  artifact→surfaces decomposition is the job of an
  [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy).
- [`IndexPlan`](#ir.base.IndexPlan) — a strategy’s output for one artifact: the
  `filter_fields` (hard-filterable metadata, *not* embedded) and the list of
  surfaces (embedded).
- [`Record`](#ir.base.Record) — a stored, embedded surface (one row in the index; maps
  directly to a `vd` `Document`).
- [`SearchHit`](#ir.base.SearchHit) — a scored record returned by retrieval, with a helper to
  collapse multiple surface-hits of the same artifact.

The split between **filter_fields** (metadata you filter on) and **surfaces**
(text you embed) is deliberate and central: good retrieval is hard metadata
filtering *and* semantic ranking, not only embeddings.

### Module Attributes

| [`FilterFields`](#ir.base.FilterFields)   | Non-embedded, hard-filterable metadata for an artifact (name, owner, tags).                                       |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [`POINTER_KEYS`](#ir.base.POINTER_KEYS)   | Metadata keys checked, in order, for a disclosure *pointer* — the file/dir whose contents are an artifact's body. |

### Functions

| [`best_per_artifact`](#ir.base.best_per_artifact)(hits)   | Collapse hits to the highest-scoring surface per artifact.                  |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`ledger_key`](#ir.base.ledger_key)(artifact_id)   | The ledger key under which an artifact's entry is filed.                    |
| [`storage_key`](#ir.base.storage_key)(\*parts)      | Stable, filesystem-safe id from arbitrary string parts (truncated SHA-256). |
| [`tag_source`](#ir.base.tag_source)(hits, source)  | Stamp *source* on every hit that doesn't already carry one.                 |

### Classes

| [`Artifact`](#ir.base.Artifact)(id, raw[, metadata])                     | A logical corpus item before decomposition into surfaces.                                                                    |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`IndexPlan`](#ir.base.IndexPlan)([filter_fields, surfaces])              | An [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy)'s output for one artifact. |
| [`Record`](#ir.base.Record)(id, artifact_id, surface_kind, ...[, ...]) | A stored, embedded surface — one row of the index.                                                                           |
| [`SearchHit`](#ir.base.SearchHit)(artifact_id, surface_kind, score, text) | A scored record returned by retrieval (higher score = closer).                                                               |
| [`Surface`](#ir.base.Surface)(artifact_id, kind, text[, ...])           | One embeddable unit derived from an artifact.                                                                                |

### *class* ir.base.Artifact(id, raw, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A logical corpus item before decomposition into surfaces.

### ir.base.FilterFields

Non-embedded, hard-filterable metadata for an artifact (name, owner, tags).

alias of [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### *class* ir.base.IndexPlan(filter_fields=<factory>, surfaces=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy)’s output for one artifact.

### ir.base.POINTER_KEYS *= ('skill_path', 'path')*

Metadata keys checked, in order, for a disclosure *pointer* — the file/dir
whose contents are an artifact’s body. A [`SearchHit`](#ir.base.SearchHit) is a
*pointer + snippet* (ir_09 §5): `text` is the snippet, the pointer is the
key a resource store dereferences to the full payload. Skills stamp
`skill_path`; packages / reports / files stamp `path` (see
[`ir.sources`](ir.sources.html.md#module-ir.sources)). [`ir.select`](ir.html.md#ir.select) re-exports this for disclosure.

### *class* ir.base.Record(id, artifact_id, surface_kind, surface_index, text, vector, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A stored, embedded surface — one row of the index.

#### *static* make_id(artifact_id, surface_kind, surface_index)

Deterministic storage id for a surface of an artifact.

`surface_index` is the surface’s **plan-global** position — its
enumeration index across *all* surfaces of the artifact’s
[`IndexPlan`](#ir.base.IndexPlan), regardless of kind — as assigned by
[`ir.index.build()`](ir.index.html.md#ir.index.build). On multi-kind strategies it therefore differs
from per-kind counters like `metadata["chunk_index"]` (e.g.
[`Package`](ir.strategy.html.md#ir.strategy.Package): the `description` surface takes
position 0, shifting `readme_chunk` *j* to `surface_index` `j+1`
— and the offset is plan-dependent, since empty surfaces are dropped).

Ids of already-built corpora are a stability contract: never re-derive
a sibling’s id from a per-kind index — address siblings through the
ledger via [`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* ir.base.SearchHit(artifact_id, surface_kind, score, text, metadata=<factory>, source=None, surface_index=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A scored record returned by retrieval (higher score = closer).

Maps onto ir_09’s `Result`: `text` is the snippet, `score` the rank
score, `metadata` the meta, and [`pointer`](#ir.base.SearchHit.pointer) the key into a resource
store (ir_09 §5). [`to_dict()`](#ir.base.SearchHit.to_dict) is the serialization-clean form for a
cross-process / subagent boundary (no numpy scalars leak).

`source` is the corpus/source name the hit came from (`None` when
unattributed — e.g. an ad-hoc corpus without a name). It is a first-class
field, not a metadata key, because `metadata` is the strategy-owned
hard-filter namespace and provenance is structural: artifact identity is
only unique *within* a source, so any cross-source operation keys on
`(source, artifact_id)` (see [`best_per_artifact()`](#ir.base.best_per_artifact)).

`surface_index` is the stored `Record.surface_index` of the hit’s
surface — its plan-global position among the artifact’s surfaces — so a
hit can name *which* surface of its artifact it is (the prerequisite for
sibling addressing and context expansion). `None` when unknown (e.g. a
hand-built hit). It is **not** the per-kind `metadata["chunk_index"]`;
see [`Record.make_id()`](#ir.base.Record.make_id) for why the two differ on multi-kind
strategies.

#### *property* pointer *: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The disclosure pointer on this hit, if any (see [`POINTER_KEYS`](#ir.base.POINTER_KEYS)).

#### to_dict()

JSON-serializable form (`score` cast to a Python `float`).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* ir.base.Surface(artifact_id, kind, text, granularity='document', metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One embeddable unit derived from an artifact.

`kind` names the surface type (e.g. `"description"`, `"synopsis"`,
`"problem_class"`, `"chunk"`) so a query can match the *right part* of
an artifact. `granularity` is a coarse hint (`"document"` / `"chunk"`
/ `"field"`). `metadata` is surface-local (e.g. chunk offsets).

### ir.base.best_per_artifact(hits)

Collapse hits to the highest-scoring surface per artifact.

Returns the surviving hits sorted by score (descending). Identity is
`(source, artifact_id)`: the same id in two different sources names two
different artifacts (the skills-corpus “dol” is not the packages-corpus
“dol”), so cross-source input never collapses them. Single-source input
(all hits sharing one `source`, or all `None`) behaves exactly as an
id-keyed collapse. Note the raw-score comparison and the final sort assume
one score scale — sound within a source; for mixed-source hits use
[`ir.retrieve.fuse_hits()`](ir.retrieve.html.md#ir.retrieve.fuse_hits), which only compares scores within a source.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](#ir.base.SearchHit)]

### ir.base.ledger_key(artifact_id)

The ledger key under which an artifact’s entry is filed.

The single source of truth for ledger keying — [`ir.index.build()`](ir.index.html.md#ir.index.build)
writes entries under this key and readers (e.g.
[`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact)) resolve them with it, so the
two can never drift apart.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.base.storage_key(\*parts)

Stable, filesystem-safe id from arbitrary string parts (truncated SHA-256).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.base.tag_source(hits, source)

Stamp *source* on every hit that doesn’t already carry one.

Existing tags win: a hit already attributed to a corpus keeps that
attribution (so re-tagging under a different registry key cannot
double-count one corpus as two sources). A `None` source is the
untagged pseudo-source — hits pass through unattributed.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](#ir.base.SearchHit)]
