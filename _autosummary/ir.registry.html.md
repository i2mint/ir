# ir.registry

Named-corpus registry — persistent, reusable corpus definitions.

A registry entry records *how* to (re)build a corpus — its `kind` (a source
preset), parameters, and embedder spec — so a corpus becomes a stable name you
can build once and query across sessions. The registry is a single JSON file
under the config dir (`~/.config/ir/corpora.json`).

Presets map to [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource) constructors:

- `skills`   → `CorpusSource.from_skills()`
- `packages` → `CorpusSource.from_packages()`
- `reports`  → `CorpusSource.from_md_reports()`
- `sessions` → `CorpusSource.from_claude_sessions()`
- `files`    → `CorpusSource.from_files()` (needs `root`; optional
  `pattern`)
- `records`  → `CorpusSource.from_records()` (needs `fetcher`, a
  `"module:attr"` reference; optional `metadata_keys` / `id_key` /
  `text_key`) — the seam for a corpus whose records another package owns

Unregistered preset names (`skills`/`packages`/`reports`/`sessions`) are
auto-registered with defaults on first use, so `ir build skills` just works.

### Module Attributes

| [`PARAMETRIC_KINDS`](#ir.registry.PARAMETRIC_KINDS)   | Kinds that need parameters, so they are never auto-registered from a bare name.   |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------|

### Functions

| [`get`](#ir.registry.get)(name)                                 | The registry entry for *name*, or `None`.                                                                                                 |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| [`policy_for`](#ir.registry.policy_for)(name)                          | The effective [`ir.policy.MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy) for corpus *name*. |
| [`register`](#ir.registry.register)(name, kind, \*[, embedder, ...]) | Register (or overwrite) a named corpus definition.                                                                                        |
| [`registered`](#ir.registry.registered)()                              | All registered corpus definitions, keyed by name.                                                                                         |
| [`retriever_for`](#ir.registry.retriever_for)(name, \*\*search_defaults)  | A [`Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) bound to the registered corpus *name*.                 |
| [`retrievers`](#ir.registry.retrievers)(\*\*search_defaults)           | A lazy `Mapping[name, Retriever]` view over the registry (ir_09 §8).                                                                      |
| [`source_for`](#ir.registry.source_for)(name)                          | Resolve *name* to a source, auto-registering a preset if needed.                                                                          |
| [`source_from_entry`](#ir.registry.source_from_entry)(name, entry)            | Reconstruct a `CorpusSource` from a registry entry.                                                                                       |
| [`unregister`](#ir.registry.unregister)(name)                          | Remove *name* from the registry (does not delete built data).                                                                             |

### ir.registry.PARAMETRIC_KINDS *= ('files', 'records')*

Kinds that need parameters, so they are never auto-registered from a bare name.

### ir.registry.get(name)

The registry entry for *name*, or `None`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.registry.policy_for(name)

The effective [`ir.policy.MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy) for corpus *name*.

Resolves the registered entry’s `maintenance` over its kind’s smart default
over the global default (see [`ir.policy.resolve_policy()`](ir.policy.html.md#ir.policy.resolve_policy)). An unregistered
name resolves to the global default policy.

### ir.registry.register(name, kind, , embedder='default', strategy=None, maintenance=None, storage=None, \*\*params)

Register (or overwrite) a named corpus definition.

Beyond the v1 `kind` / `embedder` / `params`, an entry may now carry
(all optional, with smart per-kind defaults applied at resolution time — see
[`ir.policy`](ir.policy.html.md#module-ir.policy)):

- `strategy` — an [`IndexingStrategy`](ir.strategy.html.md#ir.strategy.IndexingStrategy) (or a
  `{"name", "params"}` spec) persisted so the corpus’s *segmentation* is
  stable across rebuilds. `None` keeps the preset’s default strategy.
- `maintenance` — the background-work policy dict (`reindex` / `synopsis`;
  validated here, see [`ir.policy.MaintenancePolicy`](ir.policy.html.md#ir.policy.MaintenancePolicy)).
- `storage` — the persistence backend (default `{"backend": "local"}`).

Entries written by older `ir` (none of these keys) keep working unchanged.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.registry.registered()

All registered corpus definitions, keyed by name.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.registry.retriever_for(name, \*\*search_defaults)

A [`Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever) bound to the registered corpus *name*.

Opens the corpus (it must have been built) and wraps it with
[`ir.as_retriever()`](ir.html.md#ir.as_retriever); `search_defaults` (e.g. `mode="hybrid"`) bind to
every call.

### ir.registry.retrievers(\*\*search_defaults)

A lazy `Mapping[name, Retriever]` view over the registry (ir_09 §8).

The query-time projection of the build-recipe registry: each value is a
ready-to-call [`Retriever`](ir.retrieve.html.md#ir.retrieve.Retriever). This is the source-registry
facade an orchestration layer (`raglab`) consumes — it never opens a corpus
until the key is accessed, and always reflects the current [`registered()`](#ir.registry.registered)
set. `search_defaults` apply to every source.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ir.registry.source_for(name)

Resolve *name* to a source, auto-registering a preset if needed.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

### ir.registry.source_from_entry(name, entry)

Reconstruct a `CorpusSource` from a registry entry.

A persisted `strategy` spec (registry v2) is reconstructed and passed to
the preset constructor, so a corpus’s segmentation survives across rebuilds.
A v1 entry (no `strategy`) passes `strategy=None` and keeps the preset’s
default — unchanged behavior.

* **Return type:**
  [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource)

### ir.registry.unregister(name)

Remove *name* from the registry (does not delete built data).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)
