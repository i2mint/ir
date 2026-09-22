# ir.store

Persistence for `ir` — the repository layer over `dol` key-value views.

A [`CorpusStore`](#ir.store.CorpusStore) bundles three `MutableMapping` views, so *where* and
*how* data is persisted is swappable without touching the rest of `ir`:

- `meta`   : `record_id -> dict`  (text + metadata + filter fields), JSON.
- `vectors`: `record_id -> ndarray` (the embedding), numpy bytes.
- `ledger` : `artifact_id -> dict` (version, embedder id, record ids) —
  drives incremental maintenance.
- `config` : `key -> dict` (one entry: the corpus build settings).
- `calibration` : `mode -> dict` (a per-ranking-mode calibrated record, today
  the abstention `min_score` floor from [`ir.eval.calibrate_min_score()`](ir.eval.md#ir.eval.calibrate_min_score)).
  Kept apart from `config` on purpose — a calibration is regenerable, derived
  from an eval run, and not part of the corpus’s build identity, so it must never
  clobber (or be clobbered by) the build settings.
- `links` : `artifact_id -> {edge_type: [target, ...]}` (the semantic link
  graph — typed directed edges between artifacts; see [`ir.graph`](ir.graph.md#module-ir.graph)). Like
  `calibration` it is regenerable derived state, kept out of build identity; a
  target is a bare `artifact_id` (intra-corpus) or a `[source, artifact_id]`
  pair (cross-corpus). Optional — an absent view is simply “no edges”.

The default factory [`CorpusStore.local()`](#ir.store.CorpusStore.local) roots all six under
`~/.local/share/ir/corpora/<name>` via `dol` file stores;
[`CorpusStore.memory()`](#ir.store.CorpusStore.memory) gives a dependency-free in-memory store for tests.
Brute-force search reads vectors into a single normalized matrix
(`matrix()`), cached in-process and invalidated on writes.

### Classes

| [`CorpusStore`](#ir.store.CorpusStore)(meta, vectors, ledger, config[, ...])   | Repository bundling the meta/vectors/ledger/config views of one corpus.   |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|

### *class* ir.store.CorpusStore(meta, vectors, ledger, config, calibration=None, links=None, , packed_dir=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Repository bundling the meta/vectors/ledger/config views of one corpus.

#### calibration_modes()

The ranking modes that currently have a stored calibration.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### delete_ledger_entry(key)

Remove a ledger entry; a missing key is tolerated.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete_links(artifact_id)

Remove an artifact’s edges; a missing entry is tolerated.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### delete_record(record_id)

Remove a record’s metadata + vector; a missing id is tolerated.

The meta goes first, so the id stops being listed before its vector
disappears. Each removal tolerates the file being gone already (another
process may be deleting the same record).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_calibration(mode)

The stored calibration record for ranking `mode` (`None` if absent).

A deep copy, so a caller cannot mutate the nested `grid` back into the
stored record (in-memory stores share their objects by reference).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_config()

The persisted corpus build settings (empty dict if never written).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### get_ledger_entry(key)

The ledger entry for *key* (`None` if absent).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_links(artifact_id)

The outgoing edges of *artifact_id* — `{edge_type: [target, ...]}`.

Empty dict when the artifact has no stored edges (or no links view).
A copy, so a caller cannot mutate the persisted adjacency in place.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### get_maintenance_state()

Background-work bookkeeping (e.g. `last_maintained`); `{}` if unset.

Kept under a separate `config`-view key from the build settings: it is
regenerable scheduler state (when `ir maintain` last ran), not part of
the corpus’s build identity, so it must never clobber it.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### get_record(record_id)

Reassemble the [`Record`](ir.base.md#ir.base.Record) for *record_id* (`KeyError` if absent).

* **Return type:**
  [`Record`](ir.base.md#ir.base.Record)

#### ledger_items()

Iterate `(key, entry)` ledger pairs (the ledger may be mutated while iterating).

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### link_items()

Iterate `(artifact_id, {edge_type: [target]})` adjacency pairs.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### *classmethod* local(name)

File-backed store under `~/.local/share/ir/corpora/<name>`.

* **Return type:**
  [`CorpusStore`](#ir.store.CorpusStore)

#### matrix()

Return `(record_ids, normalized_matrix, metas)` for brute force.

Rows are L2-normalized so cosine similarity is a dot product. Empty
corpora return a `(0, 0)` matrix.

Caching is two-tier: an in-process cache (invalidated on the next write)
backed, for file-rooted stores, by an on-disk **packed** cache — one
normalized-matrix `.npy` plus its ids/metas, written once and reloaded
with a single memory-mapped read. The packed cache turns a cold reopen
from a per-record vector-file storm (thousands of tiny reads) into three
file reads; it is cleared by a writer’s first record write, and every
record write replaces a *write stamp* that a packed set must match to be
published or loaded, so a set built before any later write – by this
process or another – is never served (i2mint/ir#86).

Another process may be writing or deleting records while this one
rebuilds. A record that vanishes or is only half-written when read is
left out of the result (it is “not yet written”), with a warning; such
a partial result is cached in-process but never published as the
packed set, so a fresh process reads the records again.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `ndarray`, [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### *classmethod* memory()

In-memory store (no dependencies); ideal for tests.

* **Return type:**
  [`CorpusStore`](#ir.store.CorpusStore)

#### metas()

Return `(record_ids, metas)` **without** loading any vectors.

The vector-free counterpart of [`matrix()`](#ir.store.CorpusStore.matrix), for ranking modes that
score on text alone (`mode="lexical"`): they need candidate metadata
(text + filter fields) but never the embedding matrix, so they must not
pay its I/O. Reuses the in-process or packed cache when present; else
reads only the `meta` view (not `vectors`), skipping a record that
vanishes or is half-written mid-read (as [`matrix()`](#ir.store.CorpusStore.matrix) does).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

#### put_record(record)

Persist *record*’s metadata + vector, invalidating the search matrix.

The vector is written **before** the meta: record ids are listed from
the meta view, so a reader in another process that lists an id always
finds its vector (`delete_record` removes in the reverse order).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### record_ids()

Iterate the record ids currently stored.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### set_calibration(mode, record)

Persist a calibration `record` for ranking `mode` (one per mode).

`mode` keys a file in the calibration store, so it must be a non-empty
string with no path separator (the real modes — `dense` / `lexical` /
`hybrid` — already satisfy this).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_config(settings)

Persist the corpus build *settings* (name / embedder spec + id).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_ledger_entry(key, entry)

Write the ledger *entry* (version / embedder id / record ids) for *key*.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_links(artifact_id, edges)

Persist *artifact_id*’s outgoing *edges* (`{edge_type: [target]}`).

Empty edge-type lists are dropped; an empty result deletes the entry
(no empty adjacency rows linger). Targets are stored verbatim — a bare
`artifact_id` or a `[source, artifact_id]` pair.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### set_maintenance_state(state)

Persist the maintenance bookkeeping for this corpus.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)
