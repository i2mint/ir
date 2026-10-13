# ir.embed

Embedder resolution for `ir` — a decent local default, a light fallback.

`ir` favors a *decent local* embedding so retrieval works offline and tests
need no API keys, while keeping a *light* (numpy-only) option for when semantic
power is not what’s under test.

- `"default"` / `"local"` / `"minilm"` → `all-MiniLM-L6-v2` (384-dim)
  via `ef.embedder_adapters.sentence_transformers_embedder()`
  (`normalize=True`), wrapped in `ef.CachedEmbedder` over a `dol`
  cache under `~/.cache/ir/embeddings/<model>`. If `sentence-transformers`
  is unavailable, it degrades to the hashing embedder with a warning.
- `"light"` / `"hashing"` → `ef.HashingEmbedder` (numpy only).
- any other string → treated as a sentence-transformers model name.
- a callable / existing `Embedder` → passed through `ef.as_embedder`.

[`make_embedder()`](#ir.embed.make_embedder) returns `(embedder, embedder_id)`; the id pins the
model in the corpus ledger so a model change triggers a re-embed (the SSOT
discipline that keeps the index from silently drifting).

Importing this module sets `USE_TF=0` so `sentence-transformers` (via
`transformers`) does not import TensorFlow, which crashes on this stack’s
numpy ABI. Import `ir` before anything that imports `transformers`.

### Functions

| [`make_embedder`](#ir.embed.make_embedder)([spec, cache, strict])           | Resolve *spec* to `(embedder, embedder_id)`.                        |
|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`make_embedder_for_id`](#ir.embed.make_embedder_for_id)(embedder_id, \*[, cache]) | Re-create the embedder a corpus was built with, from its stored id. |

### Exceptions

| [`EmbedderError`](#ir.embed.EmbedderError)       | A corpus cannot be queried with the embedder at hand.       |
|----------------------------------------------------------------------|-------------------------------------------------------------|
| [`EmbedderMismatch`](#ir.embed.EmbedderMismatch)    | The query embedding's dimension differs from the index's.   |
| [`EmbedderUnavailable`](#ir.embed.EmbedderUnavailable) | The embedder a corpus was built with cannot be loaded here. |

### *exception* ir.embed.EmbedderError

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

A corpus cannot be queried with the embedder at hand.

Subclasses [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError) (the type earlier versions raised) so existing
`except ValueError` callers keep working. The message is always actionable:
it says what is wrong and the command that fixes it.

### *exception* ir.embed.EmbedderMismatch

Bases: [`EmbedderError`](#ir.embed.EmbedderError)

The query embedding’s dimension differs from the index’s.

### *exception* ir.embed.EmbedderUnavailable

Bases: [`EmbedderError`](#ir.embed.EmbedderError)

The embedder a corpus was built with cannot be loaded here.

### ir.embed.make_embedder(spec='default', , cache=True, strict=False)

Resolve *spec* to `(embedder, embedder_id)`.

`embedder` is a batch callable `Iterable[str] -> ndarray(n, dim)` that
also accepts `input_type=` (`"query"` / `"document"`).

With `strict=True` a local model that cannot be loaded raises
[`EmbedderUnavailable`](#ir.embed.EmbedderUnavailable) instead of degrading to the hashing embedder
(used when reopening a corpus, whose vectors only match its own model).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.embed.make_embedder_for_id(embedder_id, , cache=True)

Re-create the embedder a corpus was built with, from its stored id.

The ids [`make_embedder()`](#ir.embed.make_embedder) emits are invertible: `hashing__dim<N>` and
`st__<model>`. Returns `None` for an id that is not one of those forms (e.g.
`"custom"`, or empty), so the caller falls back to the stored spec. Never
degrades silently: an unavailable model raises [`EmbedderUnavailable`](#ir.embed.EmbedderUnavailable), and
an id of one of those forms that does not parse raises [`EmbedderError`](#ir.embed.EmbedderError)
(falling back to the spec there is the drift this exists to prevent).
This is what pins a corpus to its own embedder, so the *machine* querying it
(with or without `sentence-transformers`) cannot change the vector space.
