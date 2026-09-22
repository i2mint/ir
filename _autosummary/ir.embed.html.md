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

| [`make_embedder`](#ir.embed.make_embedder)([spec, cache])   | Resolve *spec* to `(embedder, embedder_id)`.   |
|---------------------------------------------------------------------------------|------------------------------------------------|

### ir.embed.make_embedder(spec='default', , cache=True)

Resolve *spec* to `(embedder, embedder_id)`.

`embedder` is a batch callable `Iterable[str] -> ndarray(n, dim)` that
also accepts `input_type=` (`"query"` / `"document"`).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
