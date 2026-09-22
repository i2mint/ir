# ir.config

Filesystem locations and process-wide defaults for `ir`.

`ir` separates three kinds of on-disk state, each under an XDG-standard base
(overridable per-install via `IR_CONFIG_DIR` / `IR_DATA_DIR` /
`IR_CACHE_DIR`, then `XDG_CONFIG_HOME` / `XDG_DATA_HOME` /
`XDG_CACHE_HOME`, then `~/.config` / `~/.local/share` / `~/.cache`):

- **config** (`~/.config/ir`) — the named-corpus registry and user settings.
- **data** (`~/.local/share/ir`) — durable corpus stores (record metadata,
  vectors, ledgers). The source of truth; losing it means rebuilding the index.
- **cache** (`~/.cache/ir`) — regenerable derived data, chiefly the embedding
  cache keyed by `(model, content_hash)`.

Every path is a plain [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) created on demand, so callers can
treat the directories as guaranteed to exist.

### Functions

| [`cache_dir`](#ir.config.cache_dir)()                    | Regenerable cache directory (default `~/.cache/ir`).               |
|---------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`config_dir`](#ir.config.config_dir)()                   | User configuration directory (default `~/.config/ir`).             |
| [`corpus_dir`](#ir.config.corpus_dir)(name)               | Durable directory holding one corpus's stores.                     |
| [`data_dir`](#ir.config.data_dir)()                     | Durable data directory (default `~/.local/share/ir`).              |
| [`embeddings_cache_dir`](#ir.config.embeddings_cache_dir)(model_id) | Cache directory for one embedding model's vectors.                 |
| [`registry_path`](#ir.config.registry_path)()                | JSON file mapping registered corpus names to their build settings. |
| [`safe_name`](#ir.config.safe_name)(name)                | Filesystem-safe slug for a corpus/model identifier.                |

### ir.config.cache_dir()

Regenerable cache directory (default `~/.cache/ir`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.config.config_dir()

User configuration directory (default `~/.config/ir`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.config.corpus_dir(name)

Durable directory holding one corpus’s stores.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.config.data_dir()

Durable data directory (default `~/.local/share/ir`).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.config.embeddings_cache_dir(model_id)

Cache directory for one embedding model’s vectors.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.config.registry_path()

JSON file mapping registered corpus names to their build settings.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.config.safe_name(name)

Filesystem-safe slug for a corpus/model identifier.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
