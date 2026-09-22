# ir.strategy

Indexing strategies — the “what do we index?” seam.

An [`IndexingStrategy`](#ir.strategy.IndexingStrategy) decomposes one artifact into an
[`IndexPlan`](ir.base.html.md#ir.base.IndexPlan): the `filter_fields` (hard-filterable metadata,
*not* embedded) and a list of [`Surface`](ir.base.html.md#ir.base.Surface) (embeddable units). This
is the central extensibility point of `ir`: a naive corpus uses
[`WholeText`](#ir.strategy.WholeText); a structured corpus (a package) decomposes into several
heterogeneous surfaces so a query can match the *right part* of an artifact,
and constrains candidates by metadata *before* semantic ranking.

Shipped strategies:

- [`WholeText`](#ir.strategy.WholeText) — one surface = the whole text. The out-of-the-box default.
- [`Chunked`](#ir.strategy.Chunked) — split the text into overlapping chunks (one surface each).
- [`Skill`](#ir.strategy.Skill) — embed `name + description` only (the body stays on disk,
  per the capability-discovery research); name/parent become filter fields.
- [`Package`](#ir.strategy.Package) — `name + description` plus README chunks as surfaces;
  name/owner/deps become filter fields (AI synopsis / problem-class surfaces
  are a documented extension).

Every strategy is a plain callable-ish object with a `decompose` method, so
custom strategies need only match the [`IndexingStrategy`](#ir.strategy.IndexingStrategy) protocol.

### Module Attributes

| [`DFLT_CHUNK_MAX_TOKENS`](#ir.strategy.DFLT_CHUNK_MAX_TOKENS)   | Content-token budget for token-aware chunking under the default embedder.                                                                                                    |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`STRATEGY_REGISTRY`](#ir.strategy.STRATEGY_REGISTRY)       | Shipped strategies addressable by name, for persisting an [`IndexingStrategy`](#ir.strategy.IndexingStrategy) in a registry entry and reconstructing it (#58). |

### Functions

| [`strategy_from_spec`](#ir.strategy.strategy_from_spec)(spec)   | Reconstruct a shipped strategy from a `{"name", "params"}` spec.   |
|-----------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`strategy_to_spec`](#ir.strategy.strategy_to_spec)(strategy) | A `{"name", "params"}` spec for a shipped, scalar-param strategy.  |
| [`text_of`](#ir.strategy.text_of)(raw[, text_key])   | Best-effort text extraction from a raw artifact payload.           |

### Classes

| [`Chunked`](#ir.strategy.Chunked)(\*[, chunk_size, overlap, text_key, ...])   | Split the artifact's text into overlapping chunk surfaces.              |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`ClaudeTurn`](#ir.strategy.ClaudeTurn)(\*[, include_full])                      | Index a Claude Code session turn-pair: user prompt + assistant summary. |
| [`IndexingStrategy`](#ir.strategy.IndexingStrategy)(\*args, \*\*kwargs)                | Decompose one artifact into filter fields + embeddable surfaces.        |
| [`Package`](#ir.strategy.Package)(\*[, chunk_size, overlap, ...])             | Package strategy: `name + description` surface plus README chunks.      |
| [`Skill`](#ir.strategy.Skill)()                                             | Capability strategy: embed `name + description` only.                   |
| [`WholeText`](#ir.strategy.WholeText)(\*[, text_key, kind])                     | One surface = the entire text.                                          |

### *class* ir.strategy.Chunked(, chunk_size=1200, overlap=200, text_key=None, kind='chunk', max_tokens=None, token_model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Split the artifact’s text into overlapping chunk surfaces.

By default chunks are packed to ~\`\`chunk_size\`\` **characters**. Set
`max_tokens` to chunk by **tokens** instead — each chunk is bounded to
`max_tokens` tokens of `token_model` (default: the default embedding
model), so no chunk overflows the embedder’s sequence limit and is silently
truncated. Token mode degrades to char mode with a warning if the tokenizer
is unavailable. The char `chunk_size` / `overlap` still set the overlap
*ratio* used in token mode.

### *class* ir.strategy.ClaudeTurn(, include_full=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Index a Claude Code session turn-pair: user prompt + assistant summary.

Two surfaces by default — `user_prompt` (what the human asked) and
`assistant_summary` (the assistant’s *final* end-of-turn text, the
highest-signal “here’s what I did”; the deliberation before it is mostly
noise) — so a query can target either side via `surfaces={"user_prompt"}` /
`{"assistant_summary"}`. With `include_full=True` a third
`assistant_full` surface (all the turn’s assistant natural-language text)
trades noise for recall — off by default. Session / project / time / model /
tool-use become hard-filter fields.

The raw artifact is a turn-pair record (see
`priv.claude_transcripts.turn_pair_records()`): a mapping with
`user_prompt` / `assistant_summary` / `assistant_full` plus metadata.

### ir.strategy.DFLT_CHUNK_MAX_TOKENS *= 250*

Content-token budget for token-aware chunking under the default embedder.
`all-MiniLM-L6-v2` caps input at 256 tokens and silently truncates beyond
it. We target 250 (not 254) to leave headroom for the two special tokens
(`[CLS]`/`[SEP]`) *and* the ±1–2 token drift when an offset-sliced
substring is re-tokenized on its own — so an embedded chunk stays within
budget and its tail is never silently truncated (see `_chunk_text()`).

### *class* ir.strategy.IndexingStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Decompose one artifact into filter fields + embeddable surfaces.

### *class* ir.strategy.Package(, chunk_size=1500, overlap=200, embed_deps=False, deps_template=None, max_tokens=None, token_model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Package strategy: `name + description` surface plus README chunks.

Filter fields capture ownership (ours vs third-party), name, deps. AI
synopsis / problem-class surfaces are a documented extension point.

With `embed_deps=True`, `decompose` additionally emits one
`Surface(kind="deps", granularity="field")` whose text is a prefix-form
serialization of the **bare** dependency names (`deps_template`, default
`_default_deps_text()`) — so a query for a domain matches a package by the
libraries it depends on (e.g. `sentence-transformers` -> embeddings,
`networkx` -> graphs), and the BM25 leg picks up exact dep-token matches. The
deps bag is kept **separate from prose** (its own surface) so a rare library
name is not diluted, and deps remain a filter field regardless. `embed_deps`
defaults `False` (today’s behavior); it folds into the strategy id, so
toggling it re-decomposes incrementally. The deps surface is appended **last**,
leaving the `description` (position 0) and `readme_chunk` indices unchanged.

Surface indexing: the `description` surface (kept whenever *name* or
*description* is non-empty) occupies plan position 0, so `readme_chunk`
*j* is stored with
`Record.surface_index == j + 1` while its surface metadata says
`chunk_index == j` — `surface_index` is plan-global, `chunk_index`
per-kind (see [`ir.base.Record.make_id()`](ir.base.html.md#ir.base.Record.make_id)). Never derive sibling record
ids from `chunk_index`; use the ledger
([`ir.retrieve.records_for_artifact()`](ir.retrieve.html.md#ir.retrieve.records_for_artifact)). `n_chunks` is stamped on
readme chunks at decompose time, but corpora built before the stamp keep
records without it until the artifact re-indexes (content / embedder /
strategy change) — read it with `metadata.get("n_chunks")`.

### ir.strategy.STRATEGY_REGISTRY *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [type](https://docs.python.org/3/builtins/functions.html#type)]* *= {'Chunked': <class 'ir.strategy.Chunked'>, 'ClaudeTurn': <class 'ir.strategy.ClaudeTurn'>, 'Package': <class 'ir.strategy.Package'>, 'Skill': <class 'ir.strategy.Skill'>, 'WholeText': <class 'ir.strategy.WholeText'>}*

Shipped strategies addressable by name, for persisting an
[`IndexingStrategy`](#ir.strategy.IndexingStrategy) in a registry entry and reconstructing it (#58).
New shipped strategies register here so a corpus can name its segmentation.

### *class* ir.strategy.Skill

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Capability strategy: embed `name + description` only.

The body (SKILL.md) is loaded post-selection and is *not* indexed; name and
parent are filter fields.

### *class* ir.strategy.WholeText(, text_key=None, kind='document')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One surface = the entire text. Sensible default for a naive corpus.

### ir.strategy.strategy_from_spec(spec)

Reconstruct a shipped strategy from a `{"name", "params"}` spec.

`None` (no persisted strategy) returns `None` so the caller falls back to
the source preset’s default strategy — the back-compatible behavior for v1
registry entries.

* **Return type:**
  [`IndexingStrategy`](#ir.strategy.IndexingStrategy) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.strategy.strategy_to_spec(strategy)

A `{"name", "params"}` spec for a shipped, scalar-param strategy.

Captures only scalar constructor parameters (the same identity surface
`ir.index._strategy_id()` stamps), so it round-trips the shipped
strategies. A custom strategy, or one wrapping another (e.g. the
[`ir.with_synopsis()`](ir.html.md#ir.with_synopsis) wrapper), is **not** captured here — those are set
programmatically at build time / by the maintenance layer, not persisted as a
segmentation spec.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ir.strategy.text_of(raw, text_key=None)

Best-effort text extraction from a raw artifact payload.

The SSOT for turning an opaque `raw` (a `str`, a `Mapping` with a
`text` field or a `text_key`, or anything else) into embeddable text —
reused by the shipped strategies *and* by [`ir.synopsis.make_llm_synthesizer()`](ir.synopsis.html.md#ir.synopsis.make_llm_synthesizer)
so an injected-free synopsis summarizes the same text a strategy would index.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
