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
- [`Fields`](#ir.strategy.Fields) — a record’s named text fields (head + body, balanced windows,
  optional normalizer); one surface per record or per field.
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

| [`balanced_windows`](#ir.strategy.balanced_windows)(text, \*, size[, overlap])   | Split *text* into the fewest windows of at most *size* chars, **near-equal in length**.                                      |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`fold_text`](#ir.strategy.fold_text)(text)                               | Case-fold *text* and strip accents from Latin and Greek letters only.                                                        |
| [`is_simple_param`](#ir.strategy.is_simple_param)(value)                        | Whether a constructor parameter is JSON-simple (scalar, or a tuple/list of scalars).                                         |
| [`strategy_from_spec`](#ir.strategy.strategy_from_spec)(spec)                      | Reconstruct a shipped strategy from a `{"name", "params"}` spec.                                                             |
| [`strategy_normalizer`](#ir.strategy.strategy_normalizer)(strategy)                 | The `normalize` an indexing *strategy* applies to its text, as given.                                                        |
| [`strategy_to_spec`](#ir.strategy.strategy_to_spec)(strategy)                    | A `{"name", "params"}` spec for a shipped, scalar-param strategy.                                                            |
| [`text_of`](#ir.strategy.text_of)(raw[, text_key])                      | Best-effort text extraction from a raw artifact payload.                                                                     |
| [`word_windows`](#ir.strategy.word_windows)(text, \*, size[, overlap])       | Like [`balanced_windows()`](#ir.strategy.balanced_windows), but each cut moves back to the nearest whitespace. |

### Classes

| [`Chunked`](#ir.strategy.Chunked)(\*[, chunk_size, overlap, text_key, ...])   | Split the artifact's text into overlapping chunk surfaces.                    |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`ClaudeTurn`](#ir.strategy.ClaudeTurn)(\*[, include_full])                      | Index a Claude Code session turn-pair: user prompt + assistant summary.       |
| [`Fields`](#ir.strategy.Fields)(fields, \*[, head, layout, normalize, ...])  | Index a record's named text fields (a catalogue item: name, place, prose...). |
| [`IndexingStrategy`](#ir.strategy.IndexingStrategy)(\*args, \*\*kwargs)                | Decompose one artifact into filter fields + embeddable surfaces.              |
| [`Package`](#ir.strategy.Package)(\*[, chunk_size, overlap, ...])             | Package strategy: `name + description` surface plus README chunks.            |
| [`Skill`](#ir.strategy.Skill)()                                             | Capability strategy: embed `name + description` only.                         |
| [`WholeText`](#ir.strategy.WholeText)(\*[, text_key, kind])                     | One surface = the entire text.                                                |

### *class* ir.strategy.Chunked(, chunk_size=1200, overlap=200, text_key=None, kind='chunk', max_tokens=None, token_model=None, split=None, split_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Split the artifact’s text into overlapping chunk surfaces.

By default chunks are packed to ~\`\`chunk_size\`\` **characters**. Set
`max_tokens` to chunk by **tokens** instead — each chunk is bounded to
`max_tokens` tokens of `token_model` (default: the default embedding
model), so no chunk overflows the embedder’s sequence limit and is silently
truncated. Token mode degrades to char mode with a warning if the tokenizer
is unavailable. The char `chunk_size` / `overlap` still set the overlap
*ratio* used in token mode.

`split` swaps the character splitter for a `(text, *, size, overlap) ->
list[str]` callable or `"module:attr"` reference, e.g.
[`balanced_windows()`](#ir.strategy.balanced_windows) (equal-sized windows, no short tail) or
[`word_windows()`](#ir.strategy.word_windows) (word-aligned cuts), with `size=chunk_size`. It is
exclusive with `max_tokens`. `split_id` names a lambda/closure for
re-indexing, as for [`Fields`](#ir.strategy.Fields).

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

### *class* ir.strategy.Fields(fields, , head=(), layout='record', normalize=None, normalize_id=None, max_chars=1200, overlap=200, kind='record', split=None, split_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Index a record’s named text fields (a catalogue item: name, place, prose…).

`fields` are the body keys, read in order; `head` are keys (a name, a
place) that lead the record. Two layouts:

- `layout="record"` (default): one surface per window of the body, with the
  head text **prefixed to every window**, so a word in the name and a word in
  the place only score together because they share a surface. Measured on 12
  judged requests over ~44k records: BM25 hit@1 0.92 vs 0.58 for `"fields"`.
- `layout="fields"`: one surface per field (head and body), `kind` = the
  field name, so a query can target one field with `surfaces={...}`.

`normalize` (a `str -> str` callable, or a `"module:attr"` reference to
one, case/accent folding…) is applied to every text before indexing, and
[`ir.build()`](ir.html.md#ir.build) records it on the corpus so every search folds its query
with the same function (see [`ir.index.Corpus`](ir.index.html.md#ir.index.Corpus)). A reference is
persisted in a registry spec and in the built corpus, and is its own
identity; a callable is not persisted (a reopened corpus warns that its
queries are no longer folded, unless `open_corpus(query_normalize=...)`
hands it back), and is identified by its qualified name or an explicit
`normalize_id` (required for a lambda or closure).
`max_chars` / `overlap` window the *body* with [`balanced_windows()`](#ir.strategy.balanced_windows)
(`split` swaps the splitter: a `(text, *, size, overlap) -> list[str]`
callable or `"module:attr"` reference, e.g. [`word_windows()`](#ir.strategy.word_windows) for
word-aligned cuts; `split_id` names a lambda/closure for re-indexing) (the head is prefixed to each window and not itself windowed, so keep it short).
`kind` names the surface in the `"record"` layout; `"fields"` uses the
field names. A key listed in both `head` and `fields` is read once.
[`reads`](#ir.strategy.Fields.reads) lists every key the strategy reads, so a change signal can cover them.

```pycon
>>> plan = Fields(fields=["about"], head=["name", "place"]).decompose(
...     "r1", {"name": "Louvre", "place": "Paris", "about": "A museum."})
>>> [(s.kind, s.text) for s in plan.surfaces]
[('record', 'Louvre\nParis\nA museum.')]
```

#### *property* reads *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

Every record key this strategy reads (for change signals).

#### *property* synopsis_fields *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

The keys a synopsis should summarize (`ir.with_synopsis` reads this).

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

#### synopsis_fields *= ('name', 'description', 'readme')*

The record fields a synopsis is written from (see [`ir.synopsis.with_synopsis()`](ir.synopsis.html.md#ir.synopsis.with_synopsis)):
prose only. The record’s `path` (a local absolute path) and `deps` never go to an LLM.

### ir.strategy.STRATEGY_REGISTRY *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [type](https://docs.python.org/3/builtins/functions.html#type)]* *= {'Chunked': <class 'ir.strategy.Chunked'>, 'ClaudeTurn': <class 'ir.strategy.ClaudeTurn'>, 'Fields': <class 'ir.strategy.Fields'>, 'Package': <class 'ir.strategy.Package'>, 'Skill': <class 'ir.strategy.Skill'>, 'WholeText': <class 'ir.strategy.WholeText'>}*

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

### ir.strategy.balanced_windows(text, , size, overlap=0)

Split *text* into the fewest windows of at most *size* chars, **near-equal in length**.

`n = ceil((L - overlap) / (size - overlap))` windows, each `ceil((L +
(n - 1) * overlap) / n)` long, so a text just past the limit becomes two
halves, not one full window plus a sliver (which length normalisation in
BM25 would favour).

```pycon
>>> balanced_windows("a" * 10, size=10)
['aaaaaaaaaa']
>>> [len(w) for w in balanced_windows("a" * 11, size=10)]
[6, 5]
>>> [len(w) for w in balanced_windows("a" * 25, size=10, overlap=2)]
[10, 10, 9]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.strategy.fold_text(text)

Case-fold *text* and strip accents from Latin and Greek letters only.

Applies NFKC first (full-width letters, ligatures and compatibility forms fold
in), then `casefold`, then removes accents of Latin/Greek letters, mapping
strokes and ligature-letters (`ø đ ł æ œ þ`) to ASCII. Safe for mixed-script
text: marks that carry meaning elsewhere are left alone (kana voicing `が`,
Hangul syllables, Devanagari virama, Thai tone marks, Cyrillic `й`/`ё`).
Use it as a `Fields(normalize=...)` normalizer (by reference:
`"ir.strategy:fold_text"`) **and** apply it to queries, e.g.
`formulate=fold_text` (a `str -> str` is a valid formulator), since folding
acts on whichever side runs it. A synopsis written by an LLM
(`ir.with_synopsis`) is not folded.

```pycon
>>> fold_text("Crème BRÛLÉE Straße Đà Nẵng Łódź")
'creme brulee strasse da nang lodz'
>>> fold_text("Ｃｒèｍｅ")
'creme'
>>> fold_text("Αθήνα")
'αθηνα'
>>> fold_text("がぎ 한국어 हिन्दी Йё ไก่")
'がぎ 한국어 हिन्दी йё ไก่'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.strategy.is_simple_param(value)

Whether a constructor parameter is JSON-simple (scalar, or a tuple/list of scalars).

The shared test behind a strategy’s persisted spec ([`strategy_to_spec()`](#ir.strategy.strategy_to_spec))
and its build identity (`ir.index._strategy_id`), so a parameter such as
[`Fields`](#ir.strategy.Fields)’s `fields` both round-trips through the registry and
re-decomposes a corpus when it changes.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ir.strategy.strategy_from_spec(spec)

Reconstruct a shipped strategy from a `{"name", "params"}` spec.

`None` (no persisted strategy) returns `None` so the caller falls back to
the source preset’s default strategy — the back-compatible behavior for v1
registry entries.

* **Return type:**
  [`IndexingStrategy`](#ir.strategy.IndexingStrategy) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.strategy.strategy_normalizer(strategy)

The `normalize` an indexing *strategy* applies to its text, as given.

That is the strategy’s `normalize` attribute (a callable or a
`"module:attr"` reference), or, for a wrapper such as
[`ir.with_synopsis()`](ir.html.md#ir.with_synopsis), that of the strategy it wraps; `None` when the
indexed text is not normalized. [`ir.build()`](ir.html.md#ir.build) records it so a search can
fold its query with the same function.

```pycon
>>> strategy_normalizer(Fields(fields=["about"], normalize="ir.strategy:fold_text"))
'ir.strategy:fold_text'
>>> strategy_normalizer(Fields(fields=["about"])) is None
True
```

* **Return type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]

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

### ir.strategy.word_windows(text, , size, overlap=0)

Like [`balanced_windows()`](#ir.strategy.balanced_windows), but each cut moves back to the nearest whitespace.

A window that would end mid-word ends at the last space inside it instead
(when there is one), and the next window starts at the following word, so no
word is split. Windows are therefore *at most* `size` long but not
near-equal. `overlap` is a character target, rounded to word starts.

```pycon
>>> word_windows("alpha beta gamma delta", size=12)
['alpha beta', 'gamma delta']
>>> word_windows("short", size=12)
['short']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
