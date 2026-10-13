# ir.rerank

The precision stage: a cross-encoder rerank over the first-stage candidates.

First-stage retrieval (dense / lexical / hybrid) is tuned for *recall*; many of its worst
false positives are artefacts a joint `(query, passage)` scorer catches – a package
whose single “DAG” bullet inflates its pooled-vector cosine, or a pair of siblings that
share jargon rather than topic. [`ir.search()`](ir.html.md#ir.search) therefore takes `rerank=`: it
over-fetches `fetch_k` candidates, hands them to the reranker, and keeps the top `k`.
This is single-shot and agent-free: a better *one* search, not the evaluator back-edge
(that belongs to `raglab`).

`ir` composes `ef` here and adds no reranker class. `ef.cross_encoder_reranker` is the
implementation; this module adds the three things `ef` cannot know about an `ir` corpus:

- **What the reranker reads.** `rerank_text="surface"` (default) scores the matched
  surface, exactly as before. `"artifact"` scores the artifact’s *whole* stored text –
  its sibling surfaces (name + description + README chunks) joined in plan order – so a
  thin description is judged against the full README. A callable
  `(corpus, meta) -> str` replaces both.
- **Opt-in, with a graceful fallback.** `rerank="cross-encoder"` (or
  `"cross-encoder:<model>"`) names the ef cross-encoder without importing it, so
  `import ir` needs no torch. When `sentence-transformers` (the `ir[rerank]` extra) is
  missing, the search warns once and returns the first-stage ranking rather than failing.
- **Calibration that stays honest.** A reranked hit’s score is the reranker’s, on a scale
  unrelated to cosine / RRF / BM25, so an abstention floor calibrated for `hybrid` would
  be mis-scaled if applied to it. [`calibration_key()`](#ir.rerank.calibration_key) gives a reranked configuration its
  own key (`hybrid+rerank-<model>+artifact`), so `calibrate_min_score(...,
  rerank=...)` and `discover(..., min_score="auto", rerank=...)` read and write a floor
  for *that* configuration, and fall back to “no floor” (with the usual warning) rather
  than a wrong one. The fallback to first-stage ranking maps back to the plain key.

Relative selection strategies (`rel_threshold`, `score_gap`…) compare score *ratios*,
so a reranker used with them must return non-negative scores. The `"cross-encoder"` spec
guarantees that: most cross-encoders (including ef’s default ms-marco model) return raw
logits that run negative, so ir passes them through a sigmoid into `(0, 1)`. (A model that already applies a sigmoid would be squashed a second time, which keeps the
order but compresses the scale until relative selectors can no longer tell hits apart: use
such a model as a callable, `ef.cross_encoder_reranker(model)` with a `rerank_id`.) A callable you pass yourself is used as is: keep it in
`[0, 1]`, or give it a stable `rerank_id` attribute so its calibrated floor is kept apart.

```pycon
>>> resolve_reranker(None) is None
True
>>> calibration_key("hybrid", None)
'hybrid'
>>> calibration_key("hybrid", rerank_tag(lambda q, s: [], "artifact"))
'hybrid+rerank-~anonymous+artifact'
```

### Module Attributes

| [`RERANK_TEXTS`](#ir.rerank.RERANK_TEXTS)          | The built-in `rerank_text` choices (a callable is also accepted).      |
|------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`DFLT_RERANK_MODEL`](#ir.rerank.DFLT_RERANK_MODEL)     | The cross-encoder `rerank="cross-encoder"` resolves to (ef's default). |
| [`DFLT_RERANK_MAX_CHARS`](#ir.rerank.DFLT_RERANK_MAX_CHARS) | Cap on the characters of one `rerank_text="artifact"` passage.         |
| [`CROSS_ENCODER`](#ir.rerank.CROSS_ENCODER)         | The `rerank=` string that names ef's cross-encoder.                    |

### Functions

| [`resolve_reranker`](#ir.rerank.resolve_reranker)(spec)                         | The reranker `spec` names, or `None` when there is none to apply.                                                                                         |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`rerank_tag`](#ir.rerank.rerank_tag)(reranker[, rerank_text, max_chars]) | A short, key-safe identity for one reranked configuration (`None` if no rerank).                                                                          |
| [`is_anonymous`](#ir.rerank.is_anonymous)(tag)                              | Does `tag` rest on an unnamed callable (so its calibration key may collide)?                                                                              |
| [`spec_tag`](#ir.rerank.spec_tag)(spec[, rerank_text, max_chars])       | The tag a `rerank` spec *would* get, without loading a model; `None` if the spec cannot run here (a `"cross-encoder"` spec whose extra is not installed). |
| [`calibration_key`](#ir.rerank.calibration_key)(mode[, tag])                   | The key an abstention floor is stored and loaded under.                                                                                                   |
| [`passage_texts`](#ir.rerank.passage_texts)(corpus, candidates[, ...])       | The text the reranker scores for each candidate's store `meta` row.                                                                                       |

### ir.rerank.CROSS_ENCODER *= 'cross-encoder'*

The `rerank=` string that names ef’s cross-encoder.

### ir.rerank.DFLT_RERANK_MAX_CHARS *= 2000*

Cap on the characters of one `rerank_text="artifact"` passage. Cross-encoders truncate
at ~512 tokens anyway; the cap keeps the pair list cheap to build and the cut at a
surface boundary-agnostic, front-loaded point (name + description come first).

### ir.rerank.DFLT_RERANK_MODEL *= 'cross-encoder/ms-marco-MiniLM-L-6-v2'*

The cross-encoder `rerank="cross-encoder"` resolves to (ef’s default).

### ir.rerank.RERANK_TEXTS *= ('surface', 'artifact')*

The built-in `rerank_text` choices (a callable is also accepted).

### ir.rerank.calibration_key(mode, tag=None)

The key an abstention floor is stored and loaded under.

`mode` alone for a first-stage ranking (today’s behaviour, so existing floors keep
working); `"<mode>+<tag>"` for a reranked one.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> calibration_key("hybrid")
'hybrid'
>>> calibration_key("hybrid", "rerank-m+artifact")
'hybrid+rerank-m+artifact'
```

### ir.rerank.is_anonymous(tag)

Does `tag` rest on an unnamed callable (so its calibration key may collide)?

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ir.rerank.passage_texts(corpus, candidates, rerank_text='surface', , max_chars=2000)

The text the reranker scores for each candidate’s store `meta` row.

`"surface"` is the matched surface’s own text. `"artifact"` stitches every stored
surface of the candidate’s artifact in plan order (chunk overlap deduped, as in
[`ir.expand`](ir.html.md#ir.expand)) (siblings looked up once per
artifact, through the ledger) and cuts at `max_chars`; an artifact the ledger does not
know falls back to its matched surface. A callable is called as `rerank_text(corpus,
meta)`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.rerank.rerank_tag(reranker, rerank_text='surface', , max_chars=2000)

A short, key-safe identity for one reranked configuration (`None` if no rerank).

Built from the reranker’s identity (see `_identity()`), the `rerank_text` choice
and, for `"artifact"`, a non-default `max_chars` – everything that changes the
score scale. An unnamed reranker or text callable puts `~anonymous` in its tag, which
ir refuses to persist a floor under ([`is_anonymous()`](#ir.rerank.is_anonymous)). A tag longer than
`MAX_TAG_CHARS` is shortened with a hash of the whole, so it stays a usable file name.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.rerank.resolve_reranker(spec)

The reranker `spec` names, or `None` when there is none to apply.

`spec` is `None` (no rerank), any `ef.Reranker` callable (returned as is), or the
string `"cross-encoder"` / `"cross-encoder:<model>"` – ef’s sentence-transformers
cross-encoder, built lazily and cached per model name (loading one is the expensive
part). If its optional dependency is absent this warns once and returns `None`, so
the caller degrades to first-stage ranking; resolve *before* deriving a calibration key
so the degraded run uses the plain key.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> resolve_reranker(len) is len
True
>>> resolve_reranker("nope")
Traceback (most recent call last):
    ...
ValueError: unknown rerank spec 'nope'; expected a callable, 'cross-encoder' or 'cross-encoder:<model>'.
```

### ir.rerank.spec_tag(spec, rerank_text='surface', , max_chars=2000)

The tag a `rerank` spec *would* get, without loading a model; `None` if the
spec cannot run here (a `"cross-encoder"` spec whose extra is not installed).

For lookups such as [`ir.eval.load_calibration()`](ir.eval.html.md#ir.eval.load_calibration), which need the key, not the model.
