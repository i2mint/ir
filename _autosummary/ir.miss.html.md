# ir.miss

Lexical miss signals: did a keyword search actually find what was asked for?

A deterministic search returns *something* for almost any query, so “got hits”
says little about whether the right thing was found. These are model-free,
auditable facts an agent’s Evaluator (raglab) can use to decide to reformulate
(ir_13 §1.3, ir_09 §3):

- `zero_hits`: retrieval returned nothing;
- `missing_required`: terms the answer must mention (an entity the user named)
  that appear in none of the committed hits;
- `missing_in_candidates`: the subset of those absent from every *retrieved*
  candidate too, which separates “retrieval missed it, reformulate” from “the
  selection was too tight, widen it”.

Matching is whole-word, case-insensitive, per `\w` boundary. Known limits: no
word boundaries in CJK text (`東京` is not found inside a sentence), no plural or
spelling folding (`invoice` vs `invoices`, `ci-cd` vs `CI/CD`), no Unicode
normalisation. Fold both sides first if that matters.

They are **signals, never directives**: ir reports them (`Selection.signals`,
`Selection.sufficient`) and never re-queries; the back-edge belongs to the
agent layer.

### Functions

| [`lexical_miss`](#ir.miss.lexical_miss)(hits, \*[, required, candidates])   | Miss signals for the committed `hits`; `{"miss": bool, ...}`.   |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|

### ir.miss.lexical_miss(hits, , required=(), candidates=None)

Miss signals for the committed `hits`; `{"miss": bool, ...}`.

`required` are terms the answer must mention. `candidates` (all retrieved
hits, a superset of `hits`) enables `zero_hits` over retrieval and
`missing_in_candidates`; without it the committed hits stand in for both.

```pycon
>>> hit = SearchHit("a", "doc", 1.0, "the acme invoice")
>>> lexical_miss([hit], required=["acme", "globex"])["missing_required"]
['globex']
>>> lexical_miss([])["miss"]
True
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
