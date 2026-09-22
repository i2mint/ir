# ir.eval_gen

LLM-backed generation of evaluation cases for [`ir.eval`](ir.eval.html.md#module-ir.eval).

This is the **build-time** companion to the offline scoring harness: it turns a
corpus into a set of [`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase)s by *back-translation* —
given a capability’s description, ask an LLM for the user intents that should
route to it. The artifact’s id is the free ground-truth label.

Two ideas make the generated set honest:

- **Name masking.** The artifact’s *name* is stripped from the description
  *before* it is shown to the generator (and any output that still leaks the
  name is dropped). Otherwise a lexical retriever (the BM25 leg of hybrid) would
  trivially match query→gold on surface name overlap and inflate scores. Many
  real descriptions contain their own name, so masking the **input** — not just
  filtering the output — is what matters.
- **An abstention slice.** A fraction of cases are “no artifact applies” intents
  (empty `gold`), so the eval can measure correct refusal, not just hits.

The LLM is **injected** (`query_generator` / `abstention_generator` callables),
so the generation *logic* — masking, gold assignment, the leakage guard, the
abstention fraction — is fully testable with a deterministic stub and no network.
The default generators are built lazily on `aix` (`aix.prompt_func`, the
multi-provider LLM facade), so `import ir.eval_gen` stays cheap and offline;
`aix` is only imported when you actually generate with the real LLM.

The output is plain [`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase) data — freeze it with
[`ir.eval.save_cases()`](ir.eval.html.md#ir.eval.save_cases) (stamping [`corpus_signature()`](#ir.eval_gen.corpus_signature) into the
`__meta__` header) and score it with [`ir.eval`](ir.eval.html.md#module-ir.eval). Generation needs a model;
scoring never does.

Quick start:

```default
import ir
from ir import eval_gen as eg

source = ir.CorpusSource.from_skills()
cases = eg.build_eval_set(source, k=5, corpus_name="skills")   # uses aix
from ir.eval import save_cases
save_cases(cases, "skills_eval.jsonl",
           meta={"corpus": "skills", "corpus_signature": eg.corpus_signature(source)})
```

### Module Attributes

| [`DFLT_QUERIES_PER_ARTIFACT`](#ir.eval_gen.DFLT_QUERIES_PER_ARTIFACT)   | Queries generated per artifact by default.                                |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`DFLT_ABSTENTION_FRAC`](#ir.eval_gen.DFLT_ABSTENTION_FRAC)        | Target share of the case set that is abstention ("no artifact applies").  |
| [`DFLT_MIN_DESCRIPTION_CHARS`](#ir.eval_gen.DFLT_MIN_DESCRIPTION_CHARS)  | Minimum description length (chars) for an artifact to be back-translated. |
| [`NAME_PLACEHOLDER`](#ir.eval_gen.NAME_PLACEHOLDER)            | What a masked name is replaced with in a description / query.             |
| [`DFLT_ABSTENTION_THEME`](#ir.eval_gen.DFLT_ABSTENTION_THEME)       | Default theme used when generating abstention ("out of scope") intents.   |
| [`QueryGenerator`](#ir.eval_gen.QueryGenerator)              | `(description, *, n) -> list[str]` (n candidate intents).                 |
| [`AbstentionGenerator`](#ir.eval_gen.AbstentionGenerator)         | `(*, n, theme) -> list[str]` (n out-of-scope intents).                    |

### Functions

| [`build_eval_set`](#ir.eval_gen.build_eval_set)(source, \*[, k, ...])               | Generate a full eval set — gold cases plus an abstention slice.      |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`corpus_signature`](#ir.eval_gen.corpus_signature)(source_or_corpus)                 | A short, order-independent hash of a corpus's artifact ids.          |
| [`generate_abstention_cases`](#ir.eval_gen.generate_abstention_cases)(n, \*[, generator, ...]) | Generate `n` abstention cases — out-of-scope intents (empty `gold`). |
| [`generate_cases`](#ir.eval_gen.generate_cases)(source, \*[, k, mask_names, ...])   | Back-translate a corpus source into gold-bearing `DiscoveryCase`s.   |
| [`make_default_abstention_generator`](#ir.eval_gen.make_default_abstention_generator)(\*[, prompt])    | Build the default abstention generator on `aix` (lazy import).       |
| [`make_default_query_generator`](#ir.eval_gen.make_default_query_generator)(\*[, prompt])         | Build the default back-translation generator on `aix` (lazy import). |
| [`mask_name`](#ir.eval_gen.mask_name)(text, name, \*[, placeholder])           | Replace occurrences of `name` in `text` with `placeholder`.          |

### ir.eval_gen.AbstentionGenerator

`(*, n, theme) -> list[str]` (n out-of-scope intents).

* **Type:**
  An abstention generator

alias of `Callable`[[…], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.eval_gen.DFLT_ABSTENTION_FRAC *= 0.15*

Target share of the case set that is abstention (“no artifact applies”).

### ir.eval_gen.DFLT_ABSTENTION_THEME *= 'software developer tools'*

Default theme used when generating abstention (“out of scope”) intents.

### ir.eval_gen.DFLT_MIN_DESCRIPTION_CHARS *= 20*

Minimum description length (chars) for an artifact to be back-translated.

### ir.eval_gen.DFLT_QUERIES_PER_ARTIFACT *= 5*

Queries generated per artifact by default.

### ir.eval_gen.NAME_PLACEHOLDER *= 'this capability'*

What a masked name is replaced with in a description / query.

### ir.eval_gen.QueryGenerator

`(description, *, n) -> list[str]` (n candidate intents).

* **Type:**
  A query generator

alias of `Callable`[[…], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.eval_gen.build_eval_set(source, , k=5, abstention_frac=0.15, query_generator=None, abstention_generator=None, theme='software developer tools', corpus_name=None, \*\*gen_kwargs)

Generate a full eval set — gold cases plus an abstention slice.

The abstention count is chosen so abstention cases make up (at least)
`abstention_frac` of the returned set: `ceil(frac * G / (1 - frac))` for
`G` gold cases. Extra `gen_kwargs` flow to [`generate_cases()`](#ir.eval_gen.generate_cases)
(`mask_names`, `min_chars`, `max_artifacts`, `describe`).

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `abstention_frac` is outside `[0, 1)` (`frac=0`
      means no abstention slice) or `k` is less than 1.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase)]

### ir.eval_gen.corpus_signature(source_or_corpus)

A short, order-independent hash of a corpus’s artifact ids.

Stamp this into a case file’s `__meta__` header so a frozen eval set can be
checked against the (live, machine-specific) corpus it was generated from.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.eval_gen.generate_abstention_cases(n, , generator=None, theme='software developer tools', corpus_name=None)

Generate `n` abstention cases — out-of-scope intents (empty `gold`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase)]

### ir.eval_gen.generate_cases(source, , k=5, mask_names=True, query_generator=None, describe=None, min_chars=20, max_artifacts=None, corpus_name=None)

Back-translate a corpus source into gold-bearing `DiscoveryCase`s.

For each artifact in `source.scope` (id → raw), extract a description,
mask the artifact’s name out of it, ask `query_generator` for `k` user
intents, and emit one case per surviving intent (gold = the artifact id).
Intents that still leak the name are dropped; artifacts whose description is
shorter than `min_chars` are skipped (and the count is warned, never
silently dropped).

* **Parameters:**
  * **source** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a [`CorpusSource`](ir.sources.html.md#ir.sources.CorpusSource) (anything with `.items()`).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – intents to request per artifact.
  * **mask_names** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – scrub the artifact name from the description before
    generating, and drop any generated intent that still contains it.
  * **query_generator** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – `(description, *, n) -> [intent, …]`. Defaults to the
    `aix`-backed back-translator (built lazily; needs a model).
  * **describe** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `raw -> description` (default: the `description` / `text`
    field, else the joined string fields).
  * **min_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – skip artifacts whose description is shorter than this.
  * **max_artifacts** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – cap how many artifacts to process (for a quick/cheap run);
    when set, artifacts are taken in sorted-id order so the subset is
    deterministic even for filesystem-ordered (`dol`-backed) scopes.
  * **corpus_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – stamped on each case’s `corpus` field.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase)]
* **Returns:**
  the generated gold cases.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `k` is less than 1.

### ir.eval_gen.make_default_abstention_generator(, prompt='You are generating "no applicable tool" cases for a tool-retrieval evaluation.\\\\n\\\\nThe tool catalog is about: {theme}.\\\\n\\\\nWrite {n} natural, plausible user requests that such a catalog should NOT be able\\\\nto satisfy because they fall outside its scope.\\\\nOne request per line. No numbering, no quotes, no extra commentary.\\\\n', \*\*prompt_function_kwargs)

Build the default abstention generator on `aix` (lazy import).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.eval_gen.make_default_query_generator(, prompt='You are generating evaluation data for a tool-retrieval system.\\\\n\\\\nBelow is a description of a capability (its name has been hidden on purpose):\\\\n\\\\n{description}\\\\n\\\\nWrite {n} natural, varied user requests that this capability should handle.\\\\nRules:\\\\n- Do NOT mention any tool, function, skill, or package name.\\\\n- Vary phrasing, specificity, and the implied (not explicit) parameters.\\\\n- One request per line. No numbering, no quotes, no extra commentary.\\\\n', \*\*prompt_function_kwargs)

Build the default back-translation generator on `aix` (lazy import).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.eval_gen.mask_name(text, name, , placeholder='this capability')

Replace occurrences of `name` in `text` with `placeholder`.

Matches the name as a contiguous phrase tolerant of the separator between its
tokens (`"ci-advisor"` / `"ci advisor"` / `"ci_advisor"` all match),
case-insensitively and whole-word (a short token like `"ci"` does not blast
through `"specific"`). If `placeholder` would itself match the name, a
neutral alternate is used so the masked text is not self-referential. Used to
scrub the artifact name out of a description *before* it is generated from.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
