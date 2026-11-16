# ir.formulate

Query formulation — the Formulator seam (ir_09 §3).

Turn a user query into one or more concrete low-level queries *before* retrieval:
rewrite, expand, paraphrase, HyDE. This is the single most-acknowledged gap in
ir’s standalone retrieval — the raw query is otherwise embedded as-is — and a
good formulation lifts recall on short, identifier-heavy capability text with
**no agent and no loop**.

The seam is **opt-in and identity by default**: `ir.search(corpus, q)` with no
`formulate=` embeds `q` verbatim (exactly today’s behavior). A formulator
returns a `str` (one query) or a sequence of strings (multi-query fan-out);
when it returns several, [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search) runs each and fuses the
candidate lists (best surface per artifact across all queries).

LOAD-BEARING BOUNDARY: a [`Formulator`](#ir.formulate.Formulator) returns **queries, never SubTasks**.
Decomposing a goal into sub-tasks + source selection is the *Planner’s* job — that
lives in the agent layer (`raglab`), not here.

[`make_llm_formulator()`](#ir.formulate.make_llm_formulator) mirrors `ir.select.make_llm_selector()`: an
injectable `rewriter` callable, built lazily on `aix` when omitted (so
importing ir stays offline), falling back to identity on any failure — a
formulator must never make retrieval *worse* than the raw query.

### Module Attributes

| [`Formulator`](#ir.formulate.Formulator)                 | a query string -> one query (`str`) or several (`Sequence[str]`).                                                           |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`FORMULATION_PROMPT`](#ir.formulate.FORMULATION_PROMPT)         | Default prompt for [`make_llm_formulator()`](#ir.formulate.make_llm_formulator) — diverse paraphrases for recall. |
| [`KEYWORD_FORMULATION_PROMPT`](#ir.formulate.KEYWORD_FORMULATION_PROMPT) | term lists for BM25 / grep, not fluent paraphrases (ir_13 §1.3).                                                            |

### Functions

| [`identity_formulator`](#ir.formulate.identity_formulator)(query)                       | The default formulator: return the query unchanged (embed it verbatim).                                             |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`make_keyword_formulator`](#ir.formulate.make_keyword_formulator)(\*[, rewriter, n, ...])  | An LLM [`Formulator`](#ir.formulate.Formulator) that writes **keyword** queries for BM25 / grep. |
| [`make_llm_formulator`](#ir.formulate.make_llm_formulator)(\*[, rewriter, prompt, ...]) | An LLM-backed [`Formulator`](#ir.formulate.Formulator) (rewrite / expand / multi-query).         |

### ir.formulate.FORMULATION_PROMPT *= 'Rewrite the search query into {n} short, diverse alternative search queries that would retrieve the same target documents: fix typos, expand jargon, and add synonyms, but keep each a terse search phrase. One query per line, no numbering.\\n\\nQuery: {query}'*

Default prompt for [`make_llm_formulator()`](#ir.formulate.make_llm_formulator) — diverse paraphrases for recall.

### ir.formulate.Formulator

a query string -> one query (`str`) or several
(`Sequence[str]`). Identity by default; an LLM rewriter / HyDE / multi-query
producer when injected.

* **Type:**
  A formulator

alias of `Callable`[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `str | Sequence[str]`]

### ir.formulate.KEYWORD_FORMULATION_PROMPT *= 'You are writing queries for a keyword (BM25 / grep) search, not a semantic one. For the request below, output {n} short keyword queries, one per line, no numbering. Each is a few terms that would literally appear in a matching document: spelling and morphological variants (deploy, deploys, deployment), synonyms and aliases, and the artifacts the answer would leave behind (file names, commands, identifiers, error messages). Never a sentence.\\n\\nRequest: {query}'*

term lists for BM25 / grep, not
fluent paraphrases (ir_13 §1.3).

* **Type:**
  Prompt for [`make_keyword_formulator()`](#ir.formulate.make_keyword_formulator)

### ir.formulate.identity_formulator(query)

The default formulator: return the query unchanged (embed it verbatim).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.formulate.make_keyword_formulator(, rewriter=None, n=4, include_original=True, \*\*kwargs)

An LLM [`Formulator`](#ir.formulate.Formulator) that writes **keyword** queries for BM25 / grep.

Like [`make_llm_formulator()`](#ir.formulate.make_llm_formulator) (same injectable `rewriter`, lazy `aix`,
identity fallback), but its default prompt asks for term lists (variants and
the artifacts an answer would leave behind) instead of fluent paraphrases,
which is what a lexical first stage can match. With `include_original` the
query itself is always one of the produced queries, so a bad rewrite can add
noise but never drop the literal request; pair it with
`search(variant_weight=...)` to keep the original on top.

```pycon
>>> f = make_keyword_formulator(rewriter=lambda q: ["deploy deployment", "ci.yml"])
>>> f("how do we ship")
['how do we ship', 'deploy deployment', 'ci.yml']
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.formulate.make_llm_formulator(, rewriter=None, prompt='Rewrite the search query into {n} short, diverse alternative search queries that would retrieve the same target documents: fix typos, expand jargon, and add synonyms, but keep each a terse search phrase. One query per line, no numbering.\\n\\nQuery: {query}', n=3, fallback=None, \*\*prompt_function_kwargs)

An LLM-backed [`Formulator`](#ir.formulate.Formulator) (rewrite / expand / multi-query).

`rewriter` is an injectable `query -> str | [str, ...]` callable (a test
double, or your own router); when omitted it is built lazily on `aix`
(`aix.prompt_func`), so importing this module stays offline. `n` is the
multi-query fan-out width. Any error or empty reply falls back to `fallback`
(default: [`identity_formulator()`](#ir.formulate.identity_formulator)).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
