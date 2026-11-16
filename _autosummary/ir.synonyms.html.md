# ir.synonyms

The synonym ledger: corpus vocabulary an agent learns, for keyword and regex search.

Lexical search (BM25, grep) misses when the searcher’s word is not the document’s
word. An LLM closes most of that gap on its own by writing variants of the query,
but it cannot know a corpus’s private vocabulary: a project’s nickname, a renamed
package, a person’s handle, in-house jargon. The ledger keeps exactly those, learned
from misses, so the next search starts where the last one ended (research report
`misc/docs/ir_13`).

One JSON object per line in `synonyms.jsonl` under ir’s config dir (override the
path with `$IR_SYNONYMS`):

```default
{"terms": ["acmecorp", "acme corp", "ac"], "note": "client; repo dir ac/", "added": "2026-11-11"}
{"terms": ["vd", "vector database"], "oneway": true, "note": "vd expands, not back"}
```

`terms` are equivalent (each expands to the others); with `oneway` only the first
expands to the rest. `corpus` (optional) limits an entry to one corpus. An entry
may carry `provenance` (`{"miss_query": ..., "found_ids": [...]}`: the search that
missed and what the vocabulary found), which lets [`ir.synonyms_eval.replay_ledger()`](ir.synonyms_eval.html.md#ir.synonyms_eval.replay_ledger)
re-test it; `verified` (the number of replays it won) and `status` (`"demoted"` entries are not
loaded by default) are maintained by that replay. The ledger
is the user’s private data: it lives in their config dir, never in a repo.

Three uses, one source: [`term_regex()`](#ir.synonyms.term_regex) builds an alternation for grep,
[`expand_query()`](#ir.synonyms.expand_query) produces query variants, and [`make_synonym_formulator()`](#ir.synonyms.make_synonym_formulator)
plugs those into [`ir.retrieve.search()`](ir.retrieve.html.md#ir.retrieve.search)’s `formulate=` seam (deterministic, no
LLM; the original query always comes first).

```pycon
>>> entries = [SynonymEntry(("acmecorp", "acme corp", "ac"))]
>>> aliases("Acme Corp", entries)
['Acme Corp', 'acmecorp', 'ac']
>>> term_regex(aliases("acmecorp", entries))
'acme[-_ ]?corp|acmecorp|\\bac\\b'
>>> expand_query("acme corp invoice", entries)
['acme corp invoice', 'acmecorp invoice', 'ac invoice']
```

### Module Attributes

| [`ACTIVE`](#ir.synonyms.ACTIVE)            | `DEMOTED` entries are kept (for audit) but not loaded.                                                                      |
|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`DEMOTED`](#ir.synonyms.DEMOTED)           | `DEMOTED` entries are kept (for audit) but not loaded.                                                                      |
| [`DFLT_LEDGER_NAME`](#ir.synonyms.DFLT_LEDGER_NAME)  | The ledger's file name inside [`ir.config.config_dir()`](ir.config.html.md#ir.config.config_dir). |
| [`SHORT_TERM_CHARS`](#ir.synonyms.SHORT_TERM_CHARS)  | Terms at most this long that are plain word characters get `\b` guards in a regex, so `ac` does not match inside `react`.   |
| [`DFLT_MAX_VARIANTS`](#ir.synonyms.DFLT_MAX_VARIANTS) | Default cap on the number of query variants (the original included).                                                        |

### Functions

| [`add_synonyms`](#ir.synonyms.add_synonyms)(terms, \*[, note, oneway, ...])     | Append one entry to the ledger (creating it) and return it.                                                                                    |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| [`aliases`](#ir.synonyms.aliases)(term, entries)                           | `term` first, then every term the ledger makes equivalent to it (deduped).                                                                     |
| [`expand_query`](#ir.synonyms.expand_query)(query, entries, \*[, max_variants]) | `query` first, then variants with one known term swapped for an alias.                                                                         |
| [`ledger_path`](#ir.synonyms.ledger_path)()                                    | Where the ledger lives: `$IR_SYNONYMS`, else `<config_dir>/synonyms.jsonl`.                                                                    |
| [`load_synonyms`](#ir.synonyms.load_synonyms)([path, corpus, include_demoted])   | The ledger's entries (those for all corpora, plus `corpus`'s own).                                                                             |
| [`make_synonym_formulator`](#ir.synonyms.make_synonym_formulator)([entries, path, ...])    | A deterministic [`ir.formulate.Formulator`](ir.formulate.html.md#ir.formulate.Formulator) that expands from the ledger. |
| [`term_regex`](#ir.synonyms.term_regex)(terms)                                | One alternation matching any of `terms`, tolerant of `-`/`_`/space.                                                                            |
| [`update_entries`](#ir.synonyms.update_entries)(changes[, path])                  | Rewrite the ledger with `(old, new)` entry changes; every other line stays verbatim.                                                           |

### Classes

| [`SynonymEntry`](#ir.synonyms.SynonymEntry)(terms[, oneway, note, added, ...])   | One ledger line: equivalent `terms` (or `oneway` from the first).   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|

### ir.synonyms.ACTIVE *= 'active'*

`DEMOTED` entries are kept (for audit) but not loaded.

* **Type:**
  Ledger entry statuses

### ir.synonyms.DEMOTED *= 'demoted'*

`DEMOTED` entries are kept (for audit) but not loaded.

* **Type:**
  Ledger entry statuses

### ir.synonyms.DFLT_LEDGER_NAME *= 'synonyms.jsonl'*

The ledger’s file name inside [`ir.config.config_dir()`](ir.config.html.md#ir.config.config_dir).

### ir.synonyms.DFLT_MAX_VARIANTS *= 6*

Default cap on the number of query variants (the original included).

### ir.synonyms.SHORT_TERM_CHARS *= 4*

Terms at most this long that are plain word characters get `\b` guards in a
regex, so `ac` does not match inside `react`.

### *class* ir.synonyms.SynonymEntry(terms, oneway=False, note='', added='', corpus=None, provenance=None, verified=0, status='active', source_line='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One ledger line: equivalent `terms` (or `oneway` from the first).

#### source_line *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= ''*

the ledger line this entry was read from (`""` if built in code); lets
[`update_entries()`](#ir.synonyms.update_entries) patch exactly that line and leave the rest verbatim.

### ir.synonyms.add_synonyms(terms, , note='', oneway=False, corpus=None, provenance=None, path=None)

Append one entry to the ledger (creating it) and return it.

`provenance` records the miss that prompted the entry
(`{"miss_query": ..., "found_ids": [...]}`, optionally `"fix_query"`) so a
later replay can verify the entry still helps.

Add an entry only after a verified find that the corpus-specific word made
possible; general-language synonyms are the searcher’s job, not the ledger’s.

* **Return type:**
  [`SynonymEntry`](#ir.synonyms.SynonymEntry)

### ir.synonyms.aliases(term, entries)

`term` first, then every term the ledger makes equivalent to it (deduped).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.synonyms.expand_query(query, entries, , max_variants=6)

`query` first, then variants with one known term swapped for an alias.

Terms are matched as whole words, case-insensitively, separator-tolerant.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.synonyms.ledger_path()

Where the ledger lives: `$IR_SYNONYMS`, else `<config_dir>/synonyms.jsonl`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### ir.synonyms.load_synonyms(path=None, , corpus=None, include_demoted=False)

The ledger’s entries (those for all corpora, plus `corpus`’s own).

Demoted entries are skipped unless `include_demoted`.

A missing ledger is an empty one; a malformed line is skipped, so a hand edit
gone wrong never breaks search.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SynonymEntry`](#ir.synonyms.SynonymEntry)]

### ir.synonyms.make_synonym_formulator(entries=None, , path=None, corpus=None, max_variants=6)

A deterministic [`ir.formulate.Formulator`](ir.formulate.html.md#ir.formulate.Formulator) that expands from the ledger.

With `entries=None` the ledger is re-read on every call, so an entry an agent
just added applies to its next search. Returns the original query alone when
nothing in it is in the ledger (identity, the seam’s default behavior).

### ir.synonyms.term_regex(terms)

One alternation matching any of `terms`, tolerant of `-`/`_`/space.

Longest first (so a regex engine prefers the full phrase); short plain words get
word boundaries.

```pycon
>>> term_regex(["db", "data base"])
'data[-_ ]?base|\\bdb\\b'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.synonyms.update_entries(changes, path=None)

Rewrite the ledger with `(old, new)` entry changes; every other line stays verbatim.

`old` must come from [`load_synonyms()`](#ir.synonyms.load_synonyms) (it carries the line it was read
from). Only `verified` and `status` are taken from `new`, and they are
patched into the original JSON object, so comments, malformed lines and unknown
keys a user hand-wrote survive. A line that changed since it was read (an edit
or another writer) is left alone. The file is replaced atomically (unique temp
file, then rename); an `ir synonyms-add` append landing between the final read
and the rename could still be lost, so keep replays out of concurrent writers.
Returns the number of lines patched.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
