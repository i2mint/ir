# ir.tools

Agent-callable tool surface over `ir` — plain functions returning JSON-ready
dicts, deliberately MCP/HTTP-agnostic.

`ir` knows nothing about MCP, HTTP, or any agent host. This module is the SSOT
for “expose ir’s retrieval as a single agent-callable tool”: a wrapper
(`py2mcp`, `qh`, a hand-written agent tool) references e.g. `ir.tools:search`
and gets a clean JSON `dict` back. The corpus is a **parameter**, so one
function serves every corpus with no per-corpus code — and [`make_search()`](#ir.tools.make_search)
returns a *corpus-bound* tool when a connector should expose exactly one corpus.

This pairs with `ir.discover` (which `ir` already calls “the single
agent-callable tool”); [`search()`](#ir.tools.search) is just its JSON-returning, tool-shaped
front door.

### Functions

| [`search`](#ir.tools.search)(query, \*, corpus[, k, mode, filter])      | Search a named `ir` corpus and return a JSON-serializable result dict.   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`make_search`](#ir.tools.make_search)(corpus, \*[, name, description, ...]) | Return a **corpus-bound** `search(query, k=...) -> dict` tool.           |

### ir.tools.make_search(corpus, , name=None, description=None, k=8, mode='hybrid')

Return a **corpus-bound** `search(query, k=...) -> dict` tool.

The returned function exposes only `query` (and `k`) — the corpus is fixed
— so a connector built over it surfaces exactly one corpus and nothing else.
Its `__name__` / `__doc__` are set so an MCP/agent host shows a clean tool
name and description. Use this when wiring a single-corpus connector; use
[`search()`](#ir.tools.search) when the caller should choose the corpus.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### ir.tools.search(query, , corpus, k=8, mode='hybrid', filter=None)

Search a named `ir` corpus and return a JSON-serializable result dict.

Thin agent-callable wrapper over [`ir.discover()`](ir.html.md#ir.discover) — returns its
`.to_dict()` (committed results, scores, disclosures), fit to hand straight
back from an MCP tool or HTTP endpoint. The *corpus* is a parameter, so this
single function serves any corpus.

* **Parameters:**
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the natural-language query.
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a registered corpus **name** (str), a list of names (federated
    search), or a built [`ir.Corpus`](ir.html.md#ir.Corpus).
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum number of results.
  * **mode** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"dense"` | `"lexical"` | `"hybrid"`.
  * **filter** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `vd` Mongo-style metadata filter (hard pre-filter).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
