# ir_10 — LLM Routing: running agentic search on a Claude subscription

**Question.** ir's optional agentic components use an LLM that, by default, bills
a pay-as-you-go API. When ir runs inside a context that *already has* a Claude
subscription — a Claude.ai/Desktop **MCP connector**, or a **Claude Code** session
— can that LLM run on the subscription instead, to avoid the metered API?

**Short answer.** Yes in principle, and **ir itself needs zero changes**: every
LLM call-site is opt-in and accepts an *injected plain callable*. The work is a
small adapter on the *host* side, and — for the connector path — one genuine
unknown: whether the connected client honours MCP *sampling*.

---

## 1. ir is offline by default; the LLM is opt-in and injectable

ir's core retrieval (dense / lexical / hybrid + selection + abstention) is **fully
offline** — no LLM, no network, no key. The LLM appears only in *optional* agentic
enhancements, and **every one injects a plain callable**; the default
(`aix`-backed) path is lazy-imported *only when the callable is omitted*, so
`import ir` stays offline.

| Seam | Factory (file:line) | Inject via | Callable shape |
|---|---|---|---|
| Query reformulation | `formulate.py:66` `make_llm_formulator` | `rewriter=` | `query:str → str \| Sequence[str]` (`Formulator`, `formulate.py:33`) |
| Synopsis at index time | `synopsis.py:97` `make_llm_synthesizer`, `synopsis.py:220` `with_synopsis` | `summarize=` / `synthesize=` | `text:str → str` / `Artifact → str` (`synopsis.py:61`) |
| LLM-based selection | `select.py:433` `make_llm_selector` | `chooser=` | `(query=, candidates=) → Sequence[str]` (chosen ids) |
| Eval-set generation | `eval_gen.py:264` `generate_cases`, `:386` `build_eval_set` | `query_generator=` / `abstention_generator=` | `(description, *, n) → Sequence[str]` (`eval_gen.py:70`) |

**ir never hardwires a provider.** Inject a callable backed by *whatever* runs the
completion, and ir routes its LLM there.

> **Doc note:** the README historically said the LLM extra is `oa`. That is drift
> — ir's `[llm]` extra is **`aix`** (`pyproject.toml`), a LiteLLM facade whose
> default is `anthropic/claude-sonnet-4` via `ANTHROPIC_API_KEY` (i.e. metered).
> `oa` (OpenAI-only) is *not* used by ir. Fixed in the README alongside this note.

## 2. The three routing targets

A subscription-routed backend is just a different `prompt → str` callable injected
into the seams above.

**(a) MCP connector → the Claude.ai/Desktop client's subscription, via *sampling*.**
MCP defines `sampling/createMessage`: a *server* tool can ask the connected
*client* to run a completion **on the client's model/subscription**. FastMCP
exposes it as `ctx.sample(...)` (`fastmcp/server/context.py:940`), and **py2mcp
needs no change** — FastMCP auto-injects the `Context` by type annotation
(`fastmcp/tools/function_parsing.py:133`), so a tool that declares
`ctx: fastmcp.Context` simply gets it. The **gate** is the client: sampling is only
available if the connected client advertises a `sampling_callback`
(`mcp/client/session.py:149`). Whether **Claude.ai remote connectors** honour
sampling is the one *must-verify* item (historically it has been a Desktop / local
feature). If they don't, this path raises and you fall back to (c) or a metered key.

**(b) Claude Code → the host session, via the Agent SDK.** Exists today. The
`claude_agent_sdk` runs completions on the host's auth (no metered key), and
**`coact`** already wraps it: `coact.realize(target, backend="sdk")`
(`coact/realize.py:57`) over `ClaudeSDKClient`, plus a bare
`LLMCallable = Callable[[str], str]` / `resolve_llm(...)` (`coact/llm.py:26`). A
thin single-turn `prompt → str` over that, injected into ir's `rewriter=` /
`chooser=` / `synthesize=`, routes ir's LLM through Claude Code.

**(c) Claude Desktop → sampling.** Same transport and glue as (a); same
client-support caveat.

## 3. Verdict

| Path | Status | Glue |
|---|---|---|
| Connector → Claude.ai (sampling) | transport **exists**; client support **must-verify** | `ctx: Context` tool + sync↔async `prompt→str` over `ctx.sample`, injected into ir seams |
| Claude Code → host (Agent SDK) | **exists today** | `prompt→str` over a single-turn `ClaudeSDKClient` / `coact.resolve_llm`, injected |
| Claude Desktop → sampling | transport **exists**; client support **must-verify** | identical to the connector row |

**Constant:** ir changes = **zero**. The adapter is a host-side concern and belongs
on the connector / agent side (it depends on `fastmcp.Context` or
`claude_agent_sdk`, not on ir). ir stays backend-agnostic.

## 4. Implications for connectors

- A **basic ir connector is free** — offline search, no LLM, no tokens. Make this
  the default; reach for the agentic layer only when reformulation / LLM-selection
  / synopsis earns its keep.
- An **agentic ir connector** can ride the user's Claude subscription via sampling
  **iff** the client supports it — otherwise it needs a metered key or the search
  degrades gracefully to offline. Until Claude.ai sampling support is confirmed,
  treat subscription-routing as *opt-in and capability-gated*, not the default.

See the planning issues in `ir` and `enlace_connector` for the adapter to build and
the client-support verification.
