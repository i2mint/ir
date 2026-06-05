# Progressive Capability Discovery for AI Agents: Give the Agent One Search Tool

*A comparative landscape survey of tool, skill, and subagent discovery architectures (2024 H2 – 2026)*

**Author: Thor Whalen**

---

## TL;DR

- **The single-search-tool pattern is now validated by primary-source numbers, not just intuition.** Anthropic's own internal testing shows tool-selection accuracy on MCP evaluations jumping from 49% to 74% (Opus 4) and 79.5% to 88.1% (Opus 4.5) when its Tool Search Tool is enabled, with an 85% token reduction; independent papers (RAG-MCP, Toolshed, ScaleMCP) show 3× accuracy gains and >50% token cuts. The pattern is real and worth adopting above ~30–50 tools.
- **Architecture should be staged, not maximal.** Single-shot semantic retrieval + a reranker is the right default; the planner→fan-out→selector multi-stage pipeline (Toolshed's "Advanced RAG-Tool Fusion") buys measurable recall gains but is over-engineered for most catalogs under a few hundred tools. Retrieval — not the LLM's final selection — is the dominant failure mode (LiveMCPBench: "retrieval errors account for nearly half of all failures").
- **Tools, skills, and subagents share one discovery primitive (metadata-first progressive disclosure) but diverge at the execution boundary.** A unified index can serve all three for *retrieval*; the difference is what gets loaded (a schema, a SKILL.md document, or an agent card). Watch the real gotchas: tool-description quality is the actual bottleneck, dynamic tool injection silently breaks prompt caching, and dynamic loading opens rug-pull/tool-poisoning attack surface.

---

## Key Findings

1. **Context degradation is the underlying physics.** Chroma's 2025 "context rot" study (Hong, Troynikov & Huber, July 14, 2025) evaluated 18 LLMs "including the state-of-the-art GPT-4.1, Claude 4, Gemini 2.5, and Qwen3 models" and found "their performance grows increasingly unreliable as input length grows" — well before the window is full — with distractors having outsized, non-uniform impact. This is *why* loading 50–100 tool schemas hurts: it is not just token cost, it is signal-to-noise collapse.

2. **The "49% baseline" family of findings is robust and converging.** RAG-MCP (arXiv 2505.03275, Gan & Sun) "significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline)." Anthropic's production numbers (49%→74%) corroborate the same shape. The breakpoint where the pattern starts paying off is widely cited at 30–50 tools.

3. **Retrieval is the bottleneck, not selection.** LiveMCPBench (70 servers, 527 tools) finds Claude-Sonnet-4 at 78.95% task success while "most models achieve only 30-50%," and "retrieval errors account for nearly half of all failures—highlighting retrieval as the dominant bottleneck." MCP-Universe (Salesforce, 231 tasks) finds even the best model (GPT-5, 43.72%) fails the majority of realistic multi-tool tasks.

4. **Skills and the single-search-tool pattern are the same idea at different layers.** Anthropic's Agent Skills (SKILL.md) implement three-level progressive disclosure — metadata (~30–50 tokens) → SKILL.md body (<5k tokens) → bundled files — which is exactly the defer-then-load shape of Tool Search.

5. **Prompt caching is the hidden cost of dynamic loading.** Anthropic's cache hierarchy is tools → system → messages; changing any tool definition invalidates the entire downstream cache. Naively injecting discovered tools mid-conversation therefore destroys cache hits. Pydantic AI's native tool-search path is explicitly designed to avoid this by keeping discovery append-only.

6. **Security is underweighted.** Dynamic tool registries enable rug-pull and tool-poisoning attacks (Invariant Labs, OWASP GenAI). The more dynamic your discovery, the larger the attack surface — and the more you need signed/versioned tool definitions (ETDI) and host-side allow-listing.

---

## Details

### 1. The single-search-tool pattern in practice

**Anthropic Tool Search Tool + `defer_loading`.** Released in beta on November 24, 2025 as part of "advanced tool use," the pattern is: provide all tool definitions to the API but mark most with `defer_loading: true`. Deferred tools are not loaded into context; Claude sees only the search tool plus your 3–5 always-on tools. Two variants ship: `tool_search_tool_regex_20251119` (Python `re.search` patterns) and `tool_search_tool_bm25_20251119` (natural-language BM25). Anthropic also documents a client-side embeddings path using `tool_reference` content blocks, so you can swap in your own vector retriever while keeping the same wire protocol.

The measured impact, from Anthropic's own internal testing (first-party, self-reported — treat as vendor claims): tool-selection accuracy on MCP evaluations rose from **49% to 74% on Opus 4** and **79.5% to 88.1% on Opus 4.5**, with the search tool preserving **191,300 tokens vs 122,800** under the traditional approach — an **85% reduction in token usage**. Anthropic notes it has "seen tool definitions consume 134K tokens before optimization," and a representative five-server setup is 58 tools / ~55K tokens before the conversation begins.

The companion **Programmatic Tool Calling** lets Claude orchestrate tools in a code-execution sandbox so intermediate results never hit context: average usage "dropped from 43,588 to 27,297 tokens, a 37% reduction on complex research tasks," with accuracy on internal knowledge retrieval improving 25.6%→28.5% and GAIA-style benchmarks 46.5%→51.2%. The related **code-execution-with-MCP** pattern reports a headline case of 150,000 → ~2,000 tokens (98.7% reduction) by turning MCP servers into code-level APIs.

**Independent corroboration on production readiness is mixed.** One marketing-automation practitioner (Growth Method) reported only ~60% retrieval accuracy in their own testing and judged it "not production-ready" for high-stakes actions — a useful counterweight to vendor numbers.

**MCP progressive disclosure proposals.** Importantly, progressive disclosure is *not* in the MCP spec — it is a pattern layered on top of `tools/list`, `tools/call`, and `notifications/tools/list_changed`. Active proposals: **SEP-1888** (Harshal Patil's "Progressive Disclosure for Typed Library Discovery & Introspection," reference impl ProDisco) proposes a standard `<library>.searchTools` meta-tool with `operations`/`types` modes; **SEP-1576** proposes embedding-based host-side tool filtering; and a community MCP "Progressive Disclosure" extension moves full tool descriptions into MCP resources fetched on demand. A widely cited MCP discussion measured a MySQL MCP server with 106 tools consuming 207KB / ~54,600 tokens on every initialization. Claude Code began rolling out MCP Tool Search (v2.1.7) that auto-triggers when MCP tool descriptions would exceed 10% of context.

**Equivalent patterns across ecosystems:**
- **OpenAI Agents SDK** added hosted **Tool Search** (`ToolSearchTool()`, requires `openai>=2.25.0`): mark function tools `defer_loading=True`, group with `tool_namespace(...)`, or defer `HostedMCPTool`. OpenAI's official guidance is "use namespaces where possible" rather than many individually deferred functions.
- **LangGraph** ships **`langgraph-bigtool`**, a library that stores tool metadata in LangGraph's long-term-memory store (in-memory or Postgres) and equips the agent with a `retrieve_tools` search tool; LangChain demonstrated it working reliably with >50 tools on local models via Ollama.
- **Pydantic AI** exposes `defer_loading=True` on tools/toolsets and `.defer_loading()` on MCP servers and `FastMCPToolset`, with an auto-injected `ToolSearch` capability that uses native provider search (Anthropic, OpenAI Responses) when available and a custom callable strategy otherwise.
- **smolagents** is tool-agnostic (MCP, LangChain, Hub Spaces as tools) but has no built-in semantic tool-retrieval layer — discovery is left to the developer.

### 2. Architecture of the search tool itself

Four reference architectures, in increasing complexity:

**(a) Flat single-shot semantic retrieval.** Embed tool docs once, embed the query, return top-k. This is RAG-MCP's core design and `langgraph-bigtool`'s default. Cheapest, lowest latency, and already captures most of the gain — RAG-MCP's jump to 43.13% accuracy (from a 13.62% blank-conditioning baseline and 18.20% actual-match), while cutting average prompt tokens to 1,084 from 2,133.84, comes from this alone. The verdict from the evidence: this is the correct default for catalogs up to a few hundred tools.

**(b) Retrieve-then-rerank.** Add a cross-encoder/LLM reranker over the top-k candidates. Toolshed (arXiv 2410.14594) folds this into "post-retrieval" corrective strategies. Worth the extra latency when tool descriptions are similar/confusable (the `notification-send-user` vs `notification-send-channel` problem Anthropic flags).

**(c) Agentic/iterative search (ReAct-style).** The agent searches, inspects, optionally re-searches. This is LiveMCPBench's MCP Copilot Agent and Pydantic AI's "search → discover → call → search again" loop. Necessary when a single query cannot express the need (multi-hop tasks), but it reintroduces latency and per-iteration inference cost.

**(d) Planner → fan-out → selector decomposition.** A query-planner subagent decomposes the request, parallel searches fan out, selector/filter subagents read enriched candidate results, and a manager makes the final selection. This is the spirit of Toolshed's "Advanced RAG-Tool Fusion" (pre-/intra-/post-retrieval ensemble), which reports **46%/56%/47% absolute Recall@5 improvements** on ToolE-single, ToolE-multi, and Seal-Tools respectively — without fine-tuning. HGMF (arXiv 2508.07602) offers a cheaper middle path: hierarchical Gaussian-mixture clustering of servers then tools.

**Verdict (opinionated):** The multi-stage pipeline is *state of the art on recall metrics* but *over-engineered relative to single-shot + rerank* for the typical practitioner. The marginal recall gains are real but come at multiplied latency, token cost, and operational complexity (multiple models, multiple failure points). Adopt (a)+(b) first; escalate to (c) only for genuinely multi-hop tasks; reserve (d) for very large (thousands of tools) catalogs where Recall@5 is business-critical. ScaleMCP's contribution is orthogonal and arguably more valuable than pipeline depth: its **Tool Document Weighted Average (TDWA)** embedding strategy and CRUD auto-synchronization (MCP servers as single source of truth) address index *freshness*, which matters more in production than squeezing recall.

### 3. Selection vs. retrieval distinction

Once candidates are retrieved, final selection strategies and their failure modes:

- **Top-k cutoff.** Simplest; Toolshed explicitly tunes top-k against the tool-count (tool-M) to trade recall vs. token cost. Failure mode: a fixed k either over-selects (distractors) or misses the needed tool when the right tool ranks k+1.
- **LLM-as-selector.** The model reasons over retrieved candidates and commits. Failure mode: distractor tools. The "Tool Preferences in Agentic LLMs are Unreliable" paper (arXiv 2505.18135) shows selection can be swung by trivial description edits — assertive phrasing alone disproportionately boosts a tool's selection rate, making the protocol "exploitable." BiasBusters (arXiv 2510.00307) documents the same metadata/ordering bias.
- **Confidence thresholds.** Return only candidates above a similarity threshold. Failure mode: brittle calibration across query types (Chroma shows needle-question similarity has non-uniform effects).
- **Learned routers / intent classification.** GeckOpt maps tasks to intents offline then restricts the toolset; AutoTool (arXiv 2511.14650) argues many tool selections are patterned enough to bypass full LLM reasoning. Failure mode: requires maintenance as the catalog drifts.

The empirically grounded conclusion: **the distractor problem is the central selection risk**, and it is worsened, not solved, by retrieving more candidates. MCP-Atlas (Scale AI) deliberately injects 5–10 distractors per task alongside 3–7 target tools precisely to measure this. Fewer, higher-precision candidates beat more candidates.

### 4. Tools vs. skills vs. subagents — different discovery strategies?

The three capability types:
- **Tool** = a typed callable with a JSON schema (name, description, input schema).
- **Skill** = a document folder (SKILL.md + optional scripts/references/assets) following the agentskills.io open standard.
- **Subagent** = a delegatable capability with its own context window, discovered via registries or A2A agent cards.

**Discovery primitive is shared; execution boundary differs.** All three use *metadata-first progressive disclosure*: load a lightweight descriptor (tool name+description ~200–800 tokens; SKILL.md frontmatter ~30–50 tokens; A2A agent card JSON) and load the heavy payload only on activation. A single vector index *can* serve retrieval across all three — you are embedding descriptions either way.

Where they diverge:
- **Skills** load a *document* into context (the SKILL.md body, then referenced files), and can bundle executable scripts the agent runs without reading into context. Discovery is fundamentally about *when to read more*, governed entirely by the quality of the name+description. The format is an open standard with **32 adopters as of March 2026** per agentskills.io — Anthropic published the spec on December 18, 2025, and within 48 hours Microsoft (VS Code) and OpenAI (ChatGPT, Codex CLI) added support; adopters also include Google (Gemini CLI), JetBrains (Junie), Sourcegraph (Amp), Block (Goose), Snowflake, Databricks, ByteDance, Mistral AI, and Spring AI. It is now governed under the Linux Foundation's Agentic AI Foundation (AAIF), which had grown to 146 member organizations by February 2026 — important reassurance for lock-in-averse readers.
- **Tools** load a *schema* the model must populate correctly; the discovery payoff is both context savings and parameterization accuracy.
- **Subagents** are delegated to, not loaded. **A2A** (Agent2Agent, now under the A2A Project) standardizes the **Agent Card** — a JSON descriptor with identity, service endpoint, capabilities, skills, and auth schemes — discoverable via a well-known URI or a central **registry/catalog** that clients query by skill/tag/capability. Delegation routing in the OpenAI Agents SDK is via **handoffs** (represented to the LLM as `transfer_to_<agent>` tools, with `handoff_description` as the discovery hint) vs. **agent-as-tool** (the manager retains control and calls subagents as tools). This is the cleanest illustration that *subagent discovery is just tool discovery with a different execution semantics* — a handoff is literally surfaced to the model as a tool.

**Verdict:** Use one unified retrieval index for discovery, but keep three distinct loaders. Treat subagent discovery as a first-class peer: register agent cards in the same searchable catalog as tools and skills, route via handoff-as-tool, and let the manager's final selection logic be identical to tool selection.

### 5. Latency, caching, and the cold-path problem

**The prompt-cache interaction is the most underappreciated production gotcha.** Anthropic's cache is a prefix cache built strictly in order tools → system → messages. Cache hits require byte-identical prefixes. Therefore:
- Changing *any* tool definition invalidates the tools cache *and* everything downstream (system + messages). Adding an MCP tool mid-session can 5× the cost of that turn.
- Non-deterministic tool serialization (dict/set ordering across Python runs) silently creates cache misses; always serialize tools in a fixed, tested order.
- The naive dynamic-loading anti-pattern — discover a tool, then inject its full definition into the tools array — invalidates the cache on every discovery turn.

Anthropic's `defer_loading` is designed to dodge this: the system expands `tool_reference` blocks in the *message* history (the cheap, append-only end of the prefix) rather than mutating the tools block, "so Claude can reuse discovered tools in subsequent turns without re-searching." Pydantic AI documents this explicitly: native provider search keeps discovery append-only and preserves the cache, whereas its local fallback (flipping `defer_loading=False` between rounds) "changes the tool-definition prefix and invalidates the cached request prefix on every discovery turn." Newer models (Opus 4.5+, Sonnet 4.6+) also support mid-conversation system messages that don't invalidate the cached prefix.

**Practical guidance:** keep your 3–5 highest-frequency tools always-loaded and stable; do discovery *once per session* where possible rather than per-turn; place per-request/dynamic data *after* the last cache breakpoint; and treat model switches as a hard cache boundary. Cache reads cost 10% of base input; writes cost 25% more — so a thrashing cache is strictly worse than no cache.

### 6. Libraries and frameworks (Python-first) — maturity, license, lock-in

| Project | What it does | Maturity | License | Lock-in risk |
|---|---|---|---|---|
| **`langgraph-bigtool`** (LangChain) | Semantic tool retrieval via LangGraph long-term store; in-memory/Postgres backends; custom retrieval fns | Released, maintained, narrow scope | MIT (OSI) | Low–medium: tied to LangGraph runtime but components swappable; Postgres backend is standard |
| **Pydantic AI toolsets** (`defer_loading`, `ToolSearch`, `FastMCPToolset`) | Native tool-search across providers + custom strategies; MCP via FastMCP | Actively developed, production-oriented | MIT (OSI) | Low: provider-agnostic, custom search callable, swappable models |
| **RAG-MCP** (reference impls: memoverflow, fintools-ai) | Tool-RAG over MCP via vector index | Research-grade / community | Open (varies by repo) | Low: pattern, not a platform |
| **ScaleMCP** (Lumer et al.) | Auto-synchronizing MCP tool retriever, TDWA embeddings, CRUD index | Research paper + concepts | Paper (no canonical OSI lib) | N/A (pattern) |
| **Toolshed / Advanced RAG-Tool Fusion** (Lumer et al.) | Tool knowledge base + pre/intra/post-retrieval ensemble | Research paper + sample code | Open sample code | Low (pattern) |
| **agentgateway** (Solo.io) | MCP gateway with progressive disclosure (`get_tool`/`invoke_tool`), reported 91% token cut | Vendor OSS | Open-source core | Medium: gateway is infra you operate; pattern portable |
| **FastMCP** | MCP server/client framework, Streamable HTTP transport | Mature, widely adopted | Apache-2.0 (OSI) | Low: standard MCP, swappable |
| **Anthropic Tool Search Tool / Skills API** | Native defer-loading + sandboxed skills | GA-track beta | Proprietary API (Skills *format* is open) | **High for the API; low for SKILL.md format** |
| **OpenAI Agents SDK ToolSearch** | Hosted tool search, namespaces, MCP | Released (`openai>=2.25.0`) | Apache-2.0 SDK, proprietary hosted search | Medium–high: hosted search is OpenAI-only |

**Lock-in verdict for a lock-in-averse reader:** The maximally swappable stack is FastMCP (Apache-2.0) for transport + Pydantic AI (MIT) for the agent/toolset layer with a *custom* `ToolSearch` strategy backed by your own embeddings/vector DB, falling back to provider-native search only as an optimization. The SKILL.md format is safe to adopt (open standard, 32 adopters, Linux Foundation governance). Anthropic's Tool Search Tool and OpenAI's hosted ToolSearch are the highest lock-in components — use them behind an abstraction so the discovery layer can be re-pointed at an OSS retriever. `langgraph-bigtool` is fine if you are already on LangGraph but couples you to that runtime.

### 7. The questions not being asked but should be

1. **Tool-description quality is the real bottleneck.** Both the BM25 and embedding paths are only as good as the descriptions. "Tool Preferences in Agentic LLMs are Unreliable" shows selection can be gamed by description edits; Anthropic's own guidance stresses documenting return formats and using clear, natural-language descriptions. Most "retrieval failures" are description failures. This deserves a dedicated quality bar and review process — descriptions are now load-bearing infrastructure, not docs.

2. **Embedding drift and index versioning.** When the tool catalog changes, embeddings computed with one model become inconsistent if you later switch embedding models, and stale indexes mis-route. ScaleMCP's CRUD auto-sync (servers as single source of truth) is the only project explicitly treating this. You need: versioned tool indexes, a re-embedding strategy on catalog change, and a pinned embedding model with migration discipline.

3. **Evaluation of discovery quality itself.** Recall@k and tool-selection accuracy (BFCL, LiveMCPBench, MCP-Universe, MCP-Atlas) measure end-task success but rarely isolate *retrieval* quality from *selection* quality. LiveMCPBench's finding that ~half of failures are retrieval errors is the exception. Inspect AI (UK AISI) ports BFCL and is a strong OSS harness for building discovery-specific evals; instrument retrieval recall separately from final-answer accuracy.

4. **Security: dynamic loading widens the attack surface.** Tool poisoning (malicious instructions in descriptions/schemas), rug pulls (a server swaps a clean tool for a malicious one after approval), and cross-server shadowing (a malicious server registers a trusted tool's name) all exploit dynamic registries — and clients typically don't re-prompt on description changes. Defenses: host-side allow-listing, static manifest scanning (Invariant Labs' mcp-scan), runtime sandboxing, and signed/versioned definitions (ETDI, arXiv 2506.01333, using OAuth-issued signed JWTs). The more dynamic your discovery, the more these matter.

5. **Multi-tenant catalog isolation.** Largely unaddressed in the literature. If one index serves multiple tenants, retrieval must be tenant-scoped (filtered vector search), descriptions must not leak across tenants, and per-tenant tool subsets must be enforced before anything reaches the model. This is an open problem practitioners hit immediately in SaaS deployments.

6. **The CLI-and-skills alternative.** A recurring thread in MCP discussions argues the answer to tool bloat isn't "5 core tools + progressive discovery" but "expose a CLI + teach workflows via skills." For some domains this is genuinely simpler and gives better progressive disclosure than MCP — worth evaluating before committing to a retrieval pipeline.

---

## Recommendations

**Stage 1 — Adopt the pattern once you cross ~30 tools.** Below ~30 tools, load everything; the retrieval overhead isn't worth it. Above it, the 49%→74%-class accuracy gains and 85%-class token savings justify a search tool. Start with single-shot semantic retrieval + a reranker (architectures a+b). Keep 3–5 high-frequency tools always-loaded.

*Threshold to escalate:* if Recall@5 on a held-out eval drops below ~0.8 or distractor-induced wrong-tool selections exceed your error budget, move to Stage 2.

**Stage 2 — Add iterative search and description hardening.** Introduce ReAct-style re-search for multi-hop tasks. Simultaneously, treat tool descriptions as load-bearing: enforce a description style guide, add consistent prefixes (`github_*`, `slack_*`), and put user-search keywords in descriptions. Most accuracy gains here come from descriptions, not pipeline depth.

*Threshold to escalate:* only if you exceed ~1,000 tools and Recall@5 is business-critical should you consider the full planner→fan-out→selector pipeline (Stage 3).

**Stage 3 — Multi-stage pipeline only for very large catalogs.** Adopt Toolshed-style pre/intra/post-retrieval fusion or HGMF hierarchical clustering. Pair with ScaleMCP-style CRUD index auto-sync. Budget for the multiplied latency and operational complexity.

**Cross-cutting, do from day one:**
- **Protect the cache.** Serialize tools in a fixed order (unit-test it); do discovery once per session; never mutate the tools block mid-conversation — use append-only `tool_reference`/native search. Prefer provider-native search *behind an abstraction* so you can swap to your own retriever.
- **Build the swappable stack:** FastMCP (Streamable HTTP) + Pydantic AI + a custom `ToolSearch` strategy over your own vector DB; adopt SKILL.md for skills; treat subagent cards (A2A) as first-class entries in the same index, routed via handoff-as-tool.
- **Instrument discovery evals separately** (Inspect AI): measure retrieval recall and selection accuracy as distinct metrics.
- **Secure the registry:** allow-list servers, scan manifests (mcp-scan), version and ideally sign tool definitions (ETDI), and enforce tenant-scoped retrieval.

---

## Caveats

- **Vendor numbers are self-reported.** Anthropic's 49%→74%, 85%-reduction, and PTC 37%-reduction figures are first-party internal tests with no published methodology or sample sizes. The independent 60%-retrieval-accuracy counter-report and the RAG-MCP/Toolshed academic numbers (which *do* publish methods) are the more trustworthy anchors for the *shape* of the gains; treat absolute percentages as indicative, not guaranteed.
- **Benchmarks measure end tasks, not discovery in isolation.** BFCL, LiveMCPBench, MCP-Universe, and MCP-Atlas conflate retrieval and selection quality except where explicitly separated. Leaderboards also move fast (MCP-Universe's top entry shifted from GPT-5 43.72% in the paper to Gemini-3-Pro-Preview 44.59% on the live board); cite the version.
- **The pattern is young.** `defer_loading` shipped in beta in late November 2025; framework support (LangChain, Pydantic AI, OpenAI SDK) was still landing through early 2026, and progressive disclosure is not yet in the MCP spec. Expect API churn.
- **"Progressive disclosure" is overloaded.** It refers to tool search, SKILL.md three-level loading, MCP resource-based lazy descriptions, and UI design simultaneously; verify which layer a given source means.
- **Some sources are vendor blogs and Medium posts.** Where used (Solo.io, Arcade, Unified.to, practitioner Medium/DEV posts), they corroborate primary docs and papers but carry promotional or anecdotal bias; the load-bearing claims here are anchored to arXiv papers and official Anthropic/OpenAI/Pydantic/MCP/A2A documentation.

---

## References

[1] [Introducing advanced tool use on the Claude Developer Platform — Anthropic](https://www.anthropic.com/engineering/advanced-tool-use)
[2] [Tool search tool — Claude API Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
[3] [RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation (arXiv 2505.03275)](https://arxiv.org/abs/2505.03275)
[4] [Toolshed: Scale Tool-Equipped Agents with Advanced RAG-Tool Fusion and Tool Knowledge Bases (arXiv 2410.14594)](https://arxiv.org/abs/2410.14594)
[5] [ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents (arXiv 2505.06416)](https://arxiv.org/abs/2505.06416)
[6] [LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? (arXiv 2508.01780)](https://arxiv.org/abs/2508.01780)
[7] [MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers (arXiv 2508.14704)](https://arxiv.org/abs/2508.14704)
[8] [MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers — Scale AI](https://scale.com/blog/open-sourcing-mcp-atlas)
[9] [Context Rot: How Increasing Input Tokens Impacts LLM Performance — Chroma](https://www.trychroma.com/research/context-rot)
[10] [langgraph-bigtool — GitHub (LangChain)](https://github.com/langchain-ai/langgraph-bigtool)
[11] [Equipping agents for the real world with Agent Skills — Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
[12] [Agent Skills Overview — agentskills.io](https://agentskills.io/home)
[13] [Agent Discovery — Agent2Agent (A2A) Protocol](https://a2a-protocol.org/latest/topics/agent-discovery/)
[14] [Handoffs — OpenAI Agents SDK](https://openai.github.io/openai-agents-python/handoffs/)
[15] [Tools — OpenAI Agents SDK](https://openai.github.io/openai-agents-python/tools/)
[16] [Toolsets — Pydantic AI Docs](https://ai.pydantic.dev/toolsets/)
[17] [Advanced Tool Features — Pydantic AI Docs](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/)
[18] [Prompt caching — Claude API Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
[19] [Code execution with MCP: building more efficient AI agents — Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)
[20] [SEP-1888: Progressive Disclosure for Typed Library Discovery & Introspection — MCP GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1888)
[21] [MCP Tool Schema Bloat: The Hidden Token Tax — Layered System](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/)
[22] [Tool Preferences in Agentic LLMs are Unreliable (arXiv 2505.18135)](https://arxiv.org/abs/2505.18135)
[23] [Less is More: Optimizing Function Calling for LLM Execution on Edge Devices (arXiv 2411.15399)](https://arxiv.org/abs/2411.15399)
[24] [Berkeley Function Calling Leaderboard (BFCL) — Gorilla](https://gorilla.cs.berkeley.edu/leaderboard.html)
[25] [MCP Security Notification: Tool Poisoning Attacks — Invariant Labs](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)
[26] [ETDI: Mitigating Tool Squatting and Rug Pull Attacks in MCP (arXiv 2506.01333)](https://arxiv.org/abs/2506.01333)
[27] [HGMF: A Hierarchical Gaussian Mixture Framework for Scalable Tool Invocation within MCP (arXiv 2508.07602)](https://arxiv.org/abs/2508.07602)
[28] [MCP Progressive Disclosure: Save Tokens, Retrieve Schemas — Solo.io](https://www.solo.io/blog/mcp-progressive-disclosure)
[29] [Anthropic's Tool Search: Not Ready for Production Marketing Workflows — Growth Method](https://growthmethod.com/anthropic-tool-search/)
[30] [smolagents — GitHub (Hugging Face)](https://github.com/huggingface/smolagents)
[31] [FastMCP Client — Pydantic AI Docs](https://pydantic.dev/docs/ai/mcp/fastmcp-client/)
[32] [AutoTool: Efficient Tool Selection for Large Language Model Agents (arXiv 2511.14650)](https://arxiv.org/abs/2511.14650)
[33] [BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models (arXiv 2510.00307)](https://arxiv.org/abs/2510.00307)
[34] [Progressive Disclosure for Typed Library Discovery & Introspection — MCP Discussion #631](https://github.com/orgs/modelcontextprotocol/discussions/631)

---

*Note for the reader: This report is delivered as Markdown and can be saved directly as a `.md` file (e.g., `progressive-capability-discovery.md`). All references include live hyperlinks in `[name](url)` format per Vancouver-style numbering.*