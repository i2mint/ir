# A Composable Search Agent: State-of-the-Art Architecture and a Role-Oriented Design

**Author:** Thor Whalen
**Date:** 2026-06-11
**Status:** Design draft

---

## 1. Scope and intent

This document specifies the architecture for a **Search Agent**: an agent whose job is to find resources relevant to a query, across one or more user-parametrized corpora (internet, local files, datasets, SQL stores, ...). The agent is responsible for:

1. **Source selection** — choosing which corpora to query, from a registered set the user parametrizes.
2. **Query formulation** — translating a sub-goal into concrete lower-level queries (search terms, filters, SQL).
3. **Execution** — running those queries against the corpora.
4. **Evaluation** — judging result relevance and sufficiency.
5. **Re-tuning** — refining queries and looping until satisfied or budget-exhausted.
6. **Presentation** — returning a ranked list of pointers to resources and/or extractions thereof.

The design goal is **structure over concretion**: the architecture should be a small set of *roles* (transformers) wired together by a control loop, with concrete tools (a specific vector DB, a specific search API) living only at the leaves. Every role is an open-closed strategy seam, injected at runtime. This document is *not* a tutorial on any particular framework; it is a reference architecture and a Python interface skeleton aligned with a functional, facade-first, dependency-injected style.

---

## 2. The consensus architecture (2026)

The field has converged on a clear reference design over the 2023–2026 period. The progression is well documented [1, 2]:

- **Static RAG** (2023): embed query → fetch top-k → stuff context → generate. One shot, no reasoning.
- **Iterative RAG** (2024): feedback loops where the model decides *when* to retrieve — IRCoT, Self-RAG, ReAct [3, 4, 5].
- **Modular RAG** (2024–2025): retrieval/indexing/generation/orchestration as composable, swappable building blocks [6, 7].
- **Agentic Search / Deep Research** (2025–2026): a planning agent decomposes the query, executes multi-step retrieval trajectories, evaluates, and re-queries — increasingly via an **orchestrator–worker** topology [8, 9, 10].

Two layers matter for our purposes, and they compose:

- The **Modular RAG layer** gives us the open-closed, dependency-injected substrate: interchangeable retrievers (vector, keyword, graph, SQL) behind a uniform interface, a pluggable generation/synthesis layer, and central orchestration coordinating data flow [6, 11]. This is where "the user parametrizes a group of sources" lives.
- The **Agentic control layer** gives us the cyclic plan → retrieve → evaluate → re-query loop that sits on top of the modular substrate [1, 8].

### 2.1 Orchestrator–worker vs. swarm

The dominant topology is **orchestrator–worker** (a.k.a. lead/subagent), popularized by Anthropic's Research feature and now the reference pattern across surveys [8, 9, 10]. A lead agent plans, optionally spawns specialized subagents in parallel, and synthesizes their findings, with a separate citation/verification pass [9].

The decisive property for a system you want to *reason about* is the **isolation boundary**. In orchestrator–worker, workers never talk to each other; every decision about what happens next lives in the orchestrator [9]. The alternative — a **swarm / peer-to-peer** model where agents share state through a common message bus or scratchpad — is more flexible but much harder to reason about [9]. For a composable, testable system, the constrained topology wins.

The empirical case for the multi-agent variant is strong but expensive: on Anthropic's internal research eval, an orchestrator–worker setup beat single-agent Claude Opus by ~90% [10, 12], at roughly **15× the token cost** of a normal chat [12]. The gain comes from spreading reasoning across independent context windows; it pays off for **breadth-first, parallelizable** tasks and is poorly suited to tightly interdependent ones [12]. Search/retrieval is breadth-first, so it sits in the sweet spot — but the cost mandates a budget governor from day one.

> **Design consequence.** Treat "single-context ReAct loop" and "parallel multi-agent" as two implementations of the *same orchestrator interface*. Start single-context (cheap, debuggable), promote to multi-agent without changing any role contract.

---

## 3. The roles (strategy seams)

The literature has stabilized on a small set of named roles [1, 8, 9, 13]. Each is an interface with swappable implementations — this is the heart of the "roles and transformers, not concrete tools" design.

| Role | Responsibility | Strategy variants |
|------|----------------|-------------------|
| **Planner / Orchestrator** | Decompose query into sub-tasks; select sources; manage budget; decide termination | ReAct, CoT planning, plan-then-execute, single-step holistic plan [13, 14] |
| **Query formulator** | Turn a sub-task + source into concrete low-level queries | rewrite, decompose, HyDE, filter-synthesis, SQL-generation [4, 11] |
| **Retriever** (per source) | Execute a low-level query against one corpus | dense/vector, BM25/sparse, hybrid, graph traversal, SQL, web API [6, 11] |
| **Evaluator / Critic** | Judge relevance + sufficiency; decide "re-query?" | LLM-as-judge, score threshold, Self-RAG reflection tokens [3, 5] |
| **Reranker** | Produce final sorted ordering | cross-encoder, reciprocal rank fusion, LLM rerank |
| **Citer / Verifier** | Confirm each pointer actually supports its claim | separate citation agent / verification pass [9, 10] |

Two atomic-improvement families from the survey literature inform the formulator and evaluator respectively: **query decomposition / rewriting / compression / selective-retrieval** at fine granularity [3], and **self-reflective retrieval** that decides when retrieved evidence is sufficient [4, 5].

---

## 4. The control loop

The system is a **cyclic, stateful graph**, not a linear pipeline [1]. The defining feature versus a static DAG is the **back-edge**: evaluator → re-formulate. Modern systems plan, retrieve, reason, critique, rewrite, and reflect in loops until confident or budget-exhausted [1].

```
                ┌─────────────────────────────────────────────┐
                │                                             │
   query ──▶ PLAN ──▶ FORMULATE ──▶ RETRIEVE ──▶ EVALUATE ──┤
                │       (per source, per sub-task)    │      │
                │                                  sufficient?
                │                                     │ no ──┘  (re-query: back-edge)
                │                                     │ yes
                └──────────────────────────▶ RERANK ──▶ CITE ──▶ results
```

State carried across the loop is a **run log / scratchpad**: the plan, the sub-tasks, accumulated results (as lightweight pointers, not full payloads), and per-round critic scores. A **budget governor** bounds the loop on three axes: max retrieval rounds, max sub-agents/sources, and a token/result ceiling per sub-task.

The substrate must therefore support: nodes (roles) + conditional edges + persistent state + cyclicity + a termination governor. Off-the-shelf, LangGraph provides exactly this abstraction — a directed cyclic graph with conditional branching, checkpoints, and human-in-the-loop interruption points [1]. You do not need LangGraph, but you need its abstraction. Note that a standard acyclic DAG-composition library is *insufficient on its own* precisely because of the required back-edge.

---

## 5. Context engineering: just-in-time, pointer-passing

The most important recent shift is away from pre-loading everything toward **lazy reference-passing** [15]. Agents maintain lightweight identifiers — file paths, stored queries, web links — and load data into context only at runtime via tools [15].

This pairs naturally with subagent isolation: a subagent may explore using tens of thousands of tokens but returns only a condensed, distilled summary (often 1–2k tokens), keeping detailed search context isolated while the lead focuses on synthesis [15].

For a **search agent specifically** — where the deliverable is *pointers + extractions*, not a synthesized essay — this is the ideal shape:

- Subagents/retrievers return **ranked pointers + snippets**, never full documents.
- The orchestrator merges and re-ranks over pointers.
- Full payloads are dereferenced only on demand (final extraction), never dragged through the lead context.

This keeps the lead context small and the system cheap, and it matches a `MutableMapping`-backed resource store: a pointer is a key, dereferencing is `store[key]`.

---

## 6. Reference Python skeleton

Everything below is a strategy seam. Roles are `Protocol`s; concrete tools are injected. The source registry is a `Mapping[source_name, Retriever]` — a facade over whatever backends exist.

```python
from __future__ import annotations
from typing import Protocol, Callable, Iterable, Sequence, Mapping, Any
from dataclasses import dataclass, field

# ---- Core value types (immutable, plain data) -------------------------------

@dataclass(frozen=True)
class Query:
    text: str
    constraints: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SubTask:
    goal: str
    sources: tuple[str, ...]          # which registered sources to use

@dataclass(frozen=True)
class LowLevelQuery:
    source: str                       # registry key
    spec: Any                         # search terms | filters | SQL | API params

@dataclass(frozen=True)
class Result:
    pointer: str                      # key into a resource store (URL, path, id)
    snippet: str                      # extraction, not full payload
    score: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Judgement:
    relevant: Sequence[Result]
    sufficient: bool
    refinement: SubTask | None = None  # if not sufficient, how to re-query

# ---- Role interfaces (the open-closed strategy seams) -----------------------

class Planner(Protocol):
    def __call__(self, query: Query, sources: Mapping[str, "Retriever"]) -> list[SubTask]: ...

class Formulator(Protocol):
    def __call__(self, task: SubTask, source: str) -> list[LowLevelQuery]: ...

class Retriever(Protocol):
    def __call__(self, q: LowLevelQuery) -> Iterable[Result]: ...

class Evaluator(Protocol):
    def __call__(self, task: SubTask, results: Sequence[Result]) -> Judgement: ...

class Reranker(Protocol):
    def __call__(self, results: Sequence[Result]) -> Sequence[Result]: ...

class Citer(Protocol):
    def __call__(self, results: Sequence[Result]) -> Sequence[Result]: ...

# ---- Budget governor --------------------------------------------------------

@dataclass(frozen=True)
class Budget:
    max_rounds: int = 3
    max_sources_per_task: int = 4
    max_results_per_task: int = 50

# ---- Orchestrator: a fixed control loop, fully parametrized by roles --------

@dataclass
class SearchAgent:
    sources: Mapping[str, Retriever]   # the user-parametrized source registry (a Mall/facade)
    planner: Planner
    formulator: Formulator
    evaluator: Evaluator
    reranker: Reranker
    citer: Citer
    budget: Budget = Budget()

    def __call__(self, query: Query) -> Sequence[Result]:
        accumulated: list[Result] = []
        for task in self.planner(query, self.sources):
            accumulated.extend(self._run_task(task))
        ranked = self.reranker(accumulated)
        return self.citer(ranked)

    def _run_task(self, task: SubTask) -> list[Result]:
        found: list[Result] = []
        current = task
        for _ in range(self.budget.max_rounds):
            for source in current.sources[: self.budget.max_sources_per_task]:
                for llq in self.formulator(current, source):
                    found.extend(self.sources[source](llq))
            judged = self.evaluator(current, found[: self.budget.max_results_per_task])
            found = list(judged.relevant)
            if judged.sufficient or judged.refinement is None:
                break
            current = judged.refinement          # the back-edge: re-query
        return found
```

Notes on the skeleton:

- **`sources` is a `Mapping`** — a `dol`-style facade / Mall. Registering a new corpus (internet, a local folder, a Parquet dataset, a Postgres DB) is adding a key. The planner sees only the registry, never concrete backends. This is the source-parametrization requirement.
- **The orchestrator is a fixed loop**; every *decision* is an injected role. Swapping single-context for multi-agent means swapping the `SearchAgent` implementation while keeping every role contract identical (see §7).
- **The back-edge** lives in `_run_task` as `current = judged.refinement`. That single line is what makes this an agent and not a DAG.
- **Results are pointers + snippets**, honoring §5. Final extraction (dereferencing `pointer`) is a separate, on-demand step you can layer on the resource store.

---

## 7. Two orchestrator implementations behind one interface

Define `SearchAgent` (or its `__call__` contract) as the seam, and provide:

1. **`SingleContextAgent`** — one ReAct-style loop, sequential sub-tasks. Cheap, fully debuggable, no inter-agent coordination. The correct *starting point*; for most enterprise search a hybrid retriever in a single loop is sufficient and agentic complexity is only warranted for genuinely multi-step, cross-source queries [2, 11].
2. **`MultiAgentAgent`** — spawns one subagent per sub-task (or per source) in parallel, each with an isolated context returning distilled pointer-sets, lead synthesizes [9, 15]. Worth the ~15× token cost only for breadth-first queries across many sources [12].

Because both consume the same `Planner / Formulator / Retriever / Evaluator / Reranker / Citer` injections, promotion from (1) to (2) changes *only the orchestrator strategy* — no role rewrite. This is the open-closed payoff.

---

## 8. Build order (recommendation)

1. **Source registry + Retriever protocol.** Get 2–3 real backends behind the `Mapping` facade (one web, one local file/vector, one SQL). This is the highest-leverage, most reusable piece.
2. **Single-context orchestrator** with trivial planner (1 sub-task) and a pass-through evaluator (`sufficient=True`). End-to-end thin slice.
3. **Formulator + Evaluator** as LLM strategies; turn on the back-edge. Now it is an agent.
4. **Reranker + Citer.** Quality/verification layer.
5. **Budget governor + observability.** Per-node metadata (critic score, round, tokens) for evaluation and regression tracking [1].
6. **Multi-agent orchestrator** as a drop-in alternative, only if breadth justifies it.

---

## 9. Open design questions

- **Cyclic substrate.** A standard acyclic DAG composition tool cannot express the evaluator→re-formulate back-edge directly. Either adopt a cyclic-graph runner or implement the loop imperatively (as in §6) and keep only the *intra-round* fan-out declarative.
- **Termination policy.** Sufficiency is the hardest role to get right; budget caps are the safety net. Consider making termination a *separate injected policy* rather than folding it into the evaluator.
- **Pointer dereferencing.** Decide early whether extractions are produced at retrieval time (eager snippets) or on-demand (lazy `store[pointer]`). The lazy form keeps context small but adds a round-trip.
- **Single vs. multi-agent default.** Default to single-context; expose multi-agent as opt-in. The 15× cost is real and only justified for breadth-first work [12].

---

## REFERENCES

[1] Rane V. Next-Generation Agentic RAG with LangGraph (2026 Edition). Medium; 2026. [https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8)

[2] 10 RAG Architectures in 2026: Enterprise Use Cases & Strategy. Techment; 2026. [https://www.techment.com/blogs/rag-architectures-enterprise-use-cases-2026/](https://www.techment.com/blogs/rag-architectures-enterprise-use-cases-2026/)

[3] GraphSearch: An Agentic Deep Searching Workflow for Graph Retrieval-Augmented Generation. arXiv; 2025. [https://arxiv.org/pdf/2509.22009](https://arxiv.org/pdf/2509.22009)

[4] Ma X, et al. Query rewriting and decomposition for retrieval (as surveyed in [3], [6]). 2023–2024.

[5] Asai A, Wu Z, Wang Y, Sil A, Hajishirzi H. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR; 2024. [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)

[6] Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG. arXiv; 2025. [https://arxiv.org/pdf/2501.09136](https://arxiv.org/pdf/2501.09136)

[7] Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search. arXiv; 2026. [https://arxiv.org/pdf/2601.04703](https://arxiv.org/pdf/2601.04703)

[8] Deep Research: A Systematic Survey. arXiv; 2025. [https://arxiv.org/pdf/2512.02038](https://arxiv.org/pdf/2512.02038)

[9] How Anthropic Built a Multi-Agent Research System. The AI Engineer; 2026. [https://theaiengineer.substack.com/p/how-anthropic-built-multi-agent-deep](https://theaiengineer.substack.com/p/how-anthropic-built-multi-agent-deep)

[10] Deep Research of Deep Research: From Transformer to Agent, From AI to AI for Science. arXiv; 2026. [https://arxiv.org/pdf/2603.28361](https://arxiv.org/pdf/2603.28361)

[11] Agentic RAG systems for enterprise-scale information retrieval. Toloka; 2026. [https://toloka.ai/blog/agentic-rag-systems-for-enterprise-scale-information-retrieval/](https://toloka.ai/blog/agentic-rag-systems-for-enterprise-scale-information-retrieval/)

[12] How Anthropic Built a Multi-Agent Research System. ByteByteGo; 2025. [https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent](https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent)

[13] Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts. arXiv; 2026. [https://arxiv.org/pdf/2604.00901](https://arxiv.org/pdf/2604.00901)

[14] Agentic Reasoning for Large Language Models. arXiv; 2026. [https://arxiv.org/pdf/2601.12538](https://arxiv.org/pdf/2601.12538)

[15] Effective context engineering for AI agents. Anthropic; 2025. [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
