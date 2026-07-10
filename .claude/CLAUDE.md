# ir — agent instructions

`ir` is an **information-retrieval substrate for agentic systems**: one uniform
"find the relevant things in this corpus" contract (retrieve → select → disclose),
built by composing the ecosystem — `ef` (embedders, content-hash, reranking,
evaluation), `vd` (BM25, RRF, metadata filters), `dol` (persistence), `oa` (LLM
ops, always lazy/opt-in). Retrieval-first; **RAG is layered on top, not here**.

## Where the design lives

- `misc/docs/ir_01`…`ir_09` — the research + design + eval-run record (read the
  relevant one before non-trivial work). Notably `ir_07` (abstention
  calibration), `ir_08` (magnitude-preserving fusion), **`ir_09` (the Composable
  Search Agent reference architecture)**.
- GitHub issues are the running journal. **Epic `#38`** is the agentic-search
  roadmap; `#1` is the overall roadmap.

## ir ↔ raglab boundary (load-bearing)

The **agent / RAG orchestration layer lives in `raglab`** (`thorwhalen/raglab`,
local `$PP/t/raglab`), **on top of ir**. The split (epic #38, decided 2026-06-11):

- **ir owns** the retrieval-adjacent seams — they each improve ir's *own*
  single-shot search: the Retriever leaf (`search`/`as_retriever`), the
  `Mapping[name, Retriever]` registry view, the injectable **Formulator** seam
  (`formulate=`, identity default), the derived **`sufficient`** signal on
  `Selection`, the pointer/resource-store contract on `disclose`, abstention
  calibration. Value types ir owns: `SearchHit` (≈ ir_09 `Result`), `Selection`,
  `Disclosure`.
- **raglab owns** the *agent*: the Planner, the **control loop with the back-edge**
  (evaluator → reformulate), the budget governor, the live multi-source registry
  across heterogeneous backends, the Citer/Verifier, and the
  single-context-vs-multi-agent orchestrators.

**Two boundaries to guard in every review:**
1. **The Formulator returns queries, never SubTasks.** Decomposition into
   goal+sources is the Planner's job → raglab.
2. **ir gains a `sufficient` *signal*, never a `refinement` directive or a loop.**
   The back-edge is what makes a system an agent; it must not live in ir.

**Dependency direction is one-way: `raglab` imports `ir`; `ir` must NEVER import
`raglab`.** This keeps ir installable-light and offline-by-default.

## Agentic search from Claude Code (no coded agent in ir)

For driving `ir` agentically **without** putting an agent in the codebase, ir
ships two Claude-Code-native artifacts (a lightweight "raglab-in-instructions"):

- `.claude/skills/ir-search/SKILL.md` — the entry-point skill: when to search
  `ir` corpora, the CLI cheat-sheet, the corpus map, and the mode/scoring/coverage
  gotchas.
- `.claude/agents/ir-discover-agent.md` — a **reviewing search subagent**: it
  formulates several queries, retrieves, **reads and grades** candidates against
  the real intent, reformulates (the back-edge), and returns only the verified
  few. It supplies the query-writing + relevance-review + reformulate loop that
  the deterministic `ir discover` pipeline lacks.

These are the correct home for the *back-edge as instructions* — they don't
violate the boundary above, because no loop or `refinement` directive enters the
`ir` package code. The coded loop still belongs to `raglab`.

## House style (inherited)

Functional > OOP; SOLID when OOP; facades, SSOT, dependency injection;
progressive disclosure (a new arg must default to today's behavior); keyword-only
beyond the 3rd positional; `collections.abc` + frozen `dataclass`es; every module
has a top-level docstring (auto-extracted for docs). `oa` stays lazy/opt-in
(`import ir` must not import `oa`). Tests hermetic with `embedder="light"`;
`USE_TF=0` is set on import. Real ruff rules (F/E/W/B) are enforced on `ir/`.
