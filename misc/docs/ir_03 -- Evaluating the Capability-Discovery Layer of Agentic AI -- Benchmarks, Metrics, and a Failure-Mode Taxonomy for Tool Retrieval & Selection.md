# Evaluating the Capability-Discovery Layer of Agentic AI: Benchmarks, Metrics, and a Failure-Mode Taxonomy for Tool Retrieval & Selection

*Author: **Thor Whalen** — Prompt C in the agentic-platform eval series. Audience: eval-methodology experts (Inspect AI, deepdiff scorers, LLM-as-judge via G-Eval/DAG). Eval fundamentals are assumed and skipped. Sources are 2024–2026, prioritizing 2025–2026.*

> **How to save this file:** copy everything below the title into a `.md` file (e.g. `capability-discovery-eval.md`). The document is self-contained Markdown with Vancouver-style numbered references and hyperlinks.

---

## TL;DR

- **Almost no existing benchmark measures what you actually need.** The field conflates three distinct things — retrieval recall (surfacing the right tool from a large catalog), selection accuracy (picking the right tool from candidates), and end-to-end task success. Only **ToolRet** [1] cleanly isolates retrieval; **MetaTool** [2] isolates selection/awareness; everything else (BFCL [3], τ²-bench [4], AppWorld [5], ToolHop [6], ACEBench [7]) bundles selection into end-task success. Build your harness around the **retrieval/selection decoupling** that the literature still lacks.
- **Distractor-robustness is the dominant, under-measured failure axis, and it directly explains your ~49% baseline.** Anthropic's own engineering data shows Claude Opus 4 going from **49%→74%** and Opus 4.5 from **79.5%→88.1%** once the Tool Search Tool (progressive disclosure via `defer_loading`) is enabled [8]; RAG-MCP shows raw selection accuracy collapsing to **13.62%** (full-schema-dump baseline) and recovering to **43.13%** with retrieval pre-filtering in a needle-in-a-haystack MCP stress test [9]; BFCL-derived experiments show calendar-task accuracy dropping **43%→2%** going from 4 tools to 51 tools across 7 domains [10]. Your 49% is a structural artifact of static tool-loading, not a model-quality ceiling.
- **Recommended stack:** treat retrieval as an IR problem scored with **recall@k / NDCG@k** (BEIR/MTEB-style [11,12]), score selection **conditionally on correct retrieval**, score parameters with a **deepdiff-based structural scorer**, and wrap everything in **Inspect AI** scorers (MIT) [13]. Generate eval cases by **back-translation from your `@command` registry** with LLM-judge validation and explicit distractor injection. Use **τ²-bench** [4] and **AppWorld** [5] for end-task sanity checks only — not for discovery-layer measurement.

---

## Key Findings

1. **The retrieval-vs-selection distinction is real and consequential.** ToolRet [1] demonstrated that IR models which excel on conventional benchmarks perform poorly on tool retrieval: "even the best model (i.e., NV-embed-v1)… achieves an nDCG@10 of only 33.83" on a 43k-tool corpus across 7.6k tasks. Most tool-use benchmarks "simplify this step by manually pre-annotating a small set of relevant tools," meaning they never test retrieval at all. With 50+ tools, the pre-annotation shortcut hides your single biggest risk.

2. **Selection accuracy degrades monotonically with catalog size and distractor count.** This is the most robustly replicated finding in the 2025–2026 literature, with consistent evidence from RAG-MCP [9], BFCL-derived studies [10], and Anthropic [8]. The mechanism: the model rarely abstains; it picks a plausible-but-wrong tool, hallucinates parameters, or confuses similarly-named tools (`notification-send-user` vs `notification-send-channel`) [8].

3. **Progressive disclosure / tool-search is the proven mitigation, and it is itself an evaluation target.** Anthropic's Tool Search Tool [8], Claude Code's MCP Tool Search (`defer_loading: true`, auto-triggers when tool schemas exceed ~10% of context) [14], and the meta-tool pattern [15] all convert the problem from "select among N" to "retrieve k then select among k." This means your harness must evaluate the *retrieval query the model writes*, not just the final selection.

4. **Published failure taxonomies now exist and map cleanly onto your selection/parameterization/ordering/hallucination classes** [16,17,18,19] — extend them with discovery-specific classes (query-planning failure, retrieval miss, selector false-negative, over-selection, namespace/granularity confusion, tool hallucination).

5. **Synthetic case generation is mature.** APIGen (three-stage verification) [20], ToolACE (self-evolution + dual-layer verification) [21], Hammer (function masking + irrelevance augmentation) [22], and back-translation pipelines [23,24] give you a well-trodden path to generate (intent → correct-capability) pairs from your registry.

---

## Details

### 1. Benchmark landscape — what each actually measures

The single most important analytical move is to classify each benchmark by which of the three sub-problems it measures:

- **(R) Retrieval recall** — can the system surface the right tool from a large, *unfiltered* catalog?
- **(S) Selection accuracy** — given a small candidate set, does the model pick the right tool / abstain correctly?
- **(E) End-to-end** — does the whole task succeed (selection bundled with parameterization, execution, ordering, multi-turn)?

#### Comparison matrix

| Benchmark | Year | Measures | Catalog scale | Credibility / maintenance | License |
|---|---|---|---|---|---|
| **ToolRet** [1] | 2025 | **R** (pure retrieval) | 43k tools, 7.6k tasks | High — only true large-scale retrieval benchmark; actively maintained | research code (GitHub) |
| **MetaTool / ToolE** [2] | ICLR 2024 | **S** + tool-use awareness/abstention | ~390 tools, 21k queries | Medium — influential for "whether to use a tool"; somewhat dated | open (GitHub) |
| **BFCL v1–v4** [3] | 2024–2026 | **S** + **E** (AST + executable + multi-turn + agentic) | thousands (AST scales) | High — canonical, updated (v4 = `bfcl-eval 2025.12.17`); some saturation | Apache-2.0 (gorilla) |
| **τ-bench / τ²-bench / τ³** [4,25] | 2024–2026 | **E** (policy adherence, multi-turn, dual-control) | small per-domain (~tens) | High — best enterprise-realism eval; actively maintained | open (sierra-research) |
| **AppWorld** [5] | ACL 2024 | **E** (stateful, code-gen, collateral damage) | 457 APIs, 9 apps | High — best stateful execution env; train/test split guards leakage | open (encrypted bundles) |
| **ToolHop** [6] | 2025 | **E** (multi-hop) | query-driven, executable | Medium-high — GPT-4o only ~49% | open |
| **ComplexFuncBench** [26] | 2025 | **E** (multi-step, long-context, constrained) | 1,000 samples | Medium — narrow (live Booking API) | open (THUDM) |
| **NESTFUL** [27] | 2025 | **E** (nested API sequences) | 1,800+ sequences | Medium — niche but rigorous; GPT-4o only 28% full-match | Apache-2.0 (IBM) |
| **ACEBench** [7] | 2025 | **S** + **E** (+ imperfect-instruction handling) | ~4,500 APIs, 8 domains | Medium-high — no-GPT eval, good error taxonomy | open |
| **StableToolBench** [28] | 2024 | **E** (stabilized ToolBench via virtual API) | 16k+ tools (simulated) | Medium — solves API-instability but GPT-judge variance | open |
| **ToolSandbox** [29] | 2024 | **S** + **E** (stateful, conversational, on-policy) | on-device sim + RapidAPI | Medium-high — Apple; strong state-dependency tasks | custom Apple license |
| **MTU-Bench** [30] | 2024 | **S** + **E** (multi-granularity, no-GPT metrics) | 136 tools, 54.8k dialogues | Medium — cheap fine-grained metrics incl. tool selection/order accuracy | open |
| **MCP-RADAR** [31] | 2025 | **S** + **E** (MCP framework) | 507 tasks, 6 domains | Medium — MCP-specific, new | open |
| **MCP-Atlas** [32] | 2026 | **E** (large-scale real MCP, claims-based rubric) | 220 tools, 36 servers, 1k tasks | New — Claude Opus 4.5 best at 62.3% | open |
| **GTA / GTA-2** [33] | 2024/2026 | **E** (real queries, deployed tools, workflow) | hierarchical | High — now evaluates harnesses too | open (open-compass) |
| **ToolBench / ToolLLM** [24] | ICLR 2024 | **R** (neural retriever) + **E** | 16,464 APIs | Declining — live-API instability; superseded by StableToolBench/ToolRet | Apache-2.0 / CC-BY-NC |
| **API-Bank** [34] | 2023 | **S** + **E** (planning/retrieving/calling) | 73 tools | Dated baseline | open |

**Opinionated assessment:**

- **For the retrieval sub-problem, ToolRet [1] is the only credible large-scale benchmark, and you should adopt its BEIR/MTEB-style framing directly.** ToolRet standardizes tasks "into a retrieval format akin to MTEB and BEIR." That is exactly the right abstraction for your `@command` registry: treat each command's `(name, description, params)` as a document, the user intent as a query, and score with NDCG@10 / Recall@10.
- **For selection, MetaTool [2] and ACEBench [7] are the most relevant** because they explicitly test "whether to use a tool" (abstention) and "which to use" with confusable/irrelevant candidates. Hammer's irrelevance augmentation (the xLAM-7.5k-Irrelevance set) is the key training-side complement [22].
- **BFCL remains the canonical function-calling leaderboard but is showing saturation** (GLM-4.5 at 0.778 on v3; top reasoning models clustered) [3,35]. It is also at contamination risk given its age and popularity; BFCL-v2 explicitly added "live, user-contributed scenarios" to combat contamination and bias [36]. Use BFCL for cross-model sanity, not for your discovery-layer signal.
- **τ²-bench [4] and AppWorld [5] are your end-task anchors.** Neither isolates discovery, but both have excellent state-based scoring (AppWorld's hash-based state diffing; τ²'s database-state comparison and `pass^k` reliability metric). Use them to confirm that discovery-layer gains translate to task success.
- **Reproducibility / contamination flags:** ToolBench's live RapidAPI dependency makes it non-reproducible (hence StableToolBench's virtual server + MirrorAPI [28]). ComplexFuncBench requires a live Booking API subscription [26]. Treat any benchmark requiring live third-party APIs as non-reproducible for regression testing.

### 2. Metrics specific to the discovery layer

Core design principle: **decouple retrieval from selection and score each conditionally.** Define:

- **Retrieval recall@k** — fraction of cases where the gold tool is in the top-k retrieved set.
- **NDCG@k** — rank-quality of retrieval (the BEIR/MTEB standard [11,12]; ToolRet's primary metric [1]).
- **Conditional selection accuracy** — `P(correct selection | gold tool was retrieved)`. This isolates the "searched, retrieved the right tool into context, but still failed to pick it" case you specifically called out.
- **End-to-end selection accuracy** = recall@k × conditional selection accuracy (the decomposition that makes the bottleneck visible).
- **Distractor-robustness curve** — selection accuracy as a function of N distractors / catalog size. The single most diagnostic metric for your platform.
- **Over-selection rate** — fraction of cases with unnecessary/extra tool calls.
- **Abstention accuracy** — correct refusal-to-act when no tool applies (MetaTool's reliability subtask [2]; Hammer's irrelevance detection [22]).
- **Parameter-filling accuracy** — scored separately and structurally (deepdiff).

MTU-Bench [30] is worth emulating here: it ships **tool selection accuracy, parameter selection accuracy, tool number accuracy, and tool order accuracy** as separate axes with *no* GPT cost — a good template for a cheap, deterministic discovery-layer scorecard.

#### recall@k, NDCG@k, and conditional selection accuracy (production Python)

```python
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import math


@dataclass(frozen=True)
class DiscoveryCase:
    intent: str
    gold_tool: str
    retrieved: Sequence[str]  # ranked list from the retriever
    selected: str | None  # tool the agent actually chose (None = abstained)
    gold_is_none: bool = False  # True when the correct behavior is to abstain


def recall_at_k(cases: Sequence[DiscoveryCase], k: int) -> float:
    """Fraction of cases whose gold tool appears in the top-k retrieved set."""
    relevant = [c for c in cases if not c.gold_is_none]
    if not relevant:
        return float("nan")
    hits = sum(c.gold_tool in c.retrieved[:k] for c in relevant)
    return hits / len(relevant)


def ndcg_at_k(cases: Sequence[DiscoveryCase], k: int) -> float:
    """Binary-relevance NDCG@k; IDCG is 1.0 since there is one gold tool."""
    relevant = [c for c in cases if not c.gold_is_none]
    if not relevant:
        return float("nan")
    total = 0.0
    for c in relevant:
        topk = list(c.retrieved[:k])
        if c.gold_tool in topk:
            rank = topk.index(c.gold_tool)  # 0-indexed
            total += 1.0 / math.log2(rank + 2)  # DCG; IDCG == 1.0
    return total / len(relevant)


def conditional_selection_accuracy(cases: Sequence[DiscoveryCase], k: int) -> float:
    """P(correct selection | gold tool retrieved into top-k).

    Isolates selector quality from retriever quality: the 'searched,
    surfaced the right tool, still failed to pick it' failure mode.
    """
    eligible = [
        c for c in cases if not c.gold_is_none and c.gold_tool in c.retrieved[:k]
    ]
    if not eligible:
        return float("nan")
    correct = sum(c.selected == c.gold_tool for c in eligible)
    return correct / len(eligible)
```

#### Distractor-robustness harness (RAG-MCP-style NIAH stress test)

```python
from collections.abc import Callable, Sequence
import random

ToolName = str
SelectFn = Callable[[str, Sequence[ToolName]], ToolName | None]


def distractor_robustness_curve(
    intent: str,
    gold_tool: ToolName,
    distractor_pool: Sequence[ToolName],
    select: SelectFn,
    sizes: Sequence[int] = (1, 4, 8, 16, 32, 64, 128),
    trials: int = 20,
    seed: int = 0,
) -> dict[int, float]:
    """Selection accuracy as catalog size grows.

    Mirrors the RAG-MCP 'needle-in-a-haystack' MCP stress test: one gold
    tool + (N-1) distractors, vary N, report accuracy at each N.
    """
    rng = random.Random(seed)
    curve: dict[int, float] = {}
    for n in sizes:
        n_distract = max(0, n - 1)
        hits = 0
        for _ in range(trials):
            sample = rng.sample(
                list(distractor_pool), min(n_distract, len(distractor_pool))
            )
            catalog = sample + [gold_tool]
            rng.shuffle(catalog)
            hits += select(intent, catalog) == gold_tool
        curve[n] = hits / trials
    return curve
```

Empirical anchors this curve should reproduce: a steep monotonic decline absent retrieval pre-filtering — RAG-MCP's full-schema-dump baseline ("Blank Conditioning") bottoms at **13.62%**, and "Actual Match" at 18.20%, versus **43.13%** for retrieval-augmented MCP-RAG [9]; Allen Chan's BFCL-derived study reports gpt-4o calendar accuracy "dropped from 43% with single domain & 4 tools to just 2% with 7 domains and 51 tools," customer-support "from 58%… to just 26%," and "llama-3-3-70b dropped from 21% to 0%" [10] — and a large recovery with progressive disclosure: Anthropic Opus 4 **49%→74%** [8].

#### deepdiff-based parameter scorer

```python
from typing import Any
from deepdiff import DeepDiff


def parameter_score(
    predicted_args: dict[str, Any],
    gold_args: dict[str, Any],
    *,
    ignore_order: bool = True,
    significant_digits: int | None = None,
) -> dict[str, float | int]:
    """Structural parameter-fill scorer.

    Decouples parameterization failures from selection failures: a tool
    can be correctly selected but mis-parameterized (your
    'parameterization' failure class). Granular sub-scores attribute
    errors to missing keys vs wrong values vs extras.
    """
    diff = DeepDiff(
        gold_args,
        predicted_args,
        ignore_order=ignore_order,
        significant_digits=significant_digits,
    )
    missing = len(diff.get("dictionary_item_removed", []))
    extra = len(diff.get("dictionary_item_added", []))
    changed = len(diff.get("values_changed", {})) + len(diff.get("type_changes", {}))
    total_keys = max(len(gold_args), 1)
    return {
        "exact_match": 1.0 if not diff else 0.0,
        "missing_keys": missing,
        "extra_keys": extra,  # signal for over-parameterization
        "changed_values": changed,
        "key_recall": 1.0 - missing / total_keys,
    }
```

#### Inspect AI scorer sketch

```python
from inspect_ai.scorer import scorer, Score, Target, accuracy, stderr
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def conditional_selection_scorer(k: int = 5):
    """CORRECT only if the gold tool was retrieved AND chosen.
    Emits metadata so retrieval-miss vs selector-false-negative are
    distinguishable in Inspect View."""

    async def score(state: TaskState, target: Target) -> Score:
        gold = target.text
        retrieved = state.metadata.get("retrieved_tools", [])[:k]
        selected = state.metadata.get("selected_tool")
        retrieved_ok = gold in retrieved
        selected_ok = selected == gold
        if not retrieved_ok:
            value, kind = "I", "retrieval_miss"
        elif not selected_ok:
            value, kind = "I", "selector_false_negative"
        else:
            value, kind = "C", "ok"
        return Score(
            value=value,
            answer=str(selected),
            metadata={"failure_class": kind, "retrieved_ok": retrieved_ok},
        )

    return score
```

### 3. Failure-mode taxonomy for the discovery layer

Mapping your existing four classes (selection / parameterization / ordering / hallucination) onto discovery-specific extensions, each with a detection signal and scorer type:

| Failure class | Definition | Detection signal | Scorer type |
|---|---|---|---|
| **Query-planning failure** *(new)* | Model writes a bad search query against the registry | Retrieved set has low overlap with gold; low BM25/embedding score | Log the search query; NDCG@k of *that* query vs an oracle query |
| **Retrieval miss** *(new → selection)* | Right tool exists but not surfaced in top-k | `gold ∉ retrieved[:k]` | recall@k; deterministic set-membership |
| **Selector false-negative** *(new → selection)* | Right tool surfaced but not chosen | `gold ∈ retrieved[:k]` AND `selected ≠ gold` | conditional selection accuracy |
| **Over-selection / over-calling** *(new)* | Too many tools chosen / unnecessary calls | `len(selected) > len(gold)`; extra calls in trace | over-selection rate; trace diff |
| **Namespace / granularity confusion** *(new)* | Confuses similar/overlapping tools (`send-user` vs `send-channel`) | Selected tool is a sibling/near-neighbor of gold | confusion matrix over tool clusters |
| **Tool hallucination** *(your hallucination class)* | Calls a tool that doesn't exist | `selected ∉ registry` | deterministic registry-membership; AST check |
| **Parameterization error** *(your class)* | Correct tool, wrong/missing/extra args | deepdiff non-empty | deepdiff structural scorer |
| **Ordering error** *(your class)* | Correct tools, wrong sequence (nested/dependent) | sequence mismatch vs gold DAG | edit-distance / DAG-topological scorer |
| **Selection error** *(your class, umbrella)* | Wrong tool given correct retrieval | as selector false-negative | conditional selection accuracy |
| **Abstention failure** *(new)* | Acts when no tool applies (or refuses when one does) | `gold_is_none` mismatch | abstention accuracy (MetaTool reliability) |

**Published taxonomies to cite and build on (2025–2026):**

- **AgentErrorTaxonomy** ("Where LLM Agents Fail and How They Can Learn From Failures") [16] — a unified failure taxonomy with a two-stage detector; explicitly moves "from descriptive taxonomies to actionable debugging," and catalogs planning brittleness, grounding/tool-use errors, and hallucination-induced cascades.
- **"Characterizing Faults in Agentic AI: A Taxonomy of Types, Symptoms, and Root Causes"** [17] — maps agentic failures to system components (orchestration, evolving internal state, environment feedback), filling the gap left by multi-agent-only analyses.
- **"LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions"** [18] — taxonomy of agent hallucinations spanning brain/perception/action modules.
- **"Internal Representations as Indicators of Hallucinations in Agent Tool Selection"** (Amazon) [19] — detects tool-calling hallucinations (incorrect tool, malformed parameters, "tool bypass") from internal representations in a single forward pass, "up to 86.4% accuracy," particularly strong at "parameter-level hallucinations and inappropriate tool selections." A concrete *detection signal* for your hallucination and parameterization classes.
- **ToolCert** (Yeon et al., Oct 2025) [37] — adversarial tool-injection certification with Clopper–Pearson bounds on selection accuracy. Use for your distractor-robustness *safety* eval.

### 4. Synthetic eval-case generation for capability retrieval

Goal: generate (intent → correct-capability) pairs at scale from your `@command` registry.

- **Back-translation / inverse generation (tool → synthetic query).** Given a command signature + docstring, prompt an LLM to generate the user intents that *should* route to it. This is the dominant pattern (ToolBench/ToolLLM's instruction generation [24], MTU-Bench's reverse construction from MultiWOZ/SGD intents [30]). The signature is your ground-truth label, so labels are free — but watch for **label leakage**: never include the tool name verbatim in the generated query. Hammer's *function masking* mitigates exactly this by forcing reliance on descriptions over names [22].
- **Schema-driven generation from tool signatures.** APIGen's three-stage pipeline (format check → actual execution → semantic verification) generated 60k verified entries from 3,673 APIs [20]. RandomWorld inverts the order: sample API-call sequences via type-guided sampling first, then populate the environment [38].
- **LLM-based user-simulator generation.** τ-bench/τ²-bench's user simulator [4] and IntellAgent's policy-graph-driven synthetic suites generate multi-turn traces; ToolACE uses multi-agent interplay (self-evolution synthesis → 26,507 APIs) with a dual-layer (rule + model) verification system [21].
- **Distractor injection.** For each (intent, gold tool), inject hard negatives: (a) sibling tools in the same namespace, (b) semantically near tools (embedding nearest neighbors — MassTool's intent-enrichment approach [39]), (c) random tools. The Hammer xLAM-7.5k-Irrelevance set is the canonical "no tool applies" augmentation [22].
- **Validation / filtering / difficulty calibration.** LLM-as-judge validation (G-Eval/DAG-style), dedup via embedding similarity, and difficulty calibration by binning on retriever score or distractor hardness. MCP-Radar used a strong baseline model (Gemini 2.5 Flash) as a *filter* to ensure tasks genuinely require tool use rather than parametric knowledge "to ensure our benchmark specifically tests tool-use rather than a model's internal knowledge — a common issue of data contamination" [31] — adopt this to avoid trivial cases.

Registry-driven generation sketch:

```python
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    params: dict[str, str]  # name -> type


def make_backtranslation_prompt(cmd: Command, n: int = 5) -> str:
    """Generate intents WITHOUT leaking the tool name (function masking)."""
    return (
        f"A capability does the following: {cmd.description}\n"
        f"It accepts parameters: {list(cmd.params)}\n"
        f"Write {n} natural user requests that this capability should "
        f"handle. Do NOT mention any function or tool name. Vary phrasing, "
        f"specificity, and implied (not explicit) parameter values."
    )


def inject_distractors(
    gold: Command,
    registry: Sequence[Command],
    k: int,
    embed,
    namespace_of,
) -> list[str]:
    """Hard-negative catalog: siblings + nearest-neighbors + random."""
    siblings = [
        c.name
        for c in registry
        if namespace_of(c) == namespace_of(gold) and c.name != gold.name
    ]
    neighbors = embed.nearest(gold.description, exclude={gold.name}, k=k)
    pool = list(dict.fromkeys(siblings + neighbors))[: k - 1]
    return pool + [gold.name]
```

### 5. Reference implementations / libraries (license-aware)

You are lock-in-averse; licenses noted.

- **Inspect AI** (UK AISI + Meridian Labs) — **MIT**, Python ≥3.10 [13]. Your harness backbone: `@scorer`-decorated functions, `TaskState` metadata for retrieval/selection traces, model-graded scorers, `multi_scorer`, MCP tool integration, and `inspect_evals` ships a **BFCL port** (V1/V2/V3 categories, AST matching from the paper's Appendix H) you can fork rather than reimplement [40].
- **BEIR** — **Apache-2.0** [11]. The standard heterogeneous zero-shot IR benchmark (18 datasets / 9 tasks); its canonical metric is **NDCG@10** ("we… compute nDCG@10 for all datasets"). Use its evaluation interface directly for your retrieval sub-problem.
- **MTEB** — **Apache-2.0** (verify SPDX in repo) [12]. Massive embedding benchmark (8 tasks / 56–58 datasets / 112 languages); its **Retrieval** task is built on BEIR and uses NDCG@10. Use to choose the embedding model for your tool retriever; the HuggingFace MTEB leaderboard is the authoritative source.
- **ToolRet code** (`mangopy/tool-retrieval-benchmark`) — research code; supports embedding + reranker eval in BEIR/MTEB format [1]. The closest existing artifact to what you're building; fork its task format.
- **BFCL harness** (`ShishirPatil/gorilla`) — **Apache-2.0** [3]. `bfcl generate` / `bfcl-eval` PyPI package; AST evaluation scales to thousands of functions; supports vLLM/sglang local serving.
- **τ²-bench / τ³-bench** (`sierra-research/tau2-bench`) — open [4]. `tau2 run` CLI, dual-control env, `pass^k` reliability metric, voice + knowledge-retrieval (BM25) domains. Best for end-task + policy-adherence anchoring.
- **AppWorld** (`StonyBrookNLP/appworld`) — open, with encrypted `.bundle` files + canary string to prevent train leakage [5]; 457 APIs, state-based unit tests, hash-based state diffing. Best stateful-execution anchor.
- **ToolSandbox** (`apple/ToolSandbox`) — **custom Apple license, NOT Apache-2.0** — review terms before redistribution [29]. Stateful, on-policy conversational eval; "open source and proprietary models have a significant performance gap" on state-dependency / canonicalization / insufficient-information tasks.
- **StableToolBench** (`THUNLP-MT`) — open; virtual API server + MirrorAPI simulator for reproducible ToolBench-style eval [28]. Use instead of raw ToolBench.
- **APIGen / xLAM data** (Salesforce) — datasets on HF (`xlam-function-calling-60k`); APIGen pipeline for verifiable generation [20]. **Hammer / xLAM-7.5k-Irrelevance** (MadeAgents) — irrelevance/abstention data + function-masking recipe [22].
- **RAG-MCP reference impls** (`fintools-ai/rag-mcp`, `memoverflow/rag-mcp`) — open; vector-index-over-tool-metadata pattern with the MCP stress test you can adapt as a distractor-robustness harness [9].

**deepdiff** itself is MIT — safe for your parameter scorer. **polyfactory** (MIT) for schema-driven synthetic param generation; **DSPy/GEPA** for optimizing the retrieval query the model writes.

### 6. Open problems and gotchas

- **Benchmark contamination & saturation.** BFCL, being old and popular, is contamination-prone and saturating (top models clustered, MMLU-style plateau) [35,41]. Systematic studies show saturation is driven by benchmark age and test-set scale as much as by capability [42]. **Mitigation:** generate your own private, registry-derived eval set; rotate/regenerate it; keep an AppWorld-style canary string to detect leakage into future training [5].
- **Static-benchmark vs production gap.** Static accuracy on a fixed catalog does not predict production capability-discovery, where the catalog grows, tools are versioned, and descriptions vary in quality. As MarkTechPost notes of Anthropic's Tool Search, "~26 percentage points of accuracy is still retrieval failure on Opus 4… Tool Search assumes the model can write a reasonable search query" [43]. Measure on *your* catalog.
- **Non-determinism / reproducibility.** LLM-judge variance (StableToolBench moved to solvable-pass-rate + an end-to-end trained evaluator to reduce it [28]), sampling temperature, and live-API instability all undermine regression testing. **Mitigation:** prefer deterministic structural scorers (deepdiff, set-membership, AST) over LLM judges wherever the label is structural; reserve LLM-judge for genuinely open-ended cases; report `pass^k`-style reliability, not just mean accuracy [4].
- **No standardized retrieval-vs-selection decoupling.** This is the field's biggest methodological gap and your biggest opportunity — almost every benchmark pre-annotates candidate tools, hiding retrieval [1]. Your conditional-selection-accuracy metric is genuinely novel contribution territory.
- **Distractor-robustness is under-measured.** Only RAG-MCP and a handful of 2025–2026 papers run the stress test systematically [9]. Make the distractor-robustness curve a first-class, always-reported metric.
- **Progressive-disclosure / tool-search systems need their own evals.** When `defer_loading` is on [8,14], you must evaluate (a) the search query the model writes, (b) whether the right tool is surfaced, (c) the post-retrieval selection — three separate scores. Standard benchmarks evaluate none of these. Note the token-tax stakes: Anthropic's five-server example is "58 tools consuming approximately 55K tokens before the conversation even begins," and tool definitions can "consume 134K tokens before optimization," with an 85% token reduction under Tool Search [8].
- **MCP-specific gaps.** MCP-RADAR [31], MCP-Atlas [32], and MCPToolBench++ are the only MCP-native benchmarks, all very new. Namespace collisions across servers (`notification-send-*`), schema-bloat token tax, and cross-server orchestration are largely unmeasured.
- **Counterfactual / safety evals.** ToolCert (adversarial tool injection with statistical bounds) [37] and SafeToolBench are the current state of the art. For the discovery layer specifically, test: (a) does injecting a malicious near-duplicate tool divert selection? (b) does the agent abstain when no safe tool applies? (c) does over-selection expose unnecessary attack surface?

---

## Recommendations

**Stage 1 — Instrument the decoupling (week 1–2).** Add `retrieved_tools` and `selected_tool` to your Inspect `TaskState.metadata` on every run. Ship recall@k, NDCG@k, conditional selection accuracy, and the distractor-robustness curve as standing metrics. *Threshold that changes the plan:* if recall@k is high (>0.9) but conditional selection accuracy is low, your bottleneck is the **selector** (invest in descriptions / few-shot / reasoning); if recall@k is low, your bottleneck is **retrieval** (invest in the embedding model / query planning).

**Stage 2 — Build the private eval set (week 2–4).** Back-translate intents from your `@command` registry with function-masking (no name leakage), inject sibling + nearest-neighbor + random distractors, and add a Hammer-style "no tool applies" abstention slice (target ≥15% of cases). Validate with an LLM judge + a strong-model filter (MCP-Radar style) to drop trivial/parametric-knowledge cases; dedup by embedding similarity. *Target:* ≥1,000 cases spanning catalog sizes 1→128.

**Stage 3 — Adopt ToolRet + BEIR framing for retrieval; reuse the BFCL Inspect port for selection (week 4–6).** Score your retriever exactly as ToolRet does (NDCG@10 over your full catalog as the corpus; remember the field-leading embedding model only hit 33.83 there — set realistic targets). For selection AST/structural checks, fork `inspect_evals/bfcl` rather than reimplementing. Use deepdiff for parameters.

**Stage 4 — Anchor to end-task, don't measure discovery with it (ongoing).** Run τ²-bench and AppWorld quarterly to confirm discovery-layer gains translate to task success and don't introduce collateral damage. Treat their scores as *guardrails*, not *targets*.

**Stage 5 — Add safety/counterfactual evals before any production catalog expansion.** Implement a ToolCert-style adversarial near-duplicate injection test and report Clopper–Pearson bounds on selection accuracy. *Threshold:* block catalog expansion if adversarial injection drops conditional selection accuracy beyond a pre-agreed margin.

**What would change these recommendations:** If a standardized retrieval-vs-selection benchmark emerges (watch ToolRet successors and MCP-Atlas), adopt it and reallocate effort from building to running. If your catalog stays under ~20 tools, the distractor problem is minor and you can defer progressive-disclosure evaluation. If Anthropic/MCP ship a native progressive-disclosure eval harness, integrate rather than rebuild.

---

## Caveats

- **Several headline numbers come from vendor/secondary sources, not peer review.** Anthropic's Tool Search figures (Opus 4: 49%→74%; Opus 4.5: 79.5%→88.1%) are from Anthropic's own engineering blog [8]; treat as directional, not independently replicated. The BFCL "43%→2%" and "58%→26%" figures are from a practitioner analysis (Allen Chan, Medium) summarizing BFCL-derived experiments, not the BFCL paper itself [10]. The "~26 pp still retrieval failure" line is MarkTechPost commentary [43], not an Anthropic statement.
- **ToolRet's "33.83 NDCG@10 for NV-Embed-v1" was verified against the primary ACL Findings 2025 / arXiv source** ("even the best model… achieves an nDCG@10 of only 33.83") [1] — safe to quote.
- **ToolSandbox's exact tool count was not verbatim-confirmed** from the primary source; do not cite a specific number without checking the arXiv PDF. Its license is a bespoke Apple license materially different from BEIR/MTEB's Apache-2.0 [29].
- **MCP-Atlas (arXiv 2602.00933) and several 2026 papers are very recent** and may not be peer-reviewed; the "Claude Opus 4.5 = 62.3%" figure is from the preprint [32].
- **Benchmark scores age fast.** All leaderboard numbers (BFCL GLM-4.5 0.778, etc.) are snapshots; re-check before relying on them.
- The author's "~49% baseline tool-selection accuracy at 50+ tools" aligns strikingly with Anthropic's pre-Tool-Search Opus 4 figure (49%) [8] and with AppWorld's GPT-4o normal-task success (~49%) [5], but these measure different things (selection vs end-task); the convergence is coincidental and should not be over-interpreted.

---

## References

1. Shi Z, Wang Y, Yan L, Ren P, Wang S, Yin D, Ren Z. Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models. ACL Findings 2025. [https://aclanthology.org/2025.findings-acl.1258/](https://aclanthology.org/2025.findings-acl.1258/) · arXiv: [https://arxiv.org/abs/2503.01763](https://arxiv.org/abs/2503.01763) · code: [https://github.com/mangopy/tool-retrieval-benchmark](https://github.com/mangopy/tool-retrieval-benchmark)
2. Huang Y, et al. MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use. ICLR 2024. [https://arxiv.org/abs/2310.03128](https://arxiv.org/abs/2310.03128) · code: [https://github.com/HowieHwong/MetaTool](https://github.com/HowieHwong/MetaTool)
3. Patil SG, Mao H, et al. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML 2025. [https://proceedings.mlr.press/v267/patil25a.html](https://proceedings.mlr.press/v267/patil25a.html) · leaderboard (V4): [https://gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html) · code: [https://github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
4. Barres V, Dong H, Ray S, Si X, Narasimhan K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. 2025. [https://arxiv.org/abs/2506.07982](https://arxiv.org/abs/2506.07982) · code: [https://github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
5. Trivedi H, Khot T, et al. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. ACL 2024. [https://aclanthology.org/2024.acl-long.850/](https://aclanthology.org/2024.acl-long.850/) · site: [https://appworld.dev/](https://appworld.dev/) · code: [https://github.com/StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld)
6. Ye J, et al. ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use. 2025. [https://arxiv.org/abs/2501.02506](https://arxiv.org/abs/2501.02506)
7. Chen C, et al. ACEBench: Who Wins the Match Point in Tool Usage? 2025. [https://arxiv.org/abs/2501.12851](https://arxiv.org/html/2501.12851)
8. Anthropic. Introducing advanced tool use on the Claude Developer Platform. 2025. [https://www.anthropic.com/engineering/advanced-tool-use](https://www.anthropic.com/engineering/advanced-tool-use)
9. Gan T, Sun Q. RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation. 2025. [https://arxiv.org/abs/2505.03275](https://arxiv.org/abs/2505.03275)
10. Chan A. How Tool Complexity Impacts AI Agents Selection Accuracy. Medium, 2025. [https://achan2013.medium.com/how-tool-complexity-impacts-ai-agents-selection-accuracy-a3b6280ddce5](https://achan2013.medium.com/how-tool-complexity-impacts-ai-agents-selection-accuracy-a3b6280ddce5)
11. Thakur N, Reimers N, Rücklé A, Srivastava A, Gurevych I. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets & Benchmarks 2021. [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) · code: [https://github.com/beir-cellar/beir](https://github.com/beir-cellar/beir)
12. Muennighoff N, Tazi N, Magne L, Reimers N. MTEB: Massive Text Embedding Benchmark. 2022. [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) · code: [https://github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)
13. UK AI Security Institute. Inspect AI: A Framework for Large Language Model Evaluations. [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/) · code: [https://github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)
14. Shihipar T (Anthropic). MCP Tool Search for Claude Code (announcement, 14 Jan 2026), via atcyrus. [https://www.atcyrus.com/stories/mcp-tool-search-claude-code-context-pollution-guide](https://www.atcyrus.com/stories/mcp-tool-search-claude-code-context-pollution-guide)
15. Synaptic Labs. The Meta-Tool Pattern: Progressive Disclosure for MCP. 2025. [https://blog.synapticlabs.ai/bounded-context-packs-meta-tool-pattern](https://blog.synapticlabs.ai/bounded-context-packs-meta-tool-pattern)
16. Where LLM Agents Fail and How They Can Learn From Failures (AgentErrorTaxonomy / AgentDebug). 2025. [https://arxiv.org/abs/2509.25370](https://arxiv.org/pdf/2509.25370)
17. Characterizing Faults in Agentic AI: A Taxonomy of Types, Symptoms, and Root Causes. 2026. [https://arxiv.org/abs/2603.06847](https://arxiv.org/html/2603.06847v1)
18. Lin X, et al. LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions. 2025. [https://arxiv.org/abs/2509.18970](https://arxiv.org/abs/2509.18970)
19. Healy K, Srinivasan B, Madathil V, Wu J (Amazon). Internal Representations as Indicators of Hallucinations in Agent Tool Selection. 2026. [https://arxiv.org/abs/2601.05214](https://arxiv.org/pdf/2601.05214)
20. Liu Z, Hoang T, Zhang J, et al. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets. NeurIPS 2024. [https://arxiv.org/abs/2406.18518](https://arxiv.org/abs/2406.18518) · site: [https://apigen-pipeline.github.io/](https://apigen-pipeline.github.io/)
21. Liu W, et al. ToolACE: Winning the Points of LLM Function Calling. 2024. [https://arxiv.org/abs/2409.00920](https://arxiv.org/html/2409.00920v2)
22. Lin Q, et al. Hammer: Robust Function-Calling for On-Device Language Models via Function Masking. ICLR 2025. [https://arxiv.org/abs/2410.04587](https://arxiv.org/abs/2410.04587) · code: [https://github.com/MadeAgents/Hammer](https://github.com/MadeAgents/Hammer)
23. Liu J, et al. (NAACL 2025) Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis (ToolFlow). [https://aclanthology.org/2025.naacl-long.214.pdf](https://aclanthology.org/2025.naacl-long.214.pdf)
24. Qin Y, et al. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (ToolBench). ICLR 2024. [https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789) · code: [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)
25. Yao S, Shinn N, Razavi P, Narasimhan K. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. 2024. [https://arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045)
26. Zhong L, Du Z, Zhang X, Hu H, Tang J. ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario. 2025. [https://arxiv.org/abs/2501.10132](https://github.com/zai-org/ComplexFuncBench)
27. Basu K, et al. NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls. EMNLP 2025. [https://aclanthology.org/2025.emnlp-main.1702/](https://aclanthology.org/2025.emnlp-main.1702/) · arXiv: [https://arxiv.org/abs/2409.03797](https://arxiv.org/abs/2409.03797)
28. Guo Z, et al. StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models. ACL Findings 2024. [https://arxiv.org/abs/2403.07714](https://arxiv.org/abs/2403.07714) · code: [https://github.com/THUNLP-MT/StableToolBench](https://github.com/THUNLP-MT/StableToolBench)
29. Lu J, Holleis T, Zhang Y, et al. (Apple). ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities. 2024. [https://arxiv.org/abs/2408.04682](https://arxiv.org/abs/2408.04682) · code: [https://github.com/apple/ToolSandbox](https://github.com/apple/ToolSandbox)
30. Wang P, et al. MTU-Bench: A Multi-granularity Tool-Use Benchmark for Large Language Models. 2024. [https://arxiv.org/abs/2410.11710](https://arxiv.org/abs/2410.11710) · site: [https://mtu-bench-team.github.io/](https://mtu-bench-team.github.io/)
31. Gao X, et al. MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models. 2025. [https://arxiv.org/abs/2505.16700](https://arxiv.org/abs/2505.16700)
32. MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP. 2026. [https://arxiv.org/pdf/2602.00933](https://arxiv.org/pdf/2602.00933)
33. Wang J, et al. GTA: A Benchmark for General Tool Agents (NeurIPS 2024 D&B) & GTA-2 (2026). [https://openreview.net/forum?id=akEt8QAa6V](https://openreview.net/forum?id=akEt8QAa6V) · code: [https://github.com/open-compass/GTA](https://github.com/open-compass/GTA)
34. Li M, et al. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. EMNLP 2023. [https://aclanthology.org/2023.emnlp-main.187/](https://aclanthology.org/2023.emnlp-main.187/)
35. iternal.ai. Which LLM to Choose in 2026? Selection Guide + Benchmarks (BFCL saturation discussion). 2026. [https://iternal.ai/llm-selection-guide](https://iternal.ai/llm-selection-guide)
36. LLM Stats. BFCL v2 Benchmark Leaderboard (contamination/bias via live scenarios). [https://llm-stats.com/benchmarks/bfcl-v2](https://llm-stats.com/benchmarks/bfcl-v2)
37. Yeon J, et al. ToolCert: Adversarial Certification of Tool-Selection Accuracy (Clopper–Pearson bounds). 2025. Summarized in Emergent Mind, Tool Selection Accuracy. [https://www.emergentmind.com/topics/tool-selection-accuracy-ts](https://www.emergentmind.com/topics/tool-selection-accuracy-ts)
38. RandomWorld: Procedural Environment Generation for Tool-Use Agents (type-guided sampling). 2026. [https://arxiv.org/html/2601.17829](https://arxiv.org/html/2601.17829)
39. Lin et al. MassTool: search-based user intent modeling for tool retrieval. 2025. Discussed in: Tools Are Under-Documented: Document Expansion Boosts Tool Retrieval. [https://arxiv.org/pdf/2510.22670](https://arxiv.org/pdf/2510.22670)
40. Inspect Evals. BFCL: Berkeley Function-Calling Leaderboard port. [https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/bfcl/](https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/bfcl/)
41. Orq.ai. LLM Benchmarks Explained: Significance, Metrics & Challenges (contamination, saturation). 2025. [https://orq.ai/blog/llm-benchmarks](https://orq.ai/blog/llm-benchmarks)
42. When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation. 2026. [https://arxiv.org/html/2602.16763v1](https://arxiv.org/html/2602.16763v1)
43. MarkTechPost. Hermes Agent Ships Tool Search for MCP: Anthropic Evals Show 49% to 74% Accuracy Gain on Opus 4. 29 May 2026. [https://www.marktechpost.com/2026/05/29/hermes-agent-ships-tool-search-for-mcp-anthropic-evals-show-49-to-74-accuracy-gain-on-opus-4/](https://www.marktechpost.com/2026/05/29/hermes-agent-ships-tool-search-for-mcp-anthropic-evals-show-49-to-74-accuracy-gain-on-opus-4/)
44. Lunar.dev. Dynamic Tool Selection for AI Agents: Solving the Context Management Problem. 2026. [https://www.lunar.dev/post/why-dynamic-tool-discovery-solves-the-context-management-problem](https://www.lunar.dev/post/why-dynamic-tool-discovery-solves-the-context-management-problem)
45. Tian Pan. The Tool Selection Problem: How Agents Choose What to Call When They Have Dozens of Tools. 2026. [https://tianpan.co/blog/2026-04-09-tool-selection-problem-agent-tool-routing-at-scale](https://tianpan.co/blog/2026-04-09-tool-selection-problem-agent-tool-routing-at-scale)