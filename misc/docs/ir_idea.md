## `ir` — An Information Retrieval Substrate for Agentic Systems

### General context and vision

`ir` is a general-purpose **information retrieval substrate**: a single, coherent abstraction for "find the relevant things in this corpus" that scales across the entire spectrum of retrieval needs. At one end, it serves as a lightweight, on-the-fly, ad hoc `find`-like function over a small or ephemeral collection. At the other, it operates as a full search engine over corpora of many millions of documents. It is designed to be **extensible into a RAG system** without being one by default — retrieval is the core competency, and generation, reranking, citation, and answer synthesis are layered capabilities composed on top rather than baked in.

The design goal is **architectural extensibility through a uniform retrieval contract**. Whatever the corpus and whatever the scale, the same facade applies: a swappable pipeline of indexing, retrieval, selection, and corpus-maintenance components behind a stable, declaratively-configured interface. `ir` can be pointed at *any* corpus, with *any* backing store (in-memory, SQLite, pgvector, a dedicated vector DB), and *any* retrieval strategy (lexical, dense, hybrid, late-interaction, reranked, agentic), without the caller's contract changing. New corpus types and retrieval strategies are added by composition, not rewriting — progressive disclosure at the architecture level: simple retrieval simple, sophisticated retrieval possible.

A first-class concern is **corpus maintenance as a defined, repeatable process** — keeping the index (ledger, embeddings, derived metadata) in sync with a living, mutable corpus. Generalized, the maintenance contract has two pluggable definitions:

- **Scope** — what the corpus *is*: how a refresh process enumerates documents in scope (paths, globs, queries against an external system, API endpoints).
- **Change detection** — what counts as *stale*: modified-date, content hash, version, ETag, or source-specific signals — driving incremental reindexing, invalidation, and additions/renames/deletions.

These are abstract slots the system exposes; concrete definitions are determined per source (and an agent can be tasked with figuring out the right scope and change-detection definitions for a new source).

### Current target: specialized search tools over maintained corpora, exposed to AI agents

The immediate focus is the **retrieval-for-orchestration layer** — using `ir` to give AI agents *specialized, maintained search tools* they can call to find exactly what they need during a task, rather than carrying everything in context. This breaks into several related corpora:

**1. Capability discovery and selection.** The progressive-disclosure problem: an agent facing a large catalog of MCP tools, skills, and subagents suffers context saturation and degraded selection accuracy when everything is loaded eagerly. `ir` provides the single meta-capability — a search-and-select tool — through which the agent discovers and dynamically binds only the relevant subset. This spans **retrieval** (ranked candidates from a heterogeneous, multi-type index) and **selection** (committing to a precise, distractor-robust subset), over artifacts that differ in kind: a skill is a document, a tool is a typed callable with a schema, a subagent is a delegatable context-bearing capability.

**2. Knowledge and development-context retrieval.** The same substrate pointed at the corpora that accumulate around real project work:

- **Research and supporting documents** — the deep-research reports and knowledge files produced during development, living untidily across `docs/`, `misc/docs/`, and shared locations, often duplicated, symlinked, or reused across projects. `ir` makes this dispersed, overlapping set searchable as a coherent corpus and keeps its ledger and embeddings current as documents are added, renamed, edited, or moved — becoming a **research-report retrieval tool** an agent calls to surface relevant prior research for a given subject.
- **Development artifacts** — GitHub issues, discussions, PRs, and commit messages, which carry useful context to pull and analyze during agentic code development. The same maintenance contract applies, with source-specific scope and change-detection definitions.

**3. Preferred-ecosystem discovery.** A search tool over the curated set of tools we *prefer to use*, so that when an agent needs to accomplish something it reaches first for sanctioned solutions rather than arbitrary ones. This corpus has two kinds of members:

- **Preferred third-party tools** — opinionated choices for specific problem classes (e.g., scikit-learn for ML, not TensorFlow). Retrieval here answers "what's our blessed tool for *this kind* of task?"
- **Our own packages** — the 200+ packages we maintain. Here retrieval serves two distinct goals: the usual "what functionality exists?" (so agents prioritize our own tooling), *and* a **generative/extensional** goal — surfacing the right general *class* of solution so that, even when the exact functionality is absent, the agent is directed to the appropriate package to *add* it, growing the ecosystem coherently rather than scattering new code arbitrarily.

The unifying claim is that all of these are the same problem — a maintained corpus, an index kept in sync, a retrieval-and-selection pipeline, and a clean agent-callable surface — differing only in concrete scope, change-detection signal, artifact representation, and index configuration. `ir` is the substrate that makes them one system; these corpora are its first concrete instantiations.

### Special attention: the ecosystem target and the "what do we index?" problem

The preferred-ecosystem target deserves separate scrutiny because it sharpens two questions that run through the whole project but are most acute here: **what do we embed**, and the recognition that **good retrieval is not all embeddings — it is also classic metadata filtering**. The ultimate goal of `ir` is to find the *appropriate resource for a specific need*; semantic similarity is only one signal toward that goal, and for richly-structured artifacts like packages it is often not the dominant one.

A single package is not one document — it is a hierarchy of indexable surfaces, and several distinct concerns coexist:

- **Structured metadata for filtering**, not embedding — package name, ownership (ours vs. third-party), domain/problem-class tags, maturity, dependencies, license. The package name in particular wants to be a first-class filterable field, and "ours vs. preferred-third-party" is a hard filter, not a fuzzy match. Much of the precision in "find the appropriate resource" comes from constraining the candidate set this way *before* or *alongside* semantic ranking.
- **Embeddable representations at multiple granularities and of multiple kinds** — a short canonical description; a longer AI-authored synopsis written by studying the package; and per-package *sub-surfaces*: individual modules, distinct functionalities, the problem classes a package addresses, how-to material. Each of these may warrant its own embedded representation so that a query can match the *right part* of a package rather than the package as an undifferentiated blob — and so that the "class of solution" goal (point 3 above) can match against the problem-classes surface specifically.

The crucial architectural stance is that **deciding what to index is a problem in its own right, and not one `ir` should solve internally.** Producing the synopsis, deciding the sub-surfaces, generating problem-class tags, choosing what becomes filterable metadata versus embedded text — these are upstream *indexing-strategy* concerns, frequently themselves AI-assisted, that vary per corpus and will evolve. `ir`'s responsibility is to expose the **seams** to plug these concerns in: a clean separation between (a) how an artifact is decomposed into filterable fields and embeddable units, (b) how those units are indexed, and (c) how retrieval combines metadata filtering with semantic ranking over them. `ir` should ship **sensible defaults** so it works out of the box on a naive corpus, while leaving every one of these decisions overridable — so that the sophisticated, package-aware indexing strategy this target ultimately needs is a pluggable extension, not a fork of the core.