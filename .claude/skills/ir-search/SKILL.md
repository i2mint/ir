---
name: ir-search
description: >-
  Find specific documents, packages, skills, or past work with the `ir`
  retrieval engine — and have the results reviewed for real relevance before
  they come back. Use whenever the user wants to search/look-up over their own
  corpora: "find the docs/report about X", "use ir (or ir discover) to find …",
  "which of my packages does Y", "search my reports/notes/sessions for Z", "what
  do I have written about …", "is there a doc on …". Covers the `ir` CLI
  (build/search/discover), the corpus map, the mode/scoring gotchas, and when to
  delegate to the `ir-discover-agent` subagent for a reviewed, reformulated
  search vs. running a quick inline lookup. `ir` is a deterministic retrieval
  substrate with no agent of its own; this skill + the subagent are the
  Claude-Code-native way to wrap it in agentic query→review→reformulate powers
  (the coded equivalent is `raglab`).
---

# ir-search — agentic lookup over your `ir` corpora

`ir` retrieves and commits; it does **not** rewrite queries or read results for
relevance. This skill adds that intelligence via Claude Code.

## Decide: inline lookup vs. the reviewing subagent

- **Quick, unambiguous lookup** (you basically know it's there, you want the id/
  path fast) → run the inline recipe below yourself.
- **"Find what I actually mean, and check it"** — the user wants candidates
  *filtered by real relevance*, the need is fuzzy, spans multiple corpora, or the
  first search may miss → **delegate to the `ir-discover-agent` subagent**
  (via the Agent tool, `subagent_type: "ir-discover-agent"`). It formulates
  several queries, reads and grades candidates, reformulates, and returns only
  the verified few — keeping all the candidate-reading out of your main context.
  This is the default for "help me find …" asks.

## The corpora

| Corpus | What's in it |
|---|---|
| `reports` | Project docs / research / design notes / ADRs — Markdown under every project's `docs/` and `misc/docs/` **and their subdirectories** (`research/`, `decisions/`, `adr/`, …). |
| `packages` | The local Python package ecosystem — one card per package (name + description + README). |
| `skills` | Agent skills (`SKILL.md` capabilities). |
| `sessions` | Past Claude Code session transcripts (turn pairs). |
| *(registered)* | Any custom `files` corpus the user registered. |

`ir ls` lists them with record counts; `ir info <corpus>` shows config, freshness,
and any calibrated abstention floors.

## Inline recipe

```bash
ir ls                                              # what corpora exist + how big
ir search reports "<intent, in natural language>" --k 15 --mode dense
ir search reports "<exact terms / identifiers>"   --k 15 --mode lexical
ir discover reports "<best query>" --disclose      # commit to a few + load their bodies
```

Then **read** the top hits (`--disclose`, or `Read` the file at the hit's path)
and judge relevance yourself before reporting — a title/score is not relevance.

## Gotchas (these bit us; encode them)

- **Mode matters, and default hybrid can under-use lexical.** Natural-language
  needs usually rank best under `--mode dense`; identifier/exact-term needs under
  `--mode lexical`. In some installs `--mode hybrid` (RRF) mirrors dense and
  under-weights BM25 — so compare **dense vs lexical explicitly** rather than
  trusting hybrid alone. `--mode hybrid --fusion blend` is a good third net when
  precision/abstention matters.
- **Scores order *within* a mode, and don't compare across modes** (dense cosine
  ≈0.3–0.7, BM25 ≈10–40, RRF ≈0.01–0.05, blend ≈0.5–0.7). Decide relevance by
  reading, not by number.
- **Coverage / freshness.** `ir` only returns what was indexed; an index can lag.
  Run `ir coverage reports` to see, at a glance, how many reports on disk are
  actually indexed and which are missing (it lists on-disk-but-unindexed files);
  rebuild to fix: `ir build <corpus>`. If a search still comes up empty for
  something you believe exists, cross-check with `rg`. (The `reports` corpus
  indexes nested subdirs — if yours predates that, rebuild once.)
- **Abstention is opt-in.** `ir discover … --min-score auto` abstains when nothing
  clears the corpus's calibrated floor (needs a prior `ir calibrate-min-score`).

## Keeping it fresh

`ir build <corpus>` is incremental (content-hash cached) — cheap to re-run to pick
up new/changed docs. `ir maintain --all` runs due background work idempotently
(cron/launchd friendly).

## Want the full coded agent?

This skill + `ir-discover-agent` are a lightweight "raglab-in-instructions". The
real thing — a coded Planner, LLM query formulator, and the evaluator→reformulate
control loop over `ir` corpora — is **[`raglab`](https://github.com/thorwhalen/raglab)**,
built on top of `ir`. Reach for it when you want the loop in code (a service, a
connector, batch runs) rather than driven turn-by-turn from Claude Code.
