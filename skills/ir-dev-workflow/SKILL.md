---
name: ir-dev-workflow
description: >-
  How to work ON the `ir` package (i2mint/ir): the module map, the hermetic test
  recipe, the CLI grammar golden, the lint and doc rules CI enforces, the
  ir-vs-raglab boundary every change must respect, and how a merge publishes.
  Use when contributing to, fixing, reviewing, or extending `ir` itself: "add a
  search option to ir", "fix this ir test", "where does fusion happen in ir",
  "update the ir CLI", "review this ir PR", "does this belong in ir or raglab".
  Not for USING ir to search corpora (that is the `ir-search` skill).
metadata:
  audience: developers
---

# ir-dev-workflow — contributing to `ir`

`ir` is a retrieval substrate: one "find the relevant things in this corpus"
contract (retrieve, select, disclose) composed from `ef` (embedders, content
hash, reranking, evaluation), `vd` (BM25, RRF, metadata filters) and `dol`
(storage). The design record is `misc/docs/ir_01` … `ir_10`; read the relevant
one before non-trivial work. GitHub issues are the running journal (epic #38).

## Module map (`ir/`)

| Module | Owns |
|---|---|
| `base` | Value types: `Artifact`, `Surface`, `IndexPlan`, `Record`, `SearchHit`; `best_per_artifact`, `tag_source`. |
| `strategy` | The "what do we index?" seam: `WholeText`, `Chunked`, `Skill`, `Package`, `ClaudeTurn`. |
| `sources` | `CorpusSource` and its factories (`from_mapping`, `from_files`, `from_skills`, `from_records`, `from_mcp_tools`, …). |
| `index` | `build` (incremental, ledger-driven) and `open_corpus` (embedder pinned to the built one). |
| `store` | `CorpusStore` over `dol` views: records, ledger, config, calibration, links. `CorpusStore.memory()` for tests. |
| `embed` | Embedder resolution; `"light"` is the numpy-only hashing embedder. |
| `retrieve` | `search` (filter, dense / lexical / hybrid, `surface_weights`, rerank, `dep_boost`), `fuse_hits`, `records_for_artifact`, `as_retriever`. |
| `rerank` | The cross-encoder precision stage and its calibration keys. |
| `select` | `select` / `disclose` / `discover`, abstention floors (`min_score="auto"`). |
| `expand`, `graph`, `traverse` | Context expansion, the typed link graph (`CorpusGraph`, reverse index), and the walk operator with its policies. |
| `formulate`, `synopsis` | Injectable LLM seams (identity / lazy `aix` defaults). |
| `eval`, `eval_gen` | Retrieval and selection metrics, calibration, `compare_indexings`; LLM case generation. |
| `registry`, `policy`, `maintenance`, `schedule`, `coverage` | Named corpora, background work, the OS job, the coverage diagnostic. |
| `cli`, `__main__`, `tools` | The `ir` CLI (dispatched by `cw`) and agent-callable functions. |

## Run the tests (as CI does)

```bash
uv venv -q .venv && . .venv/bin/activate
uv pip install -q -e ".[dev]"
python -m pytest -q --doctest-modules ir tests
ruff check ir && ruff format --check ir
```

Doctests count: every module's examples run. The suite is hermetic, so keep it
that way:

- build test corpora with `embedder="light"` and `store=CorpusStore.memory()`;
- isolate anything that touches the registry with the `IR_CONFIG_DIR`,
  `IR_DATA_DIR` and `IR_CACHE_DIR` environment variables (`monkeypatch.setenv`);
- never require network, an API key, `priv`, `$PP`, or model weights. Inject
  fetchers, synthesizers and rerankers as plain callables instead.

## Conventions CI and review enforce

- **New arguments default to today's behaviour** (progressive disclosure), are
  keyword-only beyond the third positional, and get a test proving the default
  is unchanged.
- **Every module has a top-level docstring** (ruff `D100`); ruff `F`/`E`/`W`/`B`
  run on `ir/` (not on `tests/`).
- **`import ir` stays light and offline:** no `aix`, `torch` or
  `sentence-transformers` at import time; import optional extras lazily.
- **CLI grammar is pinned** in `tests/cli_goldens/ir.json`. Adding a flag to a
  command changes its `usage:` line: update every vector for that command (its
  `--help` and its error vectors) to the new line, and check it with
  `python -m pytest -q tests/test_cli_surface.py`.
- **Don't edit the version** in `pyproject.toml`: CI bumps it on merge to
  `master`, and that merge publishes to PyPI.

## The ir ↔ raglab boundary (check it in every review)

The agent layer lives in `raglab`, which imports `ir`; `ir` never imports
`raglab`.

1. A Formulator returns **queries**, never sub-tasks (decomposition is the
   Planner's job, in raglab).
2. `ir` exposes **signals** (`Selection.sufficient`, scores, abstention), never a
   refinement directive or a control loop. The evaluate-then-reformulate
   back-edge belongs to raglab.
3. Source and library choices are **arguments** (`discover([...])`,
   `reverse_dependency_policy(seeds=...)`, `dependency_evidence_boost(libs, ...)`):
   `ir` never decides which sources or libraries matter for a goal.

## Dependents

`thorwhalen/raglab` is the dependent to keep green. Check it against a working
tree before merging anything that touches public API:

```bash
git clone -q --depth 1 https://github.com/thorwhalen/raglab /tmp/raglab
cd /tmp/raglab && uv venv -q .venv && . .venv/bin/activate
uv pip install -q -e . && uv pip install -q -e /path/to/ir pytest
python -m pytest -q --doctest-modules raglab tests
```
