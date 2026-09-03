# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [Unreleased]

### Added

- `ir schedule` — install, inspect and operate the OS job that runs `ir maintain` (launchd on macOS, cron elsewhere) ([#75](https://github.com/i2mint/ir/issues/75))

### Changed

- `ir maintain --all` is fault-isolated per corpus and exits non-zero when any corpus fails; a named `ir.maintain(name)` still raises ([#75](https://github.com/i2mint/ir/issues/75))
- `ir maintain` (CLI) takes a single-run lock so a scheduled sweep and a manual one cannot interleave writes to a corpus store ([#77](https://github.com/i2mint/ir/issues/77))

## [0.1.25] - 2026-06-18

- Add opt-in embedded deps surface to ir.strategy.Package ([#62](https://github.com/i2mint/ir/pull/62)) ([#69](https://github.com/i2mint/ir/pull/69))

## [0.1.24] - 2026-06-18

- Add graded package-relevance eval harness to ir.eval ([#66](https://github.com/i2mint/ir/pull/66)) ([#67](https://github.com/i2mint/ir/pull/67))

## [0.1.23] - 2026-06-16

- Search perf + sessions corpus + background-work policy + oa→aix ([#60](https://github.com/i2mint/ir/pull/60))

## [0.1.22] - 2026-06-13

- with_synopsis — LLM-derived synopsis surfaces as collapsed-tree routing fuel ([#48](https://github.com/i2mint/ir/pull/48))

## [0.1.21] - 2026-06-13

- traverse(query, store, policy) — query-time graph traversal with operator-enforced safety ([#47](https://github.com/i2mint/ir/pull/47))

## [0.1.20] - 2026-06-13

- links: typed-edge view on CorpusStore + GraphStore protocol ([#46](https://github.com/i2mint/ir/pull/46)) ([#53](https://github.com/i2mint/ir/pull/53))

## [0.1.19] - 2026-06-12

- expand(hit) — retrieval-time context expansion operator + neighborhood policies ([#45](https://github.com/i2mint/ir/pull/45)) ([#52](https://github.com/i2mint/ir/pull/52))

## [0.1.18] - 2026-06-12

### Fixed

- fix(strategy): _split never emits blank chunks ([#50](https://github.com/i2mint/ir/pull/50)) ([#51](https://github.com/i2mint/ir/pull/51))

## [0.1.17] - 2026-06-12

- Expansion prereq: SearchHit.surface_index + ledger-backed sibling addressing ([#44](https://github.com/i2mint/ir/pull/44)) ([#49](https://github.com/i2mint/ir/pull/49))

## [0.1.16] - 2026-06-12

- Cross-source fusion: SearchHit.source provenance, fuse_hits, federated discover ([#42](https://github.com/i2mint/ir/pull/42))

## [0.1.15] - 2026-06-11

### Added

- feat(retrieve): injectable query-formulation seam — formulate= + make_llm_formulator ([#32](https://github.com/i2mint/ir/pull/32)) ([#41](https://github.com/i2mint/ir/pull/41))

## [0.1.14] - 2026-06-11

### Added

- feat: registry.retrievers() Mapping view + Selection.sufficient signal ([#40](https://github.com/i2mint/ir/pull/40))

## [0.1.13] - 2026-06-11

### Added

- feat: agent-ready retrieval seams — as_retriever, SearchHit.to_dict, disclose(store=) ([#39](https://github.com/i2mint/ir/pull/39))

## [0.1.12] - 2026-06-11

### Added

- feat(retrieve): magnitude-preserving "blend" hybrid fusion (opt-in) ([#31](https://github.com/i2mint/ir/pull/31))

## [0.1.11] - 2026-06-11

- test+refactor: consume ef RETRIEVAL_METRICS (SSOT); frozen real fixture; guard tests ([#27](https://github.com/i2mint/ir/pull/27))

## [0.1.10] - 2026-06-11

- chore: production hardening — real lint, surfaced-defect fixes, API docs ([#26](https://github.com/i2mint/ir/pull/26))

## [0.1.9] - 2026-06-11

### Added

- feat(eval): min_score calibration — abstention floors via score separability ([#25](https://github.com/i2mint/ir/pull/25))

## [0.1.8] - 2026-06-07

### Added

- feat(eval): selector tuning — sweep_selector + tuned conservative defaults ([#24](https://github.com/i2mint/ir/pull/24))

## [0.1.7] - 2026-06-06

### Added

- feat(select): selection stage + progressive disclosure + discover tool ([#11](https://github.com/i2mint/ir/pull/11)) ([#23](https://github.com/i2mint/ir/pull/23))

## [0.1.6] - 2026-06-06

- perf(retrieve): cache BM25 index per corpus (lexical/hybrid no longer rebuild per query) ([#22](https://github.com/i2mint/ir/pull/22))

## [0.1.5] - 2026-06-06

### Docs

- docs(eval): ir_05 capability-discovery eval run findings ([#20](https://github.com/i2mint/ir/pull/20))

## [0.1.4] - 2026-06-06

### Added

- feat(eval): LLM-backed case generation (ir.eval_gen) for the eval harness ([#19](https://github.com/i2mint/ir/pull/19))

## [0.1.3] - 2026-06-06

### Added

- feat(eval): retrieval-centric capability-discovery eval harness (ir.eval)

## [0.1.2] - 2026-06-05

- Add hybrid retrieval: BM25 + RRF + optional rerank hook ([#17](https://github.com/i2mint/ir/pull/17))

## [0.1.1] - 2026-06-05

- CLI + named-corpus registry; bump to 0.1.0 ([#15](https://github.com/i2mint/ir/pull/15))
- Foundation: corpus-source substrate, persistence, retrieval (3 corpora live) ([#14](https://github.com/i2mint/ir/pull/14))
- Initial commit
