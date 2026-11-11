# Keyword and regex search done well, versus dense search, for an agent searching session transcripts

*Deep research report, 2026-11-11. Builds on the in-repo records `ir_05` (dense vs lexical vs hybrid eval run), `ir_08` (magnitude-preserving fusion) and `ir_09` (the composable search agent and its Formulator seam); it does not repeat them. Web sources retrieved on the date above. Vendor-reported numbers are marked as such.*

**What was built from this report (2026-11-11).** The synonym ledger, v1: `ir/synonyms.py` (JSONL in the config dir, `terms` + `oneway` + `note` + `added` + optional `corpus`; the provenance, use counts and eval replay of 1.3(b) are not in v1), the CLI `ir synonyms` / `ir synonyms-add`, and `--expand` on `ir search` / `ir discover` (a deterministic ledger Formulator, recommendation 1.3(c)2). The skill guidance of 1.3(a) went into the `ir-search` skill and the `ir-discover-agent` subagent. The index-free grep first stage for session transcripts landed in `astern grep` (structure-aware: it matches decoded conversational fields, and widens `\s`, `"`, `\b`, `^`, `$` in its raw-line prefilter, which also handles the JSON-escape gotcha of sec. 5.3). The rest of 1.3(c) is tracked as an issue.

---

## 1. Summary and recommendations

### 1.1 Verdict on the hypothesis

The hypothesis under test: *"regex/keyword search needs no index maintenance and yields good results if there is good intelligence in making the regex/keywords, so an agent can emulate semantic search with keyword search plus a maintained synonym ledger."*

**Verdict: mostly true for a personal transcript archive, false as a general claim, and the synonym ledger is the weakest of its three legs.** Broken into its parts:

1. **"No index maintenance": true for grep at personal scale, with a ceiling.** A warm-cache ripgrep pass over a ~9 GB, ~8.8k-file archive of coding-agent transcripts took 0.4 to 1.8 s on a laptop (measured for this report). Nothing goes stale, nothing has to be re-embedded when an embedding model changes, and new sessions are searchable the instant they are written. That is the property Claude Code's team gives for dropping its early vector index (staleness, security, privacy, reliability) [1][2]. The ceiling is latency at monorepo scale, and the fix the industry chose there is a *regex* index, not an embedding index: Cursor reports cutting regex search from 16.8 s to 13 ms with a sparse n-gram index (secondary reporting of a vendor figure) [5], in the lineage of Google Code Search's trigram index [9], zoekt [8] and GitHub's Blackbird [10]. Note also that BM25 *does* need an index (ir_05 found a per-query rebuild did not scale), so "keyword" and "no index" only coincide for grep.
2. **"Good results if the patterns are intelligent": supported, with a measurable gap to dense on paraphrase-heavy tasks.** For: an AWS study finds an agent with keyword-search tools alone reaches over 90% of a vector-RAG baseline's metrics across six datasets [15]; on BRIGHT, feeding BM25 GPT-4-written reasoning instead of the raw query roughly doubles nDCG@10 (about 14.5 to about 27, figures vary by paper version) and lets BM25 beat every raw-query dense model [17]; Augment's top SWE-bench agent found embeddings were "not the bottleneck: grep and find were sufficient" because the agent's persistence compensated [11]; Letta's agent scored 74.0% on LoCoMo conversational memory with file tools [39]. Against: on BrowseComp-Plus the same GPT-5 agent goes from 55.9% with BM25 to 70.1% with a dense retriever *and* makes fewer search calls [16]; Cursor reports +12.5% average answer accuracy (6.5% to 23.5% by model) when semantic search is added to grep, largest on big codebases (vendor eval) [4]; a 2026 repository-QA study finds vector retrieval answered 65.2% correctly against 46.2% for deep agentic grep search, at under half the cost per correct answer [14]; a structural index beat agentic grep on localization and cost per solve on SWE-bench Pro [13]; grep-only agents burn more tokens and tool calls (39% fewer tokens with an index, vendor eval) [12]. And Letta's "filesystem" agent also had a semantic `search_files` tool, so it is not a pure-grep result [39].
3. **"Emulate semantic search with a synonym ledger": only partially.** A ledger is Furnas's "unlimited aliasing" remedy for the vocabulary problem (two people pick the same word for a thing with probability under 0.20; aliasing gives several-fold gains) [20], and it works for *recurring, corpus-specific* vocabulary. But a modern LLM already knows general-English synonyms and generates them on demand (query2doc, HyDE, CoT expansion add 3% to 15% or more to BM25 with no stored ledger at all) [26][27][28]. What the LLM cannot know is *your* vocabulary: project codenames, renamed packages, private jargon, the word you used last spring for what you now call something else. That is what a ledger should hold. A ledger of general synonyms adds drift risk without adding knowledge; expansion is known to help weak retrievers and hurt strong ones [30], and full-description queries gain little even from hand-chosen thesaurus expansion [25]. No ledger rescues conceptual or affective queries with no stable surface form ("the session where the deploy kept failing and I gave up"); those need dense retrieval or summaries.

**Bottom line for transcripts.** Transcripts are unusually identifier-rich (paths, package names, error strings, commands, issue numbers, timestamps, a `cwd` field on every line), which is grep's home ground. Make grep the default first move, structured by the JSONL fields, with LLM-generated expansion on every query and a small corpus-specific alias ledger. Keep `ir`'s dense/hybrid `sessions` index as the second move for conceptual queries, and judge misses explicitly rather than looping. The evidence says agent+grep is a strong baseline that loses to agent+grep+dense by a single-digit to low-double-digit margin on paraphrase-heavy tasks, never the reverse at equal agent quality [4][13][14][16].

### 1.2 Decision table

| Query shape (examples from a session archive) | First move | If it misses | Why |
|---|---|---|---|
| Exact identifier: package, file, function, flag, issue `#123`, commit SHA | `rg -F` / `rg -w` over raw JSONL, scoped by field | Fix escaping (sec. 5.3), try separator/case variants | Lexical match is exact; embeddings blur rare tokens [18] |
| Error string, stack-trace line, command | `rg -F` on a distinctive fragment (no line numbers, no paths) | Shorter fragment; drop version-specific parts | Fixed strings avoid regex metachar mistakes |
| Named entity with known aliases (project renamed, acronym) | `rg -i` with alternation from the ledger | LLM-proposed variants, then add a ledger entry if one hits | This is exactly what a ledger is for [20] |
| Time- or place-scoped ("last week in the ir repo") | Metadata filter first (`cwd`, `timestamp`, file mtime), then content | Widen the window once | Time-aware filtering is a measured win in conversational memory [41] |
| Topical, stable vocabulary ("abstention calibration") | BM25 or `rg -i` with 3 to 6 LLM-expanded terms | Hybrid (`ir --mode hybrid`) | Text-rich corpora favor hybrid (ir_05) |
| Conceptual / paraphrased ("when did we decide not to put the loop in the library") | Hybrid over the `sessions` index (or dense) with multi-query | Grep for the decision's likely *artifacts* (file names, issue numbers) found in the top hits | Vocabulary mismatch; dense wins on paraphrase [16][17] |
| Affective / episodic ("the frustrating deploy session") | Session-level summaries or titles, then drill down | Dense over user turns | No stable surface form; summary-then-drill is the memory-system pattern [38][41] |
| "Have I ever..." (existence, need high recall) | `rg -c` / `rg -l` with broad alternation | Hybrid as a second net, union results | Grep gives exhaustive recall on its vocabulary; dense catches the rest |

### 1.3 Recommendations

**(a) Agent instructions and skills** (the `ir-search` skill and the `ir-discover-agent` subagent are the natural homes):

1. **Classify the query before choosing a tool**, with the table above. Default to grep when the query contains any rare token (identifier, path, quoted string, number, proper name); default to hybrid when it contains none.
2. **Always expand, never with one query.** Write 3 to 6 alternatives covering synonyms, morphological variants, separator and case variants, and the *artifacts* the answer would leave behind (file names, commands, issue numbers). Multi-query fusion is the cheapest reliable gain [26][28][48].
3. **Scope before scanning**: filter by project (`cwd`), time (`timestamp` prefix or file mtime) and role (`"type":"user"` vs assistant) before content matching. Count (`-c`), then list (`-l`), then extract bounded snippets (`-o '.{0,120}term.{0,120}'`); never print whole transcript lines, which can be megabytes.
4. **Respect JSON escaping** (sec. 5.3): `\b` and `-w` silently fail on words that follow an escaped newline; use `(?:\b|\\n)term` or decode the field with `jq` first.
5. **Judge misses explicitly** (sec. 7.3) and **stop** after at most 3 reformulation rounds without new candidates, or 2 lexical rounds with disjoint vocabularies on a conceptual query; then switch to hybrid rather than writing a fourth regex.
6. **Read before returning**: open the top 2 to 5 hits and check the real intent; this is the back-edge that makes grep competitive [11].
7. **Propose a ledger entry after a bridged miss** (rule in 1.3b), never mid-search, and never for general-English synonyms.

**(b) Synonym-ledger format and maintenance rule.** One small JSONL file per corpus (or per project scope), human-readable, git-tracked:

```json
{"term": "ir", "aliases": ["information retrieval substrate", "\\bir\\s+(?:package|lib)"], "kind": "equivalent", "scope": "sessions", "regex": true,
 "provenance": {"added": "2026-11-11", "miss_query": "retrieval library sessions", "fix_query": "ir package", "found": ["<session-id>#<turn>"]},
 "uses": 0, "verified_finds": 1, "last_used": null, "status": "proposed"}
```

- `kind` follows the Elasticsearch/Solr distinction [31][33]: `equivalent` (all terms interchangeable, like `a, b, c`) or `oneway` (`a => b`, for acronyms, misspellings and renames, so the old name finds new content without the reverse flooding results).
- **Add an entry only when all four hold**: (1) a query missed (zero hits, or no verified hit in the top 10); (2) a reformulation found a target; (3) the target was read and verified relevant; (4) the bridging term is corpus-specific, meaning the LLM would not have produced it unprompted (test: ask for expansions of the original query without the ledger; if the bridge term appears, do not store it). This is the search-log practice of learning substitutions from user reformulation sessions [34], applied to the agent's own reformulations.
- **Every entry is also a test case**: its `miss_query` and `found` ids form a regression eval. Re-run the ledger's queries (with and without the entry) through `ir.eval` periodically; an entry that does not raise recall, or that adds many unverified hits, is demoted.
- **Expansion hygiene**: weight original terms above expansion terms (RM3 interpolates the original query with feedback terms for this reason [22]; query2doc repeats the query 5 times for BM25 [26]); cap aliases per term (around 5); keep equivalents small and prefer one-way mappings; never chain expansions (alias of an alias). Unbounded expansion causes query drift [47].
- **Pruning**: demote `status` to `stale` when unused for 90 days or when `verified_finds / uses` falls under a floor; delete stale entries at review. Keep provenance so a bad entry can be traced and rolled back.

**(c) Features worth adding to `ir`** (each respects the ir/raglab boundary: they return queries or hits, never a loop):

1. **A grep Retriever leaf** (`mode="grep"` or a `GrepSource`): run ripgrep over a corpus's raw files and return `SearchHit`s with zero index. Score by match count with field weights and recency; push metadata filters (`cwd`, date, role) into the scan. This gives `ir` an always-fresh, index-free first stage and lets hybrid fuse grep hits with dense hits. It is the "expose your retrieval systems as tools" advice from both sides of the debate [11][4].
2. **A keyword Formulator** next to `make_llm_formulator`: a prompt that targets BM25/grep (term lists with variants and likely artifacts, CoT-style [28]) rather than fluent paraphrases, which suit dense retrieval. And a deterministic **`ledger_formulator(ledger)`** that expands from the alias file with no LLM. Both are pure `Formulator`s (query in, queries out).
3. **Weighted multi-query fusion**: today multi-query fusion keeps the best surface per artifact; add an option to weight the original query above formulations (guards against drift [30][47]).
4. **The ledger as a `dol`-backed store with an eval hook**: `ir` owns the file format, the deterministic expansion and `ir.eval` replay of each entry's provenance; the decision *when* to add an entry stays in the skill/subagent (or raglab). This is the "back-edge as instructions" pattern the repo already uses.
5. **A lexical miss signal on `Selection`**: BM25 scores have no calibrated scale (ir_07, ir_08), so feed `sufficient` with lexical evidence: zero hits, hits only via expansion terms, or the must-have entity absent from the top-k.
6. **Session-level records for drill-down**: the `sessions` source already indexes turn pairs and titles; add a per-session summary record (first user prompt plus title plus touched files) as the coarse level of a summary-then-drill search [41].

---

## 2. Theme 1: agentic grep versus embeddings

### 2.1 What coding agents chose, and why

- **Claude Code** dropped an early RAG + local vector index for "agentic search" (glob, grep, read), reporting it "generally works better" and is simpler, without the security, privacy, staleness and reliability issues of a stored index [1][2]. Anthropic's context-engineering guidance frames this as just-in-time retrieval: the agent holds lightweight identifiers (paths, queries) and loads content through tools when needed, with CLAUDE.md loaded up front [3]. The claim rests on internal usage and benchmarks, not a published controlled comparison [2].
- **Cursor** kept both and measured: +12.5% average accuracy on its internal Context Bench with semantic search added to grep (6.5% to 23.5% by model), +0.3% code retention overall and +2.6% on codebases with 1,000+ files, and 2.2% more dissatisfied follow-ups without semantic search [4]. Its conclusion: "the combination of these two leads to the best outcomes" and semantic search is "currently necessary to achieve the best results, especially in large codebases" [4]. Separately it invested heavily in making *regex* fast (sparse n-gram index) because agents use grep so much [5].
- **Sourcegraph Cody** removed embeddings for enterprise in 5.3 and moved to its existing search engine (zoekt-backed keyword search), reporting context "works as well as embeddings and is sometimes better", plus operational simplicity [7][8].
- **Aider** uses no embeddings: a tree-sitter symbol map ranked by a PageRank-style graph over references, fit to a token budget [6]. Structure, not semantics, does the work.
- **Augment** (SWE-bench): embedding tools did not help because retrieval was not the bottleneck on small repositories; the agent's persistence compensated. Embeddings still matter for large codebases, unstructured text (chats, docs), unfamiliar third-party code [11].

### 2.2 Benchmarks where an LLM agent drives the retriever

- **BrowseComp-Plus** (fixed corpus, human-verified evidence): GPT-5 55.9% accuracy with BM25 versus 70.1% with Qwen3-Embedding-8B, with fewer search calls; Search-R1 (a small trained agent) only 3.86% with BM25 [16]. The paper's framing is useful: stronger *agents* gain by making more calls; stronger *retrievers* improve both accuracy and efficiency [16]. Agent intelligence substitutes for retriever quality only partially, and at a token cost.
- **BRIGHT** (reasoning-intensive retrieval): all raw-query retrievers score poorly (best around 18 to 24 nDCG@10); replacing the query with GPT-4 reasoning roughly doubles BM25 to about 27 and helps lexical more than dense [17]. This is the strongest single piece of evidence *for* the hypothesis: an LLM that writes the query can make BM25 competitive with or better than dense on hard queries. Caveat: a later reproduction found BRIGHT's BM25 implementation differs from Anserini/Pyserini, so exact numbers are soft.
- **Keyword search is all you need** (AWS, 2025): tool-using agents with only keyword search reach over 90% of vector-RAG metrics, recommended where the knowledge base changes often [15]. Read as "close, not equal".
- **BEIR**: BM25 [46] remains a strong zero-shot baseline that many dense models fail to beat out of domain [18]; this is why hybrid rarely loses.
- **Anthropic Contextual Retrieval**: adding BM25 to contextual embeddings takes the top-20 retrieval failure reduction from 35% to 49%, and reranking to 67% [19]. Lexical adds signal dense misses even in a dense-first system.
- **2026 code studies** (disconfirming for grep-only): a structural index (vector search + call graph) improved localization and lowered cost per solved task against agentic grep with Claude Opus held fixed [13]; on repository QA, vector retrieval beat deep agentic search 65.2% to 46.2%, with 41.8% of agentic failures at the planner-to-subagent hand-off [14].

### 2.3 When each wins

- **Keyword wins**: identifiers and rare tokens (dense models tokenize and blur them), error strings, exact phrases, names, numbers and dates, negation-free existence questions needing exhaustive recall, and anything requiring freshness or zero setup. Also when the user remembers the *words* used.
- **Dense wins**: paraphrase and vocabulary mismatch, conceptual questions, cross-lingual or cross-register queries (user asks casually; text is technical), terse documents (ir_05: on one-line package descriptions dense > hybrid > lexical, lexical nDCG@10 0.31).
- **Hybrid wins** when text is keyword-rich *and* queries vary in phrasing (ir_05: skills and reports, hybrid significantly best) [19].
- An important caveat on `ir`'s own numbers: ir_05's eval masked artifact names out of descriptions and back-translated queries into natural requests, which deliberately removes lexical search's main advantage. Transcript queries written by an agent that saw the target vocabulary will favor lexical more than ir_05 suggests.

---

## 3. Theme 2: query expansion for lexical search

### 3.1 The problem it solves

Vocabulary mismatch is the core weakness of lexical search. Furnas et al. found that two people choose the same name for an object with probability under 0.20 across five domains, so a single designer-chosen term fails 80% to 90% of the time; their remedy was "unlimited aliasing", many access terms per object, which gave several-fold improvement [20].

### 3.2 Classic methods

- **Relevance feedback (Rocchio)** moves the query vector toward known-relevant documents and away from non-relevant ones [21]. **Pseudo-relevance feedback** assumes the top-k are relevant; **RM3** builds a relevance language model from them and interpolates it with the original query [22]. BM25+RM3 is a strong, often under-reported baseline [23]. Failure mode: if the first pass is off-topic, feedback amplifies the error (query drift) [47].
- **Thesaurus/WordNet expansion**: little or negative benefit when queries are already complete descriptions, even with hand-chosen concepts; real gains only for short, incomplete queries [25]. General-purpose thesauri lack domain vocabulary [24].
- **Stemming/lemmatization** (Porter [43]): cheap recall for morphological variants; in regex this becomes `stem\w*`.
- **Index-time expansion**: doc2query [45] and learned sparse models like SPLADE [44] add predicted terms to documents, moving expansion cost to indexing. A middle ground between BM25 and dense, but it reintroduces an index to maintain.

### 3.3 LLM-era methods

- **query2doc**: few-shot LLM writes a pseudo-document, concatenated with the query (repeated 5 times) for BM25: +3% to 15% on MS MARCO and TREC DL without fine-tuning [26].
- **HyDE**: LLM writes a hypothetical answer, which is embedded instead of the query [27]. Built for dense retrieval, also usable as keyword source.
- **Prompted expansion** (Jagerman et al.): chain-of-thought prompts that break the query down beat classical PRF (Bo1, Bo2, KL) and query2doc on MS MARCO and BEIR with BM25 [28].
- **Generative relevance feedback (GRF)**: build the feedback model from LLM-generated text instead of first-pass results: +5% to 19% MAP and +17% to 24% nDCG@10 over BM25+RM3 [29]. This sidesteps PRF's drift when the first pass is bad.
- **Multi-query / RAG-Fusion**: several reformulations, fused by RRF [48]; `ir`'s Formulator already supports this (ir_09).
- **Disconfirming**: across 11 expansion methods, 12 datasets and 24 retrievers, LLM expansion helps weaker retrievers and generally *harms* stronger ones, because added terms dilute the original signal; the exception is long-query shift [30]. For an agent this argues for keeping the original terms dominant and for expanding most aggressively when the base retriever is lexical (weak on paraphrase), which is exactly the grep case.

### 3.4 What this means for the hypothesis

Most of a synonym ledger's job is done on the fly by the LLM. The measured gains of LLM expansion on BM25 come with no stored vocabulary at all [26][28][29]. A ledger earns its keep only where the LLM has no knowledge: private, corpus-specific vocabulary.

---

## 4. Theme 3: synonym ledgers and controlled vocabularies over time

### 4.1 How search engines manage synonyms

- **Elasticsearch / OpenSearch**: the `synonym_graph` token filter handles multi-word synonyms correctly; rules are either equivalent lists (`computer, pc, laptop`) or explicit one-way mappings (`pc => personal computer`) [31]. Elastic recommends **search-time** synonyms, because synonym sets can then be updated without reindexing (`updateable: true` on a search analyzer, or the synonyms API) [32]. Index-time synonyms bake the vocabulary into the index and need a reindex on every change.
- **Solr** offers the same graph filter plus managed synonym resources editable through a REST API [33].
- Practical lessons that transfer: prefer query-time expansion (a ledger that is applied when searching, never baked into an index), use one-way rules for acronyms, misspellings and renames, keep multi-word synonyms in a graph-aware form (in regex: alternation of whole phrases with flexible separators).

### 4.2 Learning synonyms from behavior

- **Query reformulation logs**: Jones et al. mined pairs of queries users issued in sequence within a session to generate query substitutions, preferring this to PRF because it avoids drift [34]. The agent analogue: the pair (missed query, query that found the verified target) is a mined substitution.
- **Zero-result queries** are the standard trigger in site search: each one is a candidate synonym or content gap. For an agent: a zero-hit grep followed by a successful reformulation is the event to log.
- **Domain glossaries** maintained by people work because they encode local vocabulary (product names, internal acronyms) that no general resource contains, the same reason Furnas recommends aliasing [20].

### 4.3 Risks

- **Drift and over-expansion**: every alias adds recall and costs precision; with alternation-based regex the precision loss is silent (more hits, no score). Expansion harms strong retrievers [30], and PRF drifts when the first pass is wrong [47].
- **Ambiguity**: short aliases (`ir`, `ef`, `vd`) match everywhere; they need word boundaries, case sensitivity or context (`ir\s+(?:package|search)`).
- **Staleness in reverse**: the ledger is itself an index to maintain, just a much smaller and human-readable one. The hypothesis' "no index maintenance" becomes "small vocabulary maintenance".
- **Mitigations**: provenance per entry, the four-condition add rule, eval replay of each entry's original miss (1.3b), one-way by default, caps on alias count, and demotion of unused entries.

---

## 5. Theme 4: writing regex well for search

### 5.1 Exploit structure

Coding-agent transcripts are JSONL: one JSON object per line with `type` (`user`, `assistant`, `system`, `summary`, plus bookkeeping types), `timestamp` (ISO 8601), `sessionId`, `cwd`, `gitBranch`, and a `message` whose `content` holds text, `tool_use` blocks (with `name`, e.g. `Bash`, `Grep`, `Edit`, and `input`) and `tool_result` blocks [49]. The JSON is compact (`"type":"user"`, no spaces), which makes field anchors reliable:

- Role: lines matching `"type":"user"` are user turns (including tool results, which are carried in user lines); `"type":"assistant"` are model turns.
- Project: `"cwd":"[^"]*/ir"` restricts to sessions run in a directory named `ir`. Matching the project in the file path (`--glob '*-ir/*.jsonl'`, since project folders are munged cwd paths) is cheaper still.
- Time: `"timestamp":"2026-10-` for a month; file mtime (`--sort modified`, or `find -newermt`) for coarse recency without reading content.
- Tool use: `"name":"Bash"` with `"command":"[^"]*git push` finds pushes; `"name":"Edit"` with `"file_path":"[^"]*retrieve\.py"` finds edits to a file.
- AND across fields on the same line: pipe (`rg '"type":"user"' | rg -i 'ledger'`) or PCRE2 lookaheads (`rg -P '^(?=.*"type":"user")(?=.*(?i)ledger)'`).
- For anything beyond line-level matching, decode first: `jq -c 'select(.type=="user") | .message.content'` then grep the decoded text.

### 5.2 Pattern craft

- **Case**: `-i` by default for prose; case-sensitive for identifiers where case carries meaning (`Selection` the class versus "selection").
- **Word boundaries**: `-w` or `\b` to stop `ir` matching `first`; essential for short tokens.
- **Alternation of synonyms**: `(?i)(?:synonym|alias|thesaurus|controlled[ -]vocab\w*)`, generated from the ledger plus LLM expansion.
- **Morphology**: stem plus `\w*` (`calibrat\w*` covers calibrate, calibrated, calibration); explicit suffix sets when the stem is too short (`index(?:es|ed|ing)?`).
- **Separators and naming styles**: `synonym[-_ ]?ledger` covers `synonym ledger`, `synonym-ledger`, `synonym_ledger`, `synonymledger`; add `synonymLedger` with `-i` or explicit camelCase.
- **Fixed strings** (`-F`) for error messages and paths, which are full of regex metacharacters.
- **Avoid catastrophic backtracking**: ripgrep's default engine (Rust `regex`) uses finite automata with linear-time matching, so nested quantifiers cannot blow up [35][36]. PCRE2 (`-P`, needed for lookaround and backreferences) backtracks: avoid `(a+)+`, `(.*)*`, `.*.*` and unanchored `.*` around alternations. A single backtracking regex took down Cloudflare's network in 2019 [37].
- **Bound the output**: transcript lines embed whole tool outputs and can be megabytes. Use `-o '.{0,120}pattern.{0,120}'`, or `--max-columns 300 --max-columns-preview`, never raw line printing.

### 5.3 A transcript-specific trap (verified)

Inside JSON strings, a newline is stored as the two characters `\` and `n`. A word at the start of a line in the original text is therefore preceded by the letter `n`, so `\bterm` and `rg -w term` **do not match it**. Verified for this report: on the line `{"content":"intro\nsynonym ledger"}`, `rg 'synonym'` matches, while `rg '\bsynonym'` and `rg -w synonym` both return nothing. Use `(?:\b|\\n)synonym`, or decode with `jq` first. Similarly, quotes inside text appear as `\"`, and some characters may be `\uXXXX`-escaped, so a phrase containing quotes must be searched in its escaped form.

### 5.4 ripgrep features worth using

From ripgrep's guide [35]: `-c` (counts per file), `-l` / `--files-with-matches` (staged search), `--json` (machine-readable matches with byte offsets), `-U` / `--multiline` (patterns spanning lines; rarely needed in JSONL), `-P` (PCRE2), `-g` / `--glob` (include/exclude paths, e.g. `--glob '!**/subagents/**'` to skip subagent sidechains), `--sort modified` (newest first, disables parallelism), `--stats`, `-m` (max matches per file), `-e` repeated (several patterns, OR).

### 5.5 Staged search

1. **Cheap metadata filter**: file paths (project folder), mtimes, `cwd` and `timestamp` fields. Reduces the set by one to two orders of magnitude for free.
2. **Content grep with counts**: `rg -c` over the reduced set with the expanded alternation; rank files by count and recency.
3. **Snippet extraction** from the top files with bounded `-o` context.
4. **Read** the best 2 to 5 sessions around the matches (decoded, by turn), and verify.

### 5.6 Cost scaling

Grep is a linear scan: throughput is bounded by disk and memory bandwidth (GB/s on an SSD with a warm cache). Measured for this report: 0.4 to 1.8 s per pattern over ~9 GB of transcripts with a warm cache; a cold cache is bounded by disk read (several seconds). That is cheap per query but adds up when an agent issues dozens of patterns per task, and it grows with the archive. The middle ground that keeps regex semantics is a **trigram (n-gram) index**: extract required trigrams from the regex, intersect posting lists to get candidate files, run the real regex only on those [9]. zoekt [8], GitHub Blackbird [10] and Cursor's sparse n-gram index [5] are production versions. Such an index is content-agnostic (no model, nothing to re-embed when a model changes) and incremental, so it keeps most of grep's "no maintenance" virtue. For a personal archive under tens of GB, plain ripgrep is enough; index only when per-query latency starts to shape the agent's behavior.

---

## 6. Theme 5: searching conversations and session histories

- **MemGPT / Letta** treat past conversation as "recall memory" searchable by the agent through tools (text and date search), paging results into context [38]. Letta's LoCoMo experiment: 74.0% with GPT-4o mini, where the agent iterated its own queries with `grep` and a semantic `search_files` tool, against Mem0's self-reported 68.5% (both vendor numbers; not a pure-grep result) [39]. Letta's own lesson: agent capability in using retrieval tools matters more than the retrieval mechanism [39].
- **LoCoMo** (very long multi-session dialogues) shows long-context models and plain RAG both struggle with temporal and multi-hop questions over conversation history [40].
- **LongMemEval** gives the most transferable design results [41]: (1) **granularity**: index by round (turn) rather than whole session; decomposing into individual facts helps multi-session reasoning but loses information overall; (2) **fact-augmented key expansion**: index each round under keys enriched with extracted facts, +9.4% recall@k on average and +5.4% QA accuracy; (3) **time-aware query expansion**: extract a time range from the question and filter, +6.8% to 11.3% recall on temporal questions with a strong LLM. `ir`'s `sessions` source already indexes turn pairs and titles, which matches (1).
- **Episodic-memory tools for coding agents** (e.g. obra/episodic-memory) index local transcripts with local embeddings plus full-text search and expose `search` / `read` tools [42]; FTS5/BM25 + vector fused by RRF is the common pattern among such tools.
- **What works, consolidated**: filter by time and project first; search at turn granularity; keep a coarse session-level layer (title, first prompt, summary, touched files) for "which session" questions, then drill into turns; combine lexical (for the identifiers that dominate coding sessions) with dense (for the paraphrased recollections that dominate human questions).
- **Specific to coding-agent transcripts**: user turns are short and conversational, assistant turns are verbose and technical, and tool results are huge and noisy. Search user turns for *intent* questions ("what did I ask for"), assistant text for *decisions and explanations*, tool `input` for *actions* (commands, edited files), and exclude tool output unless hunting an error string. Subagent sidechain files duplicate content; exclude them unless needed.

---

## 7. Theme 6: a decision procedure for agents

### 7.1 Choose the tool

1. Extract the query's **anchors**: rare tokens (identifiers, paths, numbers, quoted strings, proper names), scope (project, time, role), and the **concept** (what is being looked for, in plain words).
2. Anchors present: grep (scoped), with the anchor and its variants. Concept only: hybrid over the indexed corpus, with multi-query. Both: grep on anchors to narrow, then rank or verify by concept.
3. Need exhaustive recall ("every time we..."): grep with broad alternation, plus hybrid as a second net; union the results.
4. Need "which session" rather than "which passage": search the session-level layer first, then drill down.

### 7.2 Reformulate

- Round 1: original anchors plus LLM expansion (3 to 6 variants) plus ledger aliases.
- Round 2, on a miss: change vocabulary, not just syntax. Search for the *artifacts* the answer would leave (file names, commands, issue numbers, error text), or switch role (user versus assistant turns).
- Round 3: switch modality (grep to hybrid/dense, or the reverse), or widen scope (time window, project).
- Evidence on how many rounds: stronger agents gain by issuing more searches but with diminishing efficiency [16]; persistence is what made grep sufficient on SWE-bench [11]; silent failure at hand-offs is the main risk of deep agentic search [14]. Three rounds without a new candidate is a reasonable stop.

### 7.3 Judge a miss

A round has missed when any of these hold:

- zero hits, or hits only in tool output or subagent copies;
- hits only through expansion terms, none through the original anchors (likely drift);
- the must-have entity or constraint (project, date, file) is absent from all of the top-k;
- the top 2 to 3 hits, read, do not address the intent.

Because BM25 and grep scores have no calibrated scale (ir_07, ir_08), "the top score is low" is not a usable miss signal for lexical search; these structural signals are.

### 7.4 Stop

Stop when a verified hit answers the question; when two consecutive rounds return no new candidates (saturation); or when the budget is spent. Report "nothing sufficiently relevant" with what was tried, rather than returning weak hits. After a stop on success, apply the ledger rule (1.3b) if a reformulation bridged a miss.

---

## 8. Open questions worth measuring in `ir`

1. On the `sessions` corpus, with *agent-written* queries (not name-masked back-translations), how do grep, BM25, dense and hybrid compare? ir_05's protocol biases against lexical; a transcript-specific eval set built from real "find the session where..." requests would settle it.
2. Does a keyword-targeted Formulator beat the paraphrase Formulator for `mode="lexical"`, and does weighting the original query reduce the expansion penalty on dense [30]?
3. How many ledger entries survive their own replay eval after a month of use? If few, the ledger idea should shrink to an alias list for renames and codenames only.

---

## REFERENCES

1. [Building Claude Code with Boris Cherny — The Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/building-claude-code-with-boris-cherny)
2. [Settling the RAG Debate: Why Claude Code Dropped Vector DB-Based RAG and the Reality of Code Search — SmartScope (secondary collation of Cherny's statements)](https://smartscope.blog/en/ai-development/practices/rag-debate-agentic-search-code-exploration/)
3. [Effective context engineering for AI agents — Anthropic Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
4. [Improving agent with semantic search — Cursor (Heule, Jia, Jain, 2025)](https://cursor.com/blog/semsearch)
5. [Fast regex search: indexing text for agent tools — Cursor (2026)](https://cursor.com/blog/fast-regex-search)
6. [Building a better repository map with tree sitter — Aider](https://aider.chat/2023/10/22/repomap.html)
7. [Cody FAQ (embeddings removed in Sourcegraph 5.3) — Sourcegraph docs](https://5.3.sourcegraph.com/cody/faq)
8. [zoekt: fast trigram-based code search — Sourcegraph](https://github.com/sourcegraph/zoekt)
9. [Regular Expression Matching with a Trigram Index — Russ Cox (2012)](https://swtch.com/~rsc/regexp/regexp4.html)
10. [The technology behind GitHub's new code search — GitHub Blog](https://github.blog/engineering/architecture-optimization/the-technology-behind-githubs-new-code-search/)
11. [Why grep beat embeddings in our SWE-Bench agent (lessons from Augment) — Jason Liu with Colin Flaherty](https://jxnl.co/writing/2025/09/11/why-grep-beat-embeddings-in-our-swe-bench-agent-lessons-from-augment/)
12. [Why I'm Against Claude Code's Grep-Only Retrieval? It Just Burns Too Many Tokens — Zilliz](https://zilliz.com/blog/why-im-against-claude-codes-grep-only-retrieval-it-just-burns-too-many-tokens)
13. [Bhola I, et al. Code Isn't Memory: A Structural Codebase Index Inside a Coding Agent. arXiv 2606.22417 (2026)](https://arxiv.org/abs/2606.22417)
14. [Rafiei Oskooei A, et al. Deep Agentic Search for Repository-Level Code Question Answering: An Empirical Study. arXiv 2608.01507 (2026)](https://arxiv.org/abs/2608.01507)
15. [Subramanian S, et al. Keyword search is all you need: Achieving RAG-Level Performance without vector databases using agentic tool use. arXiv 2602.23368 (2025)](https://arxiv.org/abs/2602.23368)
16. [Chen Z, et al. BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent. arXiv 2508.06600 (2025)](https://arxiv.org/abs/2508.06600)
17. [Su H, et al. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval. arXiv 2407.12883 (2024)](https://arxiv.org/abs/2407.12883)
18. [Thakur N, et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. arXiv 2104.08663 (2021)](https://arxiv.org/abs/2104.08663)
19. [Introducing Contextual Retrieval — Anthropic (2024)](https://www.anthropic.com/news/contextual-retrieval)
20. [Furnas GW, Landauer TK, Gomez LM, Dumais ST. The Vocabulary Problem in Human-System Communication. Commun ACM 30(11):964–971 (1987)](https://doi.org/10.1145/32206.32212)
21. [Manning CD, Raghavan P, Schütze H. Relevance feedback and query expansion. Introduction to Information Retrieval, ch. 9 (2008)](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-query-expansion-1.html)
22. [Lavrenko V, Croft WB. Relevance-Based Language Models. SIGIR 2001](https://doi.org/10.1145/383952.383972)
23. [Lin J. The Neural Hype and Comparisons Against Weak Baselines. SIGIR Forum 52(2) (2019)](https://doi.org/10.1145/3308774.3308781)
24. [Carpineto C, Romano G. A Survey of Automatic Query Expansion in Information Retrieval. ACM Computing Surveys 44(1) (2012)](https://doi.org/10.1145/2071389.2071390)
25. [Voorhees EM. Query Expansion Using Lexical-Semantic Relations. SIGIR 1994](https://doi.org/10.1007/978-1-4471-2099-5_7)
26. [Wang L, Yang N, Wei F. Query2doc: Query Expansion with Large Language Models. arXiv 2303.07678 (2023)](https://arxiv.org/abs/2303.07678)
27. [Gao L, Ma X, Lin J, Callan J. Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). arXiv 2212.10496 (2022)](https://arxiv.org/abs/2212.10496)
28. [Jagerman R, et al. Query Expansion by Prompting Large Language Models. arXiv 2305.03653 (2023)](https://arxiv.org/abs/2305.03653)
29. [Mackie I, Chatterjee S, Dalton J. Generative Relevance Feedback with Large Language Models. SIGIR 2023, arXiv 2304.13157](https://arxiv.org/abs/2304.13157)
30. [Weller O, et al. When do Generative Query and Document Expansions Fail? A Comprehensive Study Across Methods, Retrievers, and Datasets. Findings of EACL 2024](https://aclanthology.org/2024.findings-eacl.134)
31. [Synonym graph token filter — Elasticsearch reference](https://www.elastic.co/docs/reference/text-analysis/analysis-synonym-graph-tokenfilter)
32. [Search with synonyms — Elasticsearch reference 8.19](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/search-with-synonyms.html)
33. [Filters: Synonym Graph Filter and managed synonyms — Apache Solr Reference Guide](https://solr.apache.org/guide/solr/latest/indexing-guide/filters.html#synonym-graph-filter)
34. [Jones R, Rey B, Madani O, Greiner W. Generating Query Substitutions. WWW 2006](https://doi.org/10.1145/1135777.1135835)
35. [ripgrep user guide — BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
36. [Regular Expression Matching Can Be Simple And Fast — Russ Cox (2007)](https://swtch.com/~rsc/regexp/regexp1.html)
37. [Details of the Cloudflare outage on July 2, 2019 — Cloudflare Blog](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/)
38. [Packer C, et al. MemGPT: Towards LLMs as Operating Systems. arXiv 2310.08560 (2023)](https://arxiv.org/abs/2310.08560)
39. [Benchmarking AI Agent Memory: Is a Filesystem All You Need? — Letta (2025)](https://www.letta.com/blog/benchmarking-ai-agent-memory)
40. [Maharana A, et al. Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo). arXiv 2402.17753 (2024)](https://arxiv.org/abs/2402.17753)
41. [Wu D, et al. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR 2025, arXiv 2410.10813](https://arxiv.org/abs/2410.10813)
42. [obra/episodic-memory — GitHub](https://github.com/obra/episodic-memory)
43. [Porter MF. An algorithm for suffix stripping (the Porter stemmer). Program 14(3) (1980)](https://tartarus.org/martin/PorterStemmer/)
44. [Formal T, Piwowarski B, Clinchant S. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. arXiv 2107.05720 (2021)](https://arxiv.org/abs/2107.05720)
45. [Nogueira R, Yang W, Lin J, Cho K. Document Expansion by Query Prediction (doc2query). arXiv 1904.08375 (2019)](https://arxiv.org/abs/1904.08375)
46. [Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR 3(4) (2009)](https://doi.org/10.1561/1500000019)
47. [Mitra M, Singhal A, Buckley C. Improving Automatic Query Expansion. SIGIR 1998](https://doi.org/10.1145/290941.290995)
48. [Rackauckas Z. RAG-Fusion: a New Take on Retrieval-Augmented Generation. arXiv 2402.03367 (2024)](https://arxiv.org/abs/2402.03367)
49. [Claude Code JSONL transcript format explained — claude-dev.tools](https://claude-dev.tools/docs/jsonl-format)
