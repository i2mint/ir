# ir.synonyms_eval

Replay the synonym ledger through `ir.eval`: keep the entries that earn their place.

A ledger entry is added after one verified find. Whether it keeps helping is testable:
an entry’s `provenance` names the query that missed and the artifact ids the
vocabulary found (see [`ir.synonyms`](ir.synonyms.html.md#module-ir.synonyms)), which is a [`ir.eval.DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase).
[`replay_ledger()`](#ir.synonyms_eval.replay_ledger) scores each such case twice, with and without the entry’s
expansion, through [`ir.eval.evaluate_discovery()`](ir.eval.html.md#ir.eval.evaluate_discovery) (so the numbers are the
harness’s), and says whether the entry `helps`, has `no_effect` or `hurts`.

What it will not do is judge on thin evidence: an entry is `unverifiable` when it has
no provenance, when its gold ids are not in this corpus (a global entry replayed on
the wrong corpus, or a corpus rebuilt since), or when `vd` is unavailable so lexical
scores are not the real ones. With `apply=True` the verdicts are written back
([`apply_verdicts()`](#ir.synonyms_eval.apply_verdicts)): a win bumps `verified` and re-activates a demoted entry; a
`hurts` demotes; a `no_effect` demotes only an entry that has never won a replay
(the miss may simply be fixed by now). Demoted entries stay in the file and are no
longer loaded by default. Replay returns verdicts and optionally patches the ledger; it
never searches on its own initiative and holds no loop (ir/raglab boundary).

### Module Attributes

| [`HELPS`](#ir.synonyms_eval.HELPS)        | Verdicts of one entry's replay.                                               |
|---------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`NO_EFFECT`](#ir.synonyms_eval.NO_EFFECT)    | Verdicts of one entry's replay.                                               |
| [`HURTS`](#ir.synonyms_eval.HURTS)        | Verdicts of one entry's replay.                                               |
| [`UNVERIFIABLE`](#ir.synonyms_eval.UNVERIFIABLE) | Verdicts of one entry's replay.                                               |
| [`DFLT_EPSILON`](#ir.synonyms_eval.DFLT_EPSILON) | A replay score gain at or below this counts as no change (float noise guard). |

### Functions

| [`apply_verdicts`](#ir.synonyms_eval.apply_verdicts)(verdicts)                        | The `(old, new)` entry changes a set of verdicts implies (pure; no I/O).                                                             |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [`case_of`](#ir.synonyms_eval.case_of)(entry)                                  | The [`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase) an entry's provenance encodes, if complete. |
| [`replay_entry`](#ir.synonyms_eval.replay_entry)(corpus, entry, \*[, mode, k, ...]) | Replay one entry: NDCG@k of its miss case without, then with, its expansion.                                                         |
| [`replay_ledger`](#ir.synonyms_eval.replay_ledger)(corpus, \*[, entries, path, ...]) | Replay the ledger's entries for `corpus` (demoted ones too, so a win re-activates).                                                  |

### Classes

| [`EntryVerdict`](#ir.synonyms_eval.EntryVerdict)(entry, verdict[, without, ...])   | One entry's replay: scores of its miss case without and with the expansion.   |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### ir.synonyms_eval.DFLT_EPSILON *= 1e-09*

A replay score gain at or below this counts as no change (float noise guard).

### *class* ir.synonyms_eval.EntryVerdict(entry, verdict, without=None, with_entry=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One entry’s replay: scores of its miss case without and with the expansion.

### ir.synonyms_eval.HELPS *= 'helps'*

Verdicts of one entry’s replay.

### ir.synonyms_eval.HURTS *= 'hurts'*

Verdicts of one entry’s replay.

### ir.synonyms_eval.NO_EFFECT *= 'no_effect'*

Verdicts of one entry’s replay.

### ir.synonyms_eval.UNVERIFIABLE *= 'unverifiable'*

Verdicts of one entry’s replay.

### ir.synonyms_eval.apply_verdicts(verdicts)

The `(old, new)` entry changes a set of verdicts implies (pure; no I/O).

```pycon
>>> e = SynonymEntry(("a1", "b1"))
>>> [(n.verified, n.status) for _, n in apply_verdicts([EntryVerdict(e, HELPS)])]
[(1, 'active')]
>>> [(n.verified, n.status) for _, n in apply_verdicts([EntryVerdict(e, HURTS)])]
[(0, 'demoted')]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`SynonymEntry`](ir.synonyms.html.md#ir.synonyms.SynonymEntry), [`SynonymEntry`](ir.synonyms.html.md#ir.synonyms.SynonymEntry)]]

### ir.synonyms_eval.case_of(entry)

The [`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase) an entry’s provenance encodes, if complete.

* **Return type:**
  [`DiscoveryCase`](ir.eval.html.md#ir.eval.DiscoveryCase) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.synonyms_eval.replay_entry(corpus, entry, , mode='lexical', k=10, epsilon=1e-09, \*\*search_kw)

Replay one entry: NDCG@k of its miss case without, then with, its expansion.

`mode` defaults to `lexical` (the ledger serves keyword search; `ir.eval`
itself defaults to `hybrid`). `formulate` cannot be passed: the entry’s own
expansion is the thing under test.

* **Return type:**
  [`EntryVerdict`](#ir.synonyms_eval.EntryVerdict)

### ir.synonyms_eval.replay_ledger(corpus, , entries=None, path=None, apply=False, mode='lexical', k=10, \*\*search_kw)

Replay the ledger’s entries for `corpus` (demoted ones too, so a win re-activates).

`entries` overrides what to replay (a dry run only: `apply` with it raises,
since the ledger lines are then not the ones being judged). With `apply=True`
the verdicts are patched into the ledger at `path` line by line
([`ir.synonyms.update_entries()`](ir.synonyms.html.md#ir.synonyms.update_entries)); unverifiable entries, entries of other
corpora and every line this code does not understand are left untouched.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`EntryVerdict`](#ir.synonyms_eval.EntryVerdict)]
