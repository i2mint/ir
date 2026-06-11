"""Command-line surface for ``ir`` (argh-dispatched).

Commands operate on **named** corpora from the registry (see
:mod:`ir.registry`)::

    ir build skills                 # build/update the skills preset corpus
    ir search skills "deploy app"   # rank candidates (retrieval only)
    ir discover skills "deploy app" # retrieve -> commit to a high-precision subset
    ir discover skills "deploy app" --disclose   # + load each selected item's body
    ir ls                           # list corpora + record counts
    ir info packages                # config + stats for a corpus
    ir register notes files --root ~/notes --pattern '.*\\.md$'
    ir rm notes                     # unregister (keeps built data)
    ir eval-gen skills skills_eval.jsonl --k 5        # generate cases (needs oa/LLM)
    ir eval skills skills_eval.jsonl --mode hybrid    # score retrieval on a case file
    ir eval-select skills skills_eval.jsonl           # score the selection stage
    ir sweep-select skills skills_eval.jsonl          # tune max_k/rel for the selector
    ir calibrate-min-score skills skills_eval.jsonl --persist  # abstention floor
    ir discover skills "deploy app" --min-score auto  # use the calibrated floor
"""

from __future__ import annotations

from . import registry
from .eval import DFLT_CALIB_SENSITIVITY_WEIGHT
from .index import build as _build
from .index import open_corpus


def _empty_corpus_msg(name) -> str:
    """The shared 'corpus is empty, build it first' guard message."""
    return f"corpus {name!r} is empty; build it first: ir build {name}"


def _drift_warning(drift, name) -> str:
    """A warning suffix when cases reference gold ids absent from the corpus.

    Returns ``""`` when there is no drift, so call sites can ``out +=`` it
    unconditionally.
    """
    if not drift:
        return ""
    return (
        f"\n  WARNING: {len(drift)} case(s) reference gold ids absent from "
        f"corpus {name!r} (stale fixture?); their misses are not real."
    )


def ls():
    """List registered corpora with their kind, embedder, and record count."""
    entries = registry.registered()
    if not entries:
        return "No corpora registered. Try: ir build skills"
    lines = []
    for name, e in entries.items():
        try:
            count = len(open_corpus(name))
        except Exception:
            count = 0
        lines.append(
            f"{name:18} {e['kind']:10} {e.get('embedder', 'default'):10} records={count}"
        )
    return "\n".join(lines)


def register(name, kind, *, root=None, pattern=None, embedder="default"):
    """Register a named corpus. kind: skills | packages | reports | files."""
    params = {}
    if root:
        params["root"] = root
    if pattern:
        params["pattern"] = pattern
    registry.register(name, kind, embedder=embedder, **params)
    return f"registered {name!r} (kind={kind}, embedder={embedder})"


def build(name, *, embedder=None, full=True):
    """Build or incrementally update a registered (or preset) corpus."""
    source = registry.source_for(name)
    corpus = _build(source, embedder=embedder, full=full)
    return f"built {name!r}: {len(corpus)} records (embedder {corpus.embedder_id})"


def search(name, query, *, k=10, mode="dense", fusion="rrf"):
    """Search a built corpus and print the top-k hits.

    mode: dense (cosine) | lexical (BM25) | hybrid (dense + BM25 fused). fusion
    (hybrid only): rrf (rank-based, default) | blend (magnitude-preserving;
    better abstention separability — see ir_08).
    """
    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    lines = []
    for h in corpus.search(query, k=k, mode=mode, fusion=fusion):
        # artifact_id is the unique label across corpora (skill name[@parent],
        # package name, or a report's relative path).
        lines.append(f"{h.score:+.3f}  {h.artifact_id}  [{h.surface_kind}]")
    return "\n".join(lines) or "(no matches)"


def discover(
    name,
    query,
    *,
    k=10,
    mode="hybrid",
    strategy="conservative",
    disclose=False,
    min_score=None,
    fusion="rrf",
):
    """Search a corpus, commit to a distractor-robust subset, and show it.

    Retrieves ``k`` candidates, then *selects* the few high-precision results an
    agent should act on (or abstains). ``--disclose`` additionally loads each
    selected item's body (SKILL.md / file text) via its stored pointer.
    ``--min-score auto`` turns on absolute abstention using the floor calibrated
    by ``ir calibrate-min-score`` (or pass a float). mode: dense | lexical |
    hybrid. fusion (hybrid only): rrf (default) | blend (magnitude-preserving;
    pair with --min-score for abstention — see ir_08). strategy: conservative |
    top_k | rel_threshold | score_gap.
    """
    from .select import discover as _discover

    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    floor = (
        "auto"
        if min_score == "auto"
        else (None if min_score is None else float(min_score))
    )
    result = _discover(
        corpus,
        query,
        k=k,
        mode=mode,
        strategy=strategy,
        disclose_level="body" if disclose else "metadata",
        min_score=floor,
        fusion=fusion,
    )
    if result.abstained:
        return (
            f"(abstained: {result.reason}; {result.n_retrieved} candidates retrieved)"
        )
    lines = [
        f"selected {len(result.results)}/{result.n_retrieved} "
        f"({result.strategy} / {result.mode}, {result.reason}):"
    ]
    for d in result.results:
        lines.append(f"  {d.score:+.3f}  {d.name}")
        if disclose and d.body:
            preview = d.body.strip().replace("\n", " ")[:160]
            lines.append(f"           {preview}…")
    return "\n".join(lines)


def info(name):
    """Show a corpus's stored config, stats, and any calibrated abstention floors."""
    corpus = open_corpus(name)
    cfg = corpus.store.get_config()
    reg = registry.get(name)
    calibrated = corpus.store.calibration_modes()
    floors = {m: corpus.store.get_calibration(m).get("min_score") for m in calibrated}
    cal = f"\nmin_score floors: {floors}" if floors else ""
    return (
        f"name: {name}\nregistered: {reg}\nrecords: {len(corpus)}\nconfig: {cfg}{cal}"
    )


def rm(name):
    """Unregister a corpus (does not delete its built data)."""
    registry.unregister(name)
    return f"unregistered {name!r}"


def eval(name, cases, *, mode="hybrid", k=10):
    """Score a built corpus's retrieval against a DiscoveryCase JSONL file.

    cases: path to a JSONL file of cases (see :mod:`ir.eval`); each line is a
    ``{"query": ..., "gold": [artifact_id, ...]}`` record (empty ``gold`` = an
    abstention case). Prints recall@k / NDCG@k / MRR / MAP plus the failure-mode
    taxonomy. mode: dense | lexical | hybrid.
    """
    from .eval import evaluate_discovery, load_cases, validate_cases

    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    case_list = load_cases(cases)
    if not case_list:
        return f"no cases found in {cases!r}"
    drift = validate_cases(corpus, case_list)
    report = evaluate_discovery(
        corpus, case_list, mode=mode, primary_k=k, k_values=tuple(sorted({1, 5, k}))
    )
    out = str(report)
    out += _drift_warning(drift, name)
    return out


def eval_gen(name, out, *, k=5, abstention_frac=0.15, max_artifacts=None):
    """Generate an eval-case file for a corpus by back-translation (needs oa/LLM).

    Writes a DiscoveryCase JSONL set (gold cases + an abstention slice) for the
    registered corpus *name* to *out*, stamping a corpus-signature into the
    header so the frozen file can be checked against the live corpus later. This
    command calls an LLM via oa; scoring it afterwards (`ir eval`) is offline.
    """
    from .eval import save_cases
    from .eval_gen import build_eval_set, corpus_signature

    source = registry.source_for(name)
    kwargs = {}
    if max_artifacts is not None:
        kwargs["max_artifacts"] = int(max_artifacts)
    cases = build_eval_set(
        source, k=k, abstention_frac=abstention_frac, corpus_name=name, **kwargs
    )
    save_cases(
        cases,
        out,
        meta={"corpus": name, "corpus_signature": corpus_signature(source), "k": k},
    )
    n_gold = sum(not c.gold_is_none for c in cases)
    return (
        f"wrote {len(cases)} cases ({n_gold} gold, {len(cases) - n_gold} abstention) "
        f"to {out!r}"
    )


def eval_select(
    name,
    cases,
    *,
    strategy="conservative",
    mode="hybrid",
    k=10,
    max_k=3,
    rel=0.9,
    min_score=None,
):
    """Score a selector against a DiscoveryCase JSONL file (selection quality).

    Reports the conditional commit rate (the selection decision isolated from
    retrieval) plus selection precision / recall / F1 and abstention accuracy.
    strategy: conservative | top_k | rel_threshold | score_gap. mode: dense |
    lexical | hybrid. max_k / rel / min_score tune the commit (see `ir.select`);
    `ir sweep-select` sweeps them to find good values.
    """
    from .eval import evaluate_selection, load_cases, validate_cases

    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    case_list = load_cases(cases)
    if not case_list:
        return f"no cases found in {cases!r}"
    drift = validate_cases(corpus, case_list)
    report = evaluate_selection(
        corpus,
        case_list,
        strategy=strategy,
        mode=mode,
        k=k,
        max_k=int(max_k),
        rel=float(rel),
        min_score=None if min_score is None else float(min_score),
    )
    out = str(report)
    out += _drift_warning(drift, name)
    return out


def sweep_select(
    name,
    cases,
    *,
    strategy="conservative",
    mode="hybrid",
    k=10,
    objective="selection_f1",
):
    """Sweep selector knobs (max_k × rel) against a case file; print the grid + best.

    Retrieves once per case and reuses the candidates across the whole grid, so
    the sweep is cheap. Prints a table (one row per setting, best objective first)
    and the winning max_k / rel. Use it to pick `ir.select` defaults empirically.
    objective: selection_f1 | selection_precision | selection_recall |
    conditional_commit_rate | mean_selected_size. mode: dense | lexical | hybrid.
    """
    from .eval import load_cases, sweep_selector, validate_cases

    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    case_list = load_cases(cases)
    if not case_list:
        return f"no cases found in {cases!r}"
    drift = validate_cases(corpus, case_list)
    sweep = sweep_selector(
        corpus, case_list, strategy=strategy, mode=mode, k=k, objective=objective
    )
    out = str(sweep)
    out += _drift_warning(drift, name)
    return out


def calibrate_min_score(
    name,
    cases,
    *,
    mode="hybrid",
    k=10,
    sensitivity_weight=DFLT_CALIB_SENSITIVITY_WEIGHT,
    persist=False,
    fusion="rrf",
):
    """Calibrate the absolute abstention min_score floor for a corpus + mode.

    Separates in-scope (gold-bearing) from out-of-scope (empty-gold) query top
    scores and reports the floor that best splits them, with sensitivity /
    specificity / Youden's J. ``cases`` must include BOTH gold-bearing and
    abstention cases — generate them with ``ir eval-gen`` (it adds an abstention
    slice). ``--persist`` stores the floor so ``ir discover ... --min-score auto``
    will abstain by it. ``--sensitivity-weight`` (0..1, balanced default);
    lower it to abstain more readily (precision-leaning). For hybrid, calibrate
    with ``--fusion blend`` (rank-based RRF barely separates — see ir_08) and
    query under the same fusion. mode: dense | lexical | hybrid. fusion (hybrid
    only): rrf | blend.
    """
    from .eval import calibrate_min_score as _calibrate
    from .eval import load_cases, validate_cases

    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    case_list = load_cases(cases)
    if not case_list:
        return f"no cases found in {cases!r}"
    drift = validate_cases(corpus, case_list)
    calib = _calibrate(
        corpus,
        case_list,
        mode=mode,
        k=int(k),
        sensitivity_weight=float(sensitivity_weight),
        persist=bool(persist),
        fusion=fusion,
    )
    out = str(calib)
    if persist and calib.min_score is not None:
        fusion_hint = (
            f" --fusion {fusion}" if mode == "hybrid" and fusion != "rrf" else ""
        )
        out += (
            f"\n  persisted → ir discover {name} ... --mode {mode}{fusion_hint} "
            f"--min-score auto"
        )
    if calib.min_score is None:
        out += (
            f"\n  NOTE: no floor calibrated ({calib.reason}); a floor needs both "
            f"gold-bearing and abstention cases. Add abstention cases via ir eval-gen."
        )
    out += _drift_warning(drift, name)
    return out


COMMANDS = [
    ls,
    register,
    build,
    search,
    discover,
    info,
    rm,
    eval,
    eval_gen,
    eval_select,
    sweep_select,
    calibrate_min_score,
]
