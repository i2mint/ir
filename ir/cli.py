"""Command-line surface for ``ir`` (dispatched by cw).

Commands operate on **named** corpora from the registry (see
:mod:`ir.registry`)::

    ir build skills                 # build/update the skills preset corpus
    ir search skills "deploy app"   # rank candidates (retrieval only)
    ir discover skills "deploy app" # retrieve -> commit to a high-precision subset
    ir discover skills "deploy app" --disclose   # + load each selected item's body
    ir ls                           # list corpora + record counts
    ir coverage reports             # disk-vs-index coverage (silent-gap detector)
    ir info packages                # config + stats for a corpus
    ir maintain --all               # run due background work (idempotent)
    ir schedule                     # install/inspect the OS job that runs it
    ir register notes files --root ~/notes --pattern '.*\\.md$'
    ir rm notes                     # unregister (keeps built data)
    ir eval-gen skills skills_eval.jsonl --k 5        # generate cases (needs aix/LLM)
    ir eval skills skills_eval.jsonl --mode hybrid    # score retrieval on a case file
    ir eval-select skills skills_eval.jsonl           # score the selection stage
    ir sweep-select skills skills_eval.jsonl          # tune max_k/rel for the selector
    ir calibrate-min-score skills skills_eval.jsonl --persist  # abstention floor
    ir discover skills "deploy app" --min-score auto  # use the calibrated floor
    ir synonyms acmecorp            # a term's aliases + a grep regex, from the ledger
    ir synonyms-add "acmecorp, acme corp" --note "client nickname"
    ir search session_turns "acmecorp invoice" -m lexical --expand
"""

from __future__ import annotations

import functools
import sys

from . import registry
from .embed import EmbedderError
from .eval import DFLT_CALIB_SENSITIVITY_WEIGHT
from .index import build as _build
from .index import open_corpus


def _empty_corpus_msg(name) -> str:
    """The shared 'corpus is empty, build it first' guard message."""
    return f"corpus {name!r} is empty; build it first: ir build {name}"


def _open_ready(name):
    """Open corpus *name*; a never-built *preset* is built once, on first use.

    Presets are auto-registered by :func:`registry.source_for`, so an empty one
    means "nobody ran ``ir build`` on this machine", not "nothing to find".
    Answering "empty" there made callers silently fall back to grep; building
    (seconds with the default embedder) and saying so does not. Non-preset
    corpora are left alone: their source is the user's to choose.
    """
    corpus = open_corpus(name)
    if len(corpus) == 0 and name in registry.PRESETS:
        print(
            f"ir: preset corpus {name!r} was never built; building it now.",
            file=sys.stderr,
        )
        build(name)
        corpus = open_corpus(name)
    return corpus


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


def _report_embedder_errors(func):
    """Turn an :class:`~ir.embed.EmbedderError` into its actionable message.

    An agent cannot tell "no component found" from "broken index" behind a
    traceback; the message (what is wrong + the command that fixes it) is the
    command's output instead.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EmbedderError as error:
            # stderr + nonzero exit: a failed index is not an empty result, and
            # `&&` chains / agents must not read it as success.
            print(
                f"ir: error (broken index, not an empty result): {error}",
                file=sys.stderr,
            )
            raise SystemExit(2) from error

    return wrapper


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


def register(
    name,
    kind,
    *,
    root=None,
    pattern=None,
    embedder="default",
    reindex_on=None,
    every_hours=None,
    synopsis=False,
):
    """Register a named corpus. kind: skills | packages | reports | files.

    Background-work policy (optional; smart per-kind defaults otherwise — see
    `ir.policy`): reindex_on (source-change | interval | manual), every_hours (for
    interval), synopsis (enable LLM synopses, run only in the policy's downtime
    window by `ir maintain`).
    """
    params = {}
    if root:
        params["root"] = root
    if pattern:
        params["pattern"] = pattern
    maintenance = None
    if reindex_on or every_hours or synopsis:
        reindex = {}
        if every_hours:
            reindex = {"on": "interval", "every_hours": float(every_hours)}
        elif reindex_on:
            reindex = {"on": reindex_on}
        maintenance = {"reindex": reindex, "synopsis": {"enabled": bool(synopsis)}}
    registry.register(name, kind, embedder=embedder, maintenance=maintenance, **params)
    return f"registered {name!r} (kind={kind}, embedder={embedder})"


def build(name, *, embedder=None, full=True):
    """Build or incrementally update a registered (or preset) corpus."""
    source = registry.source_for(name)
    corpus = _build(source, embedder=embedder, full=full)
    return f"built {name!r}: {len(corpus)} records (embedder {corpus.embedder_id})"


@_report_embedder_errors
def search(
    name,
    query,
    *,
    k=10,
    mode="dense",
    fusion="rrf",
    rerank=None,
    rerank_text="surface",
    expand=False,
    filter=None,
    variant_weight=1.0,
):
    """Search a built corpus and print the top-k hits.

    mode: dense (cosine) | lexical (BM25) | hybrid (dense + BM25 fused). fusion
    (hybrid only): rrf (rank-based, default) | blend (magnitude-preserving;
    better abstention separability — see ir_08). ``--rerank cross-encoder`` adds
    the cross-encoder precision stage (needs the ``ir[rerank]`` extra; without it
    the search warns and keeps the first-stage ranking); ``--rerank-text``
    picks what it reads: surface (default) | artifact. Reranked scores are on the
    reranker's own scale. ``--expand`` also searches the query's variants from the
    synonym ledger (``ir synonyms``) and fuses them (best score per artifact across
    the variants; the original query is always one of them; ``--variant-weight 0.5``
    demotes the variants against the query itself, to keep the original on top). ``--filter`` is a
    JSON metadata filter in the Mongo-style form of ``ir.search(filter=...)``,
    e.g. ``--filter '{"project": "dol"}'`` or ``'{"source": {"$in": ["a", "b"]}}'``;
    operators: $eq $ne $in $nin $gt $gte $lt $lte $exists, plus $and/$or/$not.
    """
    corpus = _open_ready(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    lines = []
    hits = corpus.search(
        query,
        k=k,
        mode=mode,
        fusion=fusion,
        **_rerank_kw(rerank, rerank_text),
        **_expand_kw(expand, name, variant_weight),
        **_filter_kw(filter),
    )
    for h in hits:
        # artifact_id is the unique label across corpora (skill name[@parent],
        # package name, or a report's relative path).
        lines.append(f"{h.score:+.3f}  {h.artifact_id}  [{h.surface_kind}]")
    return "\n".join(lines) or "(no matches)"


def _expand_kw(expand, name, variant_weight=1.0):
    """Search kwargs for ledger-based query expansion; empty (today's behavior) when off.

    ``variant_weight`` (below 1.0) demotes the expansion variants against the query
    itself; it needs ``--expand`` (without a formulator there are no variants).
    """
    if not expand:
        if variant_weight != 1.0:
            raise SystemExit("--variant-weight needs --expand (no variants otherwise)")
        return {}
    from .synonyms import make_synonym_formulator

    kw = {"formulate": make_synonym_formulator(corpus=name)}
    if variant_weight != 1.0:
        kw["variant_weight"] = float(variant_weight)
    return kw


def _filter_kw(filter):
    """Search kwargs for a JSON metadata filter; empty (today's behavior) when unset."""
    if filter is None or not str(filter).strip():
        return {}
    import json

    try:
        parsed = json.loads(filter) if isinstance(filter, str) else filter
    except json.JSONDecodeError as e:
        raise SystemExit(
            f'--filter must be a JSON object, e.g. \'{{"k": "v"}}\': {e}'
        ) from None
    if not isinstance(parsed, dict):
        raise SystemExit("--filter must be a JSON object (a Mongo-style filter)")
    if not parsed:
        return {}
    from vd.base import UnsupportedFilterError
    from vd.filters import validate_filter

    try:
        validate_filter(parsed)
    except (UnsupportedFilterError, AttributeError, TypeError, ValueError) as e:
        # vd raises a bare AttributeError for a malformed $and/$or operand
        raise SystemExit(f"--filter: unusable filter ({e})") from None
    return {"filter": parsed}


def _rerank_kw(rerank, rerank_text):
    """Search kwargs for an optional rerank stage; empty (today's behavior) when off."""
    if not rerank:
        return {}
    return {"rerank": rerank, "rerank_text": rerank_text or "surface"}


@_report_embedder_errors
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
    rerank=None,
    rerank_text="surface",
    expand=False,
    filter=None,
    variant_weight=1.0,
):
    """Search a corpus, commit to a distractor-robust subset, and show it.

    Retrieves ``k`` candidates, then *selects* the few high-precision results an
    agent should act on (or abstains). ``--disclose`` additionally loads each
    selected item's body (SKILL.md / file text) via its stored pointer.
    ``--min-score auto`` turns on absolute abstention using the floor calibrated
    by ``ir calibrate-min-score`` (or pass a float). mode: dense | lexical |
    hybrid. fusion (hybrid only): rrf (default) | blend (magnitude-preserving;
    pair with --min-score for abstention — see ir_08). strategy: conservative |
    top_k | rel_threshold | score_gap. ``--rerank cross-encoder`` adds the
    calibrated cross-encoder precision stage (``--rerank-text`` picks what it
    reads); with ``--min-score auto`` it uses the floor calibrated under the same
    ``--rerank`` / ``--rerank-text``. ``--expand`` adds the query's variants from
    the synonym ledger (``ir synonyms``). ``--filter`` takes the JSON form of the
    metadata filter (see ``search``).
    """
    from .select import discover as _discover

    corpus = _open_ready(name)
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
        **_rerank_kw(rerank, rerank_text),
        **_expand_kw(expand, name, variant_weight),
        **_filter_kw(filter),
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


def coverage(name="reports"):
    """Report disk-vs-index coverage for a reports corpus (silent-gap detector).

    Independently walks every ``*/*/docs`` and ``*/*/misc/docs`` tree on disk and
    diffs the includable reports against what is actually indexed, listing any
    file that is on disk but missing from the index (rebuild with ``ir build`` to
    fix). Catches the ingestion gaps that back-translation evals structurally
    cannot — a doc that was never indexed can never be an eval gold id.
    """
    from .coverage import reports_coverage

    corpus = open_corpus(name)
    if len(corpus) == 0:
        return _empty_corpus_msg(name)
    return str(reports_coverage(name))


def info(name):
    """Show a corpus's stored config, stats, policy, and any abstention floors."""
    corpus = open_corpus(name)
    cfg = corpus.store.get_config()
    reg = registry.get(name)
    calibrated = corpus.store.calibration_modes()
    floors = {m: corpus.store.get_calibration(m).get("min_score") for m in calibrated}
    cal = f"\nmin_score floors: {floors}" if floors else ""
    pol = registry.policy_for(name)
    state = corpus.store.get_maintenance_state()
    last = state.get("last_maintained", "never")
    syn = pol.synopsis
    syn_str = f"enabled, scope={syn.scope}/{syn.window_days}d" if syn.enabled else "off"
    window = f", downtime={syn.downtime_hours}" if syn.downtime_hours else ""
    policy_str = (
        f"\npolicy: reindex={pol.reindex.on}"
        + (f"/{pol.reindex.every_hours}h" if pol.reindex.every_hours else "")
        + f", synopsis={syn_str}{window}\nlast maintained: {last}"
    )
    return (
        f"name: {name}\nregistered: {reg}\nrecords: {len(corpus)}\n"
        f"config: {cfg}{policy_str}{cal}"
    )


def maintain(name=None, *, all=False, dry_run=False):
    """Run due background work: incremental reindex, synopsis in its downtime window.

    With a name, maintains that corpus; with --all (or no name), every registered
    corpus. Idempotent and safe to schedule (cron/launchd): it no-ops what is not
    due. --dry-run reports what would run without doing it.
    """
    from .maintenance import MaintenanceBusy, maintain as _maintain, single_run

    try:
        if dry_run:
            results = _maintain(name=name, all=all, dry_run=True)
        else:
            # Overlapping runs can interleave writes to one corpus store; the
            # scheduled job and a manual run both arrive through here.
            with single_run():
                results = _maintain(name=name, all=all)
    except MaintenanceBusy as busy:
        return f"skipped: {busy}"
    if not results:
        return "no corpora registered"
    out = "\n".join(str(r) for r in results)
    failures = [r for r in results if r.error]
    if failures:
        # The report goes to stdout even when the run failed -- `ir maintain --all
        # > report.txt` must not lose the corpora that succeeded just because one
        # did not. Only the summary rides SystemExit, whose job here is the
        # non-zero exit code: a sweep that half-fails and exits 0 is
        # indistinguishable from a healthy one, which is how corpora go stale
        # unnoticed. `ir schedule --status` reads that exit code back.
        print(out)
        raise SystemExit(
            f"{len(failures)} of {len(results)} corpora failed to maintain."
        )
    return out


def schedule(
    *,
    every=None,
    status=False,
    restart=False,
    remove=False,
    dry_run=False,
    backend=None,
):
    """Install, inspect, and operate the OS job that runs ``ir maintain``.

    Bare ``ir schedule`` is idempotent: it creates the schedule when there is
    none, and when there already is one it reports it (interval, definition,
    whether the OS has it loaded, last run) plus how to operate it, rather than
    silently reinstalling over a schedule you tuned.

    An existing definition's environment is carried forward by every operation,
    so operating a working schedule from a shell that happens to lack $PP never
    silently re-points it at a different corpus store.

    every: interval as 15m / 2h / 1d, or a plain number of minutes.
    status: report only; never mutates.
    restart: reload the definition and re-pin it to this interpreter.
    remove: stop the schedule and delete its definition.
    dry_run: print the definition that would be written, and write nothing.
    backend: launchd | cron (default: whichever this machine provides).
    """
    from . import schedule as _schedule

    # Mutually exclusive: silently letting one flag win would make
    # `--restart --every 15m` report success while ignoring the interval.
    chosen = [
        flag
        for flag, on in (
            ("--status", status),
            ("--restart", restart),
            ("--remove", remove),
            ("--dry-run", dry_run),
        )
        if on
    ]
    if len(chosen) > 1:
        return f"ir schedule: {' and '.join(chosen)} cannot be combined"
    if every is not None and (restart or remove or status):
        return f"ir schedule: --every cannot be combined with {chosen[0]}"

    try:
        if status:
            state = _schedule.status(backend=backend)
        elif remove:
            state = _schedule.remove(backend=backend)
        elif restart:
            state = _schedule.restart(backend=backend)
        elif dry_run:
            state = _schedule.install(every=every, backend=backend, dry_run=True)
        else:
            state = _schedule.ensure(every=every, backend=backend)
    except _schedule.ScheduleError as error:
        return f"ir schedule: {error}"
    except OSError as error:
        # The whole point of this command is to report on the machine's
        # scheduler; an unreadable definition or a vanished directory has to read
        # as a message, not a traceback.
        return f"ir schedule: {error}"
    return _schedule.render(state)


def rm(name):
    """Unregister a corpus (does not delete its built data)."""
    registry.unregister(name)
    return f"unregistered {name!r}"


@_report_embedder_errors
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
    """Generate an eval-case file for a corpus by back-translation (needs aix/LLM).

    Writes a DiscoveryCase JSONL set (gold cases + an abstention slice) for the
    registered corpus *name* to *out*, stamping a corpus-signature into the
    header so the frozen file can be checked against the live corpus later. This
    command calls an LLM via aix; scoring it afterwards (`ir eval`) is offline.
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


@_report_embedder_errors
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


@_report_embedder_errors
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


@_report_embedder_errors
def calibrate_min_score(
    name,
    cases,
    *,
    mode="hybrid",
    k=10,
    sensitivity_weight=DFLT_CALIB_SENSITIVITY_WEIGHT,
    persist=False,
    fusion="rrf",
    rerank=None,
    rerank_text="surface",
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
    only): rrf | blend. ``--rerank cross-encoder`` calibrates the floor for a
    reranked search (its own floor, stored apart from the plain one); query with
    the same ``--rerank`` / ``--rerank-text``.
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
        **_rerank_kw(rerank, rerank_text),
    )
    out = str(calib)
    if persist and calib.min_score is not None:
        fusion_hint = (
            f" --fusion {fusion}" if mode == "hybrid" and fusion != "rrf" else ""
        ) + (f" --rerank {rerank}" if rerank else "")
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


def synonyms(term=None, *, corpus=None):
    """Show what the synonym ledger knows: a term's aliases and grep regex, or all entries.

    The ledger (``$IR_SYNONYMS`` or ``<config dir>/synonyms.jsonl``) holds a corpus's
    private vocabulary (nicknames, renamed packages, jargon) learned from misses. Use
    the printed regex in ``rg`` / ``astern grep``; ``ir search --expand`` uses the
    same entries. Add with ``ir synonyms-add``.
    """
    from .synonyms import aliases, ledger_path, load_synonyms, term_regex

    entries = load_synonyms(corpus=corpus)
    if term is None:
        lines = [f"# {ledger_path()} ({len(entries)} entries)"]
        for e in entries:
            arrow = " -> " if e.oneway else " = "
            scope = f"  [{e.corpus}]" if e.corpus else ""
            note = f"  # {e.note}" if e.note else ""
            lines.append(arrow.join(e.terms) + scope + note)
        return "\n".join(lines)
    terms = aliases(term, entries)
    return f"aliases: {', '.join(terms)}\nregex:   {term_regex(terms)}"


def synonyms_add(
    terms, *, note="", oneway=False, corpus=None, miss_query=None, found=None
):
    """Record corpus vocabulary in the synonym ledger: comma-separated equivalent terms.

    ``ir synonyms-add "acmecorp, acme corp, ac" --note "client nickname"``. Add only
    after a verified find that the word made possible; ``--oneway`` makes the first
    term expand to the others but not back; ``--corpus`` scopes the entry.
    ``--miss-query "the query that missed" --found "id1,id2"`` records the miss and
    what the word found, so the entry can later be replayed through ``ir.eval``
    (``ir.synonyms_eval.replay_ledger``) and demoted if it does not help.
    """
    from .synonyms import add_synonyms, ledger_path

    provenance = None
    if miss_query or found:
        if not (miss_query and found):
            raise SystemExit("--miss-query and --found go together")
        provenance = {
            "miss_query": miss_query,
            "found_ids": [i.strip() for i in str(found).split(",") if i.strip()],
            **({"corpus": corpus} if corpus else {}),
        }
    e = add_synonyms(
        [t for t in str(terms).split(",")],
        note=note,
        oneway=oneway,
        corpus=corpus,
        provenance=provenance,
    )
    return f"added to {ledger_path()}: {e.to_json()}"


@_report_embedder_errors
def synonyms_replay(name, *, apply=False, mode="lexical"):
    """Replay the synonym ledger's entries against corpus NAME and say which still help.

    Each entry recorded with ``--miss-query`` / ``--found`` is searched with and without
    its expansion through ``ir.eval``: ``helps``, ``no_effect``, ``hurts`` or
    ``unverifiable`` (no provenance, its ids are not in this corpus, or vd missing).
    Dry run by default; ``--apply`` bumps ``verified`` on a win and demotes a ``hurts``
    (or a ``no_effect`` that never won). Demoted entries stay in the file but stop
    expanding searches.
    """
    from .synonyms import update_entries
    from .synonyms_eval import apply_verdicts, replay_ledger

    corpus = _open_ready(name)
    verdicts = replay_ledger(corpus, mode=mode)
    if not verdicts:
        return f"(no ledger entries for {name!r})"
    lines = []
    for v in verdicts:
        scores = (
            f"  ndcg {v.without:.3f} -> {v.with_entry:.3f}"
            if v.without is not None
            else ""
        )
        lines.append(f"{v.verdict:<12}{' = '.join(v.entry.terms)}{scores}")
    changes = apply_verdicts(verdicts)
    if apply:
        lines.append(f"{update_entries(changes)} ledger line(s) updated")
    else:
        lines.append(f"{len(changes)} ledger line(s) would change (use --apply)")
    return "\n".join(lines)


COMMANDS = [
    ls,
    register,
    build,
    search,
    discover,
    coverage,
    info,
    maintain,
    schedule,
    rm,
    eval,
    eval_gen,
    eval_select,
    sweep_select,
    calibrate_min_score,
    synonyms,
    synonyms_add,
    synonyms_replay,
]
