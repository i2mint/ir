"""Command-line surface for ``ir`` (argh-dispatched).

Commands operate on **named** corpora from the registry (see
:mod:`ir.registry`)::

    ir build skills                 # build/update the skills preset corpus
    ir search skills "deploy app"   # query it
    ir ls                           # list corpora + record counts
    ir info packages                # config + stats for a corpus
    ir register notes files --root ~/notes --pattern '.*\\.md$'
    ir rm notes                     # unregister (keeps built data)
    ir eval-gen skills skills_eval.jsonl --k 5        # generate cases (needs oa/LLM)
    ir eval skills skills_eval.jsonl --mode hybrid   # score retrieval on a case file
"""

from __future__ import annotations

from . import registry
from .index import build as _build
from .index import open_corpus


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


def search(name, query, *, k=10, mode="dense"):
    """Search a built corpus and print the top-k hits.

    mode: dense (cosine) | lexical (BM25) | hybrid (dense + BM25 fused via RRF).
    """
    corpus = open_corpus(name)
    if len(corpus) == 0:
        return f"corpus {name!r} is empty; build it first: ir build {name}"
    lines = []
    for h in corpus.search(query, k=k, mode=mode):
        # artifact_id is the unique label across corpora (skill name[@parent],
        # package name, or a report's relative path).
        lines.append(f"{h.score:+.3f}  {h.artifact_id}  [{h.surface_kind}]")
    return "\n".join(lines) or "(no matches)"


def info(name):
    """Show a corpus's stored config and stats."""
    corpus = open_corpus(name)
    cfg = corpus.store.get_config()
    reg = registry.get(name)
    return f"name: {name}\nregistered: {reg}\nrecords: {len(corpus)}\nconfig: {cfg}"


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
        return f"corpus {name!r} is empty; build it first: ir build {name}"
    case_list = load_cases(cases)
    if not case_list:
        return f"no cases found in {cases!r}"
    drift = validate_cases(corpus, case_list)
    report = evaluate_discovery(
        corpus, case_list, mode=mode, primary_k=k, k_values=tuple(sorted({1, 5, k}))
    )
    out = str(report)
    if drift:
        out += (
            f"\n  WARNING: {len(drift)} case(s) reference gold ids absent from "
            f"corpus {name!r} (stale fixture?); their misses are not real."
        )
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


COMMANDS = [ls, register, build, search, info, rm, eval, eval_gen]
