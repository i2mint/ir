"""The #99 experiment: does a synopsis rescue thin-description packages, and at what LLM cost?

Three in-memory corpora over the labelled packages of the private ``thorwhalen/ir-eval-data``
benchmark, identical except strategy:

- **a** ``Package()``;
- **b** ``with_synopsis(Package())``: a synopsis for every package;
- **c** ``with_synopsis(Package(), when=thin_description_predicate)``: only where the
  description is empty or thin (``min_chars`` 40, and the sweep below).

The synopses use :data:`ir.synopsis.TOPICAL_SYNOPSIS_PROMPT` through ``aix``
(``--model``, default ``claude-cli/haiku``: the Claude Code CLI on a subscription). They
are computed once, in parallel, and cached in the data directory; (b) and (c) read the
same cache through a counting synthesizer, so they differ only in *which* packages get
one, and (c)'s call count is what its predicate asks for.

Reported per theme: Recall@10 and mean rank of the **thin-description relevant**
packages (``thin_description`` and labelled ``core``/``strong``), graded nDCG@20 on the
well-described packages only, and LLM calls. #99's target: (c) at most 15% of (b)'s calls.

Aggregates go to stdout (public); synopses and per-package ranks stay in the data dir.

Run: ``python misc/evals/synopsis_conditional.py <ir-eval-data dir>``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ir
from ir import eval as ev
from ir.base import Artifact
from ir.sources import package_repo_dirs
from ir.store import CorpusStore
from ir.synopsis import TOPICAL_SYNOPSIS_PROMPT, make_llm_synthesizer

sys.path.insert(0, str(Path(__file__).parent))  # the sibling harness
from dep_signal_real import load  # noqa: E402

K = 20
DEPTH = 300  # past the corpus size: every package gets a rank
RELEVANT = {"core", "strong"}
MIN_CHARS = (20, 40, 80, 160)
SYNTH_ID = "topical-synopsis-eval"


def _source(names, strategy):
    wanted, by_name, seen = set(names), {}, Counter()
    for d in package_repo_dirs():
        if d.name in wanted:
            by_name.setdefault(d.name, d)
            seen[d.name] += 1
    src = ir.CorpusSource.from_packages(
        repos=lambda: by_name.values(), exclude=[], strategy=strategy
    )
    return src, wanted - set(by_name)


def _synopses(names, cache_path: Path, *, model: str, workers: int) -> dict:
    """``{package: synopsis}`` for every labelled package, cached across runs."""
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    src, _ = _source(names, ir.Package())
    synth = make_llm_synthesizer(
        prompt=TOPICAL_SYNOPSIS_PROMPT,
        model=model,
        text_fields=ir.Package().synopsis_fields,
        synthesizer_id=SYNTH_ID,
    )
    todo = [(aid, raw) for aid, raw in src.items() if aid not in cache]

    def one(item):
        aid, raw = item
        return aid, synth(Artifact(id=aid, raw=raw))

    with ThreadPoolExecutor(workers) as pool:
        for aid, text in pool.map(one, todo):
            cache[aid] = text
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


class _Counting:
    """A synthesizer reading the cache and counting the calls a strategy makes."""

    synthesizer_id = SYNTH_ID

    def __init__(self, cache):
        self.cache, self.calls = cache, 0

    def __call__(self, artifact):
        self.calls += 1
        return self.cache.get(artifact.id, "")


def _build(names, strategy):
    src, missing = _source(names, strategy)
    return ir.build(src, store=CorpusStore.memory()), missing


def run(data_dir, *, model, workers):
    data_dir = Path(data_dir)
    cases, probes, _ = load(data_dir)
    names = [c.artifact_id for c in cases]
    cache = _synopses(
        names, data_dir / "synopses_topical.json", model=model, workers=workers
    )
    arms, calls = {}, {}
    arms["a Package()"], missing = _build(names, ir.Package())
    every = _Counting(cache)
    arms["b synopsis for all"], _ = _build(
        names, ir.with_synopsis(ir.Package(), synthesize=every)
    )
    calls["b synopsis for all"] = every.calls
    for m in MIN_CHARS:
        thin = _Counting(cache)
        label = f"c synopsis when thin (min_chars={m})"
        arms[label], _ = _build(
            names,
            ir.with_synopsis(
                ir.Package(),
                synthesize=thin,
                when=ir.make_thin_description_predicate(min_chars=m),
            ),
        )
        calls[label] = thin.calls
    cases = [c for c in cases if c.artifact_id not in missing]
    well = [c for c in cases if not c.thin_description]
    well_ids = {c.artifact_id for c in well}
    report = {
        "k": K,
        "missing_checkouts": sorted(missing),
        "calls": calls,
        "themes": {},
    }
    for theme in probes:
        thin_rel = {
            c.artifact_id
            for c in cases
            if c.thin_description and c.level(theme) in RELEVANT
        }
        rows, ranks = {}, {}
        for arm, corpus in arms.items():
            hits = ir.search(corpus, probes[theme], k=DEPTH, mode="hybrid")
            ranking = list(dict.fromkeys(h.artifact_id for h in hits))
            pos = {a: i + 1 for i, a in enumerate(ranking)}
            thin_ranks = [pos.get(a, DEPTH + 1) for a in thin_rel]
            row, _ = ev.score_ranking(
                [a for a in ranking if a in well_ids], well, theme, k=K
            )
            rows[arm] = {
                "thin relevant recall@10": (
                    sum(r <= 10 for r in thin_ranks) / len(thin_ranks)
                    if thin_ranks
                    else 0.0
                ),
                "thin relevant mean rank": statistics.mean(thin_ranks)
                if thin_ranks
                else 0.0,
                "nDCG@20 (well-described)": row["ndcg"],
                "LLM calls": calls.get(arm, 0),
            }
            ranks[arm] = {a: pos.get(a, 0) for a in thin_rel}
        report["themes"][theme] = {
            "n_thin_relevant": len(thin_rel),
            "arms": rows,
            "thin_ranks": ranks,
        }
    return report


def as_markdown(report):
    out = []
    b = report["calls"]["b synopsis for all"]
    for theme, t in report["themes"].items():
        out += [
            f"### {theme} ({t['n_thin_relevant']} thin-description relevant packages)",
            "",
            "| arm | thin recall@10 | thin mean rank | nDCG@20 (well-described) | LLM calls (% of b) |",
            "|---|---|---|---|---|",
        ]
        for arm, r in t["arms"].items():
            share = f"{r['LLM calls']} ({r['LLM calls'] / b:.0%})" if b else "0"
            out.append(
                f"| {arm} | {r['thin relevant recall@10']:.2f} | "
                f"{r['thin relevant mean rank']:.1f} | {r['nDCG@20 (well-described)']:.3f} | {share} |"
            )
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", help="a checkout of thorwhalen/ir-eval-data")
    parser.add_argument("--model", default="claude-cli/haiku")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    report = run(args.data_dir, model=args.model, workers=args.workers)
    out = Path(args.data_dir) / "synopsis_conditional_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(as_markdown(report))
    print(f"\n(per-package ranks and synopses, private: {out.parent})")


if __name__ == "__main__":
    main()
