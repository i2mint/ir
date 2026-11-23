"""The #68 sweep: does up-weighting the ``deps`` surface at fusion help on the real benchmark?

Two in-memory corpora over the labelled packages of the private ``thorwhalen/ir-eval-data``
benchmark, identical except ``Package(embed_deps=...)``. On the one with a ``deps``
surface, ``ir.search(..., mode="hybrid", surface_weights={"deps": w})`` for each ``w``.
The decision rule is #68's: the ``w`` that maximizes hard-positive recall@20 without
raising the named-distractor FP-rate@20 above the no-deps baseline; ``embed_deps`` is
worth turning on by default only if that ``w`` beats the baseline on recall.

Aggregates go to stdout (public); the per-package ranks, which name packages, go to
``<data dir>/deps_weight_sweep_report.json``.

Run: ``python misc/evals/deps_weight_sweep.py <ir-eval-data dir>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import ir
from ir import eval as ev
from ir.sources import package_repo_dirs
from ir.store import CorpusStore

sys.path.insert(0, str(Path(__file__).parent))  # the sibling harness
from dep_signal_real import load  # noqa: E402

K = 20
DEPTH = 100
WEIGHTS = (1.0, 1.5, 2.0, 3.0)
RELEVANT = {"core", "strong"}


def build(names, *, embed_deps: bool):
    """An in-memory packages corpus over *names* (first checkout wins, as in the harness)."""
    wanted, by_name, seen = set(names), {}, Counter()
    for d in package_repo_dirs():
        if d.name in wanted:
            by_name.setdefault(d.name, d)
            seen[d.name] += 1
    src = ir.CorpusSource.from_packages(
        repos=lambda: by_name.values(),
        exclude=[],
        strategy=ir.Package(embed_deps=embed_deps),
    )
    return ir.build(src, store=CorpusStore.memory()), wanted - set(by_name)


def run(data_dir):
    cases, probes, named_sets = load(data_dir)
    names = [c.artifact_id for c in cases]
    plain, missing = build(names, embed_deps=False)
    deps, _ = build(names, embed_deps=True)
    cases = [c for c in cases if c.artifact_id not in missing]
    arms = {"no deps surface": (plain, None)}
    arms |= {f"deps, w={w}": (deps, {"deps": w}) for w in WEIGHTS}
    report = {"k": K, "missing_checkouts": sorted(missing), "themes": {}}
    for theme, named in named_sets.items():
        relevant = {c.artifact_id for c in cases if c.level(theme) in RELEVANT}
        rows, ranks = {}, {}
        for arm, (corpus, weights) in arms.items():
            hits = ir.search(
                corpus, probes[theme], k=DEPTH, mode="hybrid", surface_weights=weights
            )
            ranking = list(dict.fromkeys(h.artifact_id for h in hits))
            row, cells = ev.score_ranking(ranking, cases, theme, k=K, named_sets=named)
            rows[arm] = {
                "nDCG@20": row["ndcg"],
                "recall@20": len(set(ranking[:K]) & relevant) / len(relevant),
                "hard-positive recall@20": row["hard_positive_recall"],
                "distractor FP-rate@20": row["fp_rate"],
            }
            ranks[arm] = {
                a: cells[a]["rank"] for a in (*named.hard_positives, *named.distractors)
            }
        base = rows["no deps surface"]
        ok = [
            a
            for a in rows
            if a != "no deps surface"
            and rows[a]["distractor FP-rate@20"] <= base["distractor FP-rate@20"]
        ]
        best = max(ok, key=lambda a: rows[a]["hard-positive recall@20"], default=None)
        report["themes"][theme] = {
            "n_hard_positives": len(named.hard_positives),
            "n_distractors": len(named.distractors),
            "arms": rows,
            "named_ranks": ranks,
            "best": best,
        }
    return report


def as_markdown(report):
    out = []
    for theme, t in report["themes"].items():
        out += [
            f"### {theme} ({t['n_hard_positives']} hard positives, {t['n_distractors']} distractors)",
            "",
            "| arm | nDCG@20 | recall@20 | hard-positive recall@20 | distractor FP-rate@20 |",
            "|---|---|---|---|---|",
        ]
        for arm, r in t["arms"].items():
            out.append(
                f"| {arm} | {r['nDCG@20']:.3f} | {r['recall@20']:.2f} | "
                f"{r['hard-positive recall@20']:.2f} | {r['distractor FP-rate@20']:.2f} |"
            )
        out += ["", f"Best by #68's rule: **{t['best']}**", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", help="a checkout of thorwhalen/ir-eval-data")
    args = parser.parse_args()
    report = run(args.data_dir)
    out = Path(args.data_dir) / "deps_weight_sweep_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(as_markdown(report))
    print(f"\n(per-package ranks, private: {out})")


if __name__ == "__main__":
    main()
