"""The #104 experiment: does ef's cross-encoder, reading the whole artifact, clean the top-20?

Three arms on each theme probe of the private package benchmark (``thorwhalen/ir-eval-data``):

- **A** ``ir.search(..., mode="hybrid")``;
- **B** A + ``rerank="cross-encoder"`` (the reranker reads the matched surface);
- **C** A + ``rerank="cross-encoder", rerank_text="artifact"`` (it reads the package's
  sibling surfaces joined: description, README chunks, deps).

All at ``fetch_k=50``. Metrics: graded nDCG@20, P@10 (``core``/``strong`` in the top
10), named-distractor FP-rate@20, hard-positive recall@20, and the median latency of a
query. The pre-registered success rule (#104): C cuts the distractor FP-rate@20 by at
least 40% against A, with hard-positive recall@20 at least A's.

The corpus is built in memory from the labelled packages' checkouts exactly as
``dep_signal_real.py`` builds it. Aggregates go to stdout (public); the per-package
ranks, which name packages, go to ``<data dir>/rerank_abc_report.json``.

Run: ``python misc/evals/rerank_abc.py <ir-eval-data dir>``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import ir
from ir import eval as ev
from ir.rerank import resolve_reranker

sys.path.insert(0, str(Path(__file__).parent))  # the sibling harness
from dep_signal_real import build, load  # noqa: E402

K = 20
FETCH_K = 50
RELEVANT = {"core", "strong"}
REPEATS = 5
ARMS = {
    "A hybrid": {},
    "B + cross-encoder (surface)": {
        "rerank": "cross-encoder",
        "rerank_text": "surface",
    },
    "C + cross-encoder (artifact)": {
        "rerank": "cross-encoder",
        "rerank_text": "artifact",
    },
}


def _run(corpus, probe, kw):
    t0 = time.perf_counter()
    hits = ir.search(corpus, probe, k=FETCH_K, fetch_k=FETCH_K, mode="hybrid", **kw)
    return [h.artifact_id for h in hits], time.perf_counter() - t0


def run(data_dir):
    if resolve_reranker("cross-encoder") is None:
        raise RuntimeError("the cross-encoder is unavailable (install ir[rerank])")
    cases, probes, named_sets = load(data_dir)
    corpus, missing, _ = build(c.artifact_id for c in cases)
    cases = [c for c in cases if c.artifact_id not in missing]
    report = {
        "k": K,
        "fetch_k": FETCH_K,
        "missing_checkouts": sorted(missing),
        "themes": {},
    }
    for theme, named in named_sets.items():
        probe = probes[theme]
        relevant = {c.artifact_id for c in cases if c.level(theme) in RELEVANT}
        rows, ranks = {}, {}
        for arm, kw in ARMS.items():
            _run(corpus, probe, kw)  # warm: model load and BM25 index out of the timing
            times = []
            for _ in range(REPEATS):
                ranking, dt = _run(corpus, probe, kw)
                times.append(dt)
            row, cells = ev.score_ranking(ranking, cases, theme, k=K, named_sets=named)
            rows[arm] = {
                "nDCG@20": row["ndcg"],
                "P@10": len(set(ranking[:10]) & relevant) / 10,
                "distractor FP-rate@20": row["fp_rate"],
                "hard-positive recall@20": row["hard_positive_recall"],
                "median latency (ms)": 1000 * statistics.median(times),
            }
            ranks[arm] = {
                a: cells[a]["rank"] for a in (*named.hard_positives, *named.distractors)
            }
        a, c = rows["A hybrid"], rows["C + cross-encoder (artifact)"]
        cut = (
            1 - c["distractor FP-rate@20"] / a["distractor FP-rate@20"]
            if a["distractor FP-rate@20"]
            else 0.0
        )
        report["themes"][theme] = {
            "n_hard_positives": len(named.hard_positives),
            "n_distractors": len(named.distractors),
            "arms": rows,
            "named_ranks": ranks,
            "fp_cut_C_vs_A": cut,
            "success": cut >= 0.40
            and c["hard-positive recall@20"] >= a["hard-positive recall@20"],
        }
    return report


def as_markdown(report):
    out = []
    for theme, t in report["themes"].items():
        out += [
            f"### {theme} ({t['n_hard_positives']} hard positives, {t['n_distractors']} distractors)",
            "",
            "| arm | nDCG@20 | P@10 | distractor FP-rate@20 | hard-positive recall@20 | median latency (ms) |",
            "|---|---|---|---|---|---|",
        ]
        for arm, r in t["arms"].items():
            out.append(
                f"| {arm} | {r['nDCG@20']:.3f} | {r['P@10']:.2f} | "
                f"{r['distractor FP-rate@20']:.2f} | {r['hard-positive recall@20']:.2f} | "
                f"{r['median latency (ms)']:.0f} |"
            )
        out += [
            "",
            f"FP-rate cut, C vs A: {t['fp_cut_C_vs_A']:.0%}; success rule met: "
            f"**{'yes' if t['success'] else 'no'}**",
            "",
        ]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", help="a checkout of thorwhalen/ir-eval-data")
    args = parser.parse_args()
    report = run(args.data_dir)
    out = Path(args.data_dir) / "rerank_abc_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(as_markdown(report))
    print(f"\n(per-package ranks, private: {out})")


if __name__ == "__main__":
    main()
