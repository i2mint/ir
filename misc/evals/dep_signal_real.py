"""The #64 experiment on the real package benchmark: does the dependency signal help?

The companion of ``dep_signal_synthetic.py``: the same arms, scored against the graded
gold of the private ``thorwhalen/ir-eval-data`` repo (``package_relevance_cases.jsonl``,
``named_sets_official.json``). Nothing private lives here: the labels, the named sets and
the package names are read from the data checkout, and the full report (which names
packages) is written next to it. Only aggregate metrics belong in this public repo.

The corpus is the labelled packages, read from their local checkouts with the default
``Package`` strategy, ``default_edge_extractor`` (``deps`` -> ``REF`` edges) and
``all-MiniLM-L6-v2``, built in memory: the shared ``packages`` corpus is not touched.

Arms, per theme (the seeds are the caller's, as ``ir`` requires; these are the issue's):

- the flat hybrid search (baseline);
- that fused by RRF with the reverse-dependency walk seeded on the theme's libraries;
- the search with ``dep_boost`` at several weights (the FP-rate-vs-recall ablation);
- the search reranked by ef's cross-encoder, alone and with ``dep_boost`` on its scale.

Run: ``python misc/evals/dep_signal_real.py <ir-eval-data dir> [--out FILE]``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import ir
from ir import eval as ev
from ir.graph import CorpusGraph, normalize_dist_name
from ir.sources import package_repo_dirs
from ir.store import CorpusStore
from ir.traverse import traverse

#: The issue's seed libraries per theme (#64, "Experiment").
SEEDS = {
    "embeddings": {
        "ef",
        "imbed",
        "vd",
        "grub",
        "sentence-transformers",
        "transformers",
        "torch",
        "openai",
        "oa",
        "aix",
        "scikit-learn",
    },
    "graphs": {
        "meshed",
        "linked",
        "networkx",
        "graphviz",
        "igraph",
        "dagapp",
        "cosmograph",
    },
}
K = 20
DEPTH = 1000  # ranking depth: past the corpus size, so every named id gets a rank
#: ``dep_boost`` weights on the hybrid (RRF, ~0.03) and cross-encoder (0..1) scales.
RRF_WEIGHTS = (0.005, 0.01, 0.02, 0.05)
RERANK_WEIGHTS = (0.1, 0.3)
RERANK = "cross-encoder"
RELEVANT = {"core", "strong"}  # the benchmark's "relevant" for recall@K


def load(data_dir):
    """``(cases, probes, named_sets)`` from an ``ir-eval-data`` checkout."""
    data_dir = Path(data_dir)
    path = data_dir / "package_relevance_cases.jsonl"
    named = json.loads((data_dir / "named_sets_official.json").read_text())
    return (
        ev.load_package_cases(path),
        ev.read_package_meta(path)["probes"],
        {t: ev.NamedSets.from_dict(d) for t, d in named.items()},
    )


def build(names, *, embedder="default"):
    """``(corpus, names with no checkout)``: an in-memory packages corpus over *names*."""
    wanted = set(names)
    by_name = {}  # the first checkout of each name, as the manifest orders them
    for d in package_repo_dirs():
        if d.name in wanted:
            by_name.setdefault(d.name, d)
    src = ir.CorpusSource.from_packages(repos=lambda: by_name.values(), exclude=[])
    corpus = ir.build(
        src,
        store=CorpusStore.memory(),
        embedder=embedder,
        edge_extractor=ir.default_edge_extractor,
    )
    return corpus, wanted - set(by_name)


def arms(corpus, probe, seeds, *, rerank=RERANK):
    """``{arm: ranked artifact ids}`` for one theme probe.

    ``rerank`` names the reranker of the comparison arms (``None`` drops them).
    """

    def ids(hits):
        return [h.artifact_id for h in hits]

    def search(**kw):
        return ir.search(corpus, probe, k=DEPTH, mode="hybrid", **kw)

    def boost(w):
        return ir.dependency_evidence_boost(seeds, weight=w)

    flat = search()
    walked = traverse(
        probe,
        corpus,
        policy=ir.reverse_dependency_policy(seeds=seeds),
        k=DEPTH,
        node_budget=DEPTH,
    )
    out = {
        "baseline (hybrid)": ids(flat),
        "+ reverse-dep walk (RRF)": ids(ir.fuse_hits({"flat": flat, "walk": walked})),
    }
    out |= {f"+ dep_boost w={w}": ids(search(dep_boost=boost(w))) for w in RRF_WEIGHTS}
    if rerank is not None:
        out[rerank] = ids(search(rerank=rerank))
        out |= {
            f"{rerank} + dep_boost w={w}": ids(
                search(rerank=rerank, dep_boost=boost(w))
            )
            for w in RERANK_WEIGHTS
        }
    return out


def score(ranking, cases, theme, named):
    """The issue's metrics for one ranking, plus where each named id landed."""
    from ef.evaluation import ndcg_at_k

    gains = {c.artifact_id: c.gain(theme) for c in cases if c.gain(theme) > 0}
    relevant = {c.artifact_id for c in cases if c.level(theme) in RELEVANT}
    top, top10 = set(ranking[:K]), set(ranking[:10])
    rank = {a: i + 1 for i, a in enumerate(ranking)}
    return {
        "nDCG@20": ndcg_at_k(ranking, gains, K),
        "recall@20": len(top & relevant) / len(relevant),
        "hard positives in top-20": sum(a in top for a in named.hard_positives),
        "distractor fp_rate@20": ev.fp_rate_on_distractors(
            ranking, named.distractors, k=K
        ),
        "distractors in top-10": sum(a in top10 for a in named.distractors),
        "uses-tools in top-20": sum(
            c.artifact_id in top for c in cases if c.level(theme) == "uses-tools"
        ),
    }, {a: rank.get(a, 0) for a in (*named.hard_positives, *named.distractors)}


def run(data_dir):
    """The full report: per-theme metrics and named ranks per arm, and the fan-in."""
    cases, probes, named_sets = load(data_dir)
    corpus, missing = build(c.artifact_id for c in cases)
    report = {"k": K, "missing_checkouts": sorted(missing), "themes": {}}
    for theme, named in named_sets.items():
        seeds = {normalize_dist_name(s) for s in SEEDS[theme]}
        ranked = arms(corpus, probes[theme], seeds)
        scored = {arm: score(r, cases, theme, named) for arm, r in ranked.items()}
        report["themes"][theme] = {
            "n_hard_positives": len(named.hard_positives),
            "n_distractors": len(named.distractors),
            "metrics": {arm: m for arm, (m, _) in scored.items()},
            "named_ranks": {arm: r for arm, (_, r) in scored.items()},
        }
    graph = CorpusGraph(corpus)
    seeds = set().union(*SEEDS.values())
    report["seed_fan_in"] = {s: graph.fan_in(s) for s in sorted(seeds)}
    targets = {
        t for aid in ev.corpus_artifact_ids(corpus) for t in graph.neighbors(aid)
    }
    report["fan_in_histogram"] = dict(
        sorted(Counter(graph.fan_in(t) for t in targets).items())
    )
    return report


def as_markdown(report):
    """The aggregate tables (no package names): safe for the public repo."""
    lines = []
    for theme, t in report["themes"].items():
        cols = list(next(iter(t["metrics"].values())))
        lines += [
            f"\n### {theme} ({t['n_hard_positives']} hard positives, "
            f"{t['n_distractors']} distractors)\n",
            "| arm | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for arm, m in t["metrics"].items():
            cells = (f"{v:.2f}" if isinstance(v, float) else str(v) for v in m.values())
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append(
        f"\nFan-in histogram (dependents: targets): {report['fan_in_histogram']}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", help="a checkout of thorwhalen/ir-eval-data")
    parser.add_argument("--out", help="write the full JSON report (names packages)")
    args = parser.parse_args()
    result = run(args.data_dir)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    print(as_markdown(result))
