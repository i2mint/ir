"""The #64 experiment on a synthetic package corpus: does the dependency signal help?

The real benchmark (graded gold over the fleet's packages) is private, so this runs the
same arms on a corpus generated here, with known relevance. Per theme it holds:

- *prose positives*: say the theme in their description and depend on its libraries;
- *hard positives*: depend on the theme's libraries but describe themselves in words
  that never name it (the "uses-tools" false negatives of #61);
- *distractors*: use the theme's words, but depend on unrelated libraries;
- *incidental users*: off-topic, but depend on one of the theme's libraries anyway
  (``torch`` in an audio package): the false positives the dependency signal makes;
- *background*: other topics, other libraries.

Arms: the flat hybrid search (baseline); that fused with the reverse-dependency walk
seeded on the theme's libraries; the search with ``dep_boost`` at several weights. The
seeds are the caller's, as in ``ir`` itself. Being built to the issue's description,
this checks that the mechanisms do what they claim and how the boost weight trades
recall for false positives; it says nothing about the size of the effect on real data.

Run: ``python misc/evals/dep_signal_synthetic.py`` (hermetic: light embedder, memory).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import ir
from ir.graph import CorpusGraph
from ir.store import CorpusStore
from ir.traverse import traverse

THEMES = {
    "graphs": dict(
        query="graph and network analysis library",
        words=["graph", "network", "nodes", "edges", "DAG", "topology"],
        libs=["networkx", "igraph", "graphviz", "meshlib"],
    ),
    "embeddings": dict(
        query="text embeddings and vector similarity",
        words=["embedding", "vector", "similarity", "semantic", "encoder"],
        libs=["sentence-transformers", "torch", "vectorlib", "faiss"],
    ),
}
#: Words for everything else: hard positives and background draw their prose here.
OTHER_WORDS = [
    "files", "config", "audio", "parser", "cache", "plots", "tables", "storage",
    "scheduler", "logging", "images", "http", "dates", "strings", "tasks", "async",
]
OTHER_LIBS = ["requests", "numpy", "pandas", "click", "asyncio-extras", "librosa",
              "pyyaml", "pillow", "boto3", "rich"]
K = 20
WEIGHTS = (0.0, 0.005, 0.01, 0.02, 0.05)


def make_corpus(*, seed=0, n_prose=10, n_hard=8, n_distract=8, n_incidental=8, n_background=150):
    """``(packages, gold)``: gold maps theme -> {name: relevance (2 or 1)}, plus the
    named hard positives and distractors per theme."""
    rng = random.Random(seed)
    packages, gold = {}, {}

    def prose(words, n=6):
        return " ".join(rng.choice(words) for _ in range(n))

    def add(name, description, deps):
        packages[name] = {"name": name, "description": description, "deps": deps}

    for theme, spec in THEMES.items():
        g = gold[theme] = {"relevance": {}, "hard": [], "distractors": [], "incidental": []}
        for i in range(n_prose):
            name = f"{theme}_tool_{i}"
            add(name, prose(spec["words"] + OTHER_WORDS[:4]),
                rng.sample(spec["libs"], 2) + rng.sample(OTHER_LIBS, 1))
            g["relevance"][name] = 2
        for i in range(n_hard):
            name = f"{theme}_user_{i}"
            add(name, prose(OTHER_WORDS), [rng.choice(spec["libs"])] + rng.sample(OTHER_LIBS, 2))
            g["relevance"][name] = 1
            g["hard"].append(name)
        for i in range(n_distract):
            name = f"{theme}_lookalike_{i}"
            add(name, prose(spec["words"] + OTHER_WORDS[8:]), rng.sample(OTHER_LIBS, 3))
            g["distractors"].append(name)
        for i in range(n_incidental):
            name = f"{theme}_incidental_{i}"
            add(name, prose(OTHER_WORDS), [rng.choice(spec["libs"])] + rng.sample(OTHER_LIBS, 2))
            g["incidental"].append(name)
    for i in range(n_background):
        add(f"misc_{i}", prose(OTHER_WORDS), rng.sample(OTHER_LIBS, 3))
    return packages, gold


def build(packages):
    src = ir.CorpusSource.from_mapping(packages, name="pk", strategy=ir.Package())
    return ir.build(src, store=CorpusStore.memory(), embedder="light",
                    edge_extractor=ir.default_edge_extractor)


def ndcg(ranked, relevance, k=K):
    dcg = sum(relevance.get(a, 0) / math.log2(i + 2) for i, a in enumerate(ranked[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    return dcg / sum(r / math.log2(i + 2) for i, r in enumerate(ideal))


def score(ranked, g):
    top, top10 = set(ranked[:K]), set(ranked[:10])
    rel = g["relevance"]
    return {
        "nDCG@20": ndcg(ranked, rel),
        "recall@20": len(top & set(rel)) / len(rel),
        "hard in top-20": sum(a in top for a in g["hard"]),
        "distractors in top-10": sum(a in top10 for a in g["distractors"]),
        "incidental in top-20": sum(a in top for a in g["incidental"]),
    }


def run(*, seed=0, **corpus_kw):
    """``{theme: {arm: metrics}}`` and the fan-in distribution of the corpus."""
    packages, gold = make_corpus(seed=seed, **corpus_kw)
    corpus = build(packages)
    out = {}
    for theme, spec in THEMES.items():
        q, libs, g = spec["query"], set(spec["libs"]), gold[theme]
        arms = {}
        flat = ir.search(corpus, q, k=50, mode="hybrid")
        arms["baseline (hybrid)"] = [h.artifact_id for h in flat]
        walked = traverse(q, corpus, policy=ir.reverse_dependency_policy(seeds=libs), k=50)
        fused = ir.fuse_hits({"flat": flat, "walk": walked})
        arms["+ reverse-dep walk (RRF)"] = [h.artifact_id for h in fused]
        for w in WEIGHTS[1:]:
            boosted = ir.search(corpus, q, k=50, mode="hybrid",
                                dep_boost=ir.dependency_evidence_boost(libs, weight=w))
            arms[f"+ dep_boost w={w}"] = [h.artifact_id for h in boosted]
        out[theme] = {arm: score(ranked, g) for arm, ranked in arms.items()}
    graph = CorpusGraph(corpus)
    targets = {t for p in packages.values() for t in p["deps"]}
    fan_in = Counter(graph.fan_in(t) for t in targets)
    return out, dict(sorted(fan_in.items()))


def run_seeds(seeds=range(5), **corpus_kw):
    """:func:`run` averaged over corpora drawn from several seeds."""
    runs = [run(seed=s, **corpus_kw) for s in seeds]
    mean = {
        theme: {
            arm: {m: sum(r[0][theme][arm][m] for r in runs) / len(runs) for m in metrics}
            for arm, metrics in arms.items()
        }
        for theme, arms in runs[0][0].items()
    }
    return mean, runs[0][1]


def as_markdown(results, fan_in):
    lines = []
    for theme, arms in results.items():
        cols = list(next(iter(arms.values())))
        lines += [f"\n### {theme}\n", "| arm | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for arm, m in arms.items():
            lines.append(f"| {arm} | " + " | ".join(
                f"{v:.2f}" for v in m.values()) + " |")
    lines.append(f"\nFan-in of the first corpus (dependents: how many dependencies): {fan_in}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(as_markdown(*run_seeds()))
