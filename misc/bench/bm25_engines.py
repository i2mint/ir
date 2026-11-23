"""Benchmark ir's lexical and hybrid search on vd's two BM25 engines (#171).

Builds a fixture corpus from ir's own source (or opens a registered corpus with
``--corpus NAME``), then times BM25 index build and per-query search with
``ir.retrieve.BM25_ENGINE`` set to ``"python"`` and to ``"bm25s"``, and checks
that both engines return the same top-50 for every query.

    python misc/bench/bm25_engines.py
    python misc/bench/bm25_engines.py --corpus packages --repeat 7

Times are process CPU seconds, min over ``--repeat`` runs. The hybrid column
uses the ``light`` embedder for the fixture (so the dense side is cheap) and the
corpus's own embedder for a registered corpus.
"""

import argparse
import time
from pathlib import Path

import ir
from ir import retrieve
from ir.store import CorpusStore

QUERIES = (
    "abstention floor calibration",
    "reciprocal rank fusion of dense and lexical rankings",
    "incremental reindex of a changed corpus",
    "surface weights per artifact",
    "the single run maintenance lock",
    "chunk text by tokens",
    "metadata filter on the candidate set",
    "rerank the fetched candidates",
    "register a corpus by name",
    "query normalization",
    "search for a package that parses audio files",
    "store key value data in a folder of files",
)


def fixture_corpus():
    root = Path(ir.__file__).parent
    docs = {
        str(p.relative_to(root)): {"text": p.read_text(encoding="utf-8")}
        for p in sorted(root.rglob("*.py"))
    }
    source = ir.CorpusSource.from_mapping(
        docs, name="ir-src", strategy=ir.Chunked(chunk_size=600, overlap=60)
    )
    return ir.build(source, store=CorpusStore.memory(), embedder="light")


def _cpu(fn, repeat):
    best = float("inf")
    for _ in range(repeat):
        start = time.process_time()
        fn()
        best = min(best, time.process_time() - start)
    return best


def _ids(hits):
    return [(h.artifact_id, h.surface_kind, h.surface_index) for h in hits]


def bench(corpus, *, repeat=5, k=50, modes=("lexical", "hybrid")):
    rows, tops = {}, {}
    for engine in ("python", "bm25s"):
        retrieve.BM25_ENGINE = engine

        def build():
            corpus._bm25_index_cache = None  # force a rebuild
            retrieve.search(corpus, QUERIES[0], mode="lexical", k=1)

        build_s = _cpu(build, repeat)

        def run(mode):
            return lambda: [
                retrieve.search(corpus, q, mode=mode, k=k, fetch_k=k) for q in QUERIES
            ]

        lexical_s = _cpu(run("lexical"), repeat) / len(QUERIES)
        hybrid_s = (
            _cpu(run("hybrid"), repeat) / len(QUERIES) if "hybrid" in modes else None
        )
        tops[engine] = [
            _ids(retrieve.search(corpus, q, mode="lexical", k=k, fetch_k=k))
            for q in QUERIES
        ]
        rows[engine] = (build_s, lexical_s, hybrid_s)
    same = sum(a == b for a, b in zip(tops["python"], tops["bm25s"], strict=True))
    return rows, same


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", help="a registered corpus name (default: fixture)")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="skip hybrid (when the corpus's embedder is not installed here)",
    )
    args = parser.parse_args()
    corpus = ir.open_corpus(args.corpus) if args.corpus else fixture_corpus()
    modes = ("lexical",) if args.lexical_only else ("lexical", "hybrid")
    rows, same = bench(corpus, repeat=args.repeat, modes=modes)
    print(f"corpus: {args.corpus or 'ir source fixture'} ({len(corpus)} records)")
    print("| engine | index build | lexical query | hybrid query |")
    print("|---|---|---|---|")

    def ms(seconds, digits=2):
        return "-" if seconds is None else f"{seconds * 1e3:.{digits}f} ms"

    for engine, (b, lex, hyb) in rows.items():
        print(f"| {engine} | {ms(b, 0)} | {ms(lex)} | {ms(hyb)} |")
    (_, plex, phyb), (_, slex, shyb) = rows["python"], rows["bm25s"]
    hybrid = f", hybrid {phyb / shyb:.1f}x" if phyb is not None else ""
    print(f"speed-up: lexical {plex / slex:.1f}x{hybrid}")
    print(f"same top-50: {same} of {len(QUERIES)} queries")


if __name__ == "__main__":
    main()
