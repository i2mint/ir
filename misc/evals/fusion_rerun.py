"""Re-run ir_08's fusion comparison (ir#168): abstention J and retrieval quality.

For each corpus with a case file, and each config (dense, hybrid/rrf,
hybrid/blend at the default alpha/k_sat), report the calibrated floor's Youden J
(`ir.eval.calibrate_min_score`, not persisted) and NDCG@10 / recall@10 / MRR
(`ir.eval.evaluate_discovery`). Prints a markdown table; numbers only.

    python misc/evals/fusion_rerun.py skills=skills_cases.jsonl packages=packages_cases.jsonl
"""

import json
import sys

from ir import eval as ev
from ir.index import open_corpus

CONFIGS = {
    "dense": {"mode": "dense"},
    "hybrid/rrf": {"mode": "hybrid", "fusion": "rrf"},
    "hybrid/blend": {"mode": "hybrid", "fusion": "blend"},
}


def run(corpus_name, cases_path):
    corpus = open_corpus(corpus_name)
    cases = ev.load_cases(cases_path)
    rows = {}
    for label, kw in CONFIGS.items():
        cal = ev.calibrate_min_score(corpus, cases, k=10, **kw)
        rep = ev.evaluate_discovery(corpus, cases, k_values=(1, 10), **kw)
        m = rep.retrieval.metrics
        rows[label] = {
            "youden_j": cal.youden_j,
            "ndcg@10": m.get("ndcg@10"),
            "recall@10": m.get("recall@10"),
            "mrr": m.get("mrr@10", m.get("mrr")),
            "hit@1": rep.failure_classes.get("hit_rank_1", 0) / max(rep.n_gold, 1),
            "n_gold": rep.n_gold,
            "n_abstention": rep.n_abstention,
        }
        print(corpus_name, label, json.dumps(rows[label]), file=sys.stderr, flush=True)
    return rows


def main():
    results = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        results[name] = run(name, path)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
