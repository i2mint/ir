"""Generate an ir eval case set for one registered corpus, on the Claude subscription.

Same as `ir eval-gen`, but the LLM is injected: aix's `claude-cli/` backend
(`claude -p`, billed to the subscription, no API key). Usage:

    python misc/evals/gen_cases_subscription.py CORPUS OUT.jsonl --theme "..." --max-artifacts 100
"""

import argparse
import sys

from ir import registry
from ir.eval import save_cases
from ir.eval_gen import (
    build_eval_set,
    corpus_signature,
    make_default_abstention_generator,
    make_default_query_generator,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("out")
    parser.add_argument("--theme", default="software developer tools")
    parser.add_argument("--max-artifacts", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="claude-cli/sonnet")
    args = parser.parse_args()
    source = registry.source_for(args.corpus)
    cases = build_eval_set(
        source,
        k=args.k,
        abstention_frac=0.15,
        corpus_name=args.corpus,
        theme=args.theme,
        query_generator=make_default_query_generator(model=args.model),
        abstention_generator=make_default_abstention_generator(model=args.model),
        max_artifacts=args.max_artifacts,
    )
    save_cases(
        cases,
        args.out,
        meta={
            "corpus": args.corpus,
            "corpus_signature": corpus_signature(source),
            "k": args.k,
            "generator": args.model,
        },
    )
    n_gold = sum(not c.gold_is_none for c in cases)
    print(f"{args.corpus}: {len(cases)} cases, {n_gold} gold", file=sys.stderr)


if __name__ == "__main__":
    main()
