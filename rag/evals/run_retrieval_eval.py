"""Run retrieval evaluation for a ranked predictions file.

Predictions are JSONL with this shape:

```json
{"query_id": "retrieval_001", "retrieved_chunk_ids": ["chunk_a", "chunk_b"]}
```

Every retrieval stage should write that format so the same evaluator can compare
dense-only, hybrid, and reranked retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import EVAL_REPORTS_DIR, RETRIEVAL_GOLD_PATH
from rag.evals.metrics import evaluate_retrieval, load_gold_examples, load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked RAG retrieval predictions.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=RETRIEVAL_GOLD_PATH,
        help="Path to gold retrieval JSONL dataset.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to retrieval predictions JSONL.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="K values for Recall@K, MRR@K, and nDCG@K.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_examples = load_gold_examples(args.gold)
    predictions = load_predictions(args.predictions)
    report = evaluate_retrieval(gold_examples, predictions, tuple(args.k))

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()

