"""Run Stage 7 context assembly quality evaluation from retrieval predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import (
    EVAL_REPORTS_DIR,
    PROCESSED_CHUNKS_PATH,
    RETRIEVAL_GOLD_PATH,
)
from rag.evals.context_quality import chunks_by_id, evaluate_context_quality
from rag.evals.metrics import load_gold_examples, load_predictions
from rag.ingestion.pipeline import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=RETRIEVAL_GOLD_PATH)
    parser.add_argument("--chunks", type=Path, default=PROCESSED_CHUNKS_PATH)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage4_reranked_predictions.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage7_context_metrics.json",
    )
    args = parser.parse_args()

    gold_examples = load_gold_examples(args.gold)
    predictions = load_predictions(args.predictions)
    chunks = chunks_by_id(load_chunks(args.chunks))
    report = evaluate_context_quality(
        gold_examples,
        predictions,
        chunks,
        top_k=args.top_k,
    )
    report["experiment"] = {
        "stage": "stage_7",
        "component": "context_assembly",
        "gold_dataset": str(args.gold),
        "predictions": str(args.predictions),
        "top_k": args.top_k,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
