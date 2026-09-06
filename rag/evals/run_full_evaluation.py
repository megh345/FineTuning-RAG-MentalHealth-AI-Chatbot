"""Run the Stage 7 full RAG evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import (
    EVAL_REPORTS_DIR,
    GUARDRAIL_GOLD_PATH,
    PROCESSED_CHUNKS_PATH,
    RETRIEVAL_GOLD_PATH,
)
from rag.evals.context_quality import chunks_by_id, evaluate_context_quality
from rag.evals.full_suite import build_full_report, load_json_report
from rag.evals.guardrail_eval import evaluate_guardrails, load_guardrail_examples
from rag.evals.metrics import load_gold_examples, load_predictions
from rag.ingestion.pipeline import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-gold", type=Path, default=RETRIEVAL_GOLD_PATH)
    parser.add_argument("--guardrail-gold", type=Path, default=GUARDRAIL_GOLD_PATH)
    parser.add_argument("--chunks", type=Path, default=PROCESSED_CHUNKS_PATH)
    parser.add_argument(
        "--stage2-report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage2_dense_metrics.json",
    )
    parser.add_argument(
        "--stage3-report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage3_hybrid_metrics.json",
    )
    parser.add_argument(
        "--stage4-report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage4_reranked_metrics.json",
    )
    parser.add_argument(
        "--stage4-predictions",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage4_reranked_predictions.jsonl",
    )
    parser.add_argument("--context-top-k", type=int, default=5)
    parser.add_argument(
        "--context-report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage7_context_metrics.json",
    )
    parser.add_argument(
        "--guardrail-report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage7_guardrail_metrics.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORTS_DIR / "stage7_full_evaluation_report.json",
    )
    args = parser.parse_args()

    dense_report = load_json_report(args.stage2_report)
    hybrid_report = load_json_report(args.stage3_report)
    reranked_report = load_json_report(args.stage4_report)

    gold_examples = load_gold_examples(args.retrieval_gold)
    predictions = load_predictions(args.stage4_predictions)
    chunks = chunks_by_id(load_chunks(args.chunks))
    context_report = evaluate_context_quality(
        gold_examples,
        predictions,
        chunks,
        top_k=args.context_top_k,
    )
    context_report["experiment"] = {
        "stage": "stage_7",
        "component": "context_assembly",
        "gold_dataset": str(args.retrieval_gold),
        "predictions": str(args.stage4_predictions),
        "top_k": args.context_top_k,
    }

    guardrail_examples = load_guardrail_examples(args.guardrail_gold)
    guardrail_report = evaluate_guardrails(guardrail_examples)
    guardrail_report["experiment"] = {
        "stage": "stage_7",
        "component": "guardrails",
        "gold_dataset": str(args.guardrail_gold),
    }

    full_report = build_full_report(
        dense_report=dense_report,
        hybrid_report=hybrid_report,
        reranked_report=reranked_report,
        context_report=context_report,
        guardrail_report=guardrail_report,
    )

    args.context_report.parent.mkdir(parents=True, exist_ok=True)
    args.context_report.write_text(
        json.dumps(context_report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.guardrail_report.write_text(
        json.dumps(guardrail_report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(full_report["quality_gates"], indent=2))
    print(f"Passed: {full_report['passed']}")
    print(f"Context report: {args.context_report}")
    print(f"Guardrail report: {args.guardrail_report}")
    print(f"Full report: {args.report}")


if __name__ == "__main__":
    main()
