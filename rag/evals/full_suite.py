"""Combine RAG retrieval, context, and guardrail reports into one suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_QUALITY_GATES = {
    "stage4_recall_at_5": 0.80,
    "context_relevant_context_rate": 0.80,
    "guardrail_exact_match_accuracy": 0.95,
}


def load_json_report(path: Path) -> dict[str, Any]:
    """Load a JSON report with a useful error message."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required evaluation report not found: {path}") from error


def build_full_report(
    dense_report: dict[str, Any],
    hybrid_report: dict[str, Any],
    reranked_report: dict[str, Any],
    context_report: dict[str, Any],
    guardrail_report: dict[str, Any],
    quality_gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create one Stage 7 report from component reports."""

    gates = quality_gates or DEFAULT_QUALITY_GATES
    gate_results = evaluate_quality_gates(
        reranked_report=reranked_report,
        context_report=context_report,
        guardrail_report=guardrail_report,
        quality_gates=gates,
    )
    return {
        "experiment": {
            "stage": "stage_7",
            "framework": "full_rag_evaluation",
            "components": [
                "retrieval",
                "context_assembly",
                "guardrails",
            ],
            "generation_eval_status": "pending_runtime_generator_integration",
        },
        "retrieval": {
            "stage2_dense": dense_report.get("aggregate", {}),
            "stage3_hybrid": hybrid_report.get("aggregate", {}),
            "stage4_reranked": reranked_report.get("aggregate", {}),
        },
        "context": context_report.get("aggregate", {}),
        "guardrails": guardrail_report.get("aggregate", {}),
        "quality_gates": gate_results,
        "passed": all(result["passed"] for result in gate_results.values()),
    }


def evaluate_quality_gates(
    reranked_report: dict[str, Any],
    context_report: dict[str, Any],
    guardrail_report: dict[str, Any],
    quality_gates: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Evaluate configured minimum thresholds."""

    actual_values = {
        "stage4_recall_at_5": reranked_report["aggregate"]["recall_at_5"],
        "context_relevant_context_rate": context_report["aggregate"][
            "relevant_context_rate"
        ],
        "guardrail_exact_match_accuracy": guardrail_report["aggregate"][
            "exact_match_accuracy"
        ],
    }
    return {
        gate_name: {
            "actual": actual_values[gate_name],
            "minimum": minimum,
            "passed": actual_values[gate_name] >= minimum,
        }
        for gate_name, minimum in quality_gates.items()
    }
