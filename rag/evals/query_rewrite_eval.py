"""Evaluate Stage 8 conditional query rewriting decisions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag.config import QUERY_REWRITE_GOLD_PATH
from rag.query_rewriting import ConditionalQueryRewriter, QueryRewriteStrategy


@dataclass(frozen=True)
class QueryRewriteGoldExample:
    """One expected query rewrite behavior example."""

    example_id: str
    query: str
    conversation_history: list[str]
    expected_changed: bool
    expected_strategy: QueryRewriteStrategy
    expected_terms: list[str]
    notes: str = ""


def load_query_rewrite_examples(
    path: Path = QUERY_REWRITE_GOLD_PATH,
) -> list[QueryRewriteGoldExample]:
    examples: list[QueryRewriteGoldExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            examples.append(
                QueryRewriteGoldExample(
                    example_id=payload["example_id"],
                    query=payload["query"],
                    conversation_history=list(payload.get("conversation_history", [])),
                    expected_changed=bool(payload["expected_changed"]),
                    expected_strategy=QueryRewriteStrategy(payload["expected_strategy"]),
                    expected_terms=list(payload.get("expected_terms", [])),
                    notes=payload.get("notes", ""),
                )
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid query rewrite example at {path}:{line_number}") from error
    if not examples:
        raise ValueError(f"No query rewrite examples found in {path}")
    return examples


def evaluate_query_rewriting(
    examples: list[QueryRewriteGoldExample],
    rewriter: ConditionalQueryRewriter | None = None,
) -> dict[str, Any]:
    """Score rewrite decisions against the gold dataset."""

    if not examples:
        raise ValueError("Cannot evaluate query rewriting without examples")

    current_rewriter = rewriter or ConditionalQueryRewriter()
    per_example = []
    changed_correct_count = 0
    strategy_correct_count = 0
    term_recall_count = 0
    rewrite_count = 0

    for example in examples:
        decision = current_rewriter.rewrite(
            example.query,
            conversation_history=example.conversation_history,
        )
        rewritten_lower = decision.retrieval_query.lower()
        changed_correct = decision.changed == example.expected_changed
        strategy_correct = decision.strategy == example.expected_strategy
        expected_terms_present = all(
            term.lower() in rewritten_lower for term in example.expected_terms
        )

        changed_correct_count += int(changed_correct)
        strategy_correct_count += int(strategy_correct)
        term_recall_count += int(expected_terms_present)
        rewrite_count += int(decision.changed)
        per_example.append(
            {
                "example_id": example.example_id,
                "query": example.query,
                "rewritten_query": decision.retrieval_query,
                "expected_changed": example.expected_changed,
                "actual_changed": decision.changed,
                "expected_strategy": example.expected_strategy.value,
                "actual_strategy": decision.strategy.value,
                "expected_terms": example.expected_terms,
                "changed_correct": changed_correct,
                "strategy_correct": strategy_correct,
                "expected_terms_present": expected_terms_present,
                "notes": example.notes,
            }
        )

    total = len(examples)
    return {
        "num_examples": total,
        "aggregate": {
            "changed_accuracy": changed_correct_count / total,
            "strategy_accuracy": strategy_correct_count / total,
            "expected_term_recall": term_recall_count / total,
            "rewrite_rate": rewrite_count / total,
        },
        "per_example": per_example,
    }
