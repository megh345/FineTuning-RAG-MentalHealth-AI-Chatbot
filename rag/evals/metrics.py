"""Retrieval metrics used after every retrieval stage.

The evaluator is deliberately independent from any specific retriever. Dense,
hybrid, and reranked systems all emit the same ranked `chunk_id` predictions,
then this module scores them against the gold dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GoldRetrievalExample:
    query_id: str
    query: str
    relevant_chunk_ids: set[str]
    topic: str
    query_type: str
    notes: str = ""


@dataclass(frozen=True)
class RetrievalPrediction:
    query_id: str
    retrieved_chunk_ids: list[str]


def load_gold_examples(path: Path) -> list[GoldRetrievalExample]:
    examples: list[GoldRetrievalExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        relevant_chunk_ids = set(payload["relevant_chunk_ids"])
        if not relevant_chunk_ids:
            raise ValueError(f"{path}:{line_number} has no relevant_chunk_ids")
        examples.append(
            GoldRetrievalExample(
                query_id=payload["query_id"],
                query=payload["query"],
                relevant_chunk_ids=relevant_chunk_ids,
                topic=payload["topic"],
                query_type=payload["query_type"],
                notes=payload.get("notes", ""),
            )
        )
    return examples


def load_predictions(path: Path) -> dict[str, RetrievalPrediction]:
    predictions: dict[str, RetrievalPrediction] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        query_id = payload["query_id"]
        retrieved_chunk_ids = payload["retrieved_chunk_ids"]
        if query_id in predictions:
            raise ValueError(f"{path}:{line_number} repeats query_id {query_id!r}")
        predictions[query_id] = RetrievalPrediction(
            query_id=query_id,
            retrieved_chunk_ids=list(retrieved_chunk_ids),
        )
    return predictions


def evaluate_retrieval(
    gold_examples: Iterable[GoldRetrievalExample],
    predictions: dict[str, RetrievalPrediction],
    k_values: tuple[int, ...] = (3, 5, 10),
) -> dict:
    examples = list(gold_examples)
    if not examples:
        raise ValueError("Cannot evaluate retrieval without gold examples")

    per_query = []
    aggregate = {
        f"recall_at_{k}": 0.0 for k in k_values
    } | {
        f"mrr_at_{k}": 0.0 for k in k_values
    } | {
        f"ndcg_at_{k}": 0.0 for k in k_values
    }

    for example in examples:
        retrieved = predictions.get(
            example.query_id,
            RetrievalPrediction(query_id=example.query_id, retrieved_chunk_ids=[]),
        ).retrieved_chunk_ids

        query_metrics = {
            "query_id": example.query_id,
            "topic": example.topic,
            "query_type": example.query_type,
        }
        for k in k_values:
            query_metrics[f"recall_at_{k}"] = recall_at_k(
                example.relevant_chunk_ids,
                retrieved,
                k,
            )
            query_metrics[f"mrr_at_{k}"] = mrr_at_k(
                example.relevant_chunk_ids,
                retrieved,
                k,
            )
            query_metrics[f"ndcg_at_{k}"] = ndcg_at_k(
                example.relevant_chunk_ids,
                retrieved,
                k,
            )

        per_query.append(query_metrics)
        for metric_name in aggregate:
            aggregate[metric_name] += query_metrics[metric_name]

    for metric_name in aggregate:
        aggregate[metric_name] = aggregate[metric_name] / len(examples)

    return {
        "num_queries": len(examples),
        "aggregate": aggregate,
        "per_query": per_query,
    }


def recall_at_k(relevant_chunk_ids: set[str], retrieved_chunk_ids: list[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    retrieved_at_k = set(retrieved_chunk_ids[:k])
    return len(relevant_chunk_ids.intersection(retrieved_at_k)) / len(relevant_chunk_ids)


def mrr_at_k(relevant_chunk_ids: set[str], retrieved_chunk_ids: list[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_chunk_ids: set[str], retrieved_chunk_ids: list[str], k: int) -> float:
    dcg = 0.0
    for index, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant_chunk_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant_chunk_ids), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg

