"""Map Qdrant scored points to the shared retrieval contract."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rag.schemas import RetrievalResult


def to_retrieval_results(points: Iterable[Any]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for point in points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise ValueError(f"Qdrant point {point.id} has no chunk_id payload")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"chunk_id", "text"}
        }
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                score=float(point.score),
                text=payload.get("text"),
                metadata=metadata,
            )
        )
    return results
