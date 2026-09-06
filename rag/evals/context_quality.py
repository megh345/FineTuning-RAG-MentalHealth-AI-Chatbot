"""Evaluate Stage 5 context assembly quality from retrieval predictions."""

from __future__ import annotations

from typing import Any

from rag.evals.metrics import GoldRetrievalExample, RetrievalPrediction
from rag.generation.context import ContextAssembler
from rag.guardrails.context import ContextGuardrail
from rag.schemas import DocumentChunk, RetrievalResult


def chunks_by_id(chunks: list[DocumentChunk]) -> dict[str, DocumentChunk]:
    """Index processed chunks by stable chunk ID."""

    indexed = {chunk.chunk_id: chunk for chunk in chunks}
    if len(indexed) != len(chunks):
        raise ValueError("Duplicate chunk IDs found while building context eval map")
    return indexed


def evaluate_context_quality(
    gold_examples: list[GoldRetrievalExample],
    predictions: dict[str, RetrievalPrediction],
    chunk_lookup: dict[str, DocumentChunk],
    top_k: int = 5,
    assembler: ContextAssembler | None = None,
    guardrail: ContextGuardrail | None = None,
) -> dict[str, Any]:
    """Measure whether retrieved chunks become usable cited context."""

    if not gold_examples:
        raise ValueError("Cannot evaluate context quality without gold examples")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    current_assembler = assembler or ContextAssembler(max_chunks=top_k)
    current_guardrail = guardrail or ContextGuardrail()
    per_query = []
    pass_count = 0
    relevant_context_count = 0
    citation_consistency_count = 0
    total_citations = 0
    total_context_chars = 0
    total_relevant_chunks = 0

    for example in gold_examples:
        prediction = predictions.get(
            example.query_id,
            RetrievalPrediction(query_id=example.query_id, retrieved_chunk_ids=[]),
        )
        retrieval_results = _prediction_to_results(
            prediction.retrieved_chunk_ids[:top_k],
            chunk_lookup,
        )
        context = current_assembler.assemble(
            example.query,
            retrieval_results,
            top_k=top_k,
        )
        decision = current_guardrail.validate(context)
        included_relevant = sorted(
            set(context.included_chunk_ids).intersection(example.relevant_chunk_ids)
        )
        citation_consistent = len(context.citations) == len(context.included_chunk_ids)
        has_relevant_context = bool(included_relevant)

        pass_count += int(decision.should_continue)
        relevant_context_count += int(has_relevant_context)
        citation_consistency_count += int(citation_consistent)
        total_citations += len(context.citations)
        total_context_chars += len(context.context_text)
        total_relevant_chunks += len(included_relevant)
        per_query.append(
            {
                "query_id": example.query_id,
                "topic": example.topic,
                "query_type": example.query_type,
                "guardrail_action": decision.action.value,
                "finding_categories": sorted(
                    {finding.category.value for finding in decision.findings}
                ),
                "included_chunk_ids": context.included_chunk_ids,
                "omitted_chunk_ids": context.omitted_chunk_ids,
                "relevant_included_chunk_ids": included_relevant,
                "has_relevant_context": has_relevant_context,
                "citation_count": len(context.citations),
                "citation_consistent": citation_consistent,
                "context_chars": len(context.context_text),
            }
        )

    total = len(gold_examples)
    return {
        "num_queries": total,
        "aggregate": {
            "context_pass_rate": pass_count / total,
            "abstention_rate": 1 - (pass_count / total),
            "relevant_context_rate": relevant_context_count / total,
            "citation_consistency_rate": citation_consistency_count / total,
            "average_citation_count": total_citations / total,
            "average_context_chars": total_context_chars / total,
            "average_relevant_chunks_included": total_relevant_chunks / total,
        },
        "per_query": per_query,
    }


def _prediction_to_results(
    retrieved_chunk_ids: list[str],
    chunk_lookup: dict[str, DocumentChunk],
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        results.append(
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                score=1.0 / rank,
                text=chunk.text,
                metadata={
                    "source": chunk.source,
                    "topic": chunk.topic,
                    "section": chunk.section,
                    "content_type": chunk.content_type,
                    "risk_category": chunk.risk_category,
                    "page": chunk.page,
                    "metadata": chunk.metadata,
                },
            )
        )
    return results
