"""Retrieval wrapper that conditionally rewrites queries before retrieval."""

from __future__ import annotations

from typing import Protocol

from rag.config import DEFAULT_HYBRID_CANDIDATE_K, DEFAULT_RERANK_TOP_K
from rag.query_rewriting.rewriter import ConditionalQueryRewriter
from rag.schemas import RetrievalResult


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RERANK_TOP_K,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    ) -> list[RetrievalResult]: ...


class QueryRewritingRetriever:
    """Apply conditional query rewriting before calling a base retriever."""

    def __init__(
        self,
        base_retriever: Retriever,
        rewriter: ConditionalQueryRewriter | None = None,
    ) -> None:
        self.base_retriever = base_retriever
        self.rewriter = rewriter or ConditionalQueryRewriter()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RERANK_TOP_K,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
        conversation_history: list[str] | None = None,
    ) -> list[RetrievalResult]:
        decision = self.rewriter.rewrite(
            query,
            conversation_history=conversation_history,
        )
        results = self.base_retriever.retrieve(
            decision.retrieval_query,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        rewritten_results = []
        for result in results:
            metadata = dict(result.metadata)
            metadata["query_rewrite"] = {
                "original_query": decision.original_query,
                "retrieval_query": decision.retrieval_query,
                "changed": decision.changed,
                "strategy": decision.strategy.value,
                "reason": decision.reason,
                "matched_terms": decision.matched_terms,
                "metadata": decision.metadata,
            }
            rewritten_results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=result.score,
                    text=result.text,
                    metadata=metadata,
                )
            )
        return rewritten_results

    def close(self) -> None:
        close = getattr(self.base_retriever, "close", None)
        if callable(close):
            close()
