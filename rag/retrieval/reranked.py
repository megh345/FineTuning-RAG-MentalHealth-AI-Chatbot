"""Hybrid retrieval followed by CrossEncoder-compatible reranking."""

from __future__ import annotations

from rag.config import DEFAULT_HYBRID_CANDIDATE_K, DEFAULT_RERANK_TOP_K
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerankers import Reranker
from rag.schemas import RetrievalResult


class RerankedRetriever:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RERANK_TOP_K,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")

        candidates = self.hybrid_retriever.retrieve(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def close(self) -> None:
        close = getattr(self.hybrid_retriever, "close", None)
        if callable(close):
            close()
