"""Inspect Stage 8 rewritten + reranked retrieval results for one query."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.config import DEFAULT_HYBRID_CANDIDATE_K, DEFAULT_RERANK_TOP_K
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.indexing.sparse_embeddings import FastEmbedBm25Embedder
from rag.query_rewriting.rewriter import ConditionalQueryRewriter
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranked import RerankedRetriever
from rag.retrieval.rerankers import CrossEncoderReranker
from rag.retrieval.rewritten import QueryRewritingRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_HYBRID_CANDIDATE_K,
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="Recent conversation message. Repeat for multiple turns.",
    )
    args = parser.parse_args()

    dense_embedder = SentenceTransformerEmbedder()
    sparse_embedder = FastEmbedBm25Embedder()
    reranker = CrossEncoderReranker()
    rewriter = ConditionalQueryRewriter()
    with HybridRetriever(dense_embedder, sparse_embedder) as hybrid_retriever:
        reranked = RerankedRetriever(hybrid_retriever, reranker)
        retriever = QueryRewritingRetriever(reranked, rewriter)
        results = retriever.retrieve(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            conversation_history=args.history,
        )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
