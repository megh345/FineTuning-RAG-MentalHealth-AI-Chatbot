"""Inspect Stage 5 assembled context and citations for one query."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.config import (
    DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_CONTEXT_TOP_K,
    DEFAULT_HYBRID_CANDIDATE_K,
)
from rag.generation.context import ContextAssembler
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.indexing.sparse_embeddings import FastEmbedBm25Embedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranked import RerankedRetriever
from rag.retrieval.rerankers import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONTEXT_TOP_K)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_HYBRID_CANDIDATE_K,
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_CONTEXT_MAX_CHARS,
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Use text for a prompt-readable view; json keeps citation metadata.",
    )
    args = parser.parse_args()

    dense_embedder = SentenceTransformerEmbedder()
    sparse_embedder = FastEmbedBm25Embedder()
    reranker = CrossEncoderReranker()
    assembler = ContextAssembler(
        max_context_chars=args.max_context_chars,
        max_chunks=args.top_k,
    )

    with HybridRetriever(dense_embedder, sparse_embedder) as hybrid_retriever:
        retriever = RerankedRetriever(hybrid_retriever, reranker)
        results = retriever.retrieve(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
    context = assembler.assemble(args.query, results, top_k=args.top_k)

    if args.format == "text":
        print(context.context_text)
        if context.citations:
            print("\nCitations:")
            for citation in context.citations:
                print(
                    f"{citation.marker} {citation.title} | "
                    f"{citation.source_name} | {citation.source_url}"
                )
        return

    print(json.dumps(asdict(context), indent=2))


if __name__ == "__main__":
    main()
