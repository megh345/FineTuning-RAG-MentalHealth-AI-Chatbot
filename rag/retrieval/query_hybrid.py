"""Inspect Stage 3 hybrid retrieval results for one query."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.config import DEFAULT_HYBRID_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.indexing.sparse_embeddings import FastEmbedBm25Embedder
from rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_HYBRID_CANDIDATE_K,
    )
    args = parser.parse_args()

    dense_embedder = SentenceTransformerEmbedder()
    sparse_embedder = FastEmbedBm25Embedder()
    with HybridRetriever(dense_embedder, sparse_embedder) as retriever:
        results = retriever.retrieve(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
