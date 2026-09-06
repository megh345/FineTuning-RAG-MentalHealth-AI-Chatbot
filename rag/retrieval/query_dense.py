"""Inspect Stage 2 dense retrieval results for one query."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.config import DEFAULT_RETRIEVAL_TOP_K
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.retrieval.dense import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    with DenseRetriever(embedder) as retriever:
        results = retriever.retrieve(args.query, top_k=args.top_k)
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
