"""Build the Stage 3 dense-and-sparse Qdrant collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import HYBRID_COLLECTION_NAME, QDRANT_STORAGE_DIR
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.indexing.hybrid_store import build_hybrid_index
from rag.indexing.sparse_embeddings import FastEmbedBm25Embedder
from rag.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", type=Path, default=QDRANT_STORAGE_DIR)
    parser.add_argument("--collection", default=HYBRID_COLLECTION_NAME)
    args = parser.parse_args()

    chunks = run_ingestion()
    dense_embedder = SentenceTransformerEmbedder()
    sparse_embedder = FastEmbedBm25Embedder()
    build_hybrid_index(
        chunks,
        dense_embedder,
        sparse_embedder,
        args.storage_path,
        args.collection,
    )
    print(
        f"Indexed {len(chunks)} dense+sparse chunks in '{args.collection}' "
        f"at {args.storage_path}"
    )


if __name__ == "__main__":
    main()
