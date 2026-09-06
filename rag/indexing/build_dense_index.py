"""Build the processed corpus and Stage 2 dense Qdrant index."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import DENSE_COLLECTION_NAME, QDRANT_STORAGE_DIR
from rag.indexing.embeddings import SentenceTransformerEmbedder
from rag.indexing.qdrant_store import build_dense_index
from rag.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", type=Path, default=QDRANT_STORAGE_DIR)
    parser.add_argument("--collection", default=DENSE_COLLECTION_NAME)
    args = parser.parse_args()

    chunks = run_ingestion()
    embedder = SentenceTransformerEmbedder()
    build_dense_index(chunks, embedder, args.storage_path, args.collection)
    print(
        f"Indexed {len(chunks)} chunks in '{args.collection}' "
        f"at {args.storage_path}"
    )


if __name__ == "__main__":
    main()
