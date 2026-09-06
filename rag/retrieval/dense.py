"""Dense-only retrieval against the Stage 2 Qdrant collection."""

from __future__ import annotations

from pathlib import Path

from rag.config import (
    DEFAULT_RETRIEVAL_TOP_K,
    DENSE_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    QDRANT_STORAGE_DIR,
)
from rag.indexing.embeddings import DenseEmbedder
from rag.indexing.qdrant_store import create_local_client
from rag.retrieval.qdrant_results import to_retrieval_results
from rag.schemas import RetrievalResult


class DenseRetriever:
    def __init__(
        self,
        embedder: DenseEmbedder,
        storage_path: Path = QDRANT_STORAGE_DIR,
        collection_name: str = DENSE_COLLECTION_NAME,
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = create_local_client(storage_path)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "DenseRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        points = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedder.embed_query(query).tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        ).points
        return to_retrieval_results(points)
