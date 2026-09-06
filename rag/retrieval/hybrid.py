"""Dense and BM25 sparse retrieval fused with reciprocal rank fusion."""

from __future__ import annotations

from pathlib import Path

from rag.config import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RETRIEVAL_TOP_K,
    DENSE_VECTOR_NAME,
    HYBRID_COLLECTION_NAME,
    QDRANT_STORAGE_DIR,
    SPARSE_VECTOR_NAME,
)
from rag.indexing.embeddings import DenseEmbedder
from rag.indexing.qdrant_store import create_local_client
from rag.indexing.sparse_embeddings import SparseEmbedder
from rag.retrieval.qdrant_results import to_retrieval_results
from rag.schemas import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        storage_path: Path = QDRANT_STORAGE_DIR,
        collection_name: str = HYBRID_COLLECTION_NAME,
    ) -> None:
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.collection_name = collection_name
        self.client = create_local_client(storage_path)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "HybridRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    ) -> list[RetrievalResult]:
        from qdrant_client import models

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")

        dense_vector = self.dense_embedder.embed_query(query)
        sparse_vector = self.sparse_embedder.embed_query(query)
        points = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=candidate_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=candidate_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points
        return to_retrieval_results(points)
