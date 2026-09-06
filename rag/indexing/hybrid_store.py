"""Build a Qdrant collection with dense and BM25 sparse vectors."""

from __future__ import annotations

from pathlib import Path

from rag.config import (
    DENSE_VECTOR_NAME,
    HYBRID_COLLECTION_NAME,
    QDRANT_STORAGE_DIR,
    SPARSE_VECTOR_NAME,
)
from rag.indexing.embeddings import DenseEmbedder
from rag.indexing.qdrant_store import create_local_client, point_id_for_chunk
from rag.indexing.sparse_embeddings import SparseEmbedder
from rag.schemas import DocumentChunk


def build_hybrid_index(
    chunks: list[DocumentChunk],
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
    storage_path: Path = QDRANT_STORAGE_DIR,
    collection_name: str = HYBRID_COLLECTION_NAME,
) -> None:
    """Rebuild a collection containing both dense and sparse representations."""

    from qdrant_client import models

    if not chunks:
        raise ValueError("Cannot build a hybrid index with no chunks")

    texts = [chunk.text for chunk in chunks]
    dense_vectors = dense_embedder.embed_documents(texts)
    sparse_vectors = sparse_embedder.embed_documents(texts)
    if dense_vectors.shape != (len(chunks), dense_embedder.dimension):
        raise ValueError(
            f"Unexpected dense embedding shape {dense_vectors.shape}; "
            f"expected {(len(chunks), dense_embedder.dimension)}"
        )
    if len(sparse_vectors) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} sparse vectors, got {len(sparse_vectors)}"
        )

    client = create_local_client(storage_path)
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_embedder.dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        points = []
        for chunk, dense_vector, sparse_vector in zip(
            chunks,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "topic": chunk.topic,
                "section": chunk.section,
                "content_type": chunk.content_type,
                "risk_category": chunk.risk_category,
                "page": chunk.page,
                "metadata": chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=point_id_for_chunk(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector.tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                    },
                    payload=payload,
                )
            )
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
    finally:
        client.close()
