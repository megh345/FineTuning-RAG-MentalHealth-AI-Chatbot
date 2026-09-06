"""Qdrant collection creation and chunk indexing."""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.config import (
    DENSE_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    QDRANT_STORAGE_DIR,
)
from rag.indexing.embeddings import DenseEmbedder
from rag.schemas import DocumentChunk


POINT_NAMESPACE = uuid.UUID("a770e4c0-d17b-4d97-b85a-b193bf103c62")


def point_id_for_chunk(chunk_id: str) -> str:
    """Map a readable chunk ID to the stable UUID format Qdrant accepts."""

    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def create_local_client(storage_path: Path = QDRANT_STORAGE_DIR):
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise RuntimeError(
            "Install qdrant-client from rag/requirements.txt"
        ) from error
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def build_dense_index(
    chunks: list[DocumentChunk],
    embedder: DenseEmbedder,
    storage_path: Path = QDRANT_STORAGE_DIR,
    collection_name: str = DENSE_COLLECTION_NAME,
) -> None:
    """Rebuild the Stage 2 collection from the complete processed corpus."""

    from qdrant_client import models

    if not chunks:
        raise ValueError("Cannot build a dense index with no chunks")

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    if vectors.shape != (len(chunks), embedder.dimension):
        raise ValueError(
            f"Unexpected embedding shape {vectors.shape}; "
            f"expected {(len(chunks), embedder.dimension)}"
        )

    client = create_local_client(storage_path)
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=embedder.dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
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
                    vector={DENSE_VECTOR_NAME: vector.tolist()},
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
