"""Sparse embedding adapter used by hybrid indexing and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.config import FASTEMBED_CACHE_DIR, SPARSE_EMBEDDING_MODEL


@dataclass(frozen=True)
class SparseEmbeddingData:
    """Library-independent sparse vector representation."""

    indices: list[int]
    values: list[float]


class SparseEmbedder(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseEmbeddingData]: ...

    def embed_query(self, text: str) -> SparseEmbeddingData: ...


class FastEmbedBm25Embedder:
    """Local BM25 sparse embeddings compatible with Qdrant's IDF modifier."""

    def __init__(
        self,
        model_name: str = SPARSE_EMBEDDING_MODEL,
        cache_dir: Path = FASTEMBED_CACHE_DIR,
    ) -> None:
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Install the RAG dependencies from rag/requirements.txt"
            ) from error

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseEmbeddingData]:
        vectors = self._model.passage_embed(list(texts))
        return [self._convert(vector) for vector in vectors]

    def embed_query(self, text: str) -> SparseEmbeddingData:
        vectors = list(self._model.query_embed(text))
        if len(vectors) != 1:
            raise ValueError(f"Expected one sparse query vector, got {len(vectors)}")
        return self._convert(vectors[0])

    @staticmethod
    def _convert(vector: object) -> SparseEmbeddingData:
        return SparseEmbeddingData(
            indices=[int(index) for index in vector.indices],
            values=[float(value) for value in vector.values],
        )
