"""Dense embedding adapter used by indexing and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag.config import DENSE_EMBEDDING_MODEL


class DenseEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local bi-encoder with normalized vectors for cosine similarity."""

    def __init__(self, model_name: str = DENSE_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers from rag/requirements.txt"
            ) from error

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dimension = get_dimension()
        if dimension is None:
            raise ValueError(f"Embedding dimension is unavailable for {model_name}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector)
