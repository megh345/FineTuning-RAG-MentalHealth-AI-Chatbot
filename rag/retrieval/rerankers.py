"""Second-stage rerankers for retrieval candidates."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rag.config import (
    RERANKER_BATCH_SIZE,
    RERANKER_CACHE_DIR,
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
)
from rag.schemas import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]: ...


class CrossEncoderReranker:
    """Score query-chunk pairs jointly and reorder hybrid candidates."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        batch_size: int = RERANKER_BATCH_SIZE,
        model: Any | None = None,
        revision: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model_name
        self.model_revision = revision or (
            RERANKER_MODEL_REVISION if model_name == RERANKER_MODEL else None
        )
        self.batch_size = batch_size
        if model is not None:
            self._model = model
            return

        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers from rag/requirements.txt"
            ) from error

        RERANKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(
            repo_id=model_name,
            revision=self.model_revision,
            cache_dir=str(RERANKER_CACHE_DIR),
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            ],
        )
        self._model = CrossEncoder(
            model_path,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Reranking query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > len(candidates):
            raise ValueError("top_k cannot exceed the number of candidates")

        passages = []
        for candidate in candidates:
            if not candidate.text:
                raise ValueError(
                    f"Candidate {candidate.chunk_id!r} has no text to rerank"
                )
            passages.append((query, candidate.text))

        scores = np.asarray(
            self._model.predict(
                passages,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(
                f"Expected {len(candidates)} reranker scores, got {len(scores)}"
            )

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        results = []
        for candidate, score in ranked[:top_k]:
            metadata = dict(candidate.metadata)
            metadata["retrieval_score"] = candidate.score
            metadata["reranker_model"] = self.model_name
            if self.model_revision is not None:
                metadata["reranker_model_revision"] = self.model_revision
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    score=float(score),
                    text=candidate.text,
                    metadata=metadata,
                )
            )
        return results
