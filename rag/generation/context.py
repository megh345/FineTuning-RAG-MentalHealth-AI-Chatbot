"""Assemble retrieved chunks into prompt context with citation metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rag.config import DEFAULT_CONTEXT_MAX_CHARS, DEFAULT_CONTEXT_TOP_K
from rag.schemas import AssembledContext, ContextCitation, RetrievalResult


class ContextAssembler:
    """Create a bounded context block from ranked retrieval results.

    Retrieval and reranking decide which chunks are most relevant. This class
    keeps the next responsibility separate: formatting those chunks so an answer
    generator can quote them with stable citation markers such as [C1].
    """

    def __init__(
        self,
        max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
        max_chunks: int = DEFAULT_CONTEXT_TOP_K,
    ) -> None:
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be at least 1")
        if max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")

        self.max_context_chars = max_context_chars
        self.max_chunks = max_chunks

    def assemble(
        self,
        query: str,
        results: Iterable[RetrievalResult],
        top_k: int | None = None,
    ) -> AssembledContext:
        """Return a prompt-ready context block and citation list.

        The assembler preserves retrieval order, removes duplicate chunk IDs,
        skips chunks without text, and stops adding chunks when the context
        character budget would be exceeded.
        """

        if not query.strip():
            raise ValueError("Context assembly query cannot be empty")

        requested_top_k = top_k if top_k is not None else self.max_chunks
        if requested_top_k < 1:
            raise ValueError("top_k must be at least 1")
        selected_limit = min(requested_top_k, self.max_chunks)

        context_blocks: list[str] = []
        citations: list[ContextCitation] = []
        included_chunk_ids: list[str] = []
        omitted_chunk_ids: list[str] = []
        seen_chunk_ids: set[str] = set()

        for result in results:
            if result.chunk_id in seen_chunk_ids:
                omitted_chunk_ids.append(result.chunk_id)
                continue
            if not result.text or not result.text.strip():
                omitted_chunk_ids.append(result.chunk_id)
                continue
            if len(citations) >= selected_limit:
                omitted_chunk_ids.append(result.chunk_id)
                continue

            citation_id = len(citations) + 1
            marker = f"[C{citation_id}]"
            citation = self._citation_from_result(
                citation_id=citation_id,
                marker=marker,
                result=result,
            )
            block = self._format_context_block(citation, result.text)
            candidate_context = "\n\n".join([*context_blocks, block])
            if len(candidate_context) > self.max_context_chars:
                omitted_chunk_ids.append(result.chunk_id)
                continue

            seen_chunk_ids.add(result.chunk_id)
            citations.append(citation)
            context_blocks.append(block)
            included_chunk_ids.append(result.chunk_id)

        return AssembledContext(
            query=query,
            context_text="\n\n".join(context_blocks),
            citations=citations,
            included_chunk_ids=included_chunk_ids,
            omitted_chunk_ids=omitted_chunk_ids,
            max_context_chars=self.max_context_chars,
        )

    def _citation_from_result(
        self,
        citation_id: int,
        marker: str,
        result: RetrievalResult,
    ) -> ContextCitation:
        source_metadata = _source_metadata(result.metadata)
        return ContextCitation(
            citation_id=citation_id,
            marker=marker,
            chunk_id=result.chunk_id,
            title=_string_value(
                source_metadata.get("document_title")
                or result.metadata.get("document_title"),
                default="Untitled source",
            ),
            section=_string_value(result.metadata.get("section")),
            source_name=_string_value(
                source_metadata.get("source_name")
                or result.metadata.get("source_name")
            ),
            source_url=_string_value(
                source_metadata.get("source_url")
                or result.metadata.get("source_url")
            ),
            source_path=_string_value(result.metadata.get("source")),
            reviewed_on=_optional_string(
                source_metadata.get("reviewed_on")
                or result.metadata.get("reviewed_on")
            ),
            topic=_string_value(result.metadata.get("topic")),
            risk_category=_string_value(result.metadata.get("risk_category")),
            score=result.score,
            metadata={
                "content_type": result.metadata.get("content_type"),
                "page": result.metadata.get("page"),
                "retrieval_score": result.metadata.get("retrieval_score"),
                "reranker_model": result.metadata.get("reranker_model"),
                "reranker_model_revision": result.metadata.get(
                    "reranker_model_revision"
                ),
            },
        )

    def _format_context_block(self, citation: ContextCitation, text: str) -> str:
        header = f"{citation.marker} {citation.title}"
        if citation.section:
            header = f"{header} | {citation.section}"

        details = []
        if citation.source_name:
            details.append(f"Source: {citation.source_name}")
        if citation.reviewed_on:
            details.append(f"Reviewed: {citation.reviewed_on}")
        if citation.topic:
            details.append(f"Topic: {citation.topic}")
        if citation.risk_category:
            details.append(f"Risk: {citation.risk_category}")

        lines = [header]
        if details:
            lines.append("; ".join(details))
        lines.append(_normalize_text(text))
        return "\n".join(lines)


def _source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, dict):
        return nested_metadata
    return {}


def _string_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
