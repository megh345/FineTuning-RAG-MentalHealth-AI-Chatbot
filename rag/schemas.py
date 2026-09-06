"""Shared data contracts for RAG modules.

The first implementation stages will pass these structures through parsers,
chunkers, retrievers, rerankers, context builders, and evaluators. Keeping the
contracts explicit makes the pipeline easier to test and explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    """One reviewed source document before it is split into chunks."""

    document_id: str
    title: str
    text: str
    source_path: str
    topic: str
    content_type: str
    risk_category: str
    source_name: str
    source_url: str
    reviewed_on: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """One indexed unit of knowledge.

    `chunk_id` must be stable across index rebuilds because retrieval evals use
    it as the gold label. If chunking settings change, record that as a new
    experiment instead of silently reusing incompatible IDs.
    """

    chunk_id: str
    text: str
    source: str
    topic: str
    section: str
    content_type: str
    risk_category: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval result."""

    chunk_id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalExample:
    """One gold retrieval evaluation example."""

    query_id: str
    query: str
    relevant_chunk_ids: set[str]
    topic: str
    query_type: str
    notes: str = ""


@dataclass(frozen=True)
class ContextCitation:
    """One citation marker mapped back to reviewed source metadata."""

    citation_id: int
    marker: str
    chunk_id: str
    title: str
    section: str
    source_name: str
    source_url: str
    source_path: str
    reviewed_on: str | None
    topic: str
    risk_category: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssembledContext:
    """Prompt-ready retrieved context plus machine-readable citations."""

    query: str
    context_text: str
    citations: list[ContextCitation]
    included_chunk_ids: list[str]
    omitted_chunk_ids: list[str]
    max_context_chars: int
