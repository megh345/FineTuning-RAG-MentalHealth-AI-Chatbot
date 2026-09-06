"""Structure-aware Markdown and text chunking for the Stage 1 corpus."""

from __future__ import annotations

import re

from rag.schemas import DocumentChunk, SourceDocument


HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$")
CHUNK_ID_PATTERN = re.compile(r"^<!--\s*chunk_id:\s*([a-z0-9_]+)\s*-->$")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph and list boundaries."""

    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = is_blank
    return "\n".join(cleaned).strip()


def chunk_document(document: SourceDocument) -> list[DocumentChunk]:
    """Create one stable chunk per level-two Markdown section.

    Stage 1 deliberately authors semantic sections at retrieval granularity.
    This avoids splitting safety rules across chunks and makes gold labels
    stable and reviewable.
    """

    lines = document.text.splitlines()
    sections: list[tuple[str, str, list[str]]] = []
    section_title: str | None = None
    chunk_id: str | None = None
    section_lines: list[str] = []

    def finish_section() -> None:
        nonlocal section_title, chunk_id, section_lines
        if section_title is None:
            return
        if chunk_id is None:
            raise ValueError(
                f"{document.source_path}: section '{section_title}' has no chunk_id"
            )
        sections.append((section_title, chunk_id, section_lines))

    for line in lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            finish_section()
            section_title = heading_match.group(1)
            chunk_id = None
            section_lines = []
            continue

        id_match = CHUNK_ID_PATTERN.match(line.strip())
        if id_match and section_title is not None:
            if chunk_id is not None:
                raise ValueError(
                    f"{document.source_path}: duplicate chunk_id in '{section_title}'"
                )
            chunk_id = id_match.group(1)
            continue

        if section_title is not None:
            section_lines.append(line)

    finish_section()
    if not sections:
        raise ValueError(f"No level-two sections found in {document.source_path}")

    chunks: list[DocumentChunk] = []
    for section, stable_id, body_lines in sections:
        text = clean_text("\n".join(body_lines))
        if not text:
            raise ValueError(f"{document.source_path}: empty section '{section}'")
        chunks.append(
            DocumentChunk(
                chunk_id=stable_id,
                text=text,
                source=document.source_path,
                topic=document.topic,
                section=section,
                content_type=document.content_type,
                risk_category=document.risk_category,
                metadata={
                    "document_id": document.document_id,
                    "document_title": document.title,
                    "source_name": document.source_name,
                    "source_url": document.source_url,
                    "reviewed_on": document.reviewed_on,
                    **document.metadata,
                },
            )
        )
    return chunks


def chunk_documents(documents: list[SourceDocument]) -> list[DocumentChunk]:
    """Chunk all documents and reject duplicate stable IDs."""

    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            duplicates.add(chunk.chunk_id)
        seen.add(chunk.chunk_id)
    if duplicates:
        raise ValueError(f"Duplicate chunk IDs: {sorted(duplicates)}")
    return chunks
