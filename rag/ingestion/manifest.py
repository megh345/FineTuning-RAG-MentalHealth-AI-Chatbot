"""Load and validate the controlled knowledge-base manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rag.config import KNOWLEDGE_MANIFEST_PATH, RAG_ROOT
from rag.schemas import SourceDocument


REQUIRED_FIELDS = {
    "document_id",
    "file",
    "title",
    "topic",
    "content_type",
    "risk_category",
    "source_name",
    "source_url",
    "reviewed_on",
}


def load_source_documents(
    manifest_path: Path = KNOWLEDGE_MANIFEST_PATH,
) -> list[SourceDocument]:
    """Read every manifest entry and its corresponding UTF-8 source file."""

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Manifest has no sources: {manifest_path}")

    documents: list[SourceDocument] = []
    seen_ids: set[str] = set()
    for position, entry in enumerate(sources, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest source {position} must be a mapping")

        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(
                f"Manifest source {position} is missing: {sorted(missing)}"
            )

        document_id = str(entry["document_id"])
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id in manifest: {document_id}")
        seen_ids.add(document_id)

        source_path = (manifest_path.parent / str(entry["file"])).resolve()
        if not source_path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError(
                f"Source path escapes the knowledge-base directory: {source_path}"
            )
        if source_path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(f"Unsupported Stage 1 source type: {source_path.suffix}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Manifest source does not exist: {source_path}")

        known_fields = REQUIRED_FIELDS | {"notes"}
        extra_metadata: dict[str, Any] = {
            key: value for key, value in entry.items() if key not in known_fields
        }
        documents.append(
            SourceDocument(
                document_id=document_id,
                title=str(entry["title"]),
                text=source_path.read_text(encoding="utf-8"),
                source_path=str(source_path.relative_to(RAG_ROOT)),
                topic=str(entry["topic"]),
                content_type=str(entry["content_type"]),
                risk_category=str(entry["risk_category"]),
                source_name=str(entry["source_name"]),
                source_url=str(entry["source_url"]),
                reviewed_on=str(entry["reviewed_on"]),
                metadata=extra_metadata,
            )
        )

    return documents
