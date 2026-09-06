"""Run deterministic source loading, cleaning, and chunking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.config import KNOWLEDGE_MANIFEST_PATH, PROCESSED_CHUNKS_PATH
from rag.ingestion.chunking import chunk_documents
from rag.ingestion.manifest import load_source_documents
from rag.schemas import DocumentChunk


def write_chunks(chunks: list[DocumentChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")


def load_chunks(path: Path = PROCESSED_CHUNKS_PATH) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(DocumentChunk(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid chunk at {path}:{line_number}") from error
    return chunks


def run_ingestion(
    manifest_path: Path = KNOWLEDGE_MANIFEST_PATH,
    output_path: Path = PROCESSED_CHUNKS_PATH,
) -> list[DocumentChunk]:
    documents = load_source_documents(manifest_path)
    chunks = chunk_documents(documents)
    write_chunks(chunks, output_path)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=KNOWLEDGE_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_CHUNKS_PATH)
    args = parser.parse_args()

    chunks = run_ingestion(args.manifest, args.output)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
