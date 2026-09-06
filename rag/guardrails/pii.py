"""Dependency-free PII redaction for logs and evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


@dataclass(frozen=True)
class PIIEntity:
    """One detected PII-like span."""

    entity_type: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class PIIRedactionResult:
    """Original text redacted for safer logging."""

    original_text: str
    redacted_text: str
    entities: list[PIIEntity]


class PIIRedactor(Protocol):
    """Protocol for pluggable PII redactors."""

    def redact(self, text: str) -> PIIRedactionResult: ...


class RegexPIIRedactor:
    """Simple baseline PII redactor.

    This is intentionally conservative and dependency-free. It is not a privacy
    guarantee; it only reduces accidental leakage in logs and eval traces.
    """

    PATTERNS = [
        (
            "EMAIL",
            re.compile(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                re.IGNORECASE,
            ),
        ),
        (
            "SSN",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
        (
            "PHONE_NUMBER",
            re.compile(
                r"(?<!\w)(?:\+?1[-.\s]?)?"
                r"(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
            ),
        ),
        (
            "CREDIT_CARD",
            re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        ),
    ]

    def redact(self, text: str) -> PIIRedactionResult:
        entities: list[PIIEntity] = []
        spans: list[tuple[int, int, str]] = []

        for entity_type, pattern in self.PATTERNS:
            for match in pattern.finditer(text or ""):
                start, end = match.span()
                if _overlaps_existing_span(start, end, spans):
                    continue
                spans.append((start, end, entity_type))
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        text=match.group(0),
                    )
                )

        redacted_text = text or ""
        for start, end, entity_type in sorted(spans, reverse=True):
            redacted_text = (
                f"{redacted_text[:start]}[{entity_type}]"
                f"{redacted_text[end:]}"
            )

        return PIIRedactionResult(
            original_text=text or "",
            redacted_text=redacted_text,
            entities=sorted(entities, key=lambda entity: entity.start),
        )


def _overlaps_existing_span(
    start: int,
    end: int,
    spans: list[tuple[int, int, str]],
) -> bool:
    for current_start, current_end, _ in spans:
        if start < current_end and end > current_start:
            return True
    return False
