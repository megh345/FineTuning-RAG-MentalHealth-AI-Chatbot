"""Retrieval and assembled-context guardrails."""

from __future__ import annotations

from datetime import date
from urllib.parse import urlparse

from rag.guardrails.messages import low_evidence_response
from rag.guardrails.types import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailFinding,
    GuardrailLayer,
    GuardrailSeverity,
    allow_decision,
)
from rag.schemas import AssembledContext, ContextCitation


class ContextGuardrail:
    """Validate retrieved context before generation."""

    def __init__(
        self,
        min_citations: int = 1,
        min_context_chars: int = 80,
        stale_after_days: int = 1095,
    ) -> None:
        if min_citations < 1:
            raise ValueError("min_citations must be at least 1")
        if min_context_chars < 1:
            raise ValueError("min_context_chars must be at least 1")
        if stale_after_days < 1:
            raise ValueError("stale_after_days must be at least 1")

        self.min_citations = min_citations
        self.min_context_chars = min_context_chars
        self.stale_after_days = stale_after_days

    def validate(self, context: AssembledContext) -> GuardrailDecision:
        findings: list[GuardrailFinding] = []

        if not context.context_text.strip():
            findings.append(
                _finding(
                    GuardrailCategory.LOW_EVIDENCE,
                    GuardrailSeverity.BLOCK,
                    "No assembled context is available.",
                )
            )
        if len(context.citations) < self.min_citations:
            findings.append(
                _finding(
                    GuardrailCategory.LOW_EVIDENCE,
                    GuardrailSeverity.BLOCK,
                    "Not enough cited evidence was retrieved.",
                    metadata={
                        "required_citations": self.min_citations,
                        "actual_citations": len(context.citations),
                    },
                )
            )
        if len(context.context_text.strip()) < self.min_context_chars:
            findings.append(
                _finding(
                    GuardrailCategory.LOW_EVIDENCE,
                    GuardrailSeverity.WARNING,
                    "Assembled context is very short.",
                    metadata={
                        "required_chars": self.min_context_chars,
                        "actual_chars": len(context.context_text.strip()),
                    },
                )
            )

        findings.extend(self._validate_citation_consistency(context))
        for citation in context.citations:
            findings.extend(self._validate_source_metadata(citation))

        if any(finding.severity == GuardrailSeverity.BLOCK for finding in findings):
            return GuardrailDecision(
                action=GuardrailAction.ABSTAIN,
                findings=findings,
                safe_response=low_evidence_response(),
            )
        return allow_decision(findings=findings)

    def _validate_citation_consistency(
        self,
        context: AssembledContext,
    ) -> list[GuardrailFinding]:
        findings: list[GuardrailFinding] = []
        included_chunk_ids = set(context.included_chunk_ids)
        seen_markers: set[str] = set()
        seen_chunk_ids: set[str] = set()

        for expected_index, citation in enumerate(context.citations, start=1):
            expected_marker = f"[C{expected_index}]"
            if citation.marker != expected_marker:
                findings.append(
                    _finding(
                        GuardrailCategory.CITATION_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Citation markers must be sequential and stable.",
                        matched_text=citation.marker,
                        metadata={"expected_marker": expected_marker},
                    )
                )
            if citation.marker in seen_markers:
                findings.append(
                    _finding(
                        GuardrailCategory.CITATION_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Duplicate citation marker found.",
                        matched_text=citation.marker,
                    )
                )
            if citation.chunk_id in seen_chunk_ids:
                findings.append(
                    _finding(
                        GuardrailCategory.CITATION_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Duplicate cited chunk found.",
                        matched_text=citation.chunk_id,
                    )
                )
            if citation.chunk_id not in included_chunk_ids:
                findings.append(
                    _finding(
                        GuardrailCategory.CITATION_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Citation does not map to an included chunk.",
                        matched_text=citation.chunk_id,
                    )
                )

            seen_markers.add(citation.marker)
            seen_chunk_ids.add(citation.chunk_id)

        return findings

    def _validate_source_metadata(
        self,
        citation: ContextCitation,
    ) -> list[GuardrailFinding]:
        findings: list[GuardrailFinding] = []
        required_fields = {
            "title": citation.title,
            "section": citation.section,
            "source_name": citation.source_name,
            "source_url": citation.source_url,
            "source_path": citation.source_path,
        }
        for field_name, value in required_fields.items():
            if not value:
                findings.append(
                    _finding(
                        GuardrailCategory.SOURCE_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Citation is missing required source metadata.",
                        matched_text=citation.marker,
                        metadata={"missing_field": field_name},
                    )
                )

        if citation.source_path and not citation.source_path.startswith(
            "knowledge_base/raw/"
        ):
            findings.append(
                _finding(
                    GuardrailCategory.SOURCE_POLICY,
                    GuardrailSeverity.BLOCK,
                    "Citation source path is outside the reviewed knowledge base.",
                    matched_text=citation.source_path,
                )
            )

        if citation.source_url and not _is_allowed_source_url(citation.source_url):
            findings.append(
                _finding(
                    GuardrailCategory.SOURCE_POLICY,
                    GuardrailSeverity.BLOCK,
                    "Citation source URL uses an unauthorized scheme.",
                    matched_text=citation.source_url,
                )
            )

        if not citation.reviewed_on:
            findings.append(
                _finding(
                    GuardrailCategory.SOURCE_POLICY,
                    GuardrailSeverity.WARNING,
                    "Citation has no review date.",
                    matched_text=citation.marker,
                )
            )
        else:
            reviewed_on = _parse_reviewed_on(citation.reviewed_on)
            if reviewed_on is None:
                findings.append(
                    _finding(
                        GuardrailCategory.SOURCE_POLICY,
                        GuardrailSeverity.WARNING,
                        "Citation review date could not be parsed.",
                        matched_text=citation.reviewed_on,
                    )
                )
            elif (date.today() - reviewed_on).days > self.stale_after_days:
                findings.append(
                    _finding(
                        GuardrailCategory.SOURCE_POLICY,
                        GuardrailSeverity.WARNING,
                        "Citation review date is older than the configured window.",
                        matched_text=citation.reviewed_on,
                        metadata={"stale_after_days": self.stale_after_days},
                    )
                )

        return findings


def _finding(
    category: GuardrailCategory,
    severity: GuardrailSeverity,
    message: str,
    matched_text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> GuardrailFinding:
    return GuardrailFinding(
        layer=GuardrailLayer.CONTEXT,
        category=category,
        severity=severity,
        message=message,
        matched_text=matched_text,
        metadata=metadata or {},
    )


def _is_allowed_source_url(source_url: str) -> bool:
    parsed = urlparse(source_url)
    return parsed.scheme in {"http", "https", "internal"}


def _parse_reviewed_on(reviewed_on: str) -> date | None:
    try:
        return date.fromisoformat(reviewed_on)
    except ValueError:
        return None
