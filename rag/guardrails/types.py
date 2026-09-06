"""Shared contracts for deterministic RAG guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailLayer(str, Enum):
    """Where a guardrail decision was made in the RAG flow."""

    INPUT = "input"
    CONTEXT = "context"
    OUTPUT = "output"
    PII = "pii"


class GuardrailCategory(str, Enum):
    """Supported guardrail finding categories."""

    CRISIS = "crisis"
    MEDICAL_EMERGENCY = "medical_emergency"
    PROMPT_INJECTION = "prompt_injection"
    DIAGNOSIS_BOUNDARY = "diagnosis_boundary"
    MEDICATION_BOUNDARY = "medication_boundary"
    PII = "pii"
    LOW_EVIDENCE = "low_evidence"
    SOURCE_POLICY = "source_policy"
    CITATION_POLICY = "citation_policy"
    GROUNDEDNESS = "groundedness"
    SAFETY_POLICY = "safety_policy"


class GuardrailSeverity(str, Enum):
    """How serious a guardrail finding is."""

    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


class GuardrailAction(str, Enum):
    """Recommended next action for the RAG pipeline."""

    ALLOW = "allow"
    ALLOW_WITH_WARNINGS = "allow_with_warnings"
    ROUTE_CRISIS = "route_crisis"
    ROUTE_MEDICAL = "route_medical"
    BOUNDARY_RESPONSE = "boundary_response"
    ABSTAIN = "abstain"
    BLOCK_OUTPUT = "block_output"


@dataclass(frozen=True)
class GuardrailFinding:
    """One concrete guardrail finding with optional evidence."""

    layer: GuardrailLayer
    category: GuardrailCategory
    severity: GuardrailSeverity
    message: str
    matched_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardrailDecision:
    """A guardrail layer's decision and findings."""

    action: GuardrailAction
    findings: list[GuardrailFinding] = field(default_factory=list)
    safe_response: str | None = None
    redacted_text: str | None = None

    @property
    def should_continue(self) -> bool:
        """Return True when the RAG flow can continue."""

        return self.action in {
            GuardrailAction.ALLOW,
            GuardrailAction.ALLOW_WITH_WARNINGS,
        }

    @property
    def is_blocked(self) -> bool:
        """Return True when normal RAG should not continue."""

        return not self.should_continue


def allow_decision(
    findings: list[GuardrailFinding] | None = None,
    redacted_text: str | None = None,
) -> GuardrailDecision:
    """Build an allow or allow-with-warnings decision from findings."""

    current_findings = findings or []
    action = (
        GuardrailAction.ALLOW_WITH_WARNINGS
        if current_findings
        else GuardrailAction.ALLOW
    )
    return GuardrailDecision(
        action=action,
        findings=current_findings,
        redacted_text=redacted_text,
    )
