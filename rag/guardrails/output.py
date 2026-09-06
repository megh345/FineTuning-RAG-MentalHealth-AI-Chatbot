"""Output guardrails for grounded, cited RAG answers."""

from __future__ import annotations

import re

from rag.guardrails.messages import output_blocked_response
from rag.guardrails.types import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailFinding,
    GuardrailLayer,
    GuardrailSeverity,
    allow_decision,
)
from rag.schemas import AssembledContext


class OutputGuardrail:
    """Validate generated answers against context and safety boundaries."""

    def __init__(
        self,
        require_citations: bool = True,
        min_grounding_overlap: float = 0.18,
    ) -> None:
        if min_grounding_overlap < 0 or min_grounding_overlap > 1:
            raise ValueError("min_grounding_overlap must be between 0 and 1")

        self.require_citations = require_citations
        self.min_grounding_overlap = min_grounding_overlap

    def validate(
        self,
        answer: str,
        context: AssembledContext,
        input_decision: GuardrailDecision | None = None,
    ) -> GuardrailDecision:
        findings: list[GuardrailFinding] = []
        normalized_answer = answer or ""

        if not normalized_answer.strip():
            findings.append(
                _finding(
                    GuardrailCategory.LOW_EVIDENCE,
                    GuardrailSeverity.BLOCK,
                    "Generated answer is empty.",
                )
            )
        findings.extend(self._validate_citations(normalized_answer, context))
        findings.extend(_unsafe_output_findings(normalized_answer))
        findings.extend(_crisis_policy_findings(normalized_answer, input_decision))
        findings.extend(self._groundedness_findings(normalized_answer, context))

        if any(finding.severity == GuardrailSeverity.BLOCK for finding in findings):
            return GuardrailDecision(
                action=GuardrailAction.BLOCK_OUTPUT,
                findings=findings,
                safe_response=output_blocked_response(),
            )
        return allow_decision(findings=findings)

    def _validate_citations(
        self,
        answer: str,
        context: AssembledContext,
    ) -> list[GuardrailFinding]:
        findings: list[GuardrailFinding] = []
        valid_markers = {citation.marker for citation in context.citations}
        answer_markers = set(CITATION_PATTERN.findall(answer))
        legacy_source_markers = set(LEGACY_SOURCE_PATTERN.findall(answer))

        if self.require_citations and valid_markers and _requires_citation(answer):
            if not answer_markers:
                findings.append(
                    _finding(
                        GuardrailCategory.CITATION_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Answer makes support or factual claims without citations.",
                    )
                )

        for marker in sorted(answer_markers):
            if marker not in valid_markers:
                findings.append(
                    _finding(
                        GuardrailCategory.CITATION_POLICY,
                        GuardrailSeverity.BLOCK,
                        "Answer cites a marker that was not retrieved.",
                        matched_text=marker,
                    )
                )

        for marker in sorted(legacy_source_markers):
            findings.append(
                _finding(
                    GuardrailCategory.CITATION_POLICY,
                    GuardrailSeverity.BLOCK,
                    "Answer uses an unsupported citation label.",
                    matched_text=marker,
                )
            )

        return findings

    def _groundedness_findings(
        self,
        answer: str,
        context: AssembledContext,
    ) -> list[GuardrailFinding]:
        if not answer.strip() or not context.context_text.strip():
            return []

        answer_terms = _content_terms(answer)
        if len(answer_terms) < 6:
            return []

        context_terms = _content_terms(context.context_text)
        if not context_terms:
            return [
                _finding(
                    GuardrailCategory.GROUNDEDNESS,
                    GuardrailSeverity.BLOCK,
                    "No context terms are available for groundedness validation.",
                )
            ]

        overlap = len(answer_terms & context_terms) / len(answer_terms)
        if overlap < self.min_grounding_overlap:
            return [
                _finding(
                    GuardrailCategory.GROUNDEDNESS,
                    GuardrailSeverity.WARNING,
                    "Answer has low lexical overlap with retrieved context.",
                    metadata={
                        "overlap": round(overlap, 3),
                        "min_overlap": self.min_grounding_overlap,
                    },
                )
            ]
        return []


CITATION_PATTERN = re.compile(r"\[C\d+\]")
LEGACY_SOURCE_PATTERN = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)

UNSAFE_OUTPUT_PATTERNS = [
    (
        GuardrailCategory.DIAGNOSIS_BOUNDARY,
        "Answer makes a diagnosis or definitive clinical label.",
        re.compile(
            r"\byou\s+(?:have|are diagnosed with|are suffering from)\s+"
            r"(?:depression|anxiety|panic\s+disorder|bipolar|ptsd|ocd)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GuardrailCategory.DIAGNOSIS_BOUNDARY,
        "Answer states symptoms are definitely a specific condition.",
        re.compile(
            r"\bthis\s+is\s+definitely\s+"
            r"(?:anxiety|a\s+panic\s+attack|depression|ptsd|ocd)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GuardrailCategory.MEDICATION_BOUNDARY,
        "Answer gives medication or dosage instructions.",
        re.compile(
            r"\b(?:take|start|stop|increase|decrease|change)\s+"
            r"(?:taking\s+)?"
            r"(?:\d+\s?(?:mg|milligrams)\s+)?"
            r"(?:your\s+)?(?:medication|medicine|antidepressant|dose|dosage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GuardrailCategory.MEDICATION_BOUNDARY,
        "Answer claims prescribing or medication recommendation authority.",
        re.compile(
            r"\b(?:i\s+can\s+)?(?:prescribe|recommend)\s+"
            r"(?:medication|medicine|meds?|antidepressants?|dosage|dose)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GuardrailCategory.MEDICATION_BOUNDARY,
        "Answer gives specific dosage instructions.",
        re.compile(r"\btake\s+\d+\s?(?:mg|milligrams)\b", re.IGNORECASE),
    ),
    (
        GuardrailCategory.SAFETY_POLICY,
        "Answer makes an unsupported guaranteed outcome claim.",
        re.compile(
            r"\b(?:guaranteed|will\s+(?:cure|stop|fix))\b.*"
            r"\b(?:panic|anxiety|depression|self[- ]?harm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GuardrailCategory.SAFETY_POLICY,
        "Answer promises a symptom will stop within a fixed time.",
        re.compile(
            r"\bwill\s+(?:stop|end|fix|cure)\b.*\bwithin\s+\d+\s+"
            r"(?:seconds|minutes|hours)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GuardrailCategory.SAFETY_POLICY,
        "Answer discourages needed medical or professional help.",
        re.compile(
            r"\byou\s+do\s+not\s+need\s+"
            r"(?:medical|professional|emergency|crisis)\s+help\b",
            re.IGNORECASE,
        ),
    ),
]

CRISIS_ROUTING_PATTERN = re.compile(
    r"\b(?:988|911|emergency|crisis|trusted person|immediate danger)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "being",
    "could",
    "every",
    "feeling",
    "from",
    "have",
    "help",
    "helps",
    "into",
    "just",
    "like",
    "more",
    "some",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "things",
    "this",
    "through",
    "when",
    "where",
    "while",
    "with",
    "would",
    "your",
    "yourself",
}


def _unsafe_output_findings(answer: str) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    for category, message, pattern in UNSAFE_OUTPUT_PATTERNS:
        match = pattern.search(answer)
        if match:
            findings.append(
                _finding(
                    category,
                    GuardrailSeverity.BLOCK,
                    message,
                    matched_text=match.group(0),
                )
            )
    return findings


def _crisis_policy_findings(
    answer: str,
    input_decision: GuardrailDecision | None,
) -> list[GuardrailFinding]:
    if input_decision is None:
        return []
    if input_decision.action != GuardrailAction.ROUTE_CRISIS:
        return []
    if CRISIS_ROUTING_PATTERN.search(answer):
        return []
    return [
        _finding(
            GuardrailCategory.SAFETY_POLICY,
            GuardrailSeverity.BLOCK,
            "Crisis-risk input requires crisis routing language in the answer.",
        )
    ]


def _requires_citation(answer: str) -> bool:
    content_terms = _content_terms(answer)
    return len(content_terms) >= 4


def _content_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text.lower()):
        normalized = token.strip("'")
        if len(normalized) < 4 or normalized in STOPWORDS:
            continue
        terms.add(normalized)
    return terms


def _finding(
    category: GuardrailCategory,
    severity: GuardrailSeverity,
    message: str,
    matched_text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> GuardrailFinding:
    return GuardrailFinding(
        layer=GuardrailLayer.OUTPUT,
        category=category,
        severity=severity,
        message=message,
        matched_text=matched_text,
        metadata=metadata or {},
    )
